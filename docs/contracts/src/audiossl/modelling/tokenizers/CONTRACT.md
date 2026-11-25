# Tokenizer component contract

Tokenizers convert processor feature sequences into token sequences suitable for the encoder. They may reshape time and feature axes (patchify) or pass through unchanged (`NoOpTokenizer`). Cross-cutting policy: [../CONTRACT.md](../CONTRACT.md).

---

## 1. Purpose and scope

The tokenizer:

- Accepts `AudioBatch` with rank-3 `x` `(B, T_feat, F_in)`.
- Emits `AudioBatch` with `x` `(B, N_tokens, output_dim)` and collapsed mask `(B, N_tokens)`.
- Exposes `output_dim` on its config (via `@property`) for downstream `Encoder(input_dim=...)`.

It does **not** project to `d_model`, apply positional encoding, increase token count (upsample), or run attention. Learnable parameters are forbidden unless a variant config explicitly documents them and follows runtime/init rules (default: **no** learnable params).

---

## 2. Pipeline position

```text
AudioBatch (features)  →  Tokenizer  →  AudioBatch (tokens)  →  Encoder
```

| Direction | Type |
| --- | --- |
| In | `AudioBatch`, `x` `(B, T_feat, F_in)`, mask `(B, T_feat)` |
| Out | `AudioBatch`, `x` `(B, N_tokens, output_dim)`, mask `(B, N_tokens)` |

Upstream: `F_in` typically equals `processor.output_dim`. Downstream: [embedding contract](../embeddings/CONTRACT.md) with `embedding.input_dim = tokenizer.output_dim`.

---

## 3. Config contract

Union (ideal):

```python
TokenizerConfig = PatchifyTokenizerConfig | NoOpTokenizerConfig
```

Every variant:

```python
@dataclass(kw_only=True, frozen=True)
class <Variant>TokenizerConfig:
    type: Literal["<variant>"] = "<variant>"
    ...

    @property
    def output_dim(self) -> int: ...
```

Validators **always** read `config.output_dim`, never a raw private field name.

### `PatchifyTokenizerConfig`

| Field | Contract |
| --- | --- |
| `type` | `Literal["patchify"]` |
| `patch_shape` | `tuple[int, int]` — `(patch_t, patch_f)`, each `>= 1` |
| `padding_strategy` | `Literal["pad", "truncate"]` — **required** tail policy for variable-rate tokenization |
| `stride` | `tuple[int, int] \| None` — optional `(stride_t, stride_f)`, each `>= 1`; effective stride defaults to `patch_shape` |

```python
@property
def output_dim(self) -> int:
    patch_t, patch_f = self.patch_shape
    return patch_t * patch_f
```

### `NoOpTokenizerConfig`

| Field | Contract |
| --- | --- |
| `type` | `Literal["none"]` |
| `_output_dim` | `int` — private stored value |

```python
@property
def output_dim(self) -> int:
    return self._output_dim
```

| Cross-config rule | Enforcer |
| --- | --- |
| `NoOpTokenizerConfig.output_dim == processor.output_dim` | `_validate_model` in `validate_config.py` |

### Validator registration

| Check | Enforcer |
| --- | --- |
| `patch_shape` entries `>= 1` | `_validate_patchify_tokenizer` |
| `stride` entries `>= 1` when provided | `_validate_patchify_tokenizer` |
| `padding_strategy in {"pad", "truncate"}` | validator |
| `NoOpTokenizer` output dim matches processor | `_validate_model` |

`AudioModelConfig.tokenizer` is **`TokenizerConfig` only** — no `str` sentinels (`"none"`, `"None"`, etc.).

**Constructor-time validators** handle only static config-internal consistency. **Input-dependent** shape checks (e.g. divisibility) live in `forward`, not `validate_config`.

---

## 4. Concrete component contract

### 4.1 `__init__` signature

All concrete tokenizers:

```python
def __init__(self, *, config: <TokenizerConfig variant>) -> None:
```

- **No** `d_model`.
- **No** `input_dim` argument.
- Store `self.config`.
- Construct learnable submodules only if variant config declares them (rare).
- Parent `AudioModel` handles `apply_runtime_to_module` on the full tree; tokenizer does not self-apply.

### 4.2 `forward` signature

```python
def forward(self, batch: AudioBatch) -> AudioBatch:
```

#### Shape table (general)

| Stage | `x` shape | `mask` shape | Dtype |
| --- | --- | --- | --- |
| Input | `(B, T_feat, F_in)` | `(B, T_feat)` | `float32` / `bool` |
| Output | `(B, N_tokens, output_dim)` | `(B, N_tokens)` | `float32` / `bool` |

#### Boundary validation (mandatory order)

| # | Check | Exception |
| --- | --- | --- |
| 1 | `isinstance(batch, AudioBatch)` | `TypeError` |
| 2 | `batch.x.ndim == 3` | `ValueError` |
| 3 | `batch.mask.dtype is torch.bool` | `ValueError` |
| 4 | `batch.mask.shape == batch.x.shape[:2]` | `ValueError` |
| 5+ | Variant-specific shape checks (immediate after standard block) | `ValueError` |

**Patchify variant-specific (check #5+):**

- `F_in == batch.x.shape[-1]` and `F_in >= patch_f` else `ValueError`.
- When `stride is None`, the legacy exact feature partition is preserved: `F_in % patch_f == 0` else `ValueError`.
- When `stride` is provided, feature patches use in-bounds strided windows; trailing feature columns outside the last full patch may be skipped.
- Time axis: if `T_feat` does not fit a complete `patch_t`/`stride_t` grid, apply `padding_strategy`:
  - `"pad"`: zero-pad `x`, extend `mask` with `False` on padded frames.
  - `"truncate"`: drop trailing frames outside the final complete patch.
- Token order: time-major — all feature patches for time index 0, then time index 1, …
- `N_tokens = num_time_patches * num_feat_patches`.
- After forward: `batch.x.shape[-1] == config.output_dim`.

#### `NoOpTokenizer`

- `forward(batch) -> batch` (identity on `x` and `mask`).
- Requires `config.type == "none"`.
- `config.output_dim` (via property) set at construction to `processor.output_dim` (enforced by validator).

### 4.3 Mask collapse rule

For tokenizers that **aggregate** multiple input frames into one token (patchify and any future pooling tokenizer):

> Token `n` is **valid** (`mask[n] == True`) **iff** every input frame that contributed to token `n` is valid (**AND-reduction** over contributing frames).

Padded frames count as invalid. For patchify, a time patch is valid iff all `patch_t` underlying frame masks are `True`.

`NoOpTokenizer` preserves mask unchanged (`N_tokens = T_feat`).

### 4.4 `output_lengths`

Variable-rate tokenizers **must** expose:

```python
def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
```

| Requirement | Detail |
| --- | --- |
| Signature | `output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor` |
| Input | `(B,)`, same dtype/device as passed; per-row valid **frame** counts |
| Output | `(B,)`, same dtype/device as input; per-row valid **token** counts |
| Vectorized | Pure tensor ops; no Python loops over batch required |
| Consistency | For each row `b`, `output_lengths[b] == forward(...).mask[b].sum()` for valid inputs |
| **Upsampling ban** | **`output_lengths(L) <= L` element-wise** — strict mandatory invariant |
| NoOp | `return input_lengths` (identity; satisfies `<=`) |
| Patchify | Map frame lengths through patch grid + `padding_strategy` (same rules as `forward`) |

Boundary validation in `output_lengths`:

| Check | Exception |
| --- | --- |
| `isinstance(input_lengths, torch.Tensor)` | `TypeError` |
| `input_lengths.ndim == 1` | `ValueError` |

**Disallowed in this contract revision:** tokenizers where any row has `output_lengths > input_lengths` (upsampling).

---

### 4.5 `PatchifyTokenizer.patch_grid`

`PatchifyTokenizer` exposes a public patch-grid helper:

```python
def patch_grid(self, *, T_feat: int, F_in: int) -> tuple[int, int]:
    """Return (T_patches, F_patches) for a fixed-shape feature tensor."""
```

Contract:

- Pure function of `config`, `T_feat`, and `F_in`.
- Mirrors the exact time-grid and feature-grid math used by `forward`.
- Returns `(num_time_patches, num_feat_patches)`.
- Preserves patch ordering assumptions: tokens are time-major with all frequency patches for time step 0, then all frequency patches for time step 1, etc.
- Applies the same feature-axis validation as `forward`.
- Exists only on `PatchifyTokenizer`; `NoOpTokenizer` does not implement it.

This helper is the modelling-layer API for objectives that need to recover `(T_patches, F_patches)` while keeping tokenizer `forward` output rank-3.

### 4.6 `TokenLayout`

Tokenizers expose public layout metadata through:

```python
def layout(self, *, T_feat: int, F_in: int) -> TokenLayout: ...
```

`TokenLayout` is a frozen dataclass carrying token count, time/frequency patch counts, patch shape, stride, flatten order, output dim, and source feature shape. `PatchifyTokenizer.patch_grid(...)` delegates to `layout(...)`. Objectives and layout-aware positional encodings must consume `TokenLayout` instead of reconstructing grid assumptions from private tokenizer state.

Patchify also owns patch-target reordering:

```python
reorder_patch_targets(target, *, layout: TokenLayout, order: str) -> torch.Tensor
```

Supported orders are `"time_frequency"` (identity) and `"frequency_time"` (Dasheng reconstruction order).

---

## 5. Wrapper / dispatcher contract

**Class:** `Tokenizer` (`tokenizers/tokenizer.py`)

```python
class Tokenizer(nn.Module):
    def __init__(self, config: AudioModelConfig) -> None:
        cls = get_tokenizer_registry()[config.tokenizer.type]
        if cls is None:
            raise ValueError(
                f"Tokenizer.__init__ expected registered tokenizer class, got {config.tokenizer.type!r}"
            )
        self.tokenizer = cls(config=config.tokenizer)

    def forward(self, batch: AudioBatch) -> AudioBatch:
        ...
```

| Registry key | Class |
| --- | --- |
| `"patchify"` | `PatchifyTokenizer` |
| `"none"` | `NoOpTokenizer` |

The registry entry `"None": None` and string sentinels on `AudioModelConfig.tokenizer` are **removed** in the ideal state.

Wrapper `forward`:

| # | Check | Exception |
| --- | --- | --- |
| 1 | `isinstance(batch, AudioBatch)` | `TypeError` |
| 2 | Delegate to `self.tokenizer(batch)` | — |

---

## 6. Inheritance and extension surface

| Allowed | Frozen |
| --- | --- |
| New `TokenizerConfig` + class + registry key | Projecting to `d_model` |
| `output_lengths` for non-identity time mapping (must satisfy `<= L`) | Positional encoding |
| Documented learnable params on variant config + registry init | Returning non-`AudioBatch` |
| Tail policy field on variable-rate variants | Upsampling tokenizers |

---

## 7. Non-goals

- No `d_model` projection (the embedding category owns `input_dim → d_model`; see [../embeddings/CONTRACT.md](../embeddings/CONTRACT.md)).
- No positional encoding.
- No upsampling (`output_lengths > input_lengths`).
- No default learnable parameters (unless variant config explicitly allows them).
- No raw waveform input (rank-2 `x`).
- No `cls` or loss computation.
- No device validation at boundaries.

---

## 8. Validation checklist

Flat mandatory checks. Each item is pass/fail with line citation.

1. File at `src/audiossl/modelling/tokenizers/<name>.py`.
2. Config dataclass in `src/audiossl/config/config.py`: `@dataclass(kw_only=True, frozen=True)`, `type: Literal[...]`.
3. `@property output_dim` on config; validators read `config.output_dim` only.
4. `NoOpTokenizerConfig` uses private `_output_dim` + property (not public field).
5. Validator registered in `validate_config.py`.
6. Variable-rate variant declares tail policy field (`padding_strategy` or equivalent).
7. Cross-config: `NoOpTokenizerConfig.output_dim == processor.output_dim` enforced in `_validate_model`.
8. Class subclasses `nn.Module`.
9. `__init__` signature exactly `(*, config: <variant>)` — no `d_model`, no `input_dim`.
10. `forward` signature `(batch: AudioBatch) -> AudioBatch`.
11. Four standard boundary checks at top of `forward`, in order, standardized messages.
12. Variant-specific shape checks immediately after standard block (not in `validate_config` alone).
13. `output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor` present.
14. Sentinel: for test lengths `L`, `output_lengths(L) <= L` element-wise.
15. Mask collapse uses AND-reduction over contributing frames.
16. `NoOpTokenizer.forward` is identity.
17. Registered in `get_tokenizer_registry()` under config `type` literal.
18. Wrapper rejects `None` registry entries and string sentinels on `AudioModelConfig.tokenizer`.
19. For learnable variants: no inline `nn.GELU()` / bare `nn.init.*`.
20. Output `batch.x.shape[-1] == config.output_dim` after forward.

---

## 9. How to validate a new tokenizer

1. Open `src/audiossl/modelling/tokenizers/<name>.py`.
2. Open `src/audiossl/config/config.py` — locate config dataclass; verify `@property output_dim`.
3. Open `src/audiossl/config/validate_config.py` — confirm static validators; confirm input-dependent checks are **not** only in validator.
4. Open `src/audiossl/modelling/registry.py` — confirm `get_tokenizer_registry()` entry.
5. Open `src/audiossl/modelling/tokenizers/tokenizer.py` — confirm wrapper dispatch.
6. Walk checklist items 1–20; cite line numbers.
7. In `forward`, confirm checks 1–4 precede any variant-specific check 5+.
8. Trace `output_lengths` against `forward` mask logic; verify `output_lengths(L) <= L`.
9. For patchify: confirm `padding_strategy` field on config and both code paths in `forward`.

---

## 10. Cross-references

- Processor `output_dim` and feature shapes: [../processors/CONTRACT.md](../processors/CONTRACT.md)
- Embedding `input_dim`: [../embeddings/CONTRACT.md](../embeddings/CONTRACT.md)
- Central mask semantics: [../CONTRACT.md](../CONTRACT.md)
- `AudioModel` construction: [../CONTRACT.md](../CONTRACT.md#audiomodel-orchestration)
- Error-message templates: [../CONTRACT.md](../CONTRACT.md#standardized-error-message-templates)

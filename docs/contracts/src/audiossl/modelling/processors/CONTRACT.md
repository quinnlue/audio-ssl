# Processor component contract

Processors map raw waveform batches to frame-level feature sequences. They are the first transformation in `AudioModel.pipeline` and set the feature dimension consumed by tokenizers. Cross-cutting policy: [../CONTRACT.md](../CONTRACT.md).

---

## 1. Purpose and scope

The processor:

- Accepts `AudioBatch` with mono waveform `x` `(B, T_raw)` and per-sample mask `(B, T_raw)`.
- Produces `AudioBatch` with frame features `x` `(B, T_feat, output_dim)` and mask `(B, T_feat)`.
- Recomputes masks when time is downsampled (conv strides, spectrogram hop, etc.).
- Declares expected input sample rate via config field `sr` (declaration only — no runtime validation).

It does **not** emit tokens, project to `d_model`, augment audio, validate sample rate at runtime, validate device placement, or return `EncoderOutput`.

---

## 2. Pipeline position

```text
AudioBatch (raw waveform)  →  Processor  →  AudioBatch (features)  →  Tokenizer
```

| Direction | Type |
| --- | --- |
| In | `AudioBatch`, `x` `(B, T_raw)`, `mask` `(B, T_raw)` |
| Out | `AudioBatch`, `x` `(B, T_feat, output_dim)`, `mask` `(B, T_feat)` |

Downstream: [tokenizer contract](../tokenizers/CONTRACT.md).

---

## 3. Config contract

Union:

```python
ProcessorConfig = (
    MelSpectrogramProcessorConfig
    | ConvolutionalProcessorConfig
    | Wav2Vec2FeatureExtractorProcessorConfig
)
```

Every variant:

```python
@dataclass(kw_only=True, frozen=True)
class <Variant>ProcessorConfig:
    type: Literal["<variant>"] = "<variant>"
    ...

    @property
    def output_dim(self) -> int: ...
```

| Variant | `type` | `output_dim` source | `sr` |
| --- | --- | --- | --- |
| Mel spectrogram | `"melspec"` | `n_mels` | Declared on config (default e.g. `16000`) |
| Convolutional | `"conv"` | `out_channels` | Declared if applicable |
| wav2vec 2.0 feature extractor | `"wav2vec2_feature_extractor"` | `conv_dim[-1]` | Declared on config (default `16000`) |

### `output_dim` form

Always a `@property` on the config dataclass. Validators read `config.output_dim`, never raw field names.

### Cross-config rules

| Rule | Enforcer |
| --- | --- |
| Activation name ∈ `ACTIVATION_REGISTRY` | `_validate_convolutional_processor` |
| FFT/hop/kernel static consistency | per-variant validator |

Validator registered in `validate_config.py` for each variant.

**Sample rate:** `sr` on processor config is a **declaration** of the rate the processor expects. The processor does **not** validate `sr` against input tensors. The **trainer/dataloader** must deliver audio at the declared rate. `AudioBatch` carries no sample-rate field.

---

## 4. Concrete component contract

### 4.1 `__init__` signature

All concrete processors:

```python
def __init__(self, *, d_model: int, config: <ProcessorConfig variant>) -> None:
```

| Argument | Contract |
| --- | --- |
| `d_model` | Passed for uniformity; processor may use for internal projections or ignore; kwarg must remain accepted unchanged |
| `config` | Frozen variant config |

Construction:

1. Store `self.config`.
2. Build `nn.Module` children inside the parent `runtime_seed_scope(runtime, "model/processor")` (via `AudioModel`).
3. Resolve activations via `ACTIVATION_REGISTRY[config.activation]` (where applicable).
4. Resolve inits via `INITIALIZATION_REGISTRY` / shared init helpers.
5. Processor does **not** call `apply_runtime_to_module` on itself — parent `AudioModel` applies to full tree.

**Learnable processors:** parameter init through `INITIALIZATION_REGISTRY`, activations through `ACTIVATION_REGISTRY`. Inline `nn.GELU()` / bare `nn.init.*` is a **fail**.

### 4.2 `forward` signature

```python
def forward(self, batch: AudioBatch) -> AudioBatch:
```

#### Shape table

| | Input | Output |
| --- | --- | --- |
| `batch.x` | `(B, T_raw)` `float32` | `(B, T_feat, output_dim)` `float32` |
| `batch.mask` | `(B, T_raw)` `bool` | `(B, T_feat)` `bool` |

#### Boundary validation (mandatory order)

| # | Check | Exception |
| --- | --- | --- |
| 1 | `isinstance(batch, AudioBatch)` | `TypeError` |
| 2 | `batch.x.ndim == 2` | `ValueError` |
| 3 | `batch.mask.dtype is torch.bool` | `ValueError` |
| 4 | `batch.mask.shape == batch.x.shape` | `ValueError` |
| 5 | Every per-row output mask/length has at least one valid frame (via `output_mask(batch.mask)` when implemented, otherwise `output_lengths(mask.sum(dim=1))`) | `ValueError` |

Check #5: compute the same output mask/lengths used by `forward`; if any row has no valid frames, raise `ValueError` with shortest input length (`batch.mask.sum(dim=1).min()`) in `{description}`.

#### Mask recompute rule (strict / lenient)

Default (strict):

> Output frame `t` is **valid** (`mask[b, t] == True`) **iff** every input sample required to compute that frame is **valid** in the input mask for batch row `b`.

Equivalently: invalid (padded) input samples must not contribute to valid output frames. Zero-pad samples count as **invalid** for downstream recompute.

Implementations should zero masked input before the DSP/conv stack, then set output mask from the processor's mask recompute primitive.

For processors that implement `output_mask`, after `forward`, the returned mask must equal `self.output_mask(input_mask)` where `input_mask` is the pre-forward raw mask.

For prefix-padding masks (`True...True False...False`), `mask[b].sum() == output_lengths(input_lengths)[b]` where `input_lengths = input_mask.sum(dim=1)`.

Opt-out (lenient): some external models (e.g. Dasheng) do not propagate input masks through the mel front-end and expect every STFT frame to contribute downstream. `MelSpectrogramProcessorConfig.frame_validity="overlap_any"` switches the mask recompute to mark an output frame valid iff **any** input sample in its receptive window is valid — STFT center padding alone no longer invalidates a frame. The strict default is `frame_validity="strict_window"`.

### 4.3 `output_mask`

Mel spectrogram processors **must** implement:

```python
def output_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
```

| Requirement | Detail |
| --- | --- |
| Signature | `output_mask(self, input_mask: torch.Tensor) -> torch.Tensor` |
| Input | `(B, T_raw)`, raw per-sample boolean mask |
| Output | `(B, T_feat)`, boolean feature-frame mask |
| Determinism | Pure function of `config` and `input_mask` — no dependence on waveform values |
| Shared use | Mask recompute in `forward` must call the same function |

Boundary validation in `output_mask`:

| Check | Exception |
| --- | --- |
| `isinstance(input_mask, torch.Tensor)` | `TypeError` |
| `input_mask.ndim == 2` | `ValueError` |
| `input_mask.dtype is torch.bool` | `ValueError` |

### 4.4 `output_lengths`

Processors **must** implement:

```python
def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
```

| Requirement | Detail |
| --- | --- |
| Signature | `output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor` |
| Input | `(B,)`, per-row prefix-valid sample count in the **raw** mask |
| Output | `(B,)`, same device as input; per-row valid **feature** frames after processor for prefix masks |
| Determinism | Pure function of `config` and `input_lengths` — no dependence on tensor values |
| Shared use | External tooling may use this for prefix-padding batches; processors without `output_mask` use it for `forward` mask recompute |

Boundary validation in `output_lengths`:

| Check | Exception |
| --- | --- |
| `isinstance(input_lengths, torch.Tensor)` | `TypeError` |
| `input_lengths.ndim == 1` | `ValueError` |

**Mel spectrogram:** map valid sample counts through hop/frame logic (center padding rules fixed per config). Padding introduced by `pad`, `center`, or FFT framing is invalid for mask purposes, so only frames whose full receptive sample window is valid count.

**Convolutional:** repeated conv length formula from `kernel_sizes` / `strides`.

---

### 4.5 Mel-specific intermediate methods

`MelSpectrogramProcessor` exposes public intermediate methods for objective code that needs MAE-style reconstruction targets:

```python
def compute_melspec(self, batch: AudioBatch) -> AudioBatch: ...
def compute_amplitude_to_db(self, batch: AudioBatch) -> AudioBatch: ...
def compute_batch_norm(self, batch: AudioBatch) -> AudioBatch: ...
```

| Method | Input | Output | Notes |
| --- | --- | --- | --- |
| `compute_melspec` | rank-2 waveform `AudioBatch`, `(B, T_raw)` | rank-3 mel-power `AudioBatch`, `(B, T_feat, n_mels)` | owns device move, masked waveform zero-fill, mel transform, and output mask computation |
| `compute_amplitude_to_db` | rank-3 feature `AudioBatch`, `(B, T_feat, n_mels)` | rank-3 feature `AudioBatch`, same shape | identity when `config.log_mel` is false |
| `compute_batch_norm` | rank-3 feature `AudioBatch`, `(B, T_feat, n_mels)` | rank-3 feature `AudioBatch`, same shape | identity when `config.batch_norm` is false; owns `(B,T,F) <-> (B,F,T)` transposes around `MaskedFeatureBatchNorm1d` |

These methods must compose exactly to `forward`:

```python
forward(batch) == compute_batch_norm(
    compute_amplitude_to_db(
        compute_melspec(batch)
    )
)
```

Each method performs its own boundary checks. `compute_melspec` accepts only raw rank-2 waveform batches; the latter two accept rank-3 mel feature batches whose mask matches `x.shape[:2]`.

---

## 5. Wrapper / dispatcher contract

**Class:** `Processor` (`processors/processor.py`)

```python
class Processor(nn.Module):
    def __init__(self, config: AudioModelConfig) -> None:
        cls = get_processor_registry()[config.processor.type]
        self.processor = cls(d_model=config.d_model, config=config.processor)

    def forward(self, batch: AudioBatch) -> AudioBatch:
        return self.processor(batch)
```

| `config.processor.type` | Class |
| --- | --- |
| `"melspec"` | `MelSpectrogramProcessor` |
| `"conv"` | `ConvolutionalProcessor` |
| `"wav2vec2_feature_extractor"` | `Wav2Vec2FeatureExtractorProcessor` |

Register new processors in `get_processor_registry()`.

**Advisory:** wrapper may delegate boundary validation entirely to concrete processor.

---

## 6. Inheritance and extension surface

| Allowed | Frozen |
| --- | --- |
| New `ProcessorConfig` + implementation + registry entry | Output rank other than 3 |
| Custom `output_mask` / `output_lengths` matching mask rule | Token-level outputs |
| Registry activation/init names | Returning `EncoderOutput` |
| New `sr` declaration on config | Runtime sample-rate validation |

---

## 7. Non-goals

- Do **not** project features to `d_model` at the processor boundary (`output_dim` is processor-native, e.g. `n_mels` or `out_channels`; the embedding category owns the dim transform).
- Do **not** patchify or change token semantics (tokenizer responsibility).
- Do **not** run augmentation (objective responsibility).
- Do **not** return non-`AudioBatch` types.
- Do **not** compute `cls` or training loss.
- Do **not** validate sample rate at runtime.
- Do **not** validate device at boundaries.

---

## 8. Validation checklist

Flat mandatory checks. Each item is pass/fail with line citation.

1. File at `src/audiossl/modelling/processors/<name>.py`.
2. Config dataclass in `src/audiossl/config/config.py`: `@dataclass(kw_only=True, frozen=True)`, `type: Literal[...]`.
3. `@property output_dim` on config; validators read `config.output_dim`.
4. Config declares `sr` where applicable (sample-rate declaration).
5. Validator registered in `validate_config.py` for variant.
6. If variant carries `d_model`: validator enforces equality with `AudioModelConfig.d_model`.
7. Class subclasses `nn.Module`.
8. `__init__` signature exactly `(*, d_model: int, config: <variant>)`.
9. `__init__` stores `self.config`; builds learnable submodules via registries.
10. `forward` signature `(batch: AudioBatch) -> AudioBatch`.
11. Five boundary checks in `forward`, in order (§4.2), standardized messages.
12. For mel spectrogram processors: `output_mask(self, input_mask: torch.Tensor) -> torch.Tensor` present.
13. For mel spectrogram processors: `output_mask` validates tensor type, `ndim == 2`, and boolean dtype.
14. `output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor` present.
15. `output_lengths` validates tensor type and `ndim == 1`.
16. Mask recompute uses `output_mask` when implemented, otherwise `output_lengths`, consistently with forward mask output.
17. Output `batch.x.ndim == 3` and `batch.x.shape[-1] == config.output_dim`.
18. Registered in `get_processor_registry()` under config `type` literal.
19. No inline `nn.GELU()` / `nn.ReLU()` / bare `nn.init.*` — must route through registries.
20. No runtime sample-rate validation against input.
21. No device validation at boundary.

---

## 9. How to validate a new processor

1. Open `src/audiossl/modelling/processors/<name>.py`.
2. Open `src/audiossl/config/config.py` — locate `<Variant>ProcessorConfig`; verify `@property output_dim` and `sr` if applicable.
3. Open `src/audiossl/config/validate_config.py` — confirm variant validator and cross-config `d_model` check if applicable.
4. Open `src/audiossl/modelling/registry.py` — confirm `get_processor_registry()` entry.
5. Open `src/audiossl/modelling/processors/processor.py` — confirm wrapper passes `(d_model=config.d_model, config=config.processor)`.
6. Walk checklist items 1–21; cite line numbers.
7. Trace mask recompute: confirm invalid input samples (zero-pad or mask holes) do not yield valid output frames.
8. Confirm check #5 in `forward` rejects rows with no valid frames from the mask recompute primitive.
9. Grep for inline activations/inits — must use registries.

---

## 10. Cross-references

- Tokenizer `F_in` / `NoOpTokenizerConfig.output_dim`: [../tokenizers/CONTRACT.md](../tokenizers/CONTRACT.md)
- Embedding `input_dim` (via tokenizer): [../embeddings/CONTRACT.md](../embeddings/CONTRACT.md)
- Encoder shapes (post-embedding): [../encoders/CONTRACT.md](../encoders/CONTRACT.md)
- `AudioBatch` / mask glossary: [../CONTRACT.md](../CONTRACT.md)
- Sample rate responsibility: [../CONTRACT.md](../CONTRACT.md#responsibility-splits)
- Registries: [../CONTRACT.md](../CONTRACT.md#registries)
- Error-message templates: [../CONTRACT.md](../CONTRACT.md#standardized-error-message-templates)

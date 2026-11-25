# Objective component contract

Objectives wrap an `AudioModel` and own **all training-time orchestration**: multi-view passes, augmentation, JEPA masking, teacher/student paths, projection heads, and the scalar loss. The trainer never calls `model.pipeline` directly. Cross-cutting policy: [../CONTRACT.md](../CONTRACT.md).

---

## 1. Purpose and scope

An objective is a subclass of the concrete `Objective` base class (which is an
`nn.Module`) that:

- Holds a reference to a constructed `AudioModel`.
- Owns every submodule and algorithm **not** already owned by processor / tokenizer / encoder (projection heads, predictors, mask tokens, EMA teachers).
- Receives trainer-constructed `augmentor` and/or `generator` via `build_objective`, as declared by its `uses_augmentor` / `uses_generator` class attributes.
- Exposes `forward(batch: AudioBatch) -> ObjectiveOutput` as the **only** public training entrypoint for the representation stack.
- Exposes `metrics(output, ctx) -> dict[str, torch.Tensor]` so loggers can read
  objective-specific diagnostics without inspecting each output dataclass.
  The trainer adds training telemetry such as loss, epoch, learning rate,
  gradient norm, and throughput.

An objective does **not** redefine processor, tokenizer, or encoder semantics; does not mutate `model.config` or `model.applied_runtime`; does **not** construct optimizers; does **not** accept custom batch types at the public boundary.

---

## 2. Pipeline position

```text
Trainer
  → build_objective(model, config, augmentor=..., generator=...)
  → objective.forward(AudioBatch) → ObjectiveOutput
       └─ internally may call model.pipeline or public AudioModel stage methods
```

| Direction | Type |
| --- | --- |
| In | `AudioBatch` only (typically raw waveform + mask at batch collation) |
| Out | `ObjectiveOutput` subclass (`ContrastiveOutput`, `JEPAOutput`, …) |

Sibling components are invoked **inside** the objective according to strategy; the trainer sees only the objective boundary.

When objectives need intermediate stages, they must use `AudioModel.process`,
`token_layout`, `tokenize`, `embed`, `encode_tokens`, and `encode_embedded`,
plus the staged embedding API on `model.embedding`:

| Stage | Effect |
| --- | --- |
| `prepare(batch)` | validate tokens, apply input normalization |
| `project(batch)` | map tokens to `d_model` |
| `position(batch, layout=...)` | add positional encoding (layout-aware PEs require `layout`) |
| `finalize(batch)` | dropout, subsampling, and prefix (CLS) tokens; sets `positions` / `num_prefix_tokens` |

`embed` is exactly `finalize(position(project(prepare(x))))`. Objectives that
drop or replace tokens do so between `position` and `finalize`, passing the
kept indices as `AudioBatch.positions`. Shared helpers in
[`components.py`](components.py) (`encode_with_mask_embedding`, `strip_prefix`,
`frozen_copy`, `ema_update_`, `validate_waveform_batch`, `GradMultiply`) cover
the common patterns.

Objectives must not reach through wrapper internals such as
`model.processor.processor` or `model.embedding.embedding`, and must not
`isinstance`-check concrete component configs. They declare
`required_capabilities` / `incompatible_capabilities` on their config instead
(see §3).

---

## 3. Config contract

Each `ObjectiveConfig` variant:

```python
@dataclass(kw_only=True, frozen=True)
class <Variant>ObjectiveConfig:
    type: Literal["<variant>"] = "<variant>"
    ...
```

Union: `ObjectiveConfig = ContrastiveObjectiveConfig | BYOLObjectiveConfig | DINOObjectiveConfig | JEPAObjectiveConfig | …`

| Variant | `type` | Key fields (non-exhaustive) |
| --- | --- | --- |
| Contrastive | `"contrastive"` | `view_strategy`, `projection`, `temperature`, `symmetric`, `compute_stats` |
| BYOL | `"byol"` | `projector`, `predictor`, `teacher_mode`, `teacher_ema_decay` |
| DINO | `"dino"` | `head`, `teacher_mode`, `teacher_ema_decay`, `teacher_temperature`, `student_temperature`, `center_momentum` |
| JEPA | `"jepa"` | `loss_style`, `mask_token_init_scale`, `teacher_mode`, `predictor_config`, `masking`, norm flags |
| MAE | `"mae"` | `decoder_d_model`, `decoder`, `decoder_positional_encoding`, `mask_token_init_scale`, `norm_pix_loss`, `norm_pix_eps`, `masking` |

### Capabilities

Component configs declare `capabilities` (see
`audiossl.config.capabilities`: `reconstruction_target`, `patch_grid`,
`prefix_token`, `token_subsampling`, `sequence_convolution`). Objective configs
declare what they need:

| Objective | `required_capabilities` | `incompatible_capabilities` |
| --- | --- | --- |
| MAE | `reconstruction_target`, `patch_grid` | `token_subsampling`, `sequence_convolution` |
| UFO | `patch_grid`, `prefix_token` | `token_subsampling`, `sequence_convolution` |
| wav2vec2, HuBERT | — | `token_subsampling` |

`check_objective_compatibility(model_config, objective_config)` runs during
training-config validation and again in the objective constructor
(`Objective._check_model_compatibility`). Its `ValueError` names the objective,
the missing or conflicting capability, and the component that provides it.

### Validator registration

Built-in validators live in `validate_config.py` (`_validate_contrastive_objective`, `_validate_byol_objective`, `_validate_dino_objective`, `_validate_jepa_objective`, `_validate_mae_objective`, `_validate_mae_masking`, …). Third-party configs register with `@register_config("objective")` and `@register_validator(ConfigClass)` from `audiossl.config`; `load_config` then resolves their `type` from JSON:

| Check | Failure |
| --- | --- |
| Numeric ranges (temperature > 0, mask ratios, EMA decay ∈ (0,1), …) | `ValueError` |
| Cross-field consistency (see variant tables below) | `ValueError` / `RuntimeError` |
| Unknown `type` | `TypeError` |

Objectives do **not** expose `output_dim`; their public scalar is `loss`.

---

## 4. Concrete objective contract

### 4.1 Base class

Concrete objectives subclass the concrete base class `Objective` (defined in
[`_base.py`](_base.py); also re-exported from
[`objective.py`](objective.py)). `Objective` is an `nn.Module` with:

- Class attributes `uses_augmentor` / `uses_generator` (default `False`) that
  `build_objective` reads to decide which collaborators to pass.
- No-op lifecycle methods named in `OBJECTIVE_HOOKS` (§4.5).
- `forward(batch: AudioBatch) -> ObjectiveOutput` — required override.
- `metrics(output: ObjectiveOutput, ctx: StepContext) -> dict[str, torch.Tensor]`
  returns an empty dict by default. Subclasses return detached scalar
  diagnostics with short names such as `center_norm`. The trainer adds the
  `train/{objective_type}/` prefix.

There is no `ABC` enforcement — calling `Objective.forward` directly raises
`NotImplementedError`. The base class lives in `_base.py` rather than
`objective.py` to keep the `build_objective` factory and the concrete subclass
files free of an import cycle.

### 4.2 `__init__` signatures

**Contrastive:**

```python
def __init__(
    self,
    model: AudioModel,
    config: ContrastiveObjectiveConfig,
    *,
    augmentor: AugmentationPipeline | None = None,
) -> None:
```

**BYOL:**

```python
def __init__(
    self,
    model: AudioModel,
    config: BYOLObjectiveConfig,
    *,
    augmentor: AugmentationPipeline | None = None,
) -> None:
```

**DINO:**

```python
def __init__(
    self,
    model: AudioModel,
    config: DINOObjectiveConfig,
    *,
    augmentor: AugmentationPipeline | None = None,
) -> None:
```

**JEPA:**

```python
def __init__(
    self,
    model: AudioModel,
    config: JEPAObjectiveConfig,
    *,
    generator: torch.Generator | None = None,
) -> None:
```

**MAE:**

```python
def __init__(
    self,
    model: AudioModel,
    config: MAEObjectiveConfig,
    *,
    generator: torch.Generator | None = None,
) -> None:
```

MAE requires a processor providing `reconstruction_target` and a tokenizer
providing `patch_grid`; construction raises `ValueError` otherwise (§3
Capabilities). Any processor/tokenizer pair declaring those capabilities works.

#### Construction steps (all variants)

| # | Step | Exception on failure |
| --- | --- | --- |
| 1 | `isinstance(model, AudioModel)` | `TypeError` — `ContrastiveObjective.__init__` / `JEPAObjective.__init__` |
| 2 | `isinstance(config, <Variant>ObjectiveConfig)` | `TypeError` |
| 3 | `self._check_model_compatibility(model, config)` and variant-specific cross-field checks | `ValueError` / `RuntimeError` |
| 4 | Build owned submodules (heads, teachers, mask tokens, …); teachers via `frozen_copy(model)` | — |
| 5 | `apply_runtime_to_module(submodule, model.applied_runtime)` for **every** owned learnable submodule | — |
| 6 | Stochastic parameter creation inside `runtime_seed_scope(model.config.runtime, "objective/<type>")` so objective init is independent of model init | — |
| 7 | **MUST NOT** construct `torch.optim.*` or any optimizer-class object anywhere in the file | grep fail |

<!-- #### Variant-specific `__init__` checks

| Variant | Check | Exception |
| --- | --- | --- |
| Contrastive | `view_strategy == "augmentation"` ⇒ `augmentor is not None` | `ValueError` |
| Contrastive | `view_strategy == "dropout"` ⇒ `augmentor is None` | `ValueError` |
| JEPA | `teacher_mode in {"ema", "stop_grad"}` | `ValueError` |
| JEPA EMA | no undocumented persistent float buffers on student/teacher without EMA policy | `RuntimeError` | -->

#### Trainer-owned kwargs

| Variant | Kwarg | Constructed by |
| --- | --- | --- |
| Contrastive | `augmentor: AugmentationPipeline \| None` | Trainer from `AudioAugmentorConfig` |
| BYOL | `augmentor: AugmentationPipeline \| None` | Trainer from `AudioAugmentorConfig` |
| DINO | `augmentor: AugmentationPipeline \| None` | Trainer from `AudioAugmentorConfig` |
| JEPA, MAE, UFO, wav2vec2, HuBERT | `generator: torch.Generator \| None` | Trainer; seeded from `RuntimeConfig.seed` |

`AudioModel` and objectives **never** construct augmentor/generator without trainer injection. Objective takes ownership when `build_objective` returns. Objectives that declare `uses_generator=True` **must** call `self.register_generator("generator", generator)` (and any additional named streams) during `__init__`; `build_objective` raises if none are registered. Generator state is not in `nn.Module.state_dict`; the trainer checkpoints every entry from `objective.named_generators()` under `payload["rng"]["objective_generators"]` and raises on resume when the name set mismatches.

### 4.3 `forward` signature

```python
def forward(self, batch: AudioBatch) -> <Variant>Output:
```

#### Boundary validation (mandatory, local — does NOT delegate to AudioModel)

Checks at top of `forward` **in order**:

| # | Check | Exception | Notes |
| --- | --- | --- | --- |
| 1 | `isinstance(batch, AudioBatch)` | `TypeError` | No custom batch types at public boundary |
| 2 | `batch.x.ndim` matches expected rank (typically `2` for waveform-entry objectives) | `ValueError` | |
| 3 | `batch.mask.dtype is torch.bool` | `ValueError` | |
| 4 | `batch.mask.shape == batch.x.shape` for waveform-entry objectives | `ValueError` | Rank-matched raw waveform mask |

#### Train/eval invariants

- `forward` is **allowed** in eval mode for diagnostic eval-loss logging.
- Variant-specific train-mode invariants MUST raise `RuntimeError` with standardized message when violated.

| Variant | Invariant | Exception |
| --- | --- | --- |
| Contrastive | `view_strategy == "dropout"` requires `self.training` | `RuntimeError` |
| JEPA | teacher vs student shape agreement | `RuntimeError` |

#### Orchestration (variant examples)

**Contrastive:**

- Build two views via owned view strategy (`augmentation` uses trainer `augmentor`; `dropout` uses stochastic encoder dropout in train mode).
- `out_i = model.pipeline(views.batch_i)`, `out_j = model.pipeline(views.batch_j)`.
- Use `model.get_cls(out_i)` / `model.get_cls(out_j)` so wrapper-owned pooling is applied consistently. `EncoderOutput.cls` is deprecated as a producer/consumer contract; objectives should not require it to be populated.
- Project, compute InfoNCE, return `ContrastiveOutput`.

**BYOL:**

- Build two augmented views via `create_contrastive_views(..., strategy="augmentation")`; `augmentor` is required.
- Student path: `model.pipeline(view)` → `model.get_cls(out)` → projector → predictor.
- Target path: EMA model + projector, or stop-grad student model + projector under `no_grad` and temporary `eval()`.
- Predictor is student-only; the target network never owns or updates a predictor.
- Compute symmetric BYOL cosine regression with L2 normalization inside the loss; return `BYOLOutput`.

**DINO:**

- Build two augmented global views via `create_contrastive_views(..., strategy="augmentation")`; multi-crop is out of scope for v1.
- Student path: `model.pipeline(view)` → `model.get_cls(out)` → `DINOHead`.
- Teacher path: EMA model + head, or stop-grad student model + head under `no_grad` and temporary `eval()`.
- Compute two cross-view DINO CE terms, excluding same-view pairs, using fixed scalar temperatures and the current center buffer.
- Return `DINOOutput` with `teacher_logits_uncentered` for `on_after_forward` centering.

**JEPA:**

- Student tokens: `processor` → `tokenizer` on raw batch (mask token still lives in pre-embedding, tokenizer-output space).
- Teacher: full `pipeline` or EMA copy's `pipeline` under `no_grad`.
- Objective samples a token-level target mask (scattered back through `embedding.subsampling_factor` when the embedding subsamples), replaces masked tokens, then calls `model.encode_tokens` with the context mask so mask-replaced tokens are projected into `d_model` alongside real tokens.
- Predictor head on student `z`; loss against detached teacher targets.
- Return `JEPAOutput`.

**MAE:**

- Stage the embedding up to `position` (`prepare` → `project` → `position(layout)`).
- Sample masks with `valid_mask=tokens.mask`: padding is never kept while a row
  has valid tokens left, and is never scored.
- Gather kept tokens, `finalize` them with `positions=ids_keep`, and encode with
  the kept-token mask. Prefix tokens are stripped before decoding.
- Decode by appending learned mask tokens, restoring original token order, adding
  layout-aware decoder PE, running a reused `Encoder` block stack under the
  full token mask, and projecting to patch values.
- Targets come from `processor.reconstruction_target(batch)` (mel + AmplitudeToDB
  before batch-norm for the mel processor), patchified and laid out in
  reconstruction order.
- Return `MAEOutput`; `removal_mask` marks every removed position and
  `target_mask = removal_mask & token_valid` marks the scored ones.

Internal batch-shaped dataclasses (e.g. paired-views container) are **allowed** inside the objective; the public boundary remains `AudioBatch` only.

### 4.4 `ObjectiveOutput` hierarchy

**Base** (`audiossl.config.batch_dataclasses`):

```python
@dataclass
class ObjectiveOutput:
    loss: torch.Tensor  # 0-dim scalar; requires_grad=True when training
```

Rules:

- `@dataclass` (**not** frozen).
- Subclasses **MUST** inherit from `ObjectiveOutput`.
- Subclasses may live in `audiossl.config.batch_dataclasses` or next to the
  objective that produces them; built-in variants stay in `batch_dataclasses`
  and `__all__` for import convenience.
- Every subclass **MUST** declare `loss: torch.Tensor` (0-dim scalar).

**ContrastiveOutput** (subclass) — key fields:

| Field | Shape / notes |
| --- | --- |
| `loss` | 0-dim scalar |
| `zi`, `zj` | `(B, projection_dim)` |
| `cls_i`, `cls_j` | `(B, d_model)` |
| `positive_loss`, `negative_loss` | scalar |
| `mask_i`, `mask_j` | `(B, T_tokens)` bool |
| `top1`, `pos_sim`, `neg_sim`, `align`, `uniformity` | optional diagnostics when `compute_stats=True` |

**BYOLOutput** (subclass) — key fields:

| Field | Shape / notes |
| --- | --- |
| `loss` | 0-dim scalar |
| `prediction_i`, `prediction_j` | student predictor outputs `(B, projection_dim)` |
| `target_i`, `target_j` | detached target projector outputs `(B, projection_dim)` |
| `projection_i`, `projection_j` | student projector outputs before predictor |
| `cls_i`, `cls_j` | `(B, d_model)` |
| `mask_i`, `mask_j` | `(B, T_tokens)` bool |

**DINOOutput** (subclass) — key fields:

| Field | Shape / notes |
| --- | --- |
| `loss` | 0-dim scalar |
| `student_logits_i`, `student_logits_j` | `(B, num_prototypes)` |
| `teacher_logits_i`, `teacher_logits_j` | detached uncentered teacher logits |
| `teacher_logits_uncentered` | `(2B, num_prototypes)`, consumed by `on_after_forward` |
| `cls_i`, `cls_j` | `(B, d_model)` |
| `mask_i`, `mask_j` | `(B, T_tokens)` bool |

**JEPAOutput** (subclass) — key fields:

| Field | Semantics |
| --- | --- |
| `loss` | scalar prediction loss |
| `prediction` | student predictor output `(B, T, d_model)` |
| `target` | teacher representations, detached |
| `target_mask` | bool, positions in the loss |
| `context_mask` | bool, student encoder attention mask |
| `encoder_mask` | bool, full valid token mask |
| `z` | alias of `prediction` for `(z, mask)` consumers |
| `mask` | alias of `encoder_mask` |

**MAEOutput** (subclass) — key fields:

| Field | Semantics |
| --- | --- |
| `loss` | scalar reconstruction loss over removed patches |
| `prediction` | decoder reconstruction `(B, N, patch_t * patch_f)` |
| `target` | detached patch targets in dasheng reconstruction order |
| `target_mask` | bool, True for removed **valid** positions used in the loss |
| `removal_mask` | bool, True for every removed position (superset of `target_mask`) |
| `ids_restore` | inverse permutation used to restore original token order |

### 4.5 Closed hook protocol

Recognized methods:

| Hook | Signature | Requirement |
| --- | --- | --- |
| `metrics` | `(output: ObjectiveOutput, ctx: StepContext) -> dict[str, torch.Tensor]` | Subclasses return objective-specific diagnostics. |
| `on_train_start`, `on_epoch_start`, `on_epoch_end`, `on_train_end` | `(ctx: StepContext) -> None` | Optional overrides. |
| `on_after_forward` | `(output: ObjectiveOutput, ctx: StepContext) -> None` | Override for running statistics that depend on the latest forward pass. |
| `on_before_optimizer_step` | `(ctx: StepContext) -> None` | Optional override for work such as gradient surgery after backward. |
| `on_after_optimizer_step` | `(ctx: StepContext) -> None` | Advance EMA teachers and objective schedules after a successful step. Preserve them when `ctx.step_was_skipped` is true. |
| `update_teacher` | `() -> None` | **Required** if the objective owns an EMA teacher; **forbidden** otherwise. Implement with `ema_update_`. |

The lifecycle names are `audiossl.modelling.objectives.OBJECTIVE_HOOKS`. The
base class defines all of them as no-ops; the trainer calls every one of them
unconditionally (see the trainer callbacks contract §8 for exact fire points).

`DINOObjective` owns a running center and therefore overrides `on_after_forward`.
BYOL, Contrastive, JEPA, MAE, UFO, Wav2Vec2, and HuBERT do not own
forward-dependent running statistics and must not override it. `MAEObjective`
owns no EMA teacher and therefore must not define `update_teacher`.

Rules:

- **No other method names** on the closed hook list.
- A method named like a hook but not on the list (e.g. `pre_step`, `post_forward`) is a **fail**.
- A `callbacks` attribute is a **fail**: `Trainer.__init__` raises `TypeError`.
- Adding a hook requires updating this contract and [../CONTRACT.md](../CONTRACT.md) first.

### 4.6 Training-loop binding

Prescribed trainer sequence:

| # | Step |
| --- | --- |
| 1 | Trainer constructs augmentor and/or generator as the objective's `uses_*` flags declare. |
| 2 | `objective = build_objective(model, config, augmentor=..., generator=...)`. |
| 3 | Run `output = objective(batch)` and `objective.on_after_forward(output, ctx)` for each minibatch. |
| 4 | `output.loss.backward()`. |
| 5 | Call `objective.on_before_optimizer_step(ctx)`, clip, step, advance the scheduler as appropriate, then call `objective.on_after_optimizer_step(ctx)` with the step outcome. |
| 6 | Log the diagnostics returned by `objective.metrics(output, ctx)` alongside trainer telemetry. |

The trainer **MUST NOT** call `model.pipeline` directly during training.

**Distributed training:** out of scope as of 5/20 12:16 pm GMT (THIS WILL ALMOST CERTAINLY CHANGE SOON); see [../CONTRACT.md](../CONTRACT.md).

### 4.7 `build_objective` factory

**File:** `src/audiossl/modelling/objectives/objective.py`

```python
def build_objective(
    model: AudioModel,
    config: ObjectiveConfig,
    *,
    augmentor: AugmentationPipeline | None = None,
    generator: torch.Generator | None = None,
) -> Objective:
```

| Step | Behavior |
| --- | --- |
| Lookup | `get_objective_registry()[config.type]`; a missing `type` field, an unknown type, or a registry entry that is not an `Objective` subclass raises `TypeError` |
| Augmentor | passed iff `cls.uses_augmentor`; a non-`None` augmentor for any other objective raises `ValueError` |
| Generator | passed iff `cls.uses_generator` |
| Construction | `cls(model=model, config=config, **kwargs)`; logs the objective's own trainable parameter count at `INFO` |

This is the **canonical** construction path.

---

## 5. Wrapper / dispatcher contract

Objectives are constructed only via `build_objective`. To add an objective
without editing library code:

```python
from audiossl.config import register_config, register_validator
from audiossl.modelling.registry import OBJECTIVES

@register_config("objective")
@dataclass(kw_only=True, frozen=True)
class MyObjectiveConfig:
    type: Literal["mine"] = "mine"
    required_capabilities: ClassVar[frozenset[str]] = frozenset()
    ...

@register_validator(MyObjectiveConfig)
def _validate(config, **context): ...

class MyObjective(Objective):
    uses_generator = True
    ...

OBJECTIVES.register("mine", MyObjective)
```

Duplicate registrations raise `ValueError` unless `overwrite=True` is passed.

---

## 6. Inheritance and extension surface

| Allowed | Frozen |
| --- | --- |
| New `ObjectiveConfig` + `ObjectiveOutput` subclasses | Processor/tokenizer/encoder contracts |
| New hooks **after contract update** | `AudioModel.config` / `applied_runtime` mutation |
| Owned heads and teachers | Trainer calling `model.pipeline` directly |
| Eval-mode `forward` for diagnostic loss | Custom batch types at public `forward` boundary |

---

## 7. Non-goals

- Must not redefine processor, tokenizer, or encoder I/O contracts.
- Must not mutate `model.config` or `model.applied_runtime`.
- Must not construct `torch.optim.*` anywhere in objective code.
- Must not accept custom batch types at `forward` boundary.
- Must not reach into trainer optimizer state unless explicitly passed in a future extension.
- Must not swallow validation errors.
- Must not implement distributed all-gather / FSDP behavior (out of scope).

---

## 8. Validation checklist

Flat mandatory checks. Each item is pass/fail with line citation.

1. Class subclasses `nn.Module`.
2. Implementation file under `src/audiossl/modelling/objectives/<variant>/` or `objectives/`.
3. Config dataclass in `config.py`: `@dataclass(kw_only=True, frozen=True)`, `type: Literal[...]`.
4. Validator registered in `validate_config.py` or via `register_validator`.
5. `__init__` signature matches `(model, config, *, <variant kwargs>)`.
6. `__init__` validates `isinstance(model, AudioModel)` with standardized `TypeError`.
7. `__init__` validates `isinstance(config, <Variant>ObjectiveConfig)` with standardized `TypeError`.
8. `__init__` runs variant-specific cross-field checks (§4.2 table).
9. `__init__` calls `apply_runtime_to_module` for every owned learnable submodule.
10. Stochastic construction inside `runtime_seed_scope(runtime, "objective/<type>")`.
11. File contains **no** `torch.optim` import or construction (grep `torch.optim`).
12. `forward` signature `(batch: AudioBatch) -> <Variant>Output`.
13. `forward` validates `isinstance(batch, AudioBatch)` first — standardized `TypeError`.
14. `forward` validates `batch.x.ndim` — standardized `ValueError`.
15. `forward` validates `batch.mask.dtype is torch.bool` — standardized `ValueError`.
16. `forward` validates `batch.mask.shape` against `batch.x` — standardized `ValueError`.
17. `forward` raises `RuntimeError` on variant-specific train/eval invariant violation.
18. `forward` does not accept or branch on custom public batch types.
19. `ObjectiveOutput` subclass inherits `ObjectiveOutput` (location unrestricted).
20. Built-in subclasses are exported from `batch_dataclasses.__all__` when defined there.
21. Subclass declares `loss: torch.Tensor` (0-dim scalar).
22. Registered in the objective registry under config `type`.
23. `uses_augmentor` / `uses_generator` match the constructor's keyword arguments; generators are registered via `register_generator`.
24. If EMA teacher owned: `update_teacher(self) -> None` defined and called from `on_after_optimizer_step` when `not ctx.step_was_skipped`.
25. If no EMA teacher: no method named `update_teacher`.
26. If forward-dependent running statistics are owned: `on_after_forward(self, output, ctx) -> None` overridden.
27. If no forward-dependent running statistics are owned: `on_after_forward` not overridden.
28. DINO center updates consume `DINOOutput.teacher_logits_uncentered`, not hidden `_last_*` state.
29. No method names matching unrecognized hooks from the closed list.
30. No inline `nn.GELU()` / bare `nn.init.*` in owned submodules — use registries.
31. The class subclasses `Objective` and its `metrics(output, ctx)` returns detached objective-specific scalars.
32. Component requirements are declared as config capabilities, not `isinstance` checks on component configs.

---

## 9. How to validate a new objective

1. Open implementation file (e.g. `objectives/contrastive/contrastive.py`).
2. Open `src/audiossl/config/config.py` — locate `<Variant>ObjectiveConfig`.
3. Open `src/audiossl/config/validate_config.py` — confirm validator.
4. Confirm the `ObjectiveOutput` subclass inherits `ObjectiveOutput` (built-ins: `batch_dataclasses.py`).
5. Open `src/audiossl/modelling/registry.py` — confirm the `OBJECTIVES` entry (or the plug-in's `OBJECTIVES.register` call).
6. Add a case to `tests/testkit/cases.py::OBJECTIVE_CASES`; the contract suite then runs every property in this document against it.
7. Walk checklist items 1–32 in order; record pass/fail with line numbers.
8. Grep file for `torch.optim` — must be absent.
9. Grep file for `def update_teacher` — presence must match EMA ownership.
10. Grep file for `def on_after_forward` — presence must match forward-dependent running stats ownership.
11. Confirm trainer binding: only `objective(batch)` at public boundary, not `model.pipeline`.

Reference implementations: `contrastive/contrastive.py`, `byol/byol.py`, `dino/dino.py`, `jepa/jepa.py`, `objective.py`.

---

## 10. Cross-references

- Central policy: [../CONTRACT.md](../CONTRACT.md)
- `AudioModel.pipeline` (internal use only): [../CONTRACT.md](../CONTRACT.md#audiomodel-orchestration)
- Encoder masks and wrapper-owned `cls`: [../encoders/CONTRACT.md](../encoders/CONTRACT.md)
- Raw batch shapes: [../processors/CONTRACT.md](../processors/CONTRACT.md)
- Closed hook list: [../CONTRACT.md](../CONTRACT.md#closed-objective-hook-list)
- Error-message templates: [../CONTRACT.md](../CONTRACT.md#standardized-error-message-templates)

# Callback component contract

Callbacks are passive observers that fire at fixed boundaries inside `Trainer.fit()`. They read from a mutable `TrainerState`, optionally perform I/O or side effects, and (rarely) mutate fields they own. This document is the full integration spec — agents writing new callbacks should not need to read `trainer.py` to do it correctly.

Cross-cutting style: [`agent-instructions/project/CODE_STYLE_AND_REQUIREMENTS.md`](../../../../agent-instructions/project/CODE_STYLE_AND_REQUIREMENTS.md).

---

## 1. Purpose and scope

A callback:

- Subclasses [`Callback`](callback.py) and overrides the hook methods it cares about.
- Receives a mutable [`TrainerState`](../state.py) at every hook invocation.
- May hold per-instance state on `self` (not on `TrainerState`).
- May perform I/O, log, save artifacts, or compute metrics.

It does **not**:

- Run forward/backward, take optimizer steps, or modify gradients.
- Move the model or batch between devices.
- Reseed RNG without explicit fork/restore (see §7).
- Read or write state owned by another callback (see §5).
- Communicate with sibling callbacks. If two callbacks need to coordinate, that's a redesign signal — escalate.

The trainer never catches callback exceptions — a raise aborts the run.

---

## 2. Class shape

```python
from typing import Any, ClassVar

from audiotrainer.callbacks import Callback
from audiotrainer.state import TrainerState
from audiotrainer.step import StepContext
from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput


class MyCallback(Callback):
    phase: ClassVar = "observe"  # evaluate | checkpoint | log | observe
    name: ClassVar[str] = "MyCallback"

    def __init__(self, *, ...explicit-config-args...) -> None:
        # Validate args here. Raise ValueError / TypeError on bad input.
        ...

    # Override only the hooks you need. Unoverridden hooks are no-ops.
    def on_train_start(self, state: TrainerState) -> None: ...
    def on_train_epoch_start(self, state: TrainerState) -> None: ...
    def on_train_batch_start(self, state: TrainerState, batch: AudioBatch) -> None: ...
    def on_train_batch_end(
        self, state: TrainerState, batch: AudioBatch, output: ObjectiveOutput
    ) -> None: ...
    def on_optimizer_step(self, state: TrainerState, ctx: StepContext) -> None: ...
    def on_train_epoch_end(self, state: TrainerState) -> None: ...
    def on_train_end(self, state: TrainerState) -> None: ...
    def on_validation_start(self, state: TrainerState) -> None: ...
    def on_validation_batch_end(
        self, state: TrainerState, batch: AudioBatch, output: ObjectiveOutput
    ) -> None: ...
    def on_validation_end(self, state: TrainerState) -> None: ...
    def on_checkpoint_written(self, state: TrainerState, *, path: Path) -> None: ...
    def on_exception(self, state: TrainerState, exc: BaseException) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
```

**Constructor rules:**

- All parameters are keyword-only and required (per project style: no hyperparameter defaults in function signatures).
- Validate at construction. A misconfigured callback must raise *before* `Trainer.fit()` runs.
- Store per-instance state with a leading-underscore name (`self._last_logged_step`, `self._save_dir`). Public attributes are part of the API and need a reason.
- Declaring `phase` places the callback in the shared ordering (`evaluate` → `checkpoint` → `log` → `observe`); within a phase, registration order is preserved.
- Emit cross-callback events via `self.emit(event, **payload)` after the trainer calls `attach(emit)` during construction.

---

## 3. Hook semantics

Hook fire points are precise. New callbacks must respect them:

| Hook | Fires | State invariants when called |
| --- | --- | --- |
| `on_train_start` | Once, before the first epoch | `state.epoch == start_epoch`, `state.optimizer_step` and `state.global_step` reflect any resume; `state.last_loss` and `state.current_lr` may be `None` (fresh run) or restored values (resume) |
| `on_train_epoch_start` | Once per epoch, before the loader is iterated | `state.epoch` is the index of the epoch that's about to start |
| `on_train_batch_start` | Every micro-batch, after `batch.to(device)`, before forward | `batch` is on `state.applied_runtime.device`; `state.global_step` reflects batches consumed so far, not including this one |
| `on_train_batch_end` | Every micro-batch, after forward+backward and (on step boundaries) `optimizer.step` + `scheduler.step` + objective `on_after_optimizer_step` | `state.last_loss` = the just-computed scalar loss; `state.global_step` includes this batch; `state.optimizer_step` and `state.current_lr` reflect this batch's step (if it was a step boundary) |
| `on_train_epoch_end` | Once per epoch, after the loader is exhausted | `state.last_loss` = the loss of the last micro-batch in the epoch; `state.optimizer_step` = total steps through this epoch |
| `on_train_end` | Once, after the final epoch's `on_train_epoch_end` | All counters are final |
| `on_validation_start` | Before `ValidationCallback` invokes its evaluator | `state.last_val_metrics` still contains the previous validation result, or `{}` on a fresh run |
| `on_validation_batch_end` | Reserved for evaluators that stream per-batch validation outputs | Same validation run as the surrounding validation hooks |
| `on_validation_end` | After `ValidationCallback` writes prefixed metrics to `state.last_val_metrics` | Validation metrics are available for loggers and checkpointing |
| `on_checkpoint_written` | After the `Checkpoint` callback atomically replaces a file (`last.pt`, `best.pt`, `epoch_NNNN.pt`); the trainer fans the event out to every callback in registration order | `path` is the file just written; all counters are at the values they had when `Checkpoint.on_train_epoch_end` fired |

**Step-boundary subtlety.** With `gradient_accumulation_steps > 1`, `on_train_batch_end` fires on *every* micro-batch, but `state.optimizer_step` only increments on real step boundaries (every Nth micro-batch and the last batch of the epoch). Callbacks that want to fire once per optimizer step should track `state.optimizer_step` and gate on changes — see [`StdoutLogger`](stdout_logger.py) for the pattern.

**`output.loss`** in `on_train_batch_end` is the un-scaled loss (before division by `grad_accum_steps`). The same scalar is mirrored in `state.last_loss` so callbacks needing it on hooks where `output` isn't passed (`on_train_epoch_end`) can read from state.

`ValidationCallback` fans out validation hooks through the trainer, matching the sibling-callback notification pattern used by `Checkpoint`.

---

## 4. `TrainerState` field ownership

```python
@dataclass
class TrainerState:
    # Identity — immutable for the run
    model: AudioModel
    objective: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_config: AudioSSLTrainingConfig
    config_fingerprint: str
    applied_runtime: AppliedRuntimeConfig

    # Counters — owned by Trainer
    epoch: int
    global_step: int
    optimizer_step: int

    # Telemetry — owned by Trainer
    last_loss: float | None
    current_lr: float | None
    last_grad_norm: float | None
    last_input_seconds: float | None
    last_throughput_audio_seconds_per_sec: float | None

    # Validation telemetry — owned by ValidationCallback
    last_val_metrics: dict[str, float]

    # save_best tracking — owned by the Checkpoint callback
    best_metric: float | None
    best_optimizer_step: int | None
```

| Field group | Write permissions |
| --- | --- |
| Identity | Trainer sets at `__init__`. Never mutated by anyone afterward. |
| Counters | Trainer-only. Callbacks **read**. Any callback writing these breaks the loop. |
| Telemetry | Trainer-only. Callbacks **read**. |
| `last_val_metrics` | [`ValidationCallback`](validation.py)-only. Other callbacks **read**. |
| `best_metric` / `best_optimizer_step` | [`Checkpoint`](checkpoint.py)-only. Other callbacks **read**. A second callback writing here is a contract violation — if you need a different "best" semantic, store it on `self`. |

If your callback needs a new piece of state that other callbacks should observe, do not add a field to `TrainerState` unilaterally. Propose a change to this contract first.

---

## 5. Attachment and ordering

The trainer assembles auto-attached + user callbacks, then sorts by
`Callback.phase` then registration order:

```text
evaluate   → ValidationCallback (from RunConfig.evaluation)
checkpoint → Checkpoint (from TrainerConfig.checkpoint / RunConfig)
log        → StdoutLogger, MetricsJsonlLogger, WandBLogger
observe    → user-provided Callbacks (default phase)
```

Operational logger settings live on `RunConfig` / `WandBConfig` and stay out of
`config_fingerprint`. See §9.

At shared epoch/step boundaries the trainer fires objective lifecycle hooks
**before** callback `on_*_end` hooks so epoch snapshots see objective updates.
`on_optimizer_step` fires after a successful (or counted) optimizer step and is
the preferred place for step-gated logging/checkpointing.

**Implications for new callbacks:**

- Choose `phase` so validation writes metrics before checkpoint reads them, and
  checkpoint writes paths before loggers upload artifacts.
- Cross-callback coordination uses `emit` events (`validation_start` /
  `validation_end`, `checkpoint_written`), not `isinstance` special cases.
- Order within a phase follows registration order.

---

## 6. Exceptions

- **Construction** — raise `ValueError` / `TypeError` on bad config. The trainer instantiates auto-attached callbacks in its `__init__`, so a construction-time raise becomes a `Trainer.__init__` raise.
- **Hooks** — a raise from any hook aborts `fit()`. The trainer does not catch. Choose between:
  - Raising for invariant violations that would corrupt downstream results (preferred for research-critical paths).
  - Silently no-op'ing optional behavior when prerequisites aren't met (e.g., `StdoutLogger` skips logging at `optimizer_step == 0` instead of raising).
- **Per project style** (`agent-instructions/project/CODE_STYLE_AND_REQUIREMENTS.md`): do not use broad `except` to keep training going past invalid state. Fail loud.

---

## 7. Determinism and I/O

**RNG.** A callback that consumes randomness must:

- Fork RNG state before sampling and restore on exit, OR
- Document the consumption as research-relevant and configure a generator on the callback (constructor arg).

The trainer does not re-seed between callback invocations. A callback that quietly advances global RNG between epochs changes the run.

**Filesystem.** Callbacks that persist data load-bearing for resume must write atomically:

```python
tmp = dst.with_suffix(dst.suffix + ".tmp")
with open(tmp, "wb") as fp:
    torch.save(payload, fp)
    fp.flush()
    os.fsync(fp.fileno())
os.replace(tmp, dst)
```

See [`checkpoint.py`](checkpoint.py) for the canonical implementation. Non-resume-critical artifacts (logs, plots, metrics files) may use simpler writes but should still tolerate preemption mid-write — typically by writing per-epoch rather than per-step, and by not leaving zero-byte sentinels.

**Network I/O.** [`WandBLogger`](wandb_logger.py) is the canonical network-I/O
callback: it initializes a wandb run in `on_train_start`, logs metrics in
`on_optimizer_step` (gated on `WandBConfig.log_every_n_steps`), uploads
checkpoints as artifacts in `on_checkpoint_written` when
`WandBConfig.upload_checkpoints=True`, and calls `wandb.finish()` in
`on_train_end` / `on_exception`. Fresh runs allocate a new UUID run id; resume
restores the id from `Callback.state_dict()` (`payload["callbacks"]["WandBLogger"]`).

---

## 8. Objective lifecycle hooks (not to be confused with this contract)

Every `Objective` inherits no-op lifecycle methods, listed in
`audiossl.modelling.objectives.OBJECTIVE_HOOKS`. Objectives override the ones
they need; the trainer calls all of them with a frozen `StepContext`:

| Method | Fires |
| --- | --- |
| `on_train_start(ctx)` | once, after trainer `on_train_start` callbacks |
| `on_epoch_start(ctx)` | every epoch, after `on_train_epoch_start` callbacks |
| `on_after_forward(output, ctx)` | every micro-batch, after forward and before backward |
| `on_before_optimizer_step(ctx)` | every optimizer-step boundary, after backward, before unscale/clip/step |
| `on_after_optimizer_step(ctx)` | every optimizer step, after the step and scheduler update (skipped advancement when `ctx.step_was_skipped`) |
| `on_epoch_end(ctx)` | every epoch, **before** callback `on_train_epoch_end` |
| `on_train_end(ctx)` | once, **before** callback `on_train_end` |

The trainer does not pass `TrainerState` to objectives. EMA teachers update in
`on_after_optimizer_step`; DINO updates its center in `on_after_forward`.

The legacy `objective.callbacks` dict is rejected: `Trainer.__init__` raises
`TypeError` if the objective defines a `callbacks` attribute, so an
un-migrated objective cannot silently skip its EMA update.

These hooks are **not** the place for logging, checkpointing, or metric
callbacks — those belong as `Callback` subclasses under this contract.

### Objective-owned RNG

`Checkpoint` records every generator from `objective.named_generators()` under
`payload["rng"]["objective_generators"]`. `Trainer.from_checkpoint` restores
them and raises `RuntimeError` when the set of names differs from the live
objective, because the resumed run would draw different masks or augmentations.

### Provenance

`TrainerState.provenance` (written into every checkpoint as
`payload["provenance"]`) records python/torch/torchaudio/transformers/audiossl
versions, CUDA/cuDNN versions, device name, the deterministic flag, and the git
commit and dirty flag. It is logged at `INFO` when the trainer is built. On
resume, differences are logged as a warning; they never block a resume.
`model.runtime.device` is excluded from `config_fingerprint`, so a run may
resume on different hardware.

---

## 9. Config-driven vs constructor-supplied callbacks

Two attachment paths exist:

| Path | When to use | Wiring |
| --- | --- | --- |
| **Constructor-supplied** | The callback's behavior is run-local and not part of the research config (a debug recorder, a test counter) | Pass via `Trainer(..., callbacks=[MyCallback(...)])`. The instance is the source of truth. |
| **Config-driven, auto-attached** | The callback's behavior is research-relevant and must be fingerprinted | Add a frozen config dataclass to `audiossl/config/config.py`; add a validator; add a field on `TrainerConfig`; instantiate inside `Trainer.__init__` from that field. The config goes into the run fingerprint automatically. |

**Rule:** if the callback's presence or settings change the resulting model checkpoint in a research-meaningful way, it **must** be config-driven. Constructor-supplied is reserved for run-local observers that don't affect outputs.

`Checkpoint` is config-driven (read from `TrainerConfig.checkpoint`). `StdoutLogger` is config-driven via the boolean `TrainerConfig.stdout_logging` + a hyperparameter from `TrainingLoopConfig.log_every_n_steps`. `WandBLogger` is config-driven via the separate `LoggingConfig` argument passed to `Trainer.__init__` — kept outside `AudioSSLTrainingConfig` precisely because wandb settings are observational and must not enter `config_fingerprint`.

---

## 10. Resume compatibility

The trainer supports `Trainer.from_checkpoint(...)` (see [`trainer.py`](../trainer.py)). Callbacks must not break across a resume.

**Stateless callbacks** (e.g., a `LossPrinter` that just reads `state.last_loss`) are automatically resume-safe. No action needed.

**Stateful callbacks** must choose one of:

1. **Reconstruct state from `TrainerState` at first hook fire.** Easiest. Example: `Checkpoint` reads `state.best_metric` (restored by `Trainer.from_checkpoint` from the payload), so it doesn't need its own resume hook. Prefer this when possible.
2. **Tolerate fresh start on resume.** Acceptable for callbacks whose state is purely cosmetic — e.g., `StdoutLogger._last_logged_step` resets to 0; the next log line just lands at the next step boundary regardless. Document this in a docstring.
3. **Persist via `state_dict` / `load_state_dict`.** Checkpoint stores
   `payload["callbacks"][callback.name]`. Resume raises if a saved name is
   missing from the live callback list. `WandBLogger` persists its run id this way.

A callback whose state, if lost, would change training outputs (not just logs) must use path 1. Anything else is a research-validity bug across preemption.

---

## 11. File layout

```text
src/audiossl/training/callbacks/
    __init__.py            # re-exports Callback + every shipped callback
    callback.py            # the ABC; do not modify lightly
    CONTRACT.md            # this file
    stdout_logger.py       # one callback per file
    checkpoint.py
    <new_callback>.py
```

- One callback per file. Module name matches the class in snake_case.
- Add the class to [`__init__.py`](__init__.py) so `from audiotrainer.callbacks import MyCallback` works.
- If the callback is auto-attached, also touch `Trainer.__init__` to instantiate it.

---

## 12. Testing minimums

In `tests/test_<callback_name>.py`, built on the shared [`tests/harness`](../../../../../tests/harness/__init__.py) (`DEFAULT_CASE.run(...)`, `HookRecorder`, `TrainingTrace`, `known_bug`):

- **Construction** — positive: callback constructs from valid args. Negative: each validation in `__init__` raises with the right error class.
- **Each overridden hook** — at least one test that confirms the hook fires and produces its expected effect (file written, state field updated, log line printed, etc.).
- **Non-perturbation** — add the callback to `OBSERVERS` in [`tests/test_trainer_reproducibility.py`](../../../../../tests/test_trainer_reproducibility.py) so attaching it is proven not to change the trained model (§7).
- **Resume parity** — if the callback is stateful: instantiate, save a checkpoint via the `Checkpoint` callback, restore via `Trainer.from_checkpoint`, run further, assert outputs are correct (see [`tests/test_trainer_resume.py`](../../../../../tests/test_trainer_resume.py)).
- **No-op on absent prerequisites** — e.g., when an optional state field is `None`, the callback handles it without raising (unless the callback's purpose is to raise on that condition).

Run via:

```powershell
cd trainer
.\.venv\Scripts\python.exe -m pytest tests -q
```

`python -O -m pytest` must also pass — use explicit `raise` statements, not `assert`.

---

## 13. Worked example: a minimal callback

A callback that records every loss for post-hoc plotting (run-local observer, not config-driven):

```python
# src/audiossl/training/callbacks/loss_recorder.py
from __future__ import annotations

from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput

from ..state import TrainerState
from .callback import Callback


class LossRecorder(Callback):
    """Append every micro-batch loss to an in-memory list for post-hoc inspection.

    Run-local observer. State is reset across resumes (path 2 of CONTRACT §10):
    losses from earlier epochs are lost on preemption, which is acceptable
    because this is a debug tool, not part of the research record.
    """

    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_train_batch_end(
        self,
        state: TrainerState,
        batch: AudioBatch,
        output: ObjectiveOutput,
    ) -> None:
        if state.last_loss is None:
            raise RuntimeError(
                "LossRecorder: state.last_loss should be set in on_train_batch_end"
            )
        self.losses.append(state.last_loss)
```

Wire it up:

```python
recorder = LossRecorder()
Trainer(model, objective, loader, train_config, callbacks=[recorder]).fit()
print(recorder.losses)
```

Add to [`__init__.py`](__init__.py) and `tests/test_loss_recorder.py`. Done.

---

## 14. Pre-merge checklist

Before requesting review:

- [ ] One callback per file under `audiossl/training/callbacks/`.
- [ ] Subclass of `Callback`. Only the hooks you need are overridden.
- [ ] Constructor arguments are keyword-only and validated.
- [ ] No writes to `TrainerState` outside your callback's owned fields (per §4).
- [ ] No interaction with sibling callbacks.
- [ ] Atomic writes for resume-critical artifacts (per §7).
- [ ] Re-exported from `audiossl/training/callbacks/__init__.py`.
- [ ] Tests cover construction, each overridden hook, and resume behavior if stateful.
- [ ] `python -m pytest tests/ -q` and `python -O -m pytest tests/ -q` both green.
- [ ] If config-driven: dataclass added to `audiossl/config/config.py`, validator added to `audiossl/config/validate_config.py`, `TrainerConfig` field added, `Trainer.__init__` wired.

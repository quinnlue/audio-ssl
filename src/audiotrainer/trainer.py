"""Single-device SSL training loop."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from audiossl.config import (
    AudioSSLTrainingConfig,
    MixedPrecisionConfig,
    validate_config,
)
from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput
from audiossl.config.run import RunConfig
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives._base import OBJECTIVE_HOOKS, Objective
from audiossl.runtime import apply_global_runtime, derive_seed

from .callbacks import (
    Callback,
    Checkpoint,
    MetricsJsonlLogger,
    StdoutLogger,
    ValidationCallback,
    WandBLogger,
    build_checkpoint_payload,
    sort_callbacks,
)
from .fingerprint import config_fingerprint, resume_compatible_payload
from .optim import build_optimizer, build_scheduler
from .provenance import assert_clean_git, collect_provenance
from .rng import restore_training_rng
from .state import TrainerState
from .step import StepContext

logger = logging.getLogger(__name__)

_MP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
_THROUGHPUT_EMA_ALPHA = 0.1
_SNAPSHOT_FIELDS = (
    "epoch",
    "global_step",
    "optimizer_step",
    "skipped_optimizer_steps",
    "consecutive_skipped_steps",
    "last_loss",
    "current_lr",
    "last_grad_norm",
    "last_input_seconds",
    "last_throughput_audio_seconds_per_sec",
    "last_metrics",
    "last_val_metrics",
    "best_metric",
    "best_optimizer_step",
)


def _copy_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = copy.copy(value)
        result.update((key, _copy_to_cpu(item)) for key, item in value.items())
        return result
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _configure_package_logging(run_dir: Path | None) -> None:
    root = logging.getLogger("audiotrainer")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(name)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
        logging.getLogger("audiossl").addHandler(file_handler)
        logging.getLogger("audiossl").setLevel(logging.INFO)


class Trainer:
    """Single-device SSL trainer.

    Prefer :func:`audiotrainer.build.build_trainer` / :meth:`from_config` for
    research runs. The low-level constructor accepts pre-built objects and
    asserts what it can against ``train_config``.
    """

    def __init__(
        self,
        model: AudioModel,
        objective: Objective,
        train_loader: DataLoader | None,
        train_config: AudioSSLTrainingConfig,
        *,
        run_config: RunConfig | None = None,
        data_fingerprint: str,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        if not isinstance(model, AudioModel):
            raise TypeError(f"Trainer expected AudioModel, got {type(model).__name__}")
        if not isinstance(objective, Objective):
            raise TypeError(f"Trainer expected Objective, got {type(objective).__name__}")
        if train_loader is not None and not isinstance(train_loader, DataLoader):
            raise TypeError(
                f"Trainer expected torch DataLoader or None, got {type(train_loader).__name__}"
            )
        if not isinstance(train_config, AudioSSLTrainingConfig):
            raise TypeError(
                "Trainer expected AudioSSLTrainingConfig, "
                f"got {type(train_config).__name__}"
            )
        if not isinstance(data_fingerprint, str) or not data_fingerprint:
            raise ValueError(
                f"Trainer requires a non-empty data_fingerprint, got {data_fingerprint!r}"
            )
        run_config = run_config or RunConfig()
        if not isinstance(run_config, RunConfig):
            raise TypeError(
                f"Trainer.run_config must be RunConfig, got {type(run_config).__name__}"
            )
        if not isinstance(callbacks, Sequence) or isinstance(callbacks, str):
            raise TypeError(
                f"Trainer.callbacks must be a sequence, got {type(callbacks).__name__}"
            )
        for index, callback in enumerate(callbacks):
            if not isinstance(callback, Callback):
                raise TypeError(
                    f"Trainer.callbacks[{index}] must be Callback, "
                    f"got {type(callback).__name__}"
                )

        validate_config(train_config)
        validate_config(run_config)

        if model.config != train_config.model:
            raise ValueError(
                "Trainer: passed-in model.config does not match train_config.model"
            )
        objective_config = getattr(objective, "config", None)
        if objective_config is None:
            raise ValueError(
                f"Trainer: objective must expose .config; {type(objective).__name__} does not."
            )
        if objective_config != train_config.objective:
            raise ValueError(
                "Trainer: passed-in objective.config does not match train_config.objective"
            )
        if hasattr(objective, "callbacks") and not callable(
            getattr(type(objective), "callbacks", None)
        ):
            # Reject the legacy dict attribute, not the Callback list method.
            attr = objective.callbacks
            if isinstance(attr, dict):
                raise TypeError(
                    f"Trainer: {type(objective).__name__}.callbacks is no longer supported; "
                    "override the Objective lifecycle methods instead: "
                    f"{', '.join(OBJECTIVE_HOOKS)}"
                )

        trainer_cfg = train_config.trainer
        batch_cfg = trainer_cfg.batch
        loop_cfg = trainer_cfg.loop

        if train_loader is not None and train_loader.batch_size not in (
            None,
            batch_cfg.per_device_batch_size,
        ):
            raise ValueError(
                "Trainer: train_loader.batch_size "
                f"({train_loader.batch_size}) != trainer.batch.per_device_batch_size "
                f"({batch_cfg.per_device_batch_size})"
            )
        if train_loader is not None and train_loader.drop_last not in (None, batch_cfg.drop_last):
            raise ValueError(
                "Trainer: train_loader.drop_last "
                f"({train_loader.drop_last}) != trainer.batch.drop_last ({batch_cfg.drop_last})"
            )
        aug = getattr(objective, "augmentor", None)
        aug_config = getattr(aug, "config", None)
        if aug_config is not None and trainer_cfg.augmentor is not None:
            if aug_config != trainer_cfg.augmentor:
                raise ValueError(
                    "Trainer: objective.augmentor.config does not match "
                    "train_config.trainer.augmentor"
                )
        if (
            run_config.evaluation is not None
            and not any(isinstance(cb, ValidationCallback) for cb in callbacks)
        ):
            raise ValueError(
                "RunConfig.evaluation requires a ValidationCallback in "
                "Trainer(callbacks=[...]) or use build_trainer()"
            )

        device = model.applied_runtime.device
        if trainer_cfg.mixed_precision is not None:
            if trainer_cfg.mixed_precision.dtype == "float16" and device.type != "cuda":
                raise RuntimeError(
                    "MixedPrecisionConfig.dtype='float16' requires a CUDA device, "
                    f"got device.type={device.type!r}."
                )
            if trainer_cfg.mixed_precision.grad_scaler and device.type != "cuda":
                raise RuntimeError(
                    "MixedPrecisionConfig.grad_scaler=True requires CUDA, "
                    f"got device.type={device.type!r}."
                )
            if model.config.runtime.dtype != "float32":
                raise ValueError(
                    "MixedPrecisionConfig requires model.runtime.dtype='float32' "
                    f"(master weights in float32); got {model.config.runtime.dtype!r}"
                )

        apply_global_runtime(model.config.runtime)

        self.model = model
        self.objective = objective
        self.train_loader = train_loader
        self.train_config = train_config
        self.run_config = run_config
        self._data_fingerprint = data_fingerprint
        self._device = device
        self._grad_accum_steps = batch_cfg.gradient_accumulation_steps
        self._grad_clip_norm = loop_cfg.gradient_clip_norm
        self._max_epochs = loop_cfg.max_epochs
        self._max_steps = loop_cfg.max_steps
        self._nonfinite_policy = loop_cfg.nonfinite_policy
        self._max_consecutive_skipped = loop_cfg.max_consecutive_skipped_steps
        self._mp_config: MixedPrecisionConfig | None = trainer_cfg.mixed_precision

        self.optimizer = build_optimizer(objective, trainer_cfg.optimizer)
        self._total_optimizer_steps = self._compute_total_optimizer_steps()
        self.scheduler = build_scheduler(
            self.optimizer,
            trainer_cfg.scheduler,
            total_steps=self._total_optimizer_steps,
        )

        self._scaler: torch.amp.GradScaler | None = None
        if self._mp_config is not None and self._mp_config.grad_scaler:
            self._scaler = torch.amp.GradScaler(device=device.type)

        fingerprint = config_fingerprint(
            train_config, data_fingerprint=data_fingerprint
        )
        provenance = collect_provenance(model.applied_runtime)
        if run_config.require_clean_git:
            assert_clean_git(provenance)

        run_dir = self._resolve_run_dir()
        _configure_package_logging(run_dir)
        if run_dir is not None:
            self._write_run_record(run_dir, fingerprint, provenance)

        self.state = TrainerState(
            model=model,
            objective=objective,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_config=train_config,
            run_config=run_config,
            config_fingerprint=fingerprint,
            data_fingerprint=data_fingerprint,
            applied_runtime=model.applied_runtime,
            provenance=provenance,
        )
        logger.info("run fingerprint=%s provenance=%s", fingerprint, provenance)

        auto_callbacks: list[Callback] = []
        if run_dir is not None:
            auto_callbacks.append(MetricsJsonlLogger(run_dir))
        if run_config.stdout_logging:
            auto_callbacks.append(
                StdoutLogger(log_every_n_steps=run_config.log_every_n_steps)
            )
        checkpoint_callback: Checkpoint | None = None
        if run_config.checkpoint is not None:
            checkpoint_callback = Checkpoint(run_config.checkpoint)
            auto_callbacks.append(checkpoint_callback)
        for logger_cfg in run_config.loggers:
            if isinstance(logger_cfg, type(None)):
                continue
            from audiossl.config import WandBConfig

            if isinstance(logger_cfg, WandBConfig) and not any(
                isinstance(cb, WandBLogger) for cb in callbacks
            ):
                auto_callbacks.append(
                    WandBLogger(config=logger_cfg, train_config=train_config)
                )

        self.callbacks: list[Callback] = sort_callbacks(auto_callbacks + list(callbacks))
        for callback in self.callbacks:
            callback.attach(self._emit_event)
        if checkpoint_callback is not None:
            checkpoint_callback._callback_states_provider = self._collect_callback_states

        self._sample_rate_hz = self._resolve_sample_rate()
        self._last_step_perf_counter: float | None = None
        self._throughput_ema: float | None = None
        self._pending_step_input_seconds = 0.0
        self._pending_step_loss_sum = 0.0
        self._pending_step_loss_count: float = 0
        self._start_epoch = 0
        self._resume_batch_index = 0
        self._stop = False
        self._update_session_active = False
        self._updating = False
        self._last_output: ObjectiveOutput | None = None

    @classmethod
    def from_config(
        cls,
        train_config: AudioSSLTrainingConfig,
        data,
        *,
        run_config: RunConfig | None = None,
        callbacks: Sequence[Callback] = (),
    ) -> Trainer:
        """Build a trainer from config + DataModule (see :func:`build_trainer`)."""
        from .build import build_trainer

        return build_trainer(
            train_config, data, run_config=run_config, callbacks=callbacks
        )

    def fit(self) -> TrainerState:
        if self.train_loader is None:
            raise RuntimeError("Trainer.fit requires a DataLoader; use update_session()")
        self._last_step_perf_counter = None
        self._throughput_ema = None
        self._pending_step_input_seconds = 0.0
        self._pending_step_loss_sum = 0.0
        self._pending_step_loss_count = 0
        self._stop = False

        try:
            self._call_callbacks("on_train_start")
            self._call_objective_hooks("on_train_start")
            epoch = self._start_epoch
            while not self._stop:
                if self._max_epochs is not None and epoch >= self._max_epochs:
                    break
                if self._max_steps is not None and self.state.optimizer_step >= self._max_steps:
                    break
                self.state.epoch = epoch
                self._call_callbacks("on_train_epoch_start")
                self._call_objective_hooks("on_epoch_start")
                self._train_epoch()
                self._call_objective_hooks("on_epoch_end")
                self._call_callbacks("on_train_epoch_end")
                epoch += 1
                self._resume_batch_index = 0
            self._call_objective_hooks("on_train_end")
            self._call_callbacks("on_train_end")
        except BaseException as exc:
            try:
                self._call_callbacks("on_exception", exc=exc)
            finally:
                raise
        return self.state

    @contextlib.contextmanager
    def update_session(self) -> Iterator[Trainer]:
        """Run caller-defined updates with one training and epoch lifecycle.

        Construct with ``train_loader=None`` and set ``loop.max_steps`` for
        the scheduler horizon. The caller controls the actual stop point.
        """
        if self.train_loader is not None or self._update_session_active:
            raise RuntimeError("Trainer.update_session requires an idle loader-free trainer")
        self._update_session_active = True
        self._last_step_perf_counter = time.perf_counter()
        try:
            self._call_callbacks("on_train_start")
            self._call_objective_hooks("on_train_start")
            self._call_callbacks("on_train_epoch_start")
            self._call_objective_hooks("on_epoch_start")
            self._clear_accumulation()
            yield self
            self._call_objective_hooks("on_epoch_end")
            self._call_callbacks("on_train_epoch_end")
            self._call_objective_hooks("on_train_end")
            self._call_callbacks("on_train_end")
        except BaseException as exc:
            try:
                self._call_callbacks("on_exception", exc=exc)
            finally:
                raise
        finally:
            self._update_session_active = False

    def train_update(
        self,
        microbatches: Sequence[AudioBatch],
        *,
        loss_weight: Callable[[ObjectiveOutput], float] | None = None,
        learning_rate: float | None = None,
    ) -> bool:
        """Apply one update inside ``update_session`` and return whether it stepped.

        The objective returns a microbatch mean. ``loss_weight`` supplies its
        positive denominator (for example, the number of masked frames), or
        each microbatch receives equal weight. A caller-supplied learning rate
        applies to all parameter groups and bypasses the scheduler on this step.
        """
        if not self._update_session_active or self._updating:
            raise RuntimeError("Trainer.train_update requires an active update session")
        microbatch_count = len(microbatches)
        if microbatch_count == 0:
            raise ValueError("Trainer.train_update requires at least one microbatch")
        if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate < 0):
            raise ValueError("Trainer.train_update learning_rate must be finite and nonnegative")

        self._updating = True
        initial_step = self.state.optimizer_step
        weight_sum = 0.0
        metric_sums: dict[str, float] = {}
        metric_weights: dict[str, float] = {}
        try:
            self.objective.train()
            for index, microbatch in enumerate(microbatches, start=1):
                batch, output = self._forward_batch(microbatch)
                self.state.global_step += 1
                if not torch.isfinite(output.loss).all():
                    self._handle_nonfinite("loss", float("nan"))
                    self._clear_accumulation()
                    return False
                weight = 1.0 if loss_weight is None else float(loss_weight(output))
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(
                        "Trainer.train_update loss weights must be finite and positive"
                    )
                loss = output.loss
                if loss.dtype in (torch.float16, torch.bfloat16):
                    loss = loss.float()
                self._backward(loss * weight)
                self._pending_step_loss_sum += float(output.loss.detach().item()) * weight
                weight_sum += weight
                for key, value in self._objective_metrics(output, self._step_context()).items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + weight * value
                    metric_weights[key] = metric_weights.get(key, 0.0) + weight

                if index == microbatch_count:
                    self._pending_step_loss_count = weight_sum
                    self.state.last_input_seconds = self._pending_step_input_seconds
                    self._optimizer_step_boundary(
                        grad_divisor=weight_sum,
                        learning_rate=learning_rate,
                        objective_metrics={
                            key: value / metric_weights[key] for key, value in metric_sums.items()
                        },
                    )
                self._call_callbacks("on_train_batch_end", batch=batch, output=output)
            return self.state.optimizer_step > initial_step
        except BaseException:
            self._clear_accumulation()
            raise
        finally:
            self._last_output = None
            self._updating = False

    def snapshot(self, *, data_state: Any = None) -> dict[str, Any]:
        """Copy loader-free training state to CPU between updates, including caller data state."""
        if self.train_loader is not None or self._updating:
            raise RuntimeError("Trainer.snapshot requires a loader-free update boundary")
        payload = build_checkpoint_payload(self.state)
        if getattr(self.objective, "model", None) is self.model:
            del payload["model_state"]
        else:
            payload["model_state"] = self.model.state_dict()
        payload["update_state"] = {name: getattr(self.state, name) for name in _SNAPSHOT_FIELDS}
        payload["scaler_state"] = self._scaler.state_dict() if self._scaler is not None else None
        snapshot = _copy_to_cpu(payload)
        snapshot["data_state"] = copy.deepcopy(data_state)
        return snapshot

    def restore(self, snapshot: dict[str, Any]) -> Any:
        """Restore a snapshot between updates and return an independent copy of its data state."""
        if self.train_loader is not None or self._updating:
            raise RuntimeError("Trainer.restore requires a loader-free update boundary")
        if snapshot["config_fingerprint"] != self.state.config_fingerprint:
            raise ValueError("Trainer.restore: snapshot fingerprint does not match the trainer")
        self.objective.load_state_dict(snapshot["objective_state"])
        if "model_state" in snapshot:
            self.model.load_state_dict(snapshot["model_state"])
        self.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer_state"]))
        self.scheduler.load_state_dict(copy.deepcopy(snapshot["scheduler_state"]))
        self._clear_accumulation()
        if self._scaler is not None:
            self._scaler.load_state_dict(snapshot["scaler_state"])
        for name in _SNAPSHOT_FIELDS:
            setattr(self.state, name, copy.deepcopy(snapshot["update_state"][name]))
        restore_training_rng(self.objective, snapshot["rng"])
        self._throughput_ema = self.state.last_throughput_audio_seconds_per_sec
        self._last_step_perf_counter = time.perf_counter()
        self._last_output = None
        return copy.deepcopy(snapshot["data_state"])

    def announce_artifact(self, path: str | Path) -> None:
        """Notify checkpoint consumers after the caller has finalized a file or directory."""
        path = Path(path)
        if not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(path)
        self._emit_event("checkpoint_written", path=path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        model: AudioModel,
        objective: Objective,
        train_loader: DataLoader,
        train_config: AudioSSLTrainingConfig,
        run_config: RunConfig | None = None,
        data_fingerprint: str,
        callbacks: Sequence[Callback] = (),
    ) -> Trainer:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trainer.from_checkpoint: {path} does not exist")

        payload = torch.load(
            path,
            map_location=model.applied_runtime.device,
            weights_only=False,
        )

        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
        try:
            batches_per_epoch = len(train_loader)
        except TypeError:
            batches_per_epoch = None

        if batches_per_epoch:
            batches_done_in_epoch = global_step % batches_per_epoch
            if batches_done_in_epoch == 0:
                # Exactly at an epoch boundary: start the next epoch.
                next_epoch = epoch + 1
                resume_batch_index = 0
            else:
                next_epoch = epoch
                resume_batch_index = batches_done_in_epoch
        else:
            next_epoch = epoch + 1
            resume_batch_index = 0

        if (
            train_config.trainer.loop.max_epochs is not None
            and resume_batch_index == 0
            and train_config.trainer.loop.max_epochs < next_epoch
        ):
            raise RuntimeError(
                "Trainer.from_checkpoint: train_config.trainer.loop.max_epochs "
                f"({train_config.trainer.loop.max_epochs}) is before the next epoch "
                f"to run ({next_epoch}). Increase max_epochs or use an earlier checkpoint."
            )

        expected_fingerprint = config_fingerprint(
            train_config, data_fingerprint=data_fingerprint
        )
        actual_fingerprint = payload.get("config_fingerprint")
        if actual_fingerprint != expected_fingerprint and not _matches_resume_config(
            payload.get("train_config"),
            train_config,
            data_fingerprint=data_fingerprint,
            payload_data_fingerprint=payload.get("data_fingerprint"),
        ):
            raise RuntimeError(
                "Trainer.from_checkpoint: fingerprint mismatch. "
                f"checkpoint={actual_fingerprint!r}, train_config={expected_fingerprint!r}."
            )

        objective.load_state_dict(payload["objective_state"])
        # Keep model alias in sync when objective owns the student.
        if "model_state" in payload and not any(
            key.startswith("model.") for key in payload["objective_state"]
        ):
            model.load_state_dict(payload["model_state"])

        trainer = cls(
            model,
            objective,
            train_loader,
            train_config,
            run_config=run_config,
            data_fingerprint=data_fingerprint,
            callbacks=callbacks,
        )

        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        # Rebuild horizon-dependent schedules against the new total_steps.
        scheduler_cfg = train_config.trainer.scheduler
        if getattr(scheduler_cfg, "horizon_dependent", False):
            trainer.scheduler = build_scheduler(
                trainer.optimizer,
                scheduler_cfg,
                total_steps=trainer._total_optimizer_steps,
            )
            # Advance the new schedule to the resumed optimizer step.
            for _ in range(int(payload["optimizer_step"])):
                trainer.scheduler.step()
        else:
            trainer.scheduler.load_state_dict(payload["scheduler_state"])

        saved_provenance = payload.get("provenance") or {}
        changed = sorted(
            key
            for key in set(saved_provenance) | set(trainer.state.provenance)
            if key not in {"argv", "hostname"}
            and saved_provenance.get(key) != trainer.state.provenance.get(key)
        )
        if changed:
            message = (
                "Trainer.from_checkpoint: environment differs from the checkpoint in "
                f"{', '.join(changed)}; resumed numerics may not match an uninterrupted run"
            )
            policy = trainer.run_config.provenance_mismatch
            if policy == "raise":
                raise RuntimeError(message)
            logger.warning(message)

        trainer.state.global_step = global_step
        trainer.state.optimizer_step = int(payload["optimizer_step"])
        trainer.state.skipped_optimizer_steps = int(
            payload.get("skipped_optimizer_steps", 0)
        )
        trainer.state.last_val_metrics = dict(payload.get("last_val_metrics", {}))
        trainer.state.best_metric = payload.get("best_metric")
        trainer.state.best_optimizer_step = payload.get("best_optimizer_step")
        trainer.state.epoch = next_epoch
        trainer._start_epoch = next_epoch
        trainer._resume_batch_index = resume_batch_index

        callback_states = payload.get("callbacks") or {}
        present = {cb.callback_name() for cb in trainer.callbacks}
        missing = sorted(set(callback_states) - present)
        if missing:
            raise RuntimeError(
                "Trainer.from_checkpoint: checkpoint has callback state for "
                f"{missing} but those callbacks are not attached"
            )
        for callback in trainer.callbacks:
            name = callback.callback_name()
            if name in callback_states:
                callback.load_state_dict(callback_states[name])

        restore_training_rng(objective, payload["rng"])
        return trainer

    # ---------------- internals ------------------------------------------

    def _compute_total_optimizer_steps(self) -> int:
        loop = self.train_config.trainer.loop
        if loop.max_steps is not None:
            return max(loop.max_steps, 1)
        if self.train_loader is None:
            raise ValueError("Trainer: loader-free training requires TrainingLoopConfig.max_steps")
        try:
            n_batches = len(self.train_loader)
        except TypeError as exc:
            raise ValueError(
                "Trainer: train_loader has no length; set TrainingLoopConfig.max_steps"
            ) from exc
        if n_batches == 0:
            raise ValueError("Trainer: train_loader has zero batches")
        if loop.max_epochs is None:
            raise ValueError(
                "Trainer: set TrainingLoopConfig.max_epochs or max_steps"
            )
        steps_per_epoch = max(math.ceil(n_batches / self._grad_accum_steps), 1)
        return max(steps_per_epoch * loop.max_epochs, 1)

    def _resolve_run_dir(self) -> Path | None:
        if self.run_config.run_dir is not None:
            return Path(self.run_config.run_dir)
        if self.run_config.checkpoint is not None:
            return Path(self.run_config.checkpoint.save_dir)
        return None

    def _write_run_record(
        self,
        run_dir: Path,
        fingerprint: str,
        provenance: dict[str, Any],
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        from .fingerprint import _to_plain

        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "train_config": _to_plain(asdict(self.train_config)),
                    "run_config": _to_plain(asdict(self.run_config)),
                    "config_fingerprint": fingerprint,
                    "data_fingerprint": self._data_fingerprint,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _resolve_sample_rate(self) -> float:
        processor = self.train_config.model.processor
        for attr in ("sr", "sample_rate", "target_sample_rate"):
            value = getattr(processor, attr, None)
            if value is not None:
                return float(value)
        logger.warning(
            "Trainer: model.processor has no sr/sample_rate; "
            "throughput logging assumes 16000 Hz"
        )
        return 16_000.0

    def _step_context(self, *, step_was_skipped: bool = False) -> StepContext:
        return StepContext(
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            optimizer_step=self.state.optimizer_step,
            total_optimizer_steps=self._total_optimizer_steps,
            step_was_skipped=step_was_skipped,
        )

    def _forward_batch(self, batch: AudioBatch) -> tuple[AudioBatch, ObjectiveOutput]:
        batch = self._move_batch(batch)
        input_seconds = float(batch.mask.sum().item()) / self._sample_rate_hz
        self.state.last_input_seconds = input_seconds
        self._pending_step_input_seconds += input_seconds
        self._call_callbacks("on_train_batch_start", batch=batch)
        with self._autocast_ctx():
            output = self.objective(batch)
        self._call_objective_hooks("on_after_forward", output)
        return batch, output

    def _train_epoch(self) -> None:
        assert self.train_loader is not None
        self.objective.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            n_batches = len(self.train_loader)
        except TypeError:
            n_batches = None

        # Per-epoch data RNG when the loader exposes a generator.
        data_seed = derive_seed(
            self.model.applied_runtime.seed, f"data/epoch/{self.state.epoch}"
        )
        loader_generator = getattr(self.train_loader, "generator", None)
        if isinstance(loader_generator, torch.Generator):
            loader_generator.manual_seed(data_seed)

        skip = self._resume_batch_index
        for batch_index, batch in enumerate(self.train_loader, start=1):
            if skip > 0:
                skip -= 1
                continue
            if self._max_steps is not None and self.state.optimizer_step >= self._max_steps:
                self._stop = True
                break

            batch, output = self._forward_batch(batch)
            loss = output.loss
            if not torch.isfinite(loss).all():
                self._handle_nonfinite("loss", float("nan"))
                self.state.global_step += 1
                continue

            # Scale by microbatches in this window (partial last window).
            window_size = self._window_size(batch_index, n_batches)
            scaled = loss / window_size
            self._backward(scaled)
            self._pending_step_loss_sum += float(loss.detach().item())
            self._pending_step_loss_count += 1
            self.state.global_step += 1

            is_step_boundary = (
                batch_index % self._grad_accum_steps == 0
                or (n_batches is not None and batch_index == n_batches)
            )
            if is_step_boundary:
                self._optimizer_step_boundary()

            self._call_callbacks("on_train_batch_end", batch=batch, output=output)
            if self._stop:
                break

        self._resume_batch_index = 0

    def _window_size(self, batch_index: int, n_batches: int | None) -> int:
        if n_batches is None:
            return self._grad_accum_steps
        # Size of the current accumulation window ending at or after batch_index.
        start = batch_index - ((batch_index - 1) % self._grad_accum_steps)
        end = min(start + self._grad_accum_steps - 1, n_batches)
        return end - start + 1

    def _clear_accumulation(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_step_input_seconds = 0.0
        self._pending_step_loss_sum = 0.0
        self._pending_step_loss_count = 0

    def _optimizer_step_boundary(
        self,
        *,
        grad_divisor: float | None = None,
        learning_rate: float | None = None,
        objective_metrics: dict[str, float] | None = None,
    ) -> None:
        if grad_divisor is not None:
            for param in self.objective.parameters():
                if param.grad is not None:
                    param.grad.div_(grad_divisor)
        self._call_objective_hooks("on_before_optimizer_step")
        trainable_params = [p for p in self.objective.parameters() if p.requires_grad]

        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=(
                self._grad_clip_norm if self._grad_clip_norm is not None else math.inf
            ),
            error_if_nonfinite=False,
        )
        if not torch.isfinite(grad_norm):
            self._handle_nonfinite("grad_norm", float(grad_norm))
            if self._scaler is not None:
                self._scaler.update()
            self._clear_accumulation()
            return

        self.state.last_grad_norm = float(grad_norm)
        scale_before = self._scaler.get_scale() if self._scaler is not None else None
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        self._step_optimizer()
        skipped = False
        if self._scaler is not None and scale_before is not None:
            scale_after = self._scaler.get_scale()
            if scale_after < scale_before:
                skipped = True

        if skipped:
            self._handle_nonfinite("grad_scaler", float("nan"), count_only=True)
            ctx = self._step_context(step_was_skipped=True)
            self._call_objective_hooks("on_after_optimizer_step", ctx_override=ctx)
            self._clear_accumulation()
            return

        if learning_rate is None:
            self.scheduler.step()
        self.state.optimizer_step += 1
        self.state.consecutive_skipped_steps = 0
        self.state.current_lr = float(self.optimizer.param_groups[0]["lr"])
        if self._pending_step_loss_count:
            self.state.last_loss = (
                self._pending_step_loss_sum / self._pending_step_loss_count
            )
        self._update_throughput(self._pending_step_input_seconds)
        self._clear_accumulation()

        ctx = self._step_context(step_was_skipped=False)
        self._call_objective_hooks("on_after_optimizer_step", ctx_override=ctx)
        self.state.last_metrics = self._collect_metrics(ctx, objective_metrics)
        self._call_callbacks("on_optimizer_step", ctx=ctx)

        if self._max_steps is not None and self.state.optimizer_step >= self._max_steps:
            self._stop = True

    def _collect_metrics(
        self, ctx: StepContext, objective_metrics: dict[str, float] | None = None
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "train/loss": float(self.state.last_loss or 0.0),
            "train/epoch": float(ctx.epoch),
            "train/global_step": float(ctx.global_step),
            "train/optimizer_step": float(ctx.optimizer_step),
        }
        if self.state.current_lr is not None:
            metrics["train/lr"] = float(self.state.current_lr)
        if self.state.last_grad_norm is not None:
            metrics["train/grad_norm"] = float(self.state.last_grad_norm)
        if self.state.last_input_seconds is not None:
            metrics["train/input_seconds"] = float(self.state.last_input_seconds)
        if self.state.last_throughput_audio_seconds_per_sec is not None:
            metrics["train/throughput_audio_seconds_per_sec"] = float(
                self.state.last_throughput_audio_seconds_per_sec
            )
        if objective_metrics is None and self._last_output is not None:
            objective_metrics = self._objective_metrics(self._last_output, ctx)
        if objective_metrics is not None:
            metrics.update(objective_metrics)
        return metrics

    def _objective_metrics(self, output: ObjectiveOutput, ctx: StepContext) -> dict[str, float]:
        objective_type = getattr(self.train_config.objective, "type", "objective")
        result = {}
        for key, value in self.objective.metrics(output, ctx).items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            result[f"train/{objective_type}/{key}"] = float(value)
        return result

    def _handle_nonfinite(
        self, kind: str, value: float, *, count_only: bool = False
    ) -> None:
        self.state.skipped_optimizer_steps += 1
        self.state.consecutive_skipped_steps += 1
        logger.warning(
            "non-finite %s at epoch=%s step=%s value=%s (consecutive=%s)",
            kind,
            self.state.epoch,
            self.state.optimizer_step,
            value,
            self.state.consecutive_skipped_steps,
        )
        if self._nonfinite_policy == "raise" and not count_only:
            raise RuntimeError(
                f"Trainer: non-finite {kind} at optimizer_step="
                f"{self.state.optimizer_step}"
            )
        if self.state.consecutive_skipped_steps > self._max_consecutive_skipped:
            raise RuntimeError(
                "Trainer: exceeded max_consecutive_skipped_steps="
                f"{self._max_consecutive_skipped} after non-finite {kind}"
            )

    def _update_throughput(self, input_seconds: float) -> None:
        now = time.perf_counter()
        if self._last_step_perf_counter is None:
            self._last_step_perf_counter = now
            return
        dt = now - self._last_step_perf_counter
        self._last_step_perf_counter = now
        if dt <= 0.0 or input_seconds <= 0.0:
            return
        instantaneous = input_seconds / dt
        if self._throughput_ema is None:
            self._throughput_ema = instantaneous
        else:
            self._throughput_ema = (
                _THROUGHPUT_EMA_ALPHA * instantaneous
                + (1.0 - _THROUGHPUT_EMA_ALPHA) * self._throughput_ema
            )
        self.state.last_throughput_audio_seconds_per_sec = self._throughput_ema

    def _emit_event(self, event: str, **payload: Any) -> None:
        if event == "checkpoint_written":
            path = payload["path"]
            for callback in self.callbacks:
                callback.on_checkpoint_written(self.state, path)
            return
        if event == "validation_start":
            for callback in self.callbacks:
                callback.on_validation_start(self.state)
            return
        if event == "validation_end":
            for callback in self.callbacks:
                callback.on_validation_end(self.state)
            return
        raise ValueError(f"Trainer: unknown event {event!r}")

    def _collect_callback_states(self) -> dict[str, dict[str, Any]]:
        return {
            callback.callback_name(): callback.state_dict()
            for callback in self.callbacks
        }

    def _autocast_ctx(self):
        if self._mp_config is None:
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self._device.type,
            dtype=_MP_DTYPES[self._mp_config.dtype],
        )

    def _backward(self, loss: torch.Tensor) -> None:
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def _step_optimizer(self) -> None:
        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

    def _move_batch(self, batch: Any) -> AudioBatch:
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "Trainer: train_loader must yield AudioBatch, "
                f"got {type(batch).__name__}"
            )
        return batch.to(self._device, non_blocking=True)

    def _call_callbacks(self, method_name: str, **kwargs: Any) -> None:
        for callback in self.callbacks:
            getattr(callback, method_name)(self.state, **kwargs)

    def _call_objective_hooks(
        self,
        name: str,
        *args: Any,
        ctx_override: StepContext | None = None,
    ) -> None:
        if name not in OBJECTIVE_HOOKS:
            raise ValueError(f"Trainer: unknown objective hook {name!r}")
        ctx = ctx_override or self._step_context()
        if name == "on_after_forward":
            output = args[0]
            self._last_output = output
            self.objective.on_after_forward(output, ctx)
            return
        getattr(self.objective, name)(ctx)


def _matches_resume_config(
    checkpoint_train_config: Any,
    train_config: AudioSSLTrainingConfig,
    *,
    data_fingerprint: str,
    payload_data_fingerprint: str | None,
) -> bool:
    if checkpoint_train_config is None:
        return False
    if payload_data_fingerprint not in (None, data_fingerprint):
        return False
    checkpoint_payload = resume_compatible_payload(
        checkpoint_train_config, data_fingerprint=data_fingerprint
    )
    current_payload = resume_compatible_payload(
        train_config, data_fingerprint=data_fingerprint
    )
    return checkpoint_payload == current_payload


__all__ = ["Trainer"]

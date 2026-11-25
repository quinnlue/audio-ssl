"""Crash-safe checkpoint callback with periodic snapshots, save_best, and resume payload."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, ClassVar

import torch

from audiossl.config.run import CheckpointConfig

from ..fingerprint import _to_plain
from ..rng import objective_generator_states
from ..state import TrainerState
from .callback import Callback


def _atomic_save(payload: dict[str, Any], dst: Path) -> None:
    """Crash-safe save: write to ``<dst>.tmp``, fsync, then ``os.replace``."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as fp:
        torch.save(payload, fp)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, dst)


def build_checkpoint_payload(
    state: TrainerState,
    *,
    callback_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the on-disk checkpoint payload. Exposed for testing and tooling."""

    rng = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "objective_generators": objective_generator_states(state.objective),
    }
    # Single source of truth: objective_state includes the student model.
    objective_state = state.objective.state_dict()
    from dataclasses import asdict

    return {
        "objective_state": objective_state,
        "model_state": {
            key.removeprefix("model."): value
            for key, value in objective_state.items()
            if key.startswith("model.")
        },
        "optimizer_state": state.optimizer.state_dict(),
        "scheduler_state": state.scheduler.state_dict(),
        "epoch": state.epoch,
        "optimizer_step": state.optimizer_step,
        "global_step": state.global_step,
        "skipped_optimizer_steps": state.skipped_optimizer_steps,
        # Mid-epoch resume cursor: stay on this epoch after `global_step % len` batches.
        "resume_epoch": state.epoch,
        "resume_batch_index": state.global_step,  # interpreted mod len(loader) on resume
        "epoch_complete": False,
        "last_val_metrics": dict(state.last_val_metrics),
        "best_metric": state.best_metric,
        "best_optimizer_step": state.best_optimizer_step,
        "rng": rng,
        "config_fingerprint": state.config_fingerprint,
        "data_fingerprint": state.data_fingerprint,
        "train_config": _to_plain(asdict(state.train_config)),
        "provenance": dict(state.provenance),
        "callbacks": callback_states or {},
    }


class Checkpoint(Callback):
    """Periodic + save_best + full resume-state checkpoint callback."""

    phase: ClassVar = "checkpoint"
    name: ClassVar[str] = "Checkpoint"

    def __init__(self, config: CheckpointConfig) -> None:
        super().__init__()
        self.config = config
        self.save_dir = Path(config.save_dir)
        self._last_step_checkpoint: int | None = None
        self._callback_states_provider: Any = None

    def on_optimizer_step(self, state: TrainerState, ctx) -> None:
        del ctx
        if self.config.save_every_n_steps is None:
            return
        if state.optimizer_step <= 0:
            return
        if state.optimizer_step == self._last_step_checkpoint:
            return
        if state.optimizer_step % self.config.save_every_n_steps != 0:
            return
        self._write_checkpoint(state, f"step_{state.optimizer_step:08d}.pt")
        self._last_step_checkpoint = state.optimizer_step

    def on_train_epoch_end(self, state: TrainerState) -> None:
        epoch_completed = state.epoch + 1
        if (
            self.config.save_every_n_epochs is not None
            and epoch_completed % self.config.save_every_n_epochs == 0
        ):
            self._write_checkpoint(state, f"epoch_{epoch_completed:04d}.pt")
        if self.config.save_best and self.config.monitor.startswith("train/"):
            self._maybe_write_best(state)

    def on_validation_end(self, state: TrainerState) -> None:
        if self.config.save_best and self.config.monitor.startswith("val/"):
            self._maybe_write_best(state)

    def _callback_states(self, state: TrainerState) -> dict[str, dict[str, Any]]:
        if self._callback_states_provider is not None:
            return self._callback_states_provider()
        return {}

    def _write_checkpoint(self, state: TrainerState, snapshot_name: str) -> None:
        payload = build_checkpoint_payload(
            state, callback_states=self._callback_states(state)
        )
        last_path = self.save_dir / "last.pt"
        _atomic_save(payload, last_path)
        self.emit("checkpoint_written", path=last_path)
        snapshot_path = self.save_dir / snapshot_name
        _atomic_save(payload, snapshot_path)
        self.emit("checkpoint_written", path=snapshot_path)
        self._prune_rotated(snapshot_path.name.split("_", maxsplit=1)[0])

    def _maybe_write_best(self, state: TrainerState) -> None:
        metric = self._read_metric(state)
        if metric is None:
            return
        if self._is_better(metric, state.best_metric):
            state.best_metric = metric
            state.best_optimizer_step = state.optimizer_step
            best_path = self.save_dir / "best.pt"
            _atomic_save(
                build_checkpoint_payload(
                    state, callback_states=self._callback_states(state)
                ),
                best_path,
            )
            self.emit("checkpoint_written", path=best_path)

    def _prune_rotated(self, prefix: str) -> None:
        snapshots = sorted(self.save_dir.glob(f"{prefix}_*.pt"))
        excess = len(snapshots) - self.config.keep_last_n
        for stale in snapshots[: max(excess, 0)]:
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

    def _read_metric(self, state: TrainerState) -> float | None:
        monitor = self.config.monitor
        if monitor in state.last_val_metrics:
            return state.last_val_metrics[monitor]
        if monitor in state.last_metrics:
            return state.last_metrics[monitor]
        if monitor == "train/loss" and state.last_loss is not None:
            return state.last_loss
        if monitor == "val/loss":
            return state.last_val_metrics.get("val/loss")
        return None

    def _is_better(self, candidate: float, current: float | None) -> bool:
        if current is None:
            return True
        if self.config.mode == "min":
            return candidate < current
        return candidate > current


__all__ = ["Checkpoint", "build_checkpoint_payload"]

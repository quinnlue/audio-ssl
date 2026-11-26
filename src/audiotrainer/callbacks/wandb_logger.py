"""WandB logger callback (soft dependency)."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

import torch

from audiossl.config import AudioSSLTrainingConfig, WandBConfig

from ..fingerprint import _to_plain
from ..state import TrainerState
from ..step import StepContext
from .callback import Callback

_COUNTER_PREFIXES = ("train/epoch", "train/global_step", "train/optimizer_step")


def _flatten_config(payload: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_config(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def _to_python_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


class WandBLogger(Callback):
    """Stream metrics to a managed W&B run or an existing caller-owned run.

    An existing run keeps its own lifecycle and W&B history counter. Trainer
    step values are logged as metrics so callers can use custom chart axes.
    """

    phase: ClassVar = "log"
    name: ClassVar[str] = "WandBLogger"

    def __init__(
        self,
        *,
        config: WandBConfig,
        train_config: AudioSSLTrainingConfig,
        run: Any = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, WandBConfig):
            raise TypeError(
                f"WandBLogger.config must be WandBConfig, got {type(config).__name__}"
            )
        if not isinstance(train_config, AudioSSLTrainingConfig):
            raise TypeError(
                "WandBLogger.train_config must be AudioSSLTrainingConfig, "
                f"got {type(train_config).__name__}"
            )
        self._config = config
        self._train_config = train_config
        self._provided_run = run
        self._run: Any = None
        self._run_id: str | None = None
        self._last_logged_step = 0
        self._agg_sum: dict[str, float] = {}
        self._agg_count: dict[str, int] = {}
        self._finished = False

    def state_dict(self) -> dict[str, Any]:
        return {"run_id": self._run_id}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._run_id = state.get("run_id")

    def on_train_start(self, state: TrainerState) -> None:
        self._finished = False
        self._last_logged_step = 0
        self._agg_sum.clear()
        self._agg_count.clear()
        if self._provided_run is not None:
            self._run = self._provided_run
            self._run_id = self._run.id
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandBLogger requires the 'wandb' package; install it or remove "
                "WandBConfig from RunConfig.loggers"
            ) from exc

        flattened = _flatten_config(_to_plain(asdict(self._train_config)))
        flattened["config_fingerprint"] = state.config_fingerprint
        flattened["data_fingerprint"] = state.data_fingerprint
        init_kwargs: dict[str, Any] = {
            "project": self._config.project,
            "name": self._config.name,
            "entity": self._config.entity,
            "tags": list(self._config.tags),
            "mode": self._config.mode,
            "notes": self._config.notes,
            "config": flattened,
        }
        if self._config.resume and self._run_id is not None:
            init_kwargs["id"] = self._run_id
            init_kwargs["resume"] = "allow"
        else:
            self._run_id = uuid.uuid4().hex
            init_kwargs["id"] = self._run_id
        self._run = wandb.init(**init_kwargs)

    def on_optimizer_step(self, state: TrainerState, ctx: StepContext) -> None:
        del ctx
        step = state.optimizer_step
        if self._run is None or step == 0:
            return
        if self._provided_run is None and step == self._last_logged_step:
            return
        metrics = dict(state.last_metrics)
        scalars = {key: _to_python_scalar(value) for key, value in metrics.items()}
        if self._config.aggregate_over_steps:
            for key, value in scalars.items():
                if key in _COUNTER_PREFIXES or key.startswith("train/epoch"):
                    continue
                self._agg_sum[key] = self._agg_sum.get(key, 0.0) + value
                self._agg_count[key] = self._agg_count.get(key, 0) + 1
        if step % self._config.log_every_n_steps != 0:
            return
        if self._config.aggregate_over_steps:
            payload = {
                key: self._agg_sum[key] / self._agg_count[key]
                for key in self._agg_sum
            }
            self._agg_sum.clear()
            self._agg_count.clear()
        else:
            payload = scalars
        self._log(payload, step)
        self._last_logged_step = step

    def on_checkpoint_written(self, state: TrainerState, path: Path) -> None:
        if not self._config.upload_checkpoints or self._run is None:
            return
        try:
            import wandb
        except ImportError:
            return
        artifact = wandb.Artifact(
            name=f"ckpt-{state.config_fingerprint[:8]}",
            type="model",
        )
        if path.is_dir():
            artifact.add_dir(str(path))
            alias = path.name
        else:
            artifact.add_file(str(path))
            alias = path.stem
        self._run.log_artifact(
            artifact,
            aliases=[alias, f"step-{state.optimizer_step}"],
        )

    def on_validation_end(self, state: TrainerState) -> None:
        if self._run is not None and state.last_val_metrics:
            self._log(dict(state.last_val_metrics), state.optimizer_step)

    def _log(self, payload: dict[str, float], step: int) -> None:
        if self._provided_run is None:
            self._run.log(payload, step=step)
        else:
            self._run.log({**payload, "train/optimizer_step": step})

    def on_train_end(self, state: TrainerState) -> None:
        del state
        self._finish(exit_code=0)

    def on_exception(self, state: TrainerState, exc: BaseException) -> None:
        del state, exc
        self._finish(exit_code=1)

    def _finish(self, *, exit_code: int) -> None:
        if self._finished:
            return
        self._finished = True
        run, self._run = self._run, None
        if run is not None and self._provided_run is None:
            run.finish(exit_code=exit_code)


__all__ = ["WandBLogger"]

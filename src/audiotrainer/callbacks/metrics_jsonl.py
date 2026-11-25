"""Always-on local metrics.jsonl writer under the run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, TextIO

from ..state import TrainerState
from ..step import StepContext
from .callback import Callback


class MetricsJsonlLogger(Callback):
    phase: ClassVar = "log"
    name: ClassVar[str] = "MetricsJsonlLogger"

    def __init__(self, run_dir: str | Path) -> None:
        super().__init__()
        self._run_dir = Path(run_dir)
        self._path = self._run_dir / "metrics.jsonl"
        self._fp: TextIO | None = None

    def on_train_start(self, state: TrainerState) -> None:
        del state
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("a", encoding="utf-8")

    def on_optimizer_step(self, state: TrainerState, ctx: StepContext) -> None:
        del ctx
        if self._fp is None:
            return
        payload = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "optimizer_step": state.optimizer_step,
            **state.last_metrics,
            **{f"val/{k.removeprefix('val/')}" : v for k, v in state.last_val_metrics.items()},
        }
        self._fp.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fp.flush()

    def on_validation_end(self, state: TrainerState) -> None:
        if self._fp is None or not state.last_val_metrics:
            return
        payload = {
            "event": "validation",
            "epoch": state.epoch,
            "optimizer_step": state.optimizer_step,
            **state.last_val_metrics,
        }
        self._fp.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fp.flush()

    def on_train_end(self, state: TrainerState) -> None:
        del state
        self._close()

    def on_exception(self, state: TrainerState, exc: BaseException) -> None:
        del state, exc
        self._close()

    def _close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def state_dict(self) -> dict[str, Any]:
        return {}


__all__ = ["MetricsJsonlLogger"]

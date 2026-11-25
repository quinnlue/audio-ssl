"""Validation cadence callback."""

from __future__ import annotations

from typing import Any, ClassVar

from ..evaluation import Evaluator
from ..state import TrainerState
from .callback import Callback


class ValidationCallback(Callback):
    """Run an evaluator at the cadence configured by ``RunConfig.evaluation``."""

    phase: ClassVar = "evaluate"
    name: ClassVar[str] = "ValidationCallback"

    def __init__(self, evaluator: Evaluator) -> None:
        super().__init__()
        if not hasattr(evaluator, "evaluate") or not callable(evaluator.evaluate):
            raise TypeError(
                "ValidationCallback.evaluator must provide evaluate(model, state)"
            )
        self.evaluator = evaluator
        self._last_val_step: int | None = None

    def on_train_start(self, state: TrainerState) -> None:
        config = state.run_config.evaluation
        if config is not None and config.before_train:
            self._run_validation(state)

    def on_optimizer_step(self, state: TrainerState, ctx) -> None:
        del ctx
        config = state.run_config.evaluation
        if config is None or config.every_n_steps is None:
            return
        step = state.optimizer_step
        if step <= 0 or step == self._last_val_step:
            return
        if step % config.every_n_steps == 0:
            self._run_validation(state)

    def on_train_epoch_end(self, state: TrainerState) -> None:
        config = state.run_config.evaluation
        if config is None or config.every_n_epochs is None:
            return
        if (state.epoch + 1) % config.every_n_epochs == 0:
            self._run_validation(state)

    def on_train_end(self, state: TrainerState) -> None:
        config = state.run_config.evaluation
        if config is None or not config.at_train_end:
            return
        if state.optimizer_step == self._last_val_step:
            return
        self._run_validation(state)

    def _run_validation(self, state: TrainerState) -> None:
        self.emit("validation_start")
        metrics = self.evaluator.evaluate(state.model, state)
        state.last_val_metrics = {
            key if key.startswith("val/") else f"val/{key}": float(value)
            for key, value in metrics.items()
        }
        self._last_val_step = state.optimizer_step
        self.emit("validation_end")

    def state_dict(self) -> dict[str, Any]:
        return {"last_val_step": self._last_val_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._last_val_step = state.get("last_val_step")


__all__ = ["ValidationCallback"]

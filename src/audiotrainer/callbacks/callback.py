"""Base callback class for the SSL trainer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput

from ..state import TrainerState
from ..step import StepContext

CallbackPhase = Literal["evaluate", "checkpoint", "log", "observe"]

_PHASE_ORDER: dict[str, int] = {
    "evaluate": 0,
    "checkpoint": 1,
    "log": 2,
    "observe": 3,
}


class Callback:
    """Base class for trainer callbacks.

    Subclass and override the hooks you care about; the defaults are no-ops.
    ``phase`` controls ordering relative to sibling callbacks at shared
    boundaries: evaluate -> checkpoint -> log -> observe.
    """

    phase: ClassVar[CallbackPhase] = "observe"
    name: ClassVar[str | None] = None

    def __init__(self) -> None:
        self._emit: Callable[..., None] | None = None

    def attach(self, emit: Callable[..., None]) -> None:
        """Wire the trainer's event emitter (called once during Trainer construction)."""
        self._emit = emit

    def emit(self, event: str, **payload: Any) -> None:
        if self._emit is None:
            raise RuntimeError(
                f"{type(self).__name__}.emit called before attach(); "
                "pass the callback to Trainer(callbacks=[...]) first"
            )
        self._emit(event, **payload)

    def callback_name(self) -> str:
        return self.name or type(self).__name__

    def state_dict(self) -> dict[str, Any]:
        """Optional per-callback state persisted into checkpoints."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by :meth:`state_dict`."""
        del state

    def on_train_start(self, state: TrainerState) -> None:
        pass

    def on_train_epoch_start(self, state: TrainerState) -> None:
        pass

    def on_train_batch_start(self, state: TrainerState, batch: AudioBatch) -> None:
        pass

    def on_train_batch_end(
        self,
        state: TrainerState,
        batch: AudioBatch,
        output: ObjectiveOutput,
    ) -> None:
        pass

    def on_optimizer_step(self, state: TrainerState, ctx: StepContext) -> None:
        pass

    def on_train_epoch_end(self, state: TrainerState) -> None:
        pass

    def on_train_end(self, state: TrainerState) -> None:
        pass

    def on_validation_start(self, state: TrainerState) -> None:
        pass

    def on_validation_batch_end(
        self,
        state: TrainerState,
        batch: AudioBatch,
        output: ObjectiveOutput,
    ) -> None:
        pass

    def on_validation_end(self, state: TrainerState) -> None:
        pass

    def on_checkpoint_written(self, state: TrainerState, path: Path) -> None:
        pass

    def on_exception(self, state: TrainerState, exc: BaseException) -> None:
        pass


def sort_callbacks(callbacks: list[Callback]) -> list[Callback]:
    """Stable sort by phase then original registration order."""
    indexed = list(enumerate(callbacks))
    indexed.sort(key=lambda item: (_PHASE_ORDER.get(item[1].phase, 99), item[0]))
    return [callback for _, callback in indexed]


__all__ = ["Callback", "CallbackPhase", "sort_callbacks"]

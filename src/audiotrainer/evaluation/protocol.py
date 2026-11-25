from __future__ import annotations

from typing import Protocol

from audiossl.modelling.model import AudioModel

from ..state import TrainerState


class Evaluator(Protocol):
    """Protocol for validation and downstream evaluators used by callbacks."""

    def evaluate(self, model: AudioModel, state: TrainerState) -> dict[str, float]:
        """Return a flat mapping of scalar metrics."""


__all__ = ["Evaluator"]

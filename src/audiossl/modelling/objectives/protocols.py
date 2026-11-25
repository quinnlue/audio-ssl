"""Protocols decouple objectives from the training package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from audiossl.config.batch_dataclasses import AudioBatch


class AugmentationPipeline(Protocol):
    """Trainer-provided waveform augmentation applied to ``AudioBatch`` views."""

    def __call__(
        self,
        batch: AudioBatch,
        generator: torch.Generator | None = None,
    ) -> AudioBatch: ...


@dataclass
class ObjectiveMetricsState:
    """Fields read by :meth:`Objective.metrics` (subset of full trainer state)."""

    epoch: int = 0
    global_step: int = 0
    optimizer_step: int = 0
    current_lr: float | None = None
    last_grad_norm: float | None = None
    last_input_seconds: float | None = None
    last_throughput_audio_seconds_per_sec: float | None = None


__all__ = ["AugmentationPipeline"]
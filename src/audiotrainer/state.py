"""Mutable training-loop state passed to every callback hook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from audiossl.config import AudioSSLTrainingConfig
from audiossl.config.run import RunConfig
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives._base import Objective
from audiossl.runtime import AppliedRuntimeConfig


@dataclass
class TrainerState:
    """In-memory state of a training run.

    The trainer mutates the counter and telemetry fields between hook
    invocations; callbacks should treat the identity fields as immutable for
    the lifetime of a run.

    The checkpoint callback is permitted to write ``best_metric`` and
    ``best_optimizer_step`` because it owns that policy; other callbacks
    must not.
    """

    # --- identity (immutable across the run) ----------------------------
    model: AudioModel
    objective: Objective
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_config: AudioSSLTrainingConfig
    run_config: RunConfig
    config_fingerprint: str
    data_fingerprint: str
    applied_runtime: AppliedRuntimeConfig

    # --- counters -------------------------------------------------------
    epoch: int = 0
    global_step: int = 0
    optimizer_step: int = 0
    skipped_optimizer_steps: int = 0
    consecutive_skipped_steps: int = 0

    # --- last-batch / last-step telemetry --------------------------------
    last_loss: float | None = None
    current_lr: float | None = None
    last_grad_norm: float | None = None
    last_input_seconds: float | None = None
    last_throughput_audio_seconds_per_sec: float | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)

    # --- validation telemetry (owned by ValidationCallback) -------------
    last_val_metrics: dict[str, float] = field(default_factory=dict)

    # --- save_best tracking (owned by Checkpoint callback) --------------
    best_metric: float | None = None
    best_optimizer_step: int | None = None

    # --- environment provenance (set once by the trainer) ---------------
    provenance: dict[str, Any] = field(default_factory=dict)


__all__ = ["TrainerState"]

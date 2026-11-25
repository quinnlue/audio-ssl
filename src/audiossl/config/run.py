"""Operational run configuration.

Everything here controls *how a run is operated* (where artifacts go, how
often to checkpoint, validate, and log, how many loader workers to use)
without changing the trained model. None of it enters ``config_fingerprint``,
so it may change freely between a run and its resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .logging import LoggerConfig


@dataclass(kw_only=True, frozen=True)
class LoaderConfig:
    """DataLoader worker settings used by ``audiotrainer.build_trainer``."""

    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False


@dataclass(kw_only=True, frozen=True)
class EvaluationConfig:
    """When validation runs. Validation is RNG-isolated, so cadence never changes training."""

    every_n_epochs: int | None = None
    every_n_steps: int | None = None
    before_train: bool = False
    at_train_end: bool = True


@dataclass(kw_only=True, frozen=True)
class CheckpointConfig:
    """Checkpoint directory layout and save/monitor policy.

    ``monitor`` names a logged metric: ``"val/<name>"`` metrics are read after
    each validation, ``"train/<name>"`` metrics at each epoch end.
    """

    save_dir: str
    save_every_n_epochs: int | None = 1
    save_every_n_steps: int | None = None
    keep_last_n: int = 1
    save_best: bool = False
    monitor: str = "val/loss"
    mode: Literal["min", "max"] = "min"


@dataclass(kw_only=True, frozen=True)
class RunConfig:
    """Operational settings for one training run.

    ``run_dir`` enables the run record: ``config.json``, ``provenance.json``,
    ``metrics.jsonl``, and ``train.log``. ``require_clean_git`` refuses to
    start from a dirty or untracked source tree. ``provenance_mismatch``
    decides whether resuming in a different software/hardware environment
    warns or raises.
    """

    run_dir: str | None = None
    checkpoint: CheckpointConfig | None = None
    evaluation: EvaluationConfig | None = None
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    stdout_logging: bool = True
    log_every_n_steps: int = 1
    loggers: tuple[LoggerConfig, ...] = ()
    require_clean_git: bool = False
    provenance_mismatch: Literal["warn", "raise"] = "warn"

    def __post_init__(self) -> None:
        object.__setattr__(self, "loggers", tuple(self.loggers))


__all__ = ["CheckpointConfig", "EvaluationConfig", "LoaderConfig", "RunConfig"]

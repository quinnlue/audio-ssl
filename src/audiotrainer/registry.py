"""Trainer-side component registries for optimizers, schedules, and augmentations.

Mirrors the modelling :class:`~audiossl.modelling.registry.ComponentRegistry`
pattern so third-party packages can register builders without editing
``build_optimizer`` / ``build_scheduler`` / ``AugmentationPipeline``.
Matching configs register under ``audiossl.config.register_config`` categories
``optimizer``, ``scheduler``, ``augmentation``, and ``scalar_scheduler``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from audiossl.modelling.registry import ComponentRegistry

OptimizerBuilder = Callable[[list[dict], Any], torch.optim.Optimizer]
SchedulerBuilder = Callable[
    [torch.optim.Optimizer, Any, int], torch.optim.lr_scheduler.LRScheduler
]
AugmentationBuilder = Callable[[Any], Any]
ScalarSchedulerBuilder = Callable[[Any], Any]


def _adamw(groups: list[dict], config: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )


def _sgd(groups: list[dict], config: Any) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        groups,
        lr=config.lr,
        momentum=config.momentum,
        nesterov=config.nesterov,
    )


def _lars(groups: list[dict], config: Any) -> torch.optim.Optimizer:
    from .optim import LARS

    return LARS(
        groups,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        eta=config.eta,
    )


def _constant_scheduler(
    optimizer: torch.optim.Optimizer, config: Any, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    del config, total_steps
    from .optim import ConstantScale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, ConstantScale())


def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, config: Any, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    from .optim import WarmupCosineScale

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosineScale(
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr_scale=config.min_lr_scale,
        ),
    )


def _cosine_annealing_scheduler(
    optimizer: torch.optim.Optimizer, config: Any, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    from .optim import CosineAnnealingScale

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        CosineAnnealingScale(
            total_steps=total_steps,
            min_lr_scale=config.min_lr_scale,
        ),
    )


def _step_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: Any, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    del total_steps
    from .optim import StepLRScale

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        StepLRScale(step_size=config.step_size, gamma=config.gamma),
    )


def _loudness_augmentation(config: Any) -> Any:
    from .augment.waveform import LoudnessAugmentation

    return LoudnessAugmentation(
        {
            "p": config.p,
            "min_gain_db": config.min_gain_db,
            "max_gain_db": config.max_gain_db,
        }
    )


def _constant_scalar(config: Any) -> Any:
    from audiossl.modelling.objectives.wav2vec2.scalar_scheduler import (
        ConstantScalarScheduler,
    )

    return ConstantScalarScheduler(value=config.value)


def _exponential_decay_scalar(config: Any) -> Any:
    from audiossl.modelling.objectives.wav2vec2.scalar_scheduler import (
        ExponentialDecayScalarScheduler,
    )

    return ExponentialDecayScalarScheduler(
        start=config.start,
        end=config.end,
        decay=config.decay,
    )


OPTIMIZERS: ComponentRegistry = ComponentRegistry(
    "optimizer",
    {
        "adamw": lambda: _adamw,
        "sgd": lambda: _sgd,
        "lars": lambda: _lars,
    },
)

SCHEDULERS: ComponentRegistry = ComponentRegistry(
    "scheduler",
    {
        "constant": lambda: _constant_scheduler,
        "warmup_cosine": lambda: _warmup_cosine_scheduler,
        "cosine_annealing": lambda: _cosine_annealing_scheduler,
        "step_lr": lambda: _step_lr_scheduler,
    },
)

AUGMENTATIONS: ComponentRegistry = ComponentRegistry(
    "augmentation",
    {
        "loudness": lambda: _loudness_augmentation,
    },
)

SCALAR_SCHEDULERS: ComponentRegistry = ComponentRegistry(
    "scalar_scheduler",
    {
        "constant": lambda: _constant_scalar,
        "exponential_decay": lambda: _exponential_decay_scalar,
    },
)


__all__ = [
    "AUGMENTATIONS",
    "OPTIMIZERS",
    "SCALAR_SCHEDULERS",
    "SCHEDULERS",
]

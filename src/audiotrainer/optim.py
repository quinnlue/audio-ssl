"""Optimizer, scheduler, and parameter-group builders shared by SSL training and probe eval."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from audiossl.config import (
    LRSchedulerConfig,
    OptimizerConfig,
)

logger = logging.getLogger(__name__)


class LARS(torch.optim.Optimizer):
    """LARS (You et al., 2017) optimizer with per-layer adaptive scaling."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        eta: float,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "eta": eta,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            eta = group["eta"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)
                param_norm = torch.linalg.vector_norm(param)
                grad_norm = torch.linalg.vector_norm(grad)
                if param_norm > 0 and grad_norm > 0:
                    trust_ratio = eta * param_norm / grad_norm
                else:
                    trust_ratio = torch.ones((), device=param.device, dtype=param.dtype)
                scaled_grad = grad.mul(trust_ratio)
                state = self.state[param]
                if momentum != 0.0:
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = torch.clone(scaled_grad).detach()
                        state["momentum_buffer"] = buffer
                    else:
                        buffer.mul_(momentum).add_(scaled_grad)
                    scaled_grad = buffer
                param.add_(scaled_grad, alpha=-lr)
        return loss


def build_param_groups(
    module: nn.Module,
    *,
    weight_decay: float,
    no_weight_decay_patterns: tuple[str, ...],
) -> list[dict]:
    """Split ``module``'s trainable parameters into a decay and a no-decay group.

    Raises if any individual pattern matches zero parameters, or if there are
    no trainable parameters at all.
    """

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    matched_by_pattern: dict[str, int] = {pattern: 0 for pattern in no_weight_decay_patterns}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        hit = [pattern for pattern in no_weight_decay_patterns if pattern in name]
        if hit:
            no_decay.append(param)
            for pattern in hit:
                matched_by_pattern[pattern] += 1
        else:
            decay.append(param)
    if not decay and not no_decay:
        raise ValueError(
            f"build_param_groups: no trainable parameters in {type(module).__name__}"
        )
    unmatched = [pattern for pattern, count in matched_by_pattern.items() if count == 0]
    if unmatched:
        raise ValueError(
            "build_param_groups: no_weight_decay_patterns matched zero parameters: "
            f"{unmatched!r} in {type(module).__name__}; patterns are substring-matched "
            "against named_parameters() qualified names."
        )
    logger.info(
        "param groups for %s: decay=%d no_decay=%d (patterns=%s)",
        type(module).__name__,
        len(decay),
        len(no_decay),
        matched_by_pattern,
    )
    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(
    module: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Build a torch optimizer over ``module``'s trainable parameters."""

    if not isinstance(module, nn.Module):
        raise TypeError(
            f"build_optimizer expected nn.Module, got {type(module).__name__}"
        )
    if hasattr(module, "param_groups") and callable(module.param_groups):
        groups = module.param_groups(config)
    else:
        groups = build_param_groups(
            module,
            weight_decay=config.weight_decay,
            no_weight_decay_patterns=config.no_weight_decay_patterns,
        )
    type_name = getattr(config, "type", None)
    if not isinstance(type_name, str):
        raise TypeError(
            "build_optimizer expected an optimizer config with a string 'type' field, "
            f"got {type(config).__name__}"
        )
    from .registry import OPTIMIZERS

    try:
        builder = OPTIMIZERS[type_name]
    except KeyError as exc:
        raise TypeError(str(exc.args[0])) from None
    return builder(groups, config)


def _cosine_scale(step: int, total_steps: int, min_lr_scale: float) -> float:
    if total_steps <= 1:
        return 1.0
    progress = min(float(step) / float(total_steps - 1), 1.0)
    return min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class ConstantScale:
    def __call__(self, step: int) -> float:
        return 1.0


@dataclass(frozen=True)
class WarmupCosineScale:
    warmup_steps: int
    total_steps: int
    min_lr_scale: float

    def __call__(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        cosine_step = max(step - self.warmup_steps, 0)
        cosine_total = max(self.total_steps - self.warmup_steps, 1)
        return _cosine_scale(cosine_step, cosine_total, self.min_lr_scale)


@dataclass(frozen=True)
class CosineAnnealingScale:
    total_steps: int
    min_lr_scale: float

    def __call__(self, step: int) -> float:
        return _cosine_scale(step, self.total_steps, self.min_lr_scale)


@dataclass(frozen=True)
class StepLRScale:
    step_size: int
    gamma: float

    def __call__(self, step: int) -> float:
        return self.gamma ** (step // self.step_size)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerConfig,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if getattr(config, "horizon_dependent", False):
        warmup = int(getattr(config, "warmup_steps", 0) or 0)
        if warmup >= total_steps:
            raise ValueError(
                f"{type(config).__name__}.warmup_steps={warmup} must be < total_steps={total_steps}"
            )
    type_name = getattr(config, "type", None)
    if not isinstance(type_name, str):
        raise TypeError(
            "build_scheduler expected a scheduler config with a string 'type' field, "
            f"got {type(config).__name__}"
        )
    from .registry import SCHEDULERS

    try:
        builder = SCHEDULERS[type_name]
    except KeyError as exc:
        raise TypeError(str(exc.args[0])) from None
    return builder(optimizer, config, total_steps)


__all__ = [
    "ConstantScale",
    "CosineAnnealingScale",
    "LARS",
    "StepLRScale",
    "WarmupCosineScale",
    "build_optimizer",
    "build_param_groups",
    "build_scheduler",
]

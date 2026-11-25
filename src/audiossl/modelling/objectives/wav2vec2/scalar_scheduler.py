from __future__ import annotations

import torch
import torch.nn as nn

from audiossl.config import ScalarSchedulerConfig


class ScalarScheduler(nn.Module):
    def current_value(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class ConstantScalarScheduler(ScalarScheduler):
    def __init__(self, *, value: float) -> None:
        super().__init__()
        self.value = float(value)
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))

    def current_value(self) -> float:
        return self.value

    def step(self) -> None:
        self.step_count.add_(1)

    def reset(self) -> None:
        self.step_count.zero_()


class ExponentialDecayScalarScheduler(ScalarScheduler):
    def __init__(self, *, start: float, end: float, decay: float) -> None:
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.decay = float(decay)
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))

    def current_value(self) -> float:
        value = self.start * (self.decay ** int(self.step_count.item()))
        return max(value, self.end)

    def step(self) -> None:
        self.step_count.add_(1)

    def reset(self) -> None:
        self.step_count.zero_()


def build_scalar_scheduler(config: ScalarSchedulerConfig) -> ScalarScheduler:
    type_name = getattr(config, "type", None)
    if not isinstance(type_name, str):
        raise TypeError(
            "build_scalar_scheduler expected a config with a string 'type' field, "
            f"got {type(config).__name__}"
        )
    from audiotrainer.registry import SCALAR_SCHEDULERS

    try:
        builder = SCALAR_SCHEDULERS[type_name]
    except KeyError as exc:
        raise TypeError(str(exc.args[0])) from None
    scheduler = builder(config)
    if not isinstance(scheduler, ScalarScheduler):
        raise TypeError(
            f"SCALAR_SCHEDULERS[{type_name!r}] must build a ScalarScheduler, "
            f"got {type(scheduler).__name__}"
        )
    return scheduler


__all__ = [
    "ConstantScalarScheduler",
    "ExponentialDecayScalarScheduler",
    "ScalarScheduler",
    "build_scalar_scheduler",
]
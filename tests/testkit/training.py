"""Minimal trainer loop mirroring ``audiotrainer.Trainer``'s objective-hook order.

forward -> ``on_after_forward`` -> backward -> ``on_before_optimizer_step`` ->
optimizer step -> ``on_after_optimizer_step``. Kept here (rather than importing
the real trainer) so modelling tests stay independent of data loading, logging,
and checkpoint policy.
"""

from __future__ import annotations

from typing import Any

import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.objectives._base import Objective
from audiotrainer.step import StepContext


def optimizer_for(objective: torch.nn.Module, lr: float = 3e-3) -> torch.optim.Optimizer:
    params = [p for p in objective.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def _step_context(
    *,
    optimizer_step: int = 0,
    total_optimizer_steps: int = 1,
    epoch: int = 0,
    global_step: int = 0,
) -> StepContext:
    return StepContext(
        optimizer_step=optimizer_step,
        total_optimizer_steps=total_optimizer_steps,
        epoch=epoch,
        global_step=global_step,
        step_was_skipped=False,
    )


def train_step(
    objective: torch.nn.Module,
    batch: AudioBatch,
    optimizer: torch.optim.Optimizer,
    *,
    ctx: StepContext | None = None,
) -> torch.Tensor:
    context = ctx if ctx is not None else _step_context()
    output = objective(batch)
    objective.on_after_forward(output, context)
    output.loss.backward()
    objective.on_before_optimizer_step(context)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    objective.on_after_optimizer_step(context)
    return output.loss.detach()


def generators(objective: torch.nn.Module) -> dict[str, torch.Generator]:
    """Every injected RNG an objective consumes, keyed by owner."""
    if isinstance(objective, Objective):
        return objective.named_generators()
    found: dict[str, torch.Generator] = {}
    generator = getattr(objective, "generator", None)
    if isinstance(generator, torch.Generator):
        found["objective"] = generator
    augmentor = getattr(objective, "augmentor", None)
    aug_generator = getattr(augmentor, "generator", None)
    if isinstance(aug_generator, torch.Generator):
        found["augmentor"] = aug_generator
    return found


def reseed(objective: torch.nn.Module, seed: int) -> None:
    """Make the next step's randomness (masks, views, dropout) a function of ``seed``."""
    for generator in generators(objective).values():
        generator.manual_seed(seed)
    torch.manual_seed(seed)


def rng_state(objective: torch.nn.Module) -> dict[str, Any]:
    state = {name: g.get_state() for name, g in generators(objective).items()}
    state["torch"] = torch.random.get_rng_state()
    return state


def set_rng_state(objective: torch.nn.Module, state: dict[str, Any]) -> None:
    for name, generator in generators(objective).items():
        generator.set_state(state[name])
    torch.random.set_rng_state(state["torch"])

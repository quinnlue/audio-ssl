"""Discovery, checkpointing, and restoration of objective-owned RNG streams."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from audiossl.modelling.objectives._base import Objective


def objective_generators(objective: Objective) -> dict[str, torch.Generator]:
    """Every registered ``torch.Generator`` an objective owns, keyed by name."""
    if not isinstance(objective, Objective):
        raise TypeError(
            f"objective_generators expected Objective, got {type(objective).__name__}"
        )
    return objective.named_generators()


def objective_generator_states(objective: Objective) -> dict[str, torch.Tensor]:
    return {name: g.get_state() for name, g in objective_generators(objective).items()}


def restore_objective_generators(
    objective: Objective,
    states: dict[str, torch.Tensor] | None,
) -> None:
    """Restore generator states, raising when the checkpoint and objective disagree."""
    generators = objective_generators(objective)
    states = states or {}
    if set(generators) != set(states):
        raise RuntimeError(
            "Trainer.from_checkpoint: objective generators "
            f"{sorted(generators)} do not match the checkpoint's {sorted(states)}; "
            "resuming would draw different masks/augmentations than the original run"
        )
    for name, generator in generators.items():
        generator.set_state(states[name].cpu())


def capture_training_rng(objective: Objective) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "objective_generators": objective_generator_states(objective),
    }


def restore_training_rng(objective: Objective, rng: dict[str, Any]) -> None:
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    restore_objective_generators(objective, rng.get("objective_generators"))


@contextmanager
def override_objective_generators(
    objective: Objective,
    *,
    seed: int,
) -> Iterator[None]:
    """Temporarily reseed every registered generator, then restore prior states."""
    generators = objective_generators(objective)
    saved = {name: g.get_state() for name, g in generators.items()}
    try:
        for offset, generator in enumerate(generators.values()):
            generator.manual_seed(seed + offset)
        yield
    finally:
        for name, generator in generators.items():
            generator.set_state(saved[name])


__all__ = [
    "objective_generator_states",
    "objective_generators",
    "override_objective_generators",
    "restore_objective_generators",
]

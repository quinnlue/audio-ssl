"""Device-aware ``torch.Generator`` helpers for training-time randomness."""

from __future__ import annotations

import torch


def training_generator_device(device: torch.device) -> torch.device:
    """Return the device PyTorch uses for a training RNG stream on ``device``."""
    if device.type == "cuda":
        return device
    return torch.device("cpu")


def generator_device(generator: torch.Generator | None) -> torch.device:
    """Device that must be passed to ``torch.*`` ops that take ``generator=``."""
    if generator is None:
        return torch.device("cpu")
    return getattr(generator, "device", torch.device("cpu"))


def new_manual_seed_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded generator aligned with the model's training device."""
    gen_device = training_generator_device(device)
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator


__all__ = [
    "generator_device",
    "new_manual_seed_generator",
    "training_generator_device",
]
"""EAT architecture preset registration."""

from __future__ import annotations

from audiossl.integrations.eat import make_eat_base_config

from .specs import ModelPresetSpec

EAT_BASE_PRESET = ModelPresetSpec(
    name="eat-base",
    aliases=("eat_base", "eat/base"),
    family="eat",
    config_factory=make_eat_base_config,
    description="EAT-base encoder layout as an audiossl AudioModel config.",
)

__all__ = ["EAT_BASE_PRESET"]

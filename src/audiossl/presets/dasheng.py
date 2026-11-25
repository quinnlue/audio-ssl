"""Dasheng architecture preset registration."""

from __future__ import annotations

from audiossl.integrations.dasheng import make_dasheng_base_config

from .specs import ModelPresetSpec

DASHENG_BASE_PRESET = ModelPresetSpec(
    name="dasheng-base",
    aliases=("dasheng_base", "dasheng/base"),
    family="dasheng",
    config_factory=make_dasheng_base_config,
    description="Dasheng dasheng_base encoder layout as an audiossl AudioModel config.",
)

__all__ = ["DASHENG_BASE_PRESET"]

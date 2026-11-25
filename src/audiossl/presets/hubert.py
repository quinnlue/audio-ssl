"""HuBERT architecture preset registration."""

from __future__ import annotations

from audiossl.integrations.hubert import make_hubert_base_ls960_config

from .specs import ModelPresetSpec

HUBERT_BASE_LS960_PRESET = ModelPresetSpec(
    name="hubert-base-ls960",
    aliases=(
        "hubert_base_ls960",
        "hubert/base-ls960",
        "facebook/hubert-base-ls960",
    ),
    family="hubert",
    config_factory=make_hubert_base_ls960_config,
    description=(
        "facebook/hubert-base-ls960 encoder layout as an audiossl AudioModel config."
    ),
)

__all__ = ["HUBERT_BASE_LS960_PRESET"]

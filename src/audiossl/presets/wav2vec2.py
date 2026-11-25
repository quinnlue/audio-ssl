"""wav2vec2 architecture preset registration."""

from __future__ import annotations

from audiossl.integrations.wav2vec2 import make_wav2vec2_base_config

from .specs import ModelPresetSpec

WAV2VEC2_BASE_PRESET = ModelPresetSpec(
    name="wav2vec2-base",
    aliases=("wav2vec2_base", "wav2vec2/base", "facebook/wav2vec2-base"),
    family="wav2vec2",
    config_factory=make_wav2vec2_base_config,
    description=(
        "facebook/wav2vec2-base encoder layout as an audiossl AudioModel config."
    ),
)

__all__ = ["WAV2VEC2_BASE_PRESET"]

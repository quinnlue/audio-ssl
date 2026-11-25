"""wav2vec2-base pretrained model registry entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audiossl.integrations.wav2vec2.porter import load_wav2vec2_weights
from audiossl.modelling.model import AudioModel

from .specs import PretrainedModelSpec

WAV2VEC2_BASE_HF_REPO = "facebook/wav2vec2-base"
WAV2VEC2_BASE_FILENAME = "model.safetensors"
WAV2VEC2_BASE_URL_FALLBACK = (
    "https://huggingface.co/facebook/wav2vec2-base/resolve/main/model.safetensors"
)


def _load_wav2vec2_weights(
    checkpoint_path: Path,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> None:
    """Load ``facebook/wav2vec2-base`` weights through the wav2vec2 porter."""
    load_wav2vec2_weights(checkpoint_path, model, map_location=map_location)


WAV2VEC2_BASE_PRETRAINED = PretrainedModelSpec(
    name="wav2vec2-base",
    aliases=("wav2vec2_base", "wav2vec2/base", "facebook/wav2vec2-base"),
    architecture="wav2vec2-base",
    hf_repo_id=WAV2VEC2_BASE_HF_REPO,
    checkpoint_url=WAV2VEC2_BASE_URL_FALLBACK,
    checkpoint_filename=WAV2VEC2_BASE_FILENAME,
    sha256=None,
    load_fn=_load_wav2vec2_weights,
    description="facebook/wav2vec2-base checkpoint ported into audiossl's AudioModel architecture.",
)


__all__ = [
    "WAV2VEC2_BASE_FILENAME",
    "WAV2VEC2_BASE_HF_REPO",
    "WAV2VEC2_BASE_PRETRAINED",
    "WAV2VEC2_BASE_URL_FALLBACK",
]

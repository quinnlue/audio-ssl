"""EAT-base pretrained model registry entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audiossl.integrations.eat import load_eat_weights
from audiossl.modelling.model import AudioModel

from .specs import PretrainedModelSpec


EAT_BASE_HF_REPO = "worstchan/EAT-base_epoch30_pretrain"
EAT_BASE_FILENAME = "model.safetensors"
EAT_BASE_URL_FALLBACK = (
    "https://huggingface.co/worstchan/EAT-base_epoch30_pretrain/resolve/main/model.safetensors"
)


def _load_eat_weights(
    checkpoint_path: Path,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> None:
    load_eat_weights(checkpoint_path, model, map_location=map_location)


EAT_BASE_PRETRAINED = PretrainedModelSpec(
    name="eat-base",
    aliases=("eat_base", "eat/base"),
    architecture="eat-base",
    hf_repo_id=EAT_BASE_HF_REPO,
    checkpoint_url=EAT_BASE_URL_FALLBACK,
    checkpoint_filename=EAT_BASE_FILENAME,
    sha256=None,
    load_fn=_load_eat_weights,
    description="EAT-base epoch-30 pretrain checkpoint ported into audiossl.",
)

__all__ = [
    "EAT_BASE_FILENAME",
    "EAT_BASE_HF_REPO",
    "EAT_BASE_PRETRAINED",
    "EAT_BASE_URL_FALLBACK",
]

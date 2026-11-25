"""HuBERT-base LS960 pretrained model registry entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audiossl.integrations.hubert.porter import load_hubert_weights
from audiossl.modelling.model import AudioModel

from .specs import PretrainedModelSpec

HUBERT_BASE_LS960_HF_REPO = "facebook/hubert-base-ls960"
HUBERT_BASE_LS960_FILENAME = "pytorch_model.bin"
HUBERT_BASE_LS960_URL_FALLBACK = (
    "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin"
)


def _load_hubert_weights(
    checkpoint_path: Path,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> None:
    """Load ``facebook/hubert-base-ls960`` weights through the HuBERT porter."""
    load_hubert_weights(checkpoint_path, model, map_location=map_location)


HUBERT_BASE_LS960_PRETRAINED = PretrainedModelSpec(
    name="hubert-base-ls960",
    aliases=(
        "hubert_base_ls960",
        "hubert/base-ls960",
        "facebook/hubert-base-ls960",
    ),
    architecture="hubert-base-ls960",
    hf_repo_id=HUBERT_BASE_LS960_HF_REPO,
    checkpoint_url=HUBERT_BASE_LS960_URL_FALLBACK,
    checkpoint_filename=HUBERT_BASE_LS960_FILENAME,
    sha256=None,
    load_fn=_load_hubert_weights,
    description=(
        "facebook/hubert-base-ls960 checkpoint ported into audiossl's "
        "AudioModel architecture."
    ),
)


__all__ = [
    "HUBERT_BASE_LS960_FILENAME",
    "HUBERT_BASE_LS960_HF_REPO",
    "HUBERT_BASE_LS960_PRETRAINED",
    "HUBERT_BASE_LS960_URL_FALLBACK",
]

"""Dasheng-base pretrained model registry entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audiossl.integrations.dasheng import load_dasheng_weights
from audiossl.modelling.model import AudioModel

from .specs import PretrainedModelSpec


DASHENG_BASE_CHECKPOINT_URL = (
    "https://zenodo.org/records/11511780/files/dasheng_base.pt?download=1"
)


def _load_dasheng_weights(
    checkpoint_path: Path,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> None:
    """Load the Dasheng base checkpoint through the Dasheng integration porter."""
    load_dasheng_weights(checkpoint_path, model, map_location=map_location)


DASHENG_BASE_PRETRAINED = PretrainedModelSpec(
    name="dasheng-base",
    aliases=("dasheng_base", "dasheng/base"),
    architecture="dasheng-base",
    checkpoint_url=DASHENG_BASE_CHECKPOINT_URL,
    checkpoint_filename="dasheng_base.pt",
    sha256=None,
    load_fn=_load_dasheng_weights,
    description="Dasheng base checkpoint ported into audiossl's AudioModel architecture.",
)


__all__ = ["DASHENG_BASE_CHECKPOINT_URL", "DASHENG_BASE_PRETRAINED"]

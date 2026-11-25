"""Architecture preset registry for building audiossl models."""

from __future__ import annotations

from .registry import build_model, get_config, get_preset_spec, list_presets
from .specs import ModelPresetSpec

__all__ = [
    "ModelPresetSpec",
    "build_model",
    "get_config",
    "get_preset_spec",
    "list_presets",
]

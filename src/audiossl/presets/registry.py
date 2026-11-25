"""Lookup and build :class:`AudioModel` instances from registered presets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from audiossl.config import AudioModelConfig

from .dasheng import DASHENG_BASE_PRESET
from .eat import EAT_BASE_PRESET
from .hubert import HUBERT_BASE_LS960_PRESET
from .specs import ModelPresetSpec
from .wav2vec2 import WAV2VEC2_BASE_PRESET

if TYPE_CHECKING:
    from audiossl.modelling.model import AudioModel

_PRESET_SPECS: tuple[ModelPresetSpec, ...] = (
    DASHENG_BASE_PRESET,
    EAT_BASE_PRESET,
    HUBERT_BASE_LS960_PRESET,
    WAV2VEC2_BASE_PRESET,
)


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Preset name must be str, got {type(name).__name__}")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Preset name must not be empty")
    return normalized


def _build_index(specs: tuple[ModelPresetSpec, ...]) -> dict[str, ModelPresetSpec]:
    """Build a normalized name and alias lookup table, rejecting duplicates."""
    index: dict[str, ModelPresetSpec] = {}
    for spec in specs:
        for name in (spec.name, *spec.aliases):
            normalized = _normalize_name(name)
            if normalized in index:
                raise ValueError(f"Duplicate model preset name or alias: {name!r}")
            index[normalized] = spec
    return index


_PRESET_INDEX = _build_index(_PRESET_SPECS)


def list_presets() -> tuple[str, ...]:
    """Return canonical registered architecture preset names."""
    return tuple(spec.name for spec in _PRESET_SPECS)


def get_preset_spec(name: str) -> ModelPresetSpec:
    """Return metadata for a registered architecture preset.

    Parameters
    ----------
    name
        Canonical preset name or alias. Lookup is case-insensitive.
    """
    normalized = _normalize_name(name)
    try:
        return _PRESET_INDEX[normalized]
    except KeyError as exc:
        available = ", ".join(list_presets())
        raise KeyError(
            f"Unknown model preset {name!r}. Available presets: {available}"
        ) from exc


def get_config(name: str, **config_kwargs) -> AudioModelConfig:
    """Build an :class:`AudioModelConfig` from a registered architecture preset.

    Parameters
    ----------
    name
        Canonical preset name or alias. Lookup is case-insensitive.
    **config_kwargs
        Keyword arguments forwarded to the preset ``config_factory``.
    """
    spec = get_preset_spec(name)
    return spec.config_factory(**config_kwargs)


def build_model(name: str, **config_kwargs) -> AudioModel:
    """Build an :class:`AudioModel` from a registered architecture preset.

    Parameters
    ----------
    name
        Canonical preset name or alias. Lookup is case-insensitive.
    **config_kwargs
        Keyword arguments forwarded to the preset ``config_factory``.
    """
    from audiossl.modelling.model import AudioModel

    return AudioModel(get_config(name, **config_kwargs))


__all__ = [
    "build_model",
    "get_config",
    "get_preset_spec",
    "list_presets",
]

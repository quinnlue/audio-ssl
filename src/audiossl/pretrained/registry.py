"""Builtin pretrained model registry and name lookup."""

from __future__ import annotations

from .dasheng import DASHENG_BASE_PRETRAINED
from .eat import EAT_BASE_PRETRAINED
from .hubert import HUBERT_BASE_LS960_PRETRAINED
from .specs import PretrainedModelSpec
from .wav2vec2 import WAV2VEC2_BASE_PRETRAINED

_PRETRAINED_SPECS: tuple[PretrainedModelSpec, ...] = (
    DASHENG_BASE_PRETRAINED,
    EAT_BASE_PRETRAINED,
    HUBERT_BASE_LS960_PRETRAINED,
    WAV2VEC2_BASE_PRETRAINED,
)


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Pretrained model name must be str, got {type(name).__name__}")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Pretrained model name must not be empty")
    return normalized


def _build_index(specs: tuple[PretrainedModelSpec, ...]) -> dict[str, PretrainedModelSpec]:
    index: dict[str, PretrainedModelSpec] = {}
    for spec in specs:
        for name in (spec.name, *spec.aliases):
            normalized = _normalize_name(name)
            if normalized in index:
                raise ValueError(f"Duplicate pretrained model name or alias: {name!r}")
            index[normalized] = spec
    return index


_PRETRAINED_INDEX = _build_index(_PRETRAINED_SPECS)


def list_pretrained() -> tuple[str, ...]:
    """Return canonical pretrained model names."""

    return tuple(spec.name for spec in _PRETRAINED_SPECS)


def get_pretrained_spec(name: str) -> PretrainedModelSpec:
    """Return metadata for a pretrained model name or alias.

    Parameters
    ----------
    name
        Canonical pretrained model name or registered alias.
    """
    normalized = _normalize_name(name)
    try:
        return _PRETRAINED_INDEX[normalized]
    except KeyError as exc:
        available = ", ".join(list_pretrained())
        raise KeyError(
            f"Unknown pretrained model {name!r}. Available pretrained models: {available}"
        ) from exc


__all__ = ["get_pretrained_spec", "list_pretrained"]

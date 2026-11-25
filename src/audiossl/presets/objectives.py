"""Placeholder registry for future objective presets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from audiossl.config.config import ObjectiveConfig


@dataclass(frozen=True, kw_only=True)
class ObjectivePresetSpec:
    """Registered objective preset for building :class:`ObjectiveConfig`.

    Mirrors :class:`~audiossl.presets.specs.ModelPresetSpec` for training
    objectives. No objective presets are registered yet.
    """

    name: str
    aliases: tuple[str, ...] = ()
    family: str
    config_factory: Callable[..., ObjectiveConfig]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))


def list_objective_presets() -> tuple[str, ...]:
    """Return canonical objective preset names."""
    return ()


def get_objective_preset_spec(name: str) -> ObjectivePresetSpec:
    """Return metadata for a registered objective preset.

    Parameters
    ----------
    name
        Canonical preset name or alias. Lookup is case-insensitive.

    Notes
    -----
    No objective presets are registered yet.
    """
    raise KeyError(f"Unknown objective preset {name!r}. No objective presets are registered.")


def get_objective_config(name: str, **config_kwargs) -> ObjectiveConfig:
    """Build an :class:`ObjectiveConfig` from a registered objective preset.

    Parameters
    ----------
    name
        Canonical preset name or alias. Lookup is case-insensitive.
    **config_kwargs
        Keyword arguments that would be forwarded to the preset
        ``config_factory`` once presets exist.

    Notes
    -----
    No objective presets are registered yet.
    """
    del config_kwargs
    raise KeyError(f"Unknown objective preset {name!r}. No objective presets are registered.")


__all__ = [
    "ObjectivePresetSpec",
    "get_objective_config",
    "get_objective_preset_spec",
    "list_objective_presets",
]

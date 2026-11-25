"""Dataclass metadata for architecture and objective presets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from audiossl.config import AudioModelConfig


@dataclass(frozen=True, kw_only=True)
class ModelPresetSpec:
    """Registered architecture preset for building :class:`AudioModelConfig`.

    ``config_factory`` must accept the keyword overrides forwarded by
    :func:`~audiossl.presets.get_config` and
    :func:`~audiossl.presets.build_model`. ``aliases`` are normalized to
    lowercase for lookup.
    """

    name: str
    aliases: tuple[str, ...] = ()
    family: str
    config_factory: Callable[..., AudioModelConfig]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))


__all__ = ["ModelPresetSpec"]

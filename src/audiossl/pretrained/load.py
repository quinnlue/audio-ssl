"""Public API for loading registered pretrained ``AudioModel`` checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from audiossl.presets import build_model

from .download import resolve_checkpoint
from .registry import get_pretrained_spec
from .specs import PretrainedModelSpec

if TYPE_CHECKING:
    from audiossl.modelling.model import AudioModel


def load_pretrained(
    name: str,
    *,
    source: str | Path | None = None,
    cache_dir: str | Path | None = None,
    map_location: Any = "cpu",
    config_kwargs: Mapping[str, Any] | None = None,
) -> tuple[AudioModel, PretrainedModelSpec]:
    """Build a registered architecture and load its pretrained weights.

    Parameters
    ----------
    name
        Canonical pretrained model name or alias.
    source
        Optional local checkpoint path or URL override.
    cache_dir
        Directory used when downloading remote checkpoints.
    map_location
        Torch map location passed to the model-specific loader.
    config_kwargs
        Extra keyword arguments forwarded to ``build_model``.

    Returns
    -------
    model
        Built :class:`AudioModel` with pretrained weights loaded in place.
    spec
        Resolved :class:`PretrainedModelSpec` metadata for the requested name.
    """

    spec = get_pretrained_spec(name)
    kwargs = dict(config_kwargs or {})
    model = build_model(spec.architecture, **kwargs)
    checkpoint_path = resolve_checkpoint(spec, source=source, cache_dir=cache_dir)
    spec.load_fn(checkpoint_path, model, map_location=map_location)
    return model, spec


__all__ = ["load_pretrained"]

"""Legacy Dasheng factories backed by the public preset/pretrained APIs.

These helpers keep older ``audiossl.modelling.pretrained.models.dasheng``
imports working while delegating config construction to :mod:`audiossl.presets`
and checkpoint loading to :mod:`audiossl.pretrained`.
"""

from pathlib import Path
from typing import Any

from audiossl.modelling.model import AudioModel
from audiossl.presets import build_model, get_config
from audiossl.pretrained import PretrainedModelSpec, load_pretrained


def dasheng_base_config(**config_kwargs):
    """Build the Dasheng base :class:`audiossl.config.AudioModelConfig`.

    Parameters
    ----------
    **config_kwargs
        Keyword overrides forwarded to the registered ``dasheng-base`` preset
        config factory.

    Returns
    -------
    audiossl.config.AudioModelConfig
        Frozen model config consumed by :class:`audiossl.modelling.model.AudioModel`.
    """

    return get_config("dasheng-base", **config_kwargs)


def dasheng_base(**config_kwargs) -> AudioModel:
    """Build an unweighted Dasheng base :class:`AudioModel`.

    Parameters
    ----------
    **config_kwargs
        Keyword overrides forwarded to the registered ``dasheng-base`` preset.

    Returns
    -------
    AudioModel
        Model constructed through the canonical preset path. Runtime seeding,
        validation, dtype, and device placement are owned by ``AudioModel``
        construction.
    """

    return build_model("dasheng-base", **config_kwargs)


def dasheng_base_pretrained(
    *,
    source: str | Path | None = None,
    cache_dir: str | Path | None = None,
    map_location: Any = "cpu",
    config_kwargs: dict[str, Any] | None = None,
) -> tuple[AudioModel, PretrainedModelSpec]:
    """Build a Dasheng base :class:`AudioModel` and load pretrained weights.

    Parameters
    ----------
    source
        Optional local checkpoint path or URL override. When omitted, the
        registered pretrained source is resolved by :mod:`audiossl.pretrained`.
    cache_dir
        Directory used for remote checkpoint downloads.
    map_location
        Torch map location passed to the Dasheng pretrained loader.
    config_kwargs
        Optional preset overrides forwarded while constructing the base
        ``AudioModel``.

    Returns
    -------
    model
        Dasheng base model with pretrained weights loaded in place.
    spec
        Resolved pretrained metadata for the ``dasheng-base`` entry.

    Notes
    -----
    This compatibility helper preserves the legacy Dasheng factory surface. It
    does not bypass the public preset registry or the canonical ``AudioModel``
    construction/runtime contract.
    """

    return load_pretrained(
        "dasheng-base",
        source=source,
        cache_dir=cache_dir,
        map_location=map_location,
        config_kwargs=config_kwargs,
    )


__all__ = ["dasheng_base", "dasheng_base_config", "dasheng_base_pretrained"]

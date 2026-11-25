"""Factory for constructing training objectives around an ``AudioModel``."""

from __future__ import annotations

import logging

import torch

from ..model import AudioModel
from ..registry import get_objective_registry
from ._base import Objective
from .protocols import AugmentationPipeline

logger = logging.getLogger(__name__)


def build_objective(
    model: AudioModel,
    config: object,
    *,
    augmentor: AugmentationPipeline | None = None,
    generator: torch.Generator | None = None,
) -> Objective:
    """Build the configured SSL objective.

    Parameters
    ----------
    model
        Constructed audio model wrapped by the objective. Objectives own
        training-time orchestration, losses, views, masks, heads, teacher
        paths, and metrics while reusing the model stage APIs for representation
        work.
    config
        Objective config; its ``type`` selects the class from the objective
        registry, so registered third-party objectives build the same way.
    augmentor
        Trainer-constructed augmentation pipeline. Passed only to objectives
        declaring ``uses_augmentor``; supplying one to any other objective
        raises ``ValueError``.
    generator
        Trainer-owned RNG for objective masking or negative sampling. Passed
        only to objectives declaring ``uses_generator``.

    Returns
    -------
    Objective
        Concrete objective whose public ``forward`` accepts an ``AudioBatch``
        and returns an objective-specific output dataclass.
    """

    type_name = getattr(config, "type", None)
    if not isinstance(type_name, str):
        raise TypeError(
            "build_objective expected an objective config with a string 'type' field, "
            f"got {type(config).__name__}"
        )
    try:
        objective_cls = get_objective_registry()[type_name]
    except KeyError as exc:
        raise TypeError(str(exc.args[0])) from None
    if not (isinstance(objective_cls, type) and issubclass(objective_cls, Objective)):
        raise TypeError(
            f"objective registry entry {type_name!r} must be an Objective subclass, "
            f"got {objective_cls!r}"
        )

    kwargs: dict[str, object] = {}
    if objective_cls.uses_augmentor:
        kwargs["augmentor"] = augmentor
    elif augmentor is not None:
        raise ValueError(f"{objective_cls.__name__} does not use an augmentor")
    if objective_cls.uses_generator:
        kwargs["generator"] = generator

    objective = objective_cls(model=model, config=config, **kwargs)
    if objective_cls.uses_generator:
        objective._require_generator()
    logger.info(
        "built objective %s (type=%r, trainable params=%d)",
        objective_cls.__name__,
        type_name,
        sum(p.numel() for name, p in objective.named_parameters()
            if p.requires_grad and not name.startswith("model.")),
    )
    return objective


__all__ = ["Objective", "build_objective"]

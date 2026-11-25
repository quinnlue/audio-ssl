"""Capability declarations for cross-component compatibility checks.

Component configs declare what they provide through a ``capabilities``
attribute (class-level ``ClassVar`` or a property when it depends on field
values). Objective configs declare ``required_capabilities`` and
``incompatible_capabilities``. :func:`check_objective_compatibility` compares
the two, so objectives never need ``isinstance`` checks against concrete
component configs and new components plug in by declaring capabilities.
"""

from __future__ import annotations

from typing import Any

RECONSTRUCTION_TARGET = "reconstruction_target"
"""Processor exposes ``reconstruction_target(batch)`` feature targets."""

PATCH_GRID = "patch_grid"
"""Tokenizer emits a regular (time, frequency) patch grid with ``reorder_patch_targets``."""

PREFIX_TOKEN = "prefix_token"
"""Embedding prepends at least one non-content token (for example CLS)."""

TOKEN_SUBSAMPLING = "token_subsampling"
"""Embedding ``finalize`` drops tokens, so its output axis differs from the token grid."""

SEQUENCE_CONVOLUTION = "sequence_convolution"
"""Encoder convolves over the token axis and needs tokens in contiguous order."""

ALL_CAPABILITIES = frozenset(
    {RECONSTRUCTION_TARGET, PATCH_GRID, PREFIX_TOKEN, TOKEN_SUBSAMPLING, SEQUENCE_CONVOLUTION}
)

_MODEL_COMPONENTS = ("processor", "tokenizer", "embedding", "encoder")


def component_capabilities(config: Any) -> frozenset[str]:
    """Capabilities declared by one component config (empty when undeclared)."""
    return frozenset(getattr(config, "capabilities", ()))


def model_capabilities(model_config: Any) -> dict[str, frozenset[str]]:
    """Map each model component name to the capabilities its config declares."""
    return {
        name: component_capabilities(getattr(model_config, name))
        for name in _MODEL_COMPONENTS
    }


def check_objective_compatibility(model_config: Any, objective_config: Any) -> None:
    """Raise ``ValueError`` when the objective cannot run on the model's components."""
    by_component = model_capabilities(model_config)
    provided = frozenset().union(*by_component.values())
    objective_name = type(objective_config).__name__
    required = frozenset(getattr(objective_config, "required_capabilities", ()))
    incompatible = frozenset(getattr(objective_config, "incompatible_capabilities", ()))

    missing = sorted(required - provided)
    if missing:
        raise ValueError(
            f"{objective_name} requires model capabilities {missing}, which no component "
            f"provides; model provides {_describe(by_component, model_config)}"
        )
    conflicts = sorted(incompatible & provided)
    if conflicts:
        sources = [
            f"{capability} (from {name} {getattr(model_config, name).type!r})"
            for capability in conflicts
            for name, caps in by_component.items()
            if capability in caps
        ]
        raise ValueError(f"{objective_name} is incompatible with model capabilities: {sources}")


def _describe(by_component: dict[str, frozenset[str]], model_config: Any) -> str:
    parts = [
        f"{name} {getattr(model_config, name).type!r}: {sorted(caps)}"
        for name, caps in by_component.items()
    ]
    return "; ".join(parts)


__all__ = [
    "ALL_CAPABILITIES",
    "PATCH_GRID",
    "PREFIX_TOKEN",
    "RECONSTRUCTION_TARGET",
    "SEQUENCE_CONVOLUTION",
    "TOKEN_SUBSAMPLING",
    "check_objective_compatibility",
    "component_capabilities",
    "model_capabilities",
]

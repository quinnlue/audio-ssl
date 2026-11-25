"""Initialization routing helpers for shared core modules."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..registry import INITIALIZATION_REGISTRY


Initializer = Callable[[torch.Tensor], torch.Tensor]


def resolve_initializer(name: str) -> Initializer:
    """Resolve an initialization registry key to a tensor initializer."""
    if not isinstance(name, str):
        raise ValueError(
            "initialization must be a string key, "
            f"got {name!r} ({type(name).__name__})"
        )
    try:
        return INITIALIZATION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported initialization {name!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        ) from exc


def initialize_linear(linear: nn.Linear, initialization: str) -> None:
    """Initialize a linear layer weight via the registry and zero its bias."""
    initializer = resolve_initializer(initialization)
    initializer(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def initialize_conv1d(conv: nn.Conv1d, initialization: str) -> None:
    """Initialize a Conv1d layer weight via the registry and zero its bias."""
    initializer = resolve_initializer(initialization)
    initializer(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def initialize_batch_norm(batch_norm: nn.BatchNorm1d | nn.BatchNorm2d) -> None:
    """Initialize BatchNorm affine parameters to identity transform."""
    if batch_norm.weight is not None:
        nn.init.ones_(batch_norm.weight)
    if batch_norm.bias is not None:
        nn.init.zeros_(batch_norm.bias)


def initialize_multihead_attention(
    attention: nn.MultiheadAttention,
    initialization: str,
) -> None:
    """Initialize MultiheadAttention projections through the registry."""
    initializer = resolve_initializer(initialization)

    if attention.in_proj_weight is not None:
        initializer(attention.in_proj_weight)
    else:
        for weight in (
            attention.q_proj_weight,
            attention.k_proj_weight,
            attention.v_proj_weight,
        ):
            if weight is not None:
                initializer(weight)

    if attention.in_proj_bias is not None:
        nn.init.zeros_(attention.in_proj_bias)

    initialize_linear(attention.out_proj, initialization)

    if attention.bias_k is not None:
        nn.init.zeros_(attention.bias_k)
    if attention.bias_v is not None:
        nn.init.zeros_(attention.bias_v)

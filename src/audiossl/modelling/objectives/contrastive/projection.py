"""Projection head used by contrastive objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import ProjectionHeadConfig

from ...core.initialization import initialize_linear
from ...registry import ACTIVATION_REGISTRY, build_activation


class ProjectionHead(nn.Module):
    """Configurable MLP projection or predictor head owned by objectives.

    Parameters
    ----------
    input_dim
        Feature dimension of incoming pooled representations with shape
        ``(B, input_dim)``.
    config
        Projection head config. ``depth == 1`` collapses to a single linear
        layer; ``depth >= 2`` stacks hidden linear, activation, and dropout
        blocks before the final projection.

    All Linear weights are initialized via the shared initialization registry
    (consistent with `MLPConfig` / `ProjectionConfig` semantics in the rest
    of the codebase).
    """

    def __init__(self, *, input_dim: int, config: ProjectionHeadConfig):
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise TypeError(f"input_dim must be int, got {type(input_dim).__name__}")
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not isinstance(config, ProjectionHeadConfig):
            raise TypeError(
                f"config must be ProjectionHeadConfig, got {type(config).__name__}"
            )
        if config.depth < 1:
            raise ValueError(f"ProjectionHeadConfig.depth must be >= 1, got {config.depth}")
        if config.hidden_dim < 1:
            raise ValueError(
                f"ProjectionHeadConfig.hidden_dim must be >= 1, got {config.hidden_dim}"
            )
        if config.output_dim < 1:
            raise ValueError(
                f"ProjectionHeadConfig.output_dim must be >= 1, got {config.output_dim}"
            )
        if config.activation not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported ProjectionHeadConfig.activation {config.activation!r}; "
                f"expected one of {sorted(ACTIVATION_REGISTRY)}"
            )
        if not (0.0 <= config.dropout_p <= 1.0):
            raise ValueError(
                f"ProjectionHeadConfig.dropout_p must be in [0, 1], got {config.dropout_p}"
            )

        self.input_dim = input_dim
        self.config = config

        layers: list[nn.Module] = []
        if config.depth == 1:
            final = nn.Linear(input_dim, config.output_dim)
            initialize_linear(final, config.initialization)
            layers.append(final)
        else:
            previous = input_dim
            for _ in range(config.depth - 1):
                hidden = nn.Linear(previous, config.hidden_dim)
                initialize_linear(hidden, config.initialization)
                layers.append(hidden)
                layers.append(build_activation(config.activation))
                layers.append(nn.Dropout(config.dropout_p))
                previous = config.hidden_dim
            final = nn.Linear(previous, config.output_dim)
            initialize_linear(final, config.initialization)
            layers.append(final)

        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project pooled representations from ``(B, input_dim)`` to ``(B, D)``."""

        if not isinstance(z, torch.Tensor):
            raise TypeError(f"z must be torch.Tensor, got {type(z).__name__}")
        if z.ndim != 2:
            raise ValueError(f"z must be rank-2 (B, D), got shape {tuple(z.shape)}")
        if z.shape[-1] != self.input_dim:
            raise ValueError(
                f"z has feature dim {z.shape[-1]}, expected {self.input_dim}"
            )
        out = self.layers(z)
        if self.config.final_l2_normalize:
            out = F.normalize(out, dim=-1)
        return out


__all__ = ["ProjectionHead"]

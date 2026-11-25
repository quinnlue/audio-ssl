"""DINO projection head and prototype normalization helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import DINOHeadConfig

from ...core.initialization import initialize_linear
from ...registry import ACTIVATION_REGISTRY, build_activation


class DINOHead(nn.Module):
    """DINO MLP plus weight-normalized prototype layer.

    Parameters
    ----------
    input_dim
        Dimension of pooled model representations with shape ``(B, input_dim)``.
    config
        Head config controlling MLP depth, bottleneck width, prototypes, dropout,
        activation, and initialization.
    """

    def __init__(self, *, input_dim: int, config: DINOHeadConfig) -> None:
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise TypeError(f"input_dim must be int, got {type(input_dim).__name__}")
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not isinstance(config, DINOHeadConfig):
            raise TypeError(f"config must be DINOHeadConfig, got {type(config).__name__}")
        if config.depth < 1:
            raise ValueError(f"DINOHeadConfig.depth must be >= 1, got {config.depth}")
        if config.hidden_dim < 1:
            raise ValueError(
                f"DINOHeadConfig.hidden_dim must be >= 1, got {config.hidden_dim}"
            )
        if config.bottleneck_dim < 1:
            raise ValueError(
                "DINOHeadConfig.bottleneck_dim must be >= 1, "
                f"got {config.bottleneck_dim}"
            )
        if config.num_prototypes < 1:
            raise ValueError(
                "DINOHeadConfig.num_prototypes must be >= 1, "
                f"got {config.num_prototypes}"
            )
        if config.activation not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported DINOHeadConfig.activation {config.activation!r}; "
                f"expected one of {sorted(ACTIVATION_REGISTRY)}"
            )
        if not (0.0 <= config.dropout_p <= 1.0):
            raise ValueError(
                f"DINOHeadConfig.dropout_p must be in [0, 1], got {config.dropout_p}"
            )

        self.input_dim = input_dim
        self.config = config

        layers: list[nn.Module] = []
        if config.depth == 1:
            final_body = nn.Linear(input_dim, config.bottleneck_dim)
            initialize_linear(final_body, config.initialization)
            layers.append(final_body)
        else:
            previous = input_dim
            for _ in range(config.depth - 1):
                hidden = nn.Linear(previous, config.hidden_dim)
                initialize_linear(hidden, config.initialization)
                layers.append(hidden)
                layers.append(build_activation(config.activation))
                layers.append(nn.Dropout(config.dropout_p))
                previous = config.hidden_dim
            bottleneck = nn.Linear(previous, config.bottleneck_dim)
            initialize_linear(bottleneck, config.initialization)
            layers.append(bottleneck)

        self.body = nn.Sequential(*layers)
        prototypes = nn.Linear(config.bottleneck_dim, config.num_prototypes, bias=False)
        initialize_linear(prototypes, config.initialization)
        self.prototypes = nn.utils.weight_norm(prototypes)
        self.prototypes.weight_g.requires_grad_(False)

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        """Return normalized bottleneck features with shape ``(B, bottleneck_dim)``."""

        if not isinstance(z, torch.Tensor):
            raise TypeError(f"z must be torch.Tensor, got {type(z).__name__}")
        if z.ndim != 2:
            raise ValueError(f"z must be rank-2 (B, D), got shape {tuple(z.shape)}")
        if z.shape[-1] != self.input_dim:
            raise ValueError(
                f"z has feature dim {z.shape[-1]}, expected {self.input_dim}"
            )
        return F.normalize(self.body(z), dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return prototype logits with shape ``(B, num_prototypes)``."""

        return self.prototypes(self.forward_features(z))


__all__ = ["DINOHead"]

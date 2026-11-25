"""Classification head for supervised fine-tuning objectives."""

from __future__ import annotations

import torch
import torch.nn as nn

from audiossl.config import SFTHeadConfig

from ...core.initialization import initialize_linear
from ...registry import ACTIVATION_REGISTRY, build_activation


class ClassificationHead(nn.Module):
    """MLP head mapping pooled representations to class logits."""

    def __init__(self, *, input_dim: int, num_classes: int, config: SFTHeadConfig) -> None:
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise TypeError(f"input_dim must be int, got {type(input_dim).__name__}")
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise TypeError(f"num_classes must be int, got {type(num_classes).__name__}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not isinstance(config, SFTHeadConfig):
            raise TypeError(f"config must be SFTHeadConfig, got {type(config).__name__}")
        if config.depth < 1:
            raise ValueError(f"SFTHeadConfig.depth must be >= 1, got {config.depth}")
        if config.depth > 1 and config.hidden_dim < 1:
            raise ValueError(
                f"SFTHeadConfig.hidden_dim must be >= 1 when depth > 1, got {config.hidden_dim}"
            )
        if config.activation not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported SFTHeadConfig.activation {config.activation!r}; "
                f"expected one of {sorted(ACTIVATION_REGISTRY)}"
            )
        if not (0.0 <= config.dropout_p <= 1.0):
            raise ValueError(
                f"SFTHeadConfig.dropout_p must be in [0, 1], got {config.dropout_p}"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config

        layers: list[nn.Module] = []
        if config.depth == 1:
            final = nn.Linear(input_dim, num_classes)
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
            final = nn.Linear(previous, num_classes)
            initialize_linear(final, config.initialization)
            layers.append(final)

        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"z must be torch.Tensor, got {type(z).__name__}")
        if z.ndim != 2:
            raise ValueError(f"z must be rank-2 (B, D), got shape {tuple(z.shape)}")
        if z.shape[-1] != self.input_dim:
            raise ValueError(
                f"z has feature dim {z.shape[-1]}, expected {self.input_dim}"
            )
        return self.layers(z)


__all__ = ["ClassificationHead"]
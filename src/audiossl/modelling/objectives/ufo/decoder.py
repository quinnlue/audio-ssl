"""CNN decoder for EAT UFO frame-level target prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from audiossl.config import UFODecoderConfig
from audiossl.config.batch_dataclasses import TokenLayout

from ...registry import build_activation


class UFODecoder(nn.Module):
    """Six-layer 2-D CNN decoder over restored patch grids."""

    def __init__(self, *, d_model: int, config: UFODecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        layers: list[nn.Module] = []
        for _ in range(config.num_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=config.kernel_size,
                        padding=config.kernel_size // 2,
                    ),
                    nn.LayerNorm([d_model], eps=config.layer_norm_eps),
                    build_activation(config.activation),
                ]
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        layout: TokenLayout,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"UFODecoder.forward expected tokens rank-3, got {tuple(tokens.shape)}"
            )
        batch_size, num_tokens, dim = tokens.shape
        expected = layout.time_patches * layout.frequency_patches
        if num_tokens != expected:
            raise ValueError(
                "UFODecoder.forward expected patch token count "
                f"{expected}, got {num_tokens}"
            )

        x = tokens.reshape(
            batch_size,
            layout.time_patches,
            layout.frequency_patches,
            dim,
        ).permute(0, 3, 1, 2)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = x.permute(0, 2, 3, 1)
                x = layer(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = layer(x)
        return x.permute(0, 2, 3, 1).reshape(batch_size, num_tokens, dim)


__all__ = ["UFODecoder"]

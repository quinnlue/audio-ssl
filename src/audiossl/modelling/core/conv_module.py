"""Conformer convolution branch used inside encoder blocks."""

import torch
import torch.nn as nn

from audiossl.config import ConvModuleConfig

from ..registry import build_activation


class ConvModule(nn.Module):
    """Conformer convolution module (Gulati et al., 2020, Fig. 2).

    LN -> pointwise (D -> expansion * D) -> GLU -> depthwise (inner -> inner)
    -> BatchNorm -> activation -> pointwise (inner -> D) -> dropout.

    Inputs and outputs have shape `(B, N, D)`. Boolean attention masks have
    shape `(B, N)`, where `True` means valid. Invalid frames are zeroed before
    the depthwise convolution and again after the final pointwise projection.
    """

    def __init__(self, *, d_model: int, config: ConvModuleConfig, layer_norm_eps: float = 1e-5):
        super().__init__()
        kernel_size = config.kernel_size
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "ConvModule kernel_size must be a positive odd integer, "
                f"got {kernel_size}"
            )

        expansion_factor = config.expansion_factor
        if expansion_factor <= 0 or expansion_factor % 2 != 0:
            raise ValueError(
                "ConvModule expansion_factor must be a positive even integer "
                f"(GLU halves the pointwise channels), got {expansion_factor}"
            )

        expanded = d_model * expansion_factor
        inner = expanded // 2

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.pw1 = nn.Conv1d(d_model, expanded, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            inner,
            inner,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner,
        )
        self.batch_norm = nn.BatchNorm1d(inner)
        self.activation = build_activation(config.activation)
        self.pw2 = nn.Conv1d(inner, d_model, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply the convolution branch to `(B, N, D)` encoder states."""
        mask = attention_mask.unsqueeze(1).to(dtype=x.dtype)

        x = self.norm(x).transpose(1, 2)
        x = self.pw1(x)
        x = self.glu(x)
        x = x * mask
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pw2(x)
        x = self.dropout(x)
        return x.transpose(1, 2) * attention_mask.unsqueeze(-1).to(dtype=x.dtype)

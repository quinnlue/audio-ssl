"""EAT / data2vec-style AltTransformer encoder block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import AltTransformerEncoderConfig
from ..core.initialization import initialize_linear
from ..registry import build_activation


class AltAttention(nn.Module):
    """Manual multi-head self-attention matching EAT ``AltAttention``."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        qkv_bias: bool,
        attn_dropout_p: float,
        proj_dropout_p: float,
        initialization: str,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                "AltAttention d_model must be divisible by num_heads, "
                f"got {d_model} and {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_dropout_p = float(attn_dropout_p)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout_p)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_dropout_p)
        initialize_linear(self.qkv, initialization)
        initialize_linear(self.proj, initialization)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Fused scaled-dot-product attention (flash / mem-efficient kernel):
        # no explicit (B, H, S, S) score matrix, no float32 softmax round-trip.
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]  # True = key participates
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        return self.proj_drop(out)


class AltMLP(nn.Module):
    """Feed-forward block matching timm ``Mlp`` used by EAT ``AltBlock``."""

    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        activation: str,
        mlp_dropout_p: float,
        initialization: str,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = build_activation(activation)
        self.drop = nn.Dropout(mlp_dropout_p)
        self.fc2 = nn.Linear(d_ff, d_model)
        initialize_linear(self.fc1, initialization)
        initialize_linear(self.fc2, initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class AltTransformerBlock(nn.Module):
    """EAT ``AltBlock`` with optional pre-norm or post-norm layout."""

    def __init__(self, *, d_model: int, config: AltTransformerEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, AltTransformerEncoderConfig):
            raise TypeError(
                "AltTransformerBlock requires AltTransformerEncoderConfig, "
                f"got {type(config).__name__}"
            )
        self.layer_norm_first = config.layer_norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.attn = AltAttention(
            d_model=d_model,
            num_heads=config.mha.num_heads,
            qkv_bias=config.qkv_bias,
            attn_dropout_p=config.mha.dropout_p,
            proj_dropout_p=config.residual_dropout_p,
            initialization=config.mha.initialization,
        )
        self.mlp = AltMLP(
            d_model=d_model,
            d_ff=config.mlp.d_ff,
            activation=config.mlp.activation,
            mlp_dropout_p=config.mlp.dropout_p,
            initialization=config.mlp.initialization,
        )
        self.post_mlp_dropout = nn.Dropout(config.post_mlp_dropout_p)
        self.residual_dropout = nn.Dropout(config.residual_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the EAT block; ``positions`` is accepted for API uniformity (no RoPE)."""
        if self.layer_norm_first:
            x = x + self.residual_dropout(
                self.attn(self.norm1(x), attention_mask=attention_mask)
            )
            residual = x
            x = self.mlp(self.norm2(x))
            x = residual + self.residual_dropout(self.post_mlp_dropout(x))
            return x

        x = x + self.residual_dropout(self.attn(x, attention_mask=attention_mask))
        residual = self.norm1(x)
        x = self.mlp(residual)
        x = self.norm2(residual + self.residual_dropout(self.post_mlp_dropout(x)))
        return x


__all__ = ["AltAttention", "AltMLP", "AltTransformerBlock"]

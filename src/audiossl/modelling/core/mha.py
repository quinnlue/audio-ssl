"""Attention helpers for encoder blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import (
    MultiHeadAttentionConfig,
    PerBlockPositionalEncodingConfig,
    RopePositionalEncodingConfig,
)
from .initialization import initialize_multihead_attention
from .positional_encoding import RoPE


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional per-block RoPE.

    The wrapper preserves `(B, N, D)` states and consumes boolean attention
    masks with shape `(B, N)`, where `True` means valid. Absolute or entry
    positional encoding is intentionally outside this module.

    Parameters
    ----------
    d_model
        Model dimension `D`.
    config
        Attention head count, dropout, and initialization route.
    positional_encoding
        Optional per-block positional encoding config. Only RoPE is supported.
    """

    def __init__(
        self,
        *,
        d_model: int,
        config: MultiHeadAttentionConfig,
        positional_encoding: PerBlockPositionalEncodingConfig | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = config.num_heads
        self.head_dim = d_model // config.num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout_p,
            batch_first=True,
        )

        self.use_positional_encoding = positional_encoding is not None
        self.use_rope = False
        if positional_encoding is None:
            initialize_multihead_attention(self.mha, config.initialization)
            return

        self.use_rope = positional_encoding.type == "rope"
        if self.use_rope:
            if not isinstance(positional_encoding, RopePositionalEncodingConfig):
                raise TypeError(
                    "Expected RopePositionalEncodingConfig when positional_encoding.type == 'rope'"
                )
            self.rope = RoPE(positional_encoding, d_model=self.head_dim)
        else:
            raise ValueError(
                f"Unsupported positional_encoding.type {positional_encoding.type!r}"
            )
        initialize_multihead_attention(self.mha, config.initialization)

    def _rope_attention(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply scaled dot-product attention after RoPE on query/key heads."""
        q, k, v = F.linear(
            x,
            self.mha.in_proj_weight,
            self.mha.in_proj_bias,
        ).chunk(3, dim=-1)

        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        head_positions = (
            positions.repeat_interleave(self.num_heads, dim=0) if positions is not None else None
        )
        q = self.rope(
            q.reshape(batch_size * self.num_heads, seq_len, self.head_dim), head_positions
        )
        k = self.rope(
            k.reshape(batch_size * self.num_heads, seq_len, self.head_dim), head_positions
        )
        q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)

        dropout_p = self.mha.dropout if self.training else 0.0
        attn_mask = attention_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.mha.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over `(B, N, D)` states using a `(B, N)` validity mask.

        ``positions`` (integer ``(B, N)``) only affects per-block RoPE; absolute
        position information lives in the embedding.
        """
        if self.use_rope:
            return self._rope_attention(x, attention_mask, positions)
        out, _ = self.mha(x, x, x, need_weights=False, key_padding_mask=~attention_mask)
        return out

"""Transformer encoder block implementation."""

import torch
import torch.nn as nn

from audiossl.config import TransformerEncoderConfig
from ..core.mha import MultiHeadAttention
from ..core.mlp import MLP


class TransformerBlock(nn.Module):
    """Transformer encoder block with optional per-block positional encoding.

    The block preserves token states with shape `(B, N, d_model)`. It consumes a
    boolean attention mask with shape `(B, N)`, where `True` means valid.
    Positional encoding, when configured, is passed into attention per block;
    entry positional encoding and pooling are outside the encoder block.

    Parameters
    ----------
    d_model
        Token embedding dimension consumed and produced by the block.
    config
        Transformer encoder configuration for attention, MLP, normalization,
        dropout, and optional per-block positional encoding.
    """

    def __init__(self, *, d_model: int, config: TransformerEncoderConfig) -> None:
        """Create a Transformer encoder block."""

        super().__init__()
        if not isinstance(config, TransformerEncoderConfig):
            raise TypeError(
                "TransformerBlock requires TransformerEncoderConfig, "
                f"got {type(config).__name__}"
            )

        self.pre_norm = config.pre_norm
        self.norm1 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.mha = MultiHeadAttention(
            d_model=d_model,
            config=config.mha,
            positional_encoding=config.positional_encoding,
        )
        self.mlp = MLP(d_model=d_model, config=config.mlp)
        self.residual_dropout = nn.Dropout(config.residual_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention and MLP sublayers.

        Parameters
        ----------
        x
            Token states with shape `(B, N, d_model)`.
        attention_mask
            Boolean validity mask with shape `(B, N)`, where `True` means valid.
        positions
            Optional integer token positions `(B, N)` for per-block RoPE.

        Returns
        -------
        torch.Tensor
            Updated token states with shape `(B, N, d_model)`.
        """
        if self.pre_norm:
            x = x + self.residual_dropout(
                self.mha(self.norm1(x), attention_mask=attention_mask, positions=positions)
            )
            x = x + self.residual_dropout(self.mlp(self.norm2(x)))
            return x

        x = self.norm1(
            x
            + self.residual_dropout(
                self.mha(x, attention_mask=attention_mask, positions=positions)
            )
        )
        x = self.norm2(x + self.residual_dropout(self.mlp(x)))
        return x

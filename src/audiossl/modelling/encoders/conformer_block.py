"""Conformer encoder block implementation."""

import torch
import torch.nn as nn

from audiossl.config import ConformerEncoderConfig

from ..core.conv_module import ConvModule
from ..core.mha import MultiHeadAttention
from ..core.mlp import MLP


class ConformerBlock(nn.Module):
    """Conformer encoder block with macaron FFNs and convolution.

    The block preserves token states with shape `(B, N, d_model)`. It applies
    two half-scaled feed-forward sublayers around attention and convolution.
    Attention may receive optional per-block positional encoding through the
    configuration; entry positional encoding and pooling are outside the block.

    Parameters
    ----------
    d_model
        Token embedding dimension consumed and produced by the block.
    config
        Conformer encoder configuration for attention, convolution, MLP,
        normalization, dropout, and optional per-block positional encoding.
    """

    def __init__(self, *, d_model: int, config: ConformerEncoderConfig) -> None:
        """Create a Conformer encoder block."""
        super().__init__()
        if not isinstance(config, ConformerEncoderConfig):
            raise TypeError(
                "ConformerBlock requires ConformerEncoderConfig, "
                f"got {type(config).__name__}"
            )

        self.pre_norm = config.pre_norm
        self.ffn_scale = 0.5
        self.norm_ffn1 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.norm_attn = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.norm_conv = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.norm_ffn2 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.final_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)

        self.ffn1 = MLP(d_model=d_model, config=config.mlp)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            config=config.mha,
            positional_encoding=config.positional_encoding,
        )
        self.conv = ConvModule(
            d_model=d_model,
            config=config.conv,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.ffn2 = MLP(d_model=d_model, config=config.mlp)
        self.residual_dropout = nn.Dropout(config.residual_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Conformer sublayers while preserving token shape.

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
            x = x + self.ffn_scale * self.residual_dropout(self.ffn1(self.norm_ffn1(x)))
            x = x + self.residual_dropout(
                self.attn(self.norm_attn(x), attention_mask=attention_mask, positions=positions)
            )
            x = x + self.residual_dropout(self.conv(self.norm_conv(x), attention_mask))
            x = x + self.ffn_scale * self.residual_dropout(self.ffn2(self.norm_ffn2(x)))
            return self.final_norm(x)

        x = self.norm_ffn1(x + self.ffn_scale * self.residual_dropout(self.ffn1(x)))
        x = self.norm_attn(
            x
            + self.residual_dropout(
                self.attn(x, attention_mask=attention_mask, positions=positions)
            )
        )
        x = self.norm_conv(x + self.residual_dropout(self.conv(x, attention_mask)))
        x = self.final_norm(x + self.ffn_scale * self.residual_dropout(self.ffn2(x)))
        return x

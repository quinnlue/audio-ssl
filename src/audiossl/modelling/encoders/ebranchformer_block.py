"""Enhanced Branchformer encoder block implementation."""

import torch
import torch.nn as nn

from audiossl.config import EBranchformerEncoderConfig

from ..core.cgmlp import CGMLP
from ..core.e_branch_merge import EBranchMerger
from ..core.mha import MultiHeadAttention
from ..core.mlp import MLP


class EBranchformerBlock(nn.Module):
    """E-Branchformer block (Kim et al., 2022).

    Parallel global (MHA) and local (cgMLP) branches with enhanced DwConv merge.
    Optionally stacks macaron-style FFN before branches and post-merge FFN after,
    matching ESPnet's ``EBranchformerEncoderLayer``. The block preserves token
    states with shape `(B, N, d_model)` and consumes a boolean attention mask
    with shape `(B, N)`, where `True` means valid. Optional positional encoding
    is per-block through attention only; entry positional encoding and pooling
    are outside the block.

    Parameters
    ----------
    d_model
        Token embedding dimension consumed and produced by the block.
    config
        E-Branchformer encoder configuration for branch modules, enhanced merge
        policy, optional FFNs, normalization, dropout, and optional per-block
        positional encoding.
    """

    def __init__(self, *, d_model: int, config: EBranchformerEncoderConfig) -> None:
        """Create an E-Branchformer encoder block."""
        super().__init__()
        if not isinstance(config, EBranchformerEncoderConfig):
            raise TypeError(
                "EBranchformerBlock requires EBranchformerEncoderConfig, "
                f"got {type(config).__name__}"
            )

        self.pre_norm = config.pre_norm
        self.use_macaron_ffn = config.macaron_ffn
        self.use_ffn = config.use_ffn
        self.ffn_scale = 0.5 if self.use_macaron_ffn else 1.0

        self.norm_attn = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.final_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)

        if self.use_macaron_ffn:
            self.norm_ff_macaron = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
            self.ffn_macaron = MLP(d_model=d_model, config=config.mlp)  # type: ignore[arg-type]
        if self.use_ffn:
            self.norm_ff = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
            self.ffn = MLP(d_model=d_model, config=config.mlp)  # type: ignore[arg-type]

        self.attn = MultiHeadAttention(
            d_model=d_model,
            config=config.mha,
            positional_encoding=config.positional_encoding,
        )
        self.cgmlp = CGMLP(
            d_model=d_model,
            config=config.cgmlp,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.merger = EBranchMerger(d_model=d_model, config=config.merge)
        self.residual_dropout = nn.Dropout(config.residual_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply enhanced global and local branches.

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
            if self.use_macaron_ffn:
                x = x + self.ffn_scale * self.residual_dropout(
                    self.ffn_macaron(self.norm_ff_macaron(x))
                )

            g = self.attn(self.norm_attn(x), attention_mask=attention_mask, positions=positions)
            l = self.cgmlp(x, attention_mask=attention_mask)

            g = self.residual_dropout(g)
            l = self.residual_dropout(l)

            x = x + self.residual_dropout(self.merger(g, l, attention_mask))

            if self.use_ffn:
                x = x + self.residual_dropout(self.ffn(self.norm_ff(x)))

            return self.final_norm(x)

        if self.use_macaron_ffn:
            x = self.norm_ff_macaron(
                x + self.ffn_scale * self.residual_dropout(self.ffn_macaron(x))
            )

        g = self.attn(self.norm_attn(x), attention_mask=attention_mask, positions=positions)
        l = self.cgmlp(x, attention_mask=attention_mask)
        g = self.residual_dropout(g)
        l = self.residual_dropout(l)

        x = x + self.residual_dropout(self.merger(g, l, attention_mask))

        if self.use_ffn:
            x = self.norm_ff(x + self.residual_dropout(self.ffn(x)))

        return self.final_norm(x)

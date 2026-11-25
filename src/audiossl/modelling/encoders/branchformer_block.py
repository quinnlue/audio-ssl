"""Branchformer encoder block implementation."""

import torch
import torch.nn as nn

from audiossl.config import BranchformerEncoderConfig

from ..core.branch_merge import BranchMerger
from ..core.cgmlp import CGMLP
from ..core.mha import MultiHeadAttention


class _StochasticDepth(nn.Module):
    """Per-sample stochastic depth used for per-branch drop in Branchformer.

    At training time, each sample's branch output is zeroed with probability
    ``drop_p`` and otherwise scaled by ``1 / (1 - drop_p)`` to preserve the
    expected value. At eval time this is a no-op. ``drop_p == 0.0`` short-
    circuits; ``drop_p == 1.0`` always zeros (useful for tests).
    """

    def __init__(self, drop_p: float):
        super().__init__()
        if not 0.0 <= drop_p <= 1.0:
            raise ValueError(
                f"_StochasticDepth drop_p must be in [0, 1], got {drop_p!r}"
            )
        self.drop_p = float(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Drop whole per-sample residual branches during training."""
        if not self.training or self.drop_p == 0.0:
            return x
        if self.drop_p >= 1.0:
            return torch.zeros_like(x)
        keep_prob = 1.0 - self.drop_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob).to(dtype=x.dtype)
        return x.div(keep_prob) * mask


class BranchformerBlock(nn.Module):
    """Vanilla Branchformer block (Peng et al., 2022).

    Parallel global (MHA) and local (cgMLP) branches are merged once per layer.
    The block preserves token states with shape `(B, N, d_model)` and consumes
    a boolean attention mask with shape `(B, N)`, where `True` means valid.
    Pre-norm applies LayerNorm before MHA while cgMLP self-normalizes.
    Optional positional encoding is per-block through attention only; entry
    positional encoding and pooling are outside the block.

    Parameters
    ----------
    d_model
        Token embedding dimension consumed and produced by the block.
    config
        Branchformer encoder configuration for branch modules, merge policy,
        normalization, dropout, stochastic depth, and optional per-block
        positional encoding.
    """

    def __init__(self, *, d_model: int, config: BranchformerEncoderConfig) -> None:
        """Create a Branchformer encoder block."""
        super().__init__()
        if not isinstance(config, BranchformerEncoderConfig):
            raise TypeError(
                "BranchformerBlock requires BranchformerEncoderConfig, "
                f"got {type(config).__name__}"
            )

        self.pre_norm = config.pre_norm
        self.norm_attn = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.final_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)

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
        self.merger = BranchMerger(d_model=d_model, config=config.merge)

        self.attn_branch_drop = _StochasticDepth(config.attention_branch_drop_p)
        self.cgmlp_branch_drop = _StochasticDepth(config.cgmlp_branch_drop_p)
        self.residual_dropout = nn.Dropout(config.residual_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply global and local branches and merge their outputs.

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
            g = self.attn(self.norm_attn(x), attention_mask=attention_mask, positions=positions)
        else:
            g = self.attn(x, attention_mask=attention_mask, positions=positions)

        l = self.cgmlp(x, attention_mask=attention_mask)

        g = self.attn_branch_drop(g)
        l = self.cgmlp_branch_drop(l)

        merged = self.merger(g, l)

        if self.pre_norm:
            x = x + self.residual_dropout(merged)
            return self.final_norm(x)

        return self.final_norm(x + self.residual_dropout(merged))

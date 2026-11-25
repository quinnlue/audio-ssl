"""Branchformer branch merge policies."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import BranchMergeConfig


class BranchMerger(nn.Module):
    """Merge two parallel branches of shape `(B, N, D)` into one `(B, N, D)`.

    Supports three modes from the Branchformer paper (Peng et al., 2022):

    - `concat`: concatenate along the feature dim and project back to `D` via a
      learned `Linear(2D -> D)`.
    - `learned_ave`: weighted sum `w_g * g + w_l * l` where the weights are
      a softmax over a learned `nn.Parameter` of shape `(2,)`. Initialized to
      uniform (equal weights).
    - `fixed_ave`: convex combination `alpha * g + (1 - alpha) * l` where
      `alpha` is fixed at construction time and registered as a non-trainable
      buffer.
    """

    def __init__(self, *, d_model: int, config: BranchMergeConfig):
        super().__init__()
        if config.method not in {"concat", "learned_ave", "fixed_ave"}:
            raise ValueError(
                f"Unsupported BranchMergeConfig.method {config.method!r}; "
                "expected one of {'concat', 'learned_ave', 'fixed_ave'}"
            )
        self.method = config.method

        if self.method == "concat":
            self.projection = nn.Linear(2 * d_model, d_model)
        elif self.method == "learned_ave":
            self.merge_weights = nn.Parameter(torch.zeros(2))
        else:
            if not 0.0 <= config.fixed_weight <= 1.0:
                raise ValueError(
                    "BranchMergeConfig.fixed_weight must be in [0, 1], "
                    f"got {config.fixed_weight!r}"
                )
            self.register_buffer(
                "fixed_weight",
                torch.tensor(float(config.fixed_weight)),
                persistent=False,
            )

    def forward(self, global_branch: torch.Tensor, local_branch: torch.Tensor) -> torch.Tensor:
        """Merge same-shaped global and local branch outputs."""
        if global_branch.shape != local_branch.shape:
            raise ValueError(
                "BranchMerger: global and local branch tensors must have identical shapes, "
                f"got {tuple(global_branch.shape)} and {tuple(local_branch.shape)}"
            )

        if self.method == "concat":
            merged = torch.cat([global_branch, local_branch], dim=-1)
            return self.projection(merged)

        if self.method == "learned_ave":
            weights = F.softmax(self.merge_weights, dim=0)
            return weights[0] * global_branch + weights[1] * local_branch

        fixed_weight = self.fixed_weight
        assert isinstance(fixed_weight, torch.Tensor)
        alpha = fixed_weight.to(dtype=global_branch.dtype)
        return alpha * global_branch + (1.0 - alpha) * local_branch

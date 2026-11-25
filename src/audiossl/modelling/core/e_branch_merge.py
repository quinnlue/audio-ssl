"""Enhanced Branchformer branch merge block."""

import torch
import torch.nn as nn

from audiossl.config import EBranchMergeConfig


class EBranchMerger(nn.Module):
    """Enhanced merge for E-Branchformer (Kim et al., 2022).

    Concatenates global and local branch outputs with shape `(B, N, D)`,
    applies mask-aware depthwise temporal convolution for local refinement, then
    projects back to `d_model`:

        `Y_C = Concat(Y_G, Y_L)`
        `Y_D = DwConv(Y_C)`
        `Y = Linear(Y_C + Y_D)`

    Boolean attention masks have shape `(B, N)`, where `True` means valid.
    """

    def __init__(self, *, d_model: int, config: EBranchMergeConfig):
        super().__init__()
        kernel_size = config.conv_kernel_size
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "EBranchMerger conv_kernel_size must be a positive odd integer, "
                f"got {kernel_size}"
            )

        merged_dim = 2 * d_model
        self.depthwise = nn.Conv1d(
            merged_dim,
            merged_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=merged_dim,
            bias=True,
        )
        self.projection = nn.Linear(merged_dim, d_model)

    def forward(
        self,
        global_branch: torch.Tensor,
        local_branch: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Merge global and local `(B, N, D)` branches using a validity mask."""
        if global_branch.shape != local_branch.shape:
            raise ValueError(
                "EBranchMerger: global and local branch tensors must have identical shapes, "
                f"got {tuple(global_branch.shape)} and {tuple(local_branch.shape)}"
            )

        mask_btD = attention_mask.unsqueeze(-1).to(dtype=global_branch.dtype)
        mask_bDt = attention_mask.unsqueeze(1).to(dtype=global_branch.dtype)

        x_concat = torch.cat([global_branch, local_branch], dim=-1)
        x_concat = x_concat * mask_btD

        x_tmp = x_concat.transpose(1, 2)
        x_tmp = x_tmp * mask_bDt
        x_tmp = self.depthwise(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)

        return self.projection(x_concat + x_tmp)

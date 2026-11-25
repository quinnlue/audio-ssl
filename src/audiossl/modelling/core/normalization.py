"""Normalization helpers for mask-aware feature sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from .initialization import initialize_batch_norm


class MaskedFeatureBatchNorm1d(nn.Module):
    """BatchNorm over feature bins for masked `(B, F, T)` tensors.

    Boolean masks have shape `(B, T)`, where `True` means valid. During
    training, mean and variance are computed only over valid frames; invalid
    output frames are zero-filled.
    """

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True,
        )
        initialize_batch_norm(self.batch_norm)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize `(B, F, T)` features using a `(B, T)` validity mask."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "MaskedFeatureBatchNorm1d.forward expected torch.Tensor, "
                f"got {type(x).__name__}"
            )
        if x.ndim != 3:
            raise ValueError(
                "MaskedFeatureBatchNorm1d.forward expected x shape (B, F, T), "
                f"got {tuple(x.shape)}"
            )
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                "MaskedFeatureBatchNorm1d.forward expected mask torch.Tensor, "
                f"got {type(mask).__name__}"
            )
        if mask.ndim != 2:
            raise ValueError(
                "MaskedFeatureBatchNorm1d.forward expected mask shape (B, T), "
                f"got {tuple(mask.shape)}"
            )
        if mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError(
                "MaskedFeatureBatchNorm1d.forward expected mask to match x batch/time, "
                f"got {tuple(mask.shape)} and {(x.shape[0], x.shape[2])}"
            )
        if mask.dtype != torch.bool:
            raise ValueError(
                "MaskedFeatureBatchNorm1d.forward expected mask dtype torch.bool, "
                f"got {mask.dtype}"
            )

        valid = mask.unsqueeze(1).to(device=x.device, dtype=x.dtype)
        if self.training:
            count = valid.sum()
            if count.item() <= 0:
                raise ValueError(
                    "MaskedFeatureBatchNorm1d.forward expected at least one valid frame, got 0"
                )
            mean = (x * valid).sum(dim=(0, 2)) / count
            centered = (x - mean.view(1, -1, 1)) * valid
            var = centered.square().sum(dim=(0, 2)) / count
            self._update_running_stats(mean, var, count)
        else:
            assert self.batch_norm.running_mean is not None
            assert self.batch_norm.running_var is not None
            mean = self.batch_norm.running_mean.to(device=x.device, dtype=x.dtype)
            var = self.batch_norm.running_var.to(device=x.device, dtype=x.dtype)

        x = (x - mean.view(1, -1, 1)) * torch.rsqrt(var.view(1, -1, 1) + self.batch_norm.eps)
        if self.batch_norm.weight is not None:
            weight = self.batch_norm.weight.to(device=x.device, dtype=x.dtype)
            x = x * weight.view(1, -1, 1)
        if self.batch_norm.bias is not None:
            bias = self.batch_norm.bias.to(device=x.device, dtype=x.dtype)
            x = x + bias.view(1, -1, 1)
        return x.masked_fill(~mask.unsqueeze(1).to(device=x.device), 0.0)

    @torch.no_grad()
    def _update_running_stats(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        count: torch.Tensor,
    ) -> None:
        """Update BatchNorm running statistics from masked batch statistics."""
        num_batches_tracked = self.batch_norm.num_batches_tracked
        if num_batches_tracked is not None:
            num_batches_tracked.add_(1)

        momentum = self.batch_norm.momentum
        if momentum is None:
            assert num_batches_tracked is not None
            momentum = 1.0 / float(num_batches_tracked.item())

        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        assert running_mean is not None
        assert running_var is not None
        running_mean.mul_(1.0 - momentum).add_(mean.to(dtype=running_mean.dtype), alpha=momentum)

        correction = torch.where(
            count > 1.0,
            count / (count - 1.0).clamp_min(1.0),
            torch.ones_like(count),
        )
        running_var_value = var * correction
        running_var.mul_(1.0 - momentum).add_(
            running_var_value.to(dtype=running_var.dtype),
            alpha=momentum,
        )


__all__ = ["MaskedFeatureBatchNorm1d"]

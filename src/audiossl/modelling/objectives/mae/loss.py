"""MAE reconstruction loss helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class MAELossOutput:
    """Scalar MAE loss and per-patch reconstruction diagnostics."""

    loss: torch.Tensor
    per_patch_loss: torch.Tensor


def mae_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    norm_pix_loss: Literal["per_patch", "global", "none"],
    eps: float = 1e-6,
) -> MAELossOutput:
    """fp32 reconstruction loss mirroring dasheng MAELoss.

    Parameters
    ----------
    prediction, target
        Patch reconstructions and targets with shape ``(B, N, patch_t * patch_f)``.
    target_mask
        Objective-specific boolean mask with shape ``(B, N)``. Here ``True``
        means a removed patch contributes to the reconstruction loss.
    norm_pix_loss
        Pixel normalization mode before squared error.
    eps
        Numerical epsilon for normalization.

    Returns
    -------
    MAELossOutput
        Scalar masked reconstruction loss and per-patch losses with shape
        ``(B, N)``.
    """

    if not isinstance(prediction, torch.Tensor):
        raise TypeError(
            "mae_reconstruction_loss expected prediction torch.Tensor, "
            f"got {type(prediction).__name__}"
        )
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            "mae_reconstruction_loss expected target torch.Tensor, "
            f"got {type(target).__name__}"
        )
    if not isinstance(target_mask, torch.Tensor):
        raise TypeError(
            "mae_reconstruction_loss expected target_mask torch.Tensor, "
            f"got {type(target_mask).__name__}"
        )
    if prediction.ndim != 3:
        raise ValueError(
            "mae_reconstruction_loss expected prediction rank-3 [B, N, P], "
            f"got shape {tuple(prediction.shape)}"
        )
    if target.ndim != 3:
        raise ValueError(
            "mae_reconstruction_loss expected target rank-3 [B, N, P], "
            f"got shape {tuple(target.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "mae_reconstruction_loss expected prediction and target to match, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if target_mask.dtype != torch.bool:
        raise ValueError(
            f"mae_reconstruction_loss expected target_mask dtype bool, got {target_mask.dtype}"
        )
    if target_mask.shape != prediction.shape[:2]:
        raise ValueError(
            "mae_reconstruction_loss expected target_mask shape (B, N), "
            f"got {tuple(target_mask.shape)} and prediction has "
            f"{tuple(prediction.shape[:2])}"
        )
    if not bool(target_mask.any().item()):
        raise ValueError("mae_reconstruction_loss expected target_mask to select >= 1 position")
    if norm_pix_loss not in {"per_patch", "global", "none"}:
        raise ValueError(
            "mae_reconstruction_loss expected norm_pix_loss in "
            f"{{'per_patch','global','none'}}, got {norm_pix_loss!r}"
        )

    device_type = "cuda" if prediction.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        prediction_fp32 = prediction.float()
        target_fp32 = target.float()

        if norm_pix_loss == "per_patch":
            mean = target_fp32.mean(dim=-1, keepdim=True)
            var = target_fp32.var(dim=-1, keepdim=True)
            target_fp32 = (target_fp32 - mean) / (var + eps).sqrt()
        elif norm_pix_loss == "global":
            mean = target_fp32.mean()
            var = target_fp32.var()
            target_fp32 = (target_fp32 - mean) / (var + eps).sqrt()

        per_feature = (prediction_fp32 - target_fp32) ** 2
        per_patch = per_feature.mean(dim=-1)
        loss = per_patch[target_mask].mean()

    return MAELossOutput(loss=loss, per_patch_loss=per_patch)


__all__ = ["MAELossOutput", "mae_reconstruction_loss"]

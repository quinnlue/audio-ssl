"""JEPA prediction loss helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JEPALossOutput:
    """Scalar JEPA loss and unreduced per-token diagnostics."""

    loss: torch.Tensor
    per_position_loss: torch.Tensor


def jepa_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    loss_style: Literal["mse", "mae"],
) -> JEPALossOutput:
    """Masked latent-prediction loss for JEPA.

    Parameters
    ----------
    prediction, target
        Latent sequences with shape ``(B, N, D)``.
    target_mask
        Objective-specific boolean mask with shape ``(B, N)``. Here ``True``
        means the position is selected as a JEPA target and contributes to the
        scalar loss.
    loss_style
        Elementwise distance, either ``"mse"`` or ``"mae"``.

    Returns
    -------
    JEPALossOutput
        Scalar loss and per-position loss with shape ``(B, N)``.
    """

    if not isinstance(prediction, torch.Tensor):
        raise TypeError(
            f"prediction must be torch.Tensor, got {type(prediction).__name__}"
        )
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"target must be torch.Tensor, got {type(target).__name__}")
    if not isinstance(target_mask, torch.Tensor):
        raise TypeError(
            f"target_mask must be torch.Tensor, got {type(target_mask).__name__}"
        )
    if prediction.ndim != 3:
        raise ValueError(
            f"prediction must be rank-3 [B, S, D], got shape {tuple(prediction.shape)}"
        )
    if target.ndim != 3:
        raise ValueError(f"target must be rank-3 [B, S, D], got shape {tuple(target.shape)}")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shape, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if target_mask.dtype != torch.bool:
        raise ValueError(f"target_mask must be torch.bool, got {target_mask.dtype}")
    if target_mask.shape != prediction.shape[:2]:
        raise ValueError(
            "target_mask must match prediction batch/time dimensions, "
            f"got {tuple(target_mask.shape)} and {tuple(prediction.shape[:2])}"
        )
    if not bool(target_mask.any().item()):
        raise ValueError("target_mask selects zero positions")
    if loss_style not in {"mse", "mae"}:
        raise ValueError(f"Unsupported JEPA loss_style {loss_style!r}")

    if loss_style == "mse":
        per_feature = F.mse_loss(prediction, target, reduction="none")
    else:
        per_feature = F.l1_loss(prediction, target, reduction="none")
    per_position = per_feature.mean(dim=-1)
    loss = per_position[target_mask].mean()
    return JEPALossOutput(loss=loss, per_position_loss=per_position)


__all__ = ["JEPALossOutput", "jepa_prediction_loss"]

"""BYOL cosine regression loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_pair(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if not isinstance(lhs, torch.Tensor):
        raise TypeError(f"{name}[0] must be torch.Tensor, got {type(lhs).__name__}")
    if not isinstance(rhs, torch.Tensor):
        raise TypeError(f"{name}[1] must be torch.Tensor, got {type(rhs).__name__}")
    if lhs.ndim != 2:
        raise ValueError(f"{name}[0] must be rank-2 (B, D), got shape {tuple(lhs.shape)}")
    if rhs.ndim != 2:
        raise ValueError(f"{name}[1] must be rank-2 (B, D), got shape {tuple(rhs.shape)}")
    if lhs.shape != rhs.shape:
        raise ValueError(
            f"{name} tensors must have identical shape, "
            f"got {tuple(lhs.shape)} and {tuple(rhs.shape)}"
        )


def _regression_direction(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (prediction * target).sum(dim=-1)


def byol_regression_loss(
    prediction_i: torch.Tensor,
    target_j: torch.Tensor,
    prediction_j: torch.Tensor,
    target_i: torch.Tensor,
) -> torch.Tensor:
    """Symmetric BYOL cosine regression loss.

    Parameters
    ----------
    prediction_i, prediction_j
        Student predictor outputs with shape ``(B, projection_dim)``.
    target_i, target_j
        Detached target projector outputs with shape ``(B, projection_dim)``.

    Returns
    -------
    torch.Tensor
        Scalar loss. Inputs are normalized inside the loss so heads do not need
        to own final L2 normalization.
    """

    _validate_pair("prediction_i/target_j", prediction_i, target_j)
    _validate_pair("prediction_j/target_i", prediction_j, target_i)
    if prediction_i.shape != prediction_j.shape:
        raise ValueError(
            "BYOL prediction directions must share shape, "
            f"got {tuple(prediction_i.shape)} and {tuple(prediction_j.shape)}"
        )

    loss_i = _regression_direction(prediction_i, target_j).mean()
    loss_j = _regression_direction(prediction_j, target_i).mean()
    return 0.5 * (loss_i + loss_j)


__all__ = ["byol_regression_loss"]

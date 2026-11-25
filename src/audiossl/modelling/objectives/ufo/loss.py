"""Loss helpers for the EAT UFO objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class UFOLossOutput:
    loss: torch.Tensor
    frame_loss: torch.Tensor
    utterance_loss: torch.Tensor


def ufo_frame_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    removal_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE over masked patch positions."""
    if prediction.shape != target.shape:
        raise ValueError(
            "ufo_frame_loss expected matching prediction/target shapes, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if removal_mask.dtype != torch.bool:
        raise TypeError("ufo_frame_loss expected boolean removal_mask")
    if removal_mask.shape != prediction.shape[:2]:
        raise ValueError(
            "ufo_frame_loss expected removal_mask to match prediction batch/time axes"
        )
    if not removal_mask.any():
        return prediction.new_zeros(())
    per_feature = F.mse_loss(prediction, target, reduction="none")
    per_patch = per_feature.mean(dim=-1)
    return per_patch[removal_mask].mean()


def ufo_utterance_loss(
    student_cls: torch.Tensor,
    teacher_target: torch.Tensor,
) -> torch.Tensor:
    """MSE between student CLS and teacher utterance target."""
    if student_cls.shape != teacher_target.shape:
        raise ValueError(
            "ufo_utterance_loss expected matching CLS shapes, "
            f"got {tuple(student_cls.shape)} and {tuple(teacher_target.shape)}"
        )
    return F.mse_loss(student_cls, teacher_target)


def combine_ufo_loss(
    *,
    frame_loss: torch.Tensor,
    utterance_loss: torch.Tensor,
    utterance_loss_weight: float,
) -> UFOLossOutput:
    total = frame_loss + float(utterance_loss_weight) * utterance_loss
    return UFOLossOutput(
        loss=total,
        frame_loss=frame_loss,
        utterance_loss=utterance_loss,
    )


__all__ = [
    "UFOLossOutput",
    "combine_ufo_loss",
    "ufo_frame_loss",
    "ufo_utterance_loss",
]

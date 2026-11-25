"""DINO cross-view loss helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DINOLossOutput:
    """Scalar DINO loss terms for the two cross-view directions."""

    loss: torch.Tensor
    loss_i_to_j: torch.Tensor
    loss_j_to_i: torch.Tensor


def _validate_logits(name: str, logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(logits).__name__}")
    if logits.ndim != 2:
        raise ValueError(f"{name} must be rank-2 (B, K), got shape {tuple(logits.shape)}")


def _validate_temperature(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return result


def _cross_entropy(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    center: torch.Tensor,
    teacher_temperature: float,
    student_temperature: float,
) -> torch.Tensor:
    teacher_probs = F.softmax(
        (teacher_logits - center.view(1, -1)) / teacher_temperature,
        dim=-1,
    ).detach()
    student_log_probs = F.log_softmax(student_logits / student_temperature, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


def dino_cross_entropy_loss(
    student_logits_i: torch.Tensor,
    student_logits_j: torch.Tensor,
    teacher_logits_i: torch.Tensor,
    teacher_logits_j: torch.Tensor,
    center: torch.Tensor,
    *,
    teacher_temperature: float,
    student_temperature: float,
) -> DINOLossOutput:
    """Two-global-view DINO cross-entropy, excluding same-view pairs.

    Parameters
    ----------
    student_logits_i, student_logits_j
        Student head logits with shape ``(B, num_prototypes)``.
    teacher_logits_i, teacher_logits_j
        Detached teacher logits with shape ``(B, num_prototypes)`` before
        centering.
    center
        Running center with shape ``(num_prototypes,)``.

    Returns
    -------
    DINOLossOutput
        Scalar mean loss and both cross-view direction losses.
    """

    for name, logits in (
        ("student_logits_i", student_logits_i),
        ("student_logits_j", student_logits_j),
        ("teacher_logits_i", teacher_logits_i),
        ("teacher_logits_j", teacher_logits_j),
    ):
        _validate_logits(name, logits)
    if student_logits_i.shape != student_logits_j.shape:
        raise ValueError(
            "student logits must share shape, "
            f"got {tuple(student_logits_i.shape)} and {tuple(student_logits_j.shape)}"
        )
    if teacher_logits_i.shape != teacher_logits_j.shape:
        raise ValueError(
            "teacher logits must share shape, "
            f"got {tuple(teacher_logits_i.shape)} and {tuple(teacher_logits_j.shape)}"
        )
    if student_logits_i.shape != teacher_logits_i.shape:
        raise ValueError(
            "student and teacher logits must share shape, "
            f"got {tuple(student_logits_i.shape)} and {tuple(teacher_logits_i.shape)}"
        )
    if not isinstance(center, torch.Tensor):
        raise TypeError(f"center must be torch.Tensor, got {type(center).__name__}")
    if center.ndim != 1:
        raise ValueError(f"center must be rank-1 (K,), got shape {tuple(center.shape)}")
    if center.shape[0] != student_logits_i.shape[-1]:
        raise ValueError(
            "center size must match logits prototype dimension, "
            f"got {center.shape[0]} and {student_logits_i.shape[-1]}"
        )

    teacher_temperature = _validate_temperature(
        "teacher_temperature",
        teacher_temperature,
    )
    student_temperature = _validate_temperature(
        "student_temperature",
        student_temperature,
    )

    loss_i_to_j = _cross_entropy(
        teacher_logits=teacher_logits_i,
        student_logits=student_logits_j,
        center=center,
        teacher_temperature=teacher_temperature,
        student_temperature=student_temperature,
    )
    loss_j_to_i = _cross_entropy(
        teacher_logits=teacher_logits_j,
        student_logits=student_logits_i,
        center=center,
        teacher_temperature=teacher_temperature,
        student_temperature=student_temperature,
    )
    loss = 0.5 * (loss_i_to_j + loss_j_to_i)
    return DINOLossOutput(loss=loss, loss_i_to_j=loss_i_to_j, loss_j_to_i=loss_j_to_i)


__all__ = ["DINOLossOutput", "dino_cross_entropy_loss"]

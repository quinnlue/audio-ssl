"""InfoNCE loss helpers for contrastive audio objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InfoNCELossOutput:
    """Decomposed InfoNCE loss tensors.

    ``loss`` is the value to backprop. ``positive_loss`` and ``negative_loss`` are
    diagnostic, computed only on the i->j direction so the dataclass carries
    one canonical decomposition even when ``symmetric=True``.

    ``logits`` is the ``(B, B)`` cosine-similarity matrix of ``zi`` versus ``zj``
    divided by
    temperature. It is reused by the objective for ablation stats so the
    matmul is not recomputed.
    """

    loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    logits: torch.Tensor


def _negative_log_sum_exp_off_diagonal(logits: torch.Tensor) -> torch.Tensor:
    """log_sum_exp over the off-diagonal entries of each row.

    Implementation detail: replace diagonal with -inf so it does not contribute,
    then log_sum_exp the row.
    """
    batch_size = logits.shape[0]
    diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(diagonal_mask, float("-inf"))
    return torch.logsumexp(masked, dim=-1)


def contrastive_info_nce_loss(
    zi: torch.Tensor,
    zj: torch.Tensor,
    *,
    temperature: float,
    symmetric: bool,
) -> InfoNCELossOutput:
    """SimCLR-style InfoNCE on two views of a batch.

    Parameters
    ----------
    zi, zj
        Projected view representations with shape ``(B, D)``. Inputs are
        L2-normalized internally.
    temperature
        Positive scalar divisor for cosine-similarity logits.
    symmetric
        When ``True``, average i->j and j->i cross-entropy. When ``False``, use
        only the i->j direction.

    Returns
    -------
    InfoNCELossOutput
        Scalar loss terms plus ``logits`` with shape ``(B, B)``.
    """
    if not isinstance(zi, torch.Tensor):
        raise TypeError(f"zi must be torch.Tensor, got {type(zi).__name__}")
    if not isinstance(zj, torch.Tensor):
        raise TypeError(f"zj must be torch.Tensor, got {type(zj).__name__}")
    if zi.ndim != 2:
        raise ValueError(f"zi must be rank-2 (B, D), got shape {tuple(zi.shape)}")
    if zj.ndim != 2:
        raise ValueError(f"zj must be rank-2 (B, D), got shape {tuple(zj.shape)}")
    if zi.shape != zj.shape:
        raise ValueError(
            "zi and zj must have identical shape, "
            f"got {tuple(zi.shape)} and {tuple(zj.shape)}"
        )
    if zi.shape[0] < 2:
        raise ValueError(
            "contrastive InfoNCE requires batch size >= 2 for in-batch negatives, "
            f"got {zi.shape[0]}"
        )
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError(f"temperature must be a number, got {type(temperature).__name__}")
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    if not isinstance(symmetric, bool):
        raise TypeError(f"symmetric must be bool, got {type(symmetric).__name__}")

    batch_size = zi.shape[0]

    zi_norm = F.normalize(zi, dim=-1)
    zj_norm = F.normalize(zj, dim=-1)

    logits_ij = (zi_norm @ zj_norm.t()) / temperature_value
    targets = torch.arange(batch_size, device=zi.device)

    loss_ij = F.cross_entropy(logits_ij, targets)
    if symmetric:
        loss_ji = F.cross_entropy(logits_ij.t(), targets)
        loss = 0.5 * (loss_ij + loss_ji)
    else:
        loss = loss_ij

    diagonal_logits = logits_ij.diagonal()
    log_partition = torch.logsumexp(logits_ij, dim=-1)
    positive_loss = (-diagonal_logits + log_partition).mean()
    negative_loss = _negative_log_sum_exp_off_diagonal(logits_ij).mean()

    return InfoNCELossOutput(
        loss=loss,
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        logits=logits_ij,
    )


__all__ = ["InfoNCELossOutput", "contrastive_info_nce_loss"]

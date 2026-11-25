"""Base protocol and validation helpers for training-only masking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MaskingStrategy(nn.Module, ABC):
    """Base class for objective-owned training-time masking strategies.

    Masking strategies are invoked by objectives, not by the inference pipeline.
    Implementations compute boolean masks of exactly the requested shape and do
    not change tensor values or shapes themselves.
    """

    @abstractmethod
    def compute_mask(
        self,
        *,
        shape: tuple[int, ...] | torch.Size,
        valid_mask: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Compute a boolean training mask with the requested shape.

        Parameters
        ----------
        shape
            Requested rank-2 mask shape, usually `(B, T)` or `(B, F)`.
        valid_mask
            Optional boolean validity mask with shape matching `shape`. `True`
            means valid and eligible for masking.
        generator
            Optional `torch.Generator` that drives all strategy randomness when
            supplied, so objective-level seeding remains reproducible.

        Returns
        -------
        torch.Tensor
            Boolean mask with exactly `shape`. The meaning of `True` is
            strategy-specific selection, not validity.
        """
        raise NotImplementedError


def validate_mask_inputs(
    *,
    shape: tuple[int, ...] | torch.Size,
    valid_mask: torch.Tensor | None,
) -> tuple[int, int, torch.device]:
    """Validate rank-2 mask inputs and return normalized shape metadata.

    Parameters
    ----------
    shape
        Requested mask shape `(B, T)` or `(B, F)`.
    valid_mask
        Optional boolean validity mask with the same shape as `shape`. `True`
        means valid and eligible for masking.

    Returns
    -------
    batch
        Batch size `B`.
    length
        Axis length `T` or `F`.
    device
        Device on which the output mask should be allocated.
    """
    if len(shape) != 2:
        raise ValueError(f"masking shape must be rank-2, got {tuple(shape)}")
    batch, length = int(shape[0]), int(shape[1])
    if batch <= 0 or length <= 0:
        raise ValueError(f"masking shape entries must be positive, got {tuple(shape)}")
    if valid_mask is not None:
        if valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}")
        if tuple(valid_mask.shape) != (batch, length):
            raise ValueError(
                "valid_mask shape must match requested mask shape, "
                f"got {tuple(valid_mask.shape)} and {(batch, length)}"
            )
        device = valid_mask.device
    else:
        device = torch.device("cpu")
    return batch, length, device

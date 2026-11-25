"""Spectrogram-mask augmentations are not implemented yet."""

from __future__ import annotations

import torch


def time_mask(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "audiotrainer.augment.mask.time_mask is not implemented; "
        "register a typed AugmentationConfig instead of calling this stub"
    )


def frequency_mask(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "audiotrainer.augment.mask.frequency_mask is not implemented; "
        "register a typed AugmentationConfig instead of calling this stub"
    )


def random_mask(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "audiotrainer.augment.mask.random_mask is not implemented; "
        "register a typed AugmentationConfig instead of calling this stub"
    )


__all__ = ["frequency_mask", "random_mask", "time_mask"]

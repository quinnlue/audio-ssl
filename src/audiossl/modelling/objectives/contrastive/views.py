"""View construction helpers for contrastive-style objectives."""

from __future__ import annotations

from typing import Literal

import torch

from audiossl.config.batch_dataclasses import AudioBatch, ContrastiveAudioBatch
from ..protocols import AugmentationPipeline


def _validate_audio_batch(batch: AudioBatch) -> None:
    if not isinstance(batch, AudioBatch):
        raise TypeError(f"batch must be AudioBatch, got {type(batch).__name__}")
    if not isinstance(batch.x, torch.Tensor):
        raise TypeError(f"batch.x must be torch.Tensor, got {type(batch.x).__name__}")
    if not isinstance(batch.mask, torch.Tensor):
        raise TypeError(f"batch.mask must be torch.Tensor, got {type(batch.mask).__name__}")
    if batch.x.ndim != 2:
        raise ValueError(
            f"batch.x must be rank-2 (B, T) waveform tensor, got shape {tuple(batch.x.shape)}"
        )
    if batch.mask.shape != batch.x.shape:
        raise ValueError(
            "batch.mask must match batch.x shape, "
            f"got {tuple(batch.mask.shape)} and {tuple(batch.x.shape)}"
        )
    if batch.mask.dtype != torch.bool:
        raise ValueError(f"batch.mask must be torch.bool, got {batch.mask.dtype}")


def create_contrastive_views(
    batch: AudioBatch,
    *,
    strategy: Literal["augmentation", "dropout"],
    augmentor: AugmentationPipeline | None,
) -> ContrastiveAudioBatch:
    """Produce two AudioBatch views for contrastive training.

    Parameters
    ----------
    batch
        Raw waveform batch with ``x`` and ``mask`` shape ``(B, S)``. The mask is
        boolean and follows the default convention: ``True`` means valid.
    strategy
        ``"augmentation"`` applies ``augmentor`` twice. ``"dropout"`` returns
        two wrappers around the same tensors so stochastic model dropout can
        produce SimCSE-style views downstream.
    augmentor
        Trainer-owned augmentation pipeline. Required for ``"augmentation"``
        and forbidden for ``"dropout"``.

    Returns
    -------
    ContrastiveAudioBatch
        Two internal ``AudioBatch`` views. The public objective boundary remains
        ``AudioBatch`` only.
    """
    _validate_audio_batch(batch)

    if strategy == "augmentation":
        if augmentor is None:
            raise ValueError(
                "view_strategy='augmentation' requires a non-None augmentor"
            )
        view_i = augmentor(batch)
        view_j = augmentor(batch)
        return ContrastiveAudioBatch(batch_i=view_i, batch_j=view_j)

    if strategy == "dropout":
        if augmentor is not None:
            raise ValueError(
                "view_strategy='dropout' forbids passing an augmentor; "
                "the two views must be produced by stochastic model dropout"
            )
        return ContrastiveAudioBatch(batch_i=batch, batch_j=batch)

    raise ValueError(
        f"Unsupported contrastive view strategy {strategy!r}; "
        "expected 'augmentation' or 'dropout'"
    )


__all__ = ["create_contrastive_views"]

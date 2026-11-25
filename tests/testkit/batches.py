"""Deterministic waveform batches.

All randomness comes from a local ``torch.Generator`` so building a batch
never perturbs (or depends on) global RNG state.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from audiossl.config.batch_dataclasses import AudioBatch


def waveforms(lengths: Sequence[int], *, seed: int = 0) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [0.1 * torch.randn(int(n), generator=generator) for n in lengths]


def pad(items: Sequence[torch.Tensor], *, total: int | None = None) -> AudioBatch:
    """Right-pad rank-1 waveforms with zeros into a prefix-valid ``AudioBatch``."""
    width = max(int(item.shape[0]) for item in items) if total is None else total
    x = torch.zeros(len(items), width)
    mask = torch.zeros(len(items), width, dtype=torch.bool)
    for row, item in enumerate(items):
        x[row, : item.shape[0]] = item
        mask[row, : item.shape[0]] = True
    return AudioBatch(x=x, mask=mask)


def waveform_batch(lengths: Sequence[int], *, seed: int = 0) -> AudioBatch:
    return pad(waveforms(lengths, seed=seed))


def row(batch: AudioBatch, index: int) -> AudioBatch:
    """Return row ``index`` as an unpadded single-item batch."""
    length = int(batch.mask[index].sum())
    return AudioBatch(
        x=batch.x[index : index + 1, :length].clone(),
        mask=batch.mask[index : index + 1, :length].clone(),
    )


def with_garbage_padding(
    batch: AudioBatch,
    *,
    extra: int = 0,
    scale: float = 1e3,
    seed: int = 1234,
) -> AudioBatch:
    """Extend padding by ``extra`` samples and fill every invalid sample with noise.

    Valid samples are untouched. Any computation that honours the mask must
    produce identical valid outputs for the original and the returned batch.
    """
    generator = torch.Generator().manual_seed(seed)
    batch_size, width = batch.x.shape
    x = torch.zeros(batch_size, width + extra)
    mask = torch.zeros(batch_size, width + extra, dtype=torch.bool)
    x[:, :width] = batch.x
    mask[:, :width] = batch.mask
    noise = scale * torch.randn(x.shape, generator=generator)
    x = torch.where(mask, x, noise)
    return AudioBatch(x=x, mask=mask)


def clone(batch: AudioBatch) -> AudioBatch:
    return AudioBatch(x=batch.x.clone(), mask=batch.mask.clone())

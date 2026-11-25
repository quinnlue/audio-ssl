"""Config-driven waveform augmentation pipeline."""

from __future__ import annotations

from typing import Protocol

import torch

from audiossl.config import (
    AudioAugmentorConfig,
    AugmentationConfig,
    validate_config,
)
from audiossl.config.batch_dataclasses import AudioBatch

from audiotrainer.registry import AUGMENTATIONS


class _WaveformAugmentation(Protocol):
    def __call__(
        self,
        x: list[torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> list[torch.Tensor]: ...


def _build_augmentation(config: AugmentationConfig) -> _WaveformAugmentation:
    type_name = getattr(config, "type", None)
    if not isinstance(type_name, str):
        raise TypeError(
            "augmentation config must have a string 'type' field, "
            f"got {type(config).__name__}"
        )
    try:
        builder = AUGMENTATIONS[type_name]
    except KeyError as exc:
        raise TypeError(str(exc.args[0])) from None
    return builder(config)


class AugmentationPipeline:
    """Apply an ordered sequence of waveform augmentations to an ``AudioBatch``.

    Returns a new batch on the same device/dtype as the input (padding preserved
    via a device-aware collate). The optional ``generator`` is used for every
    stochastic draw and is discovered by the trainer for checkpointing.
    """

    def __init__(
        self,
        config: AudioAugmentorConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        if not isinstance(config, AudioAugmentorConfig):
            raise TypeError(
                "AugmentationPipeline expected AudioAugmentorConfig, "
                f"got {type(config).__name__}"
            )
        validate_config(config)
        self.config = config
        self.generator = generator
        self.augmentations: list[_WaveformAugmentation] = [
            _build_augmentation(aug) for aug in config.augmentations
        ]

    def __call__(
        self,
        batch: AudioBatch,
        generator: torch.Generator | None = None,
    ) -> AudioBatch:
        if not isinstance(batch, AudioBatch):
            raise TypeError(f"Expected AudioBatch, got {type(batch).__name__}")
        lengths = batch.mask.sum(dim=1)
        waveforms = [
            sample[: int(length.item())]
            for sample, length in zip(batch.x, lengths, strict=True)
        ]
        effective_generator = generator if generator is not None else self.generator
        for augmentation in self.augmentations:
            waveforms = augmentation(waveforms, generator=effective_generator)
        return _collate_on_device(waveforms, device=batch.x.device, dtype=batch.x.dtype)


def _collate_on_device(
    waveforms: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AudioBatch:
    if not waveforms:
        raise ValueError("AugmentationPipeline produced an empty waveform list")
    lengths = torch.tensor(
        [item.shape[0] for item in waveforms],
        device=device,
        dtype=torch.long,
    )
    max_length = int(lengths.max().item())
    x = torch.zeros((len(waveforms), max_length), device=device, dtype=dtype)
    mask = torch.zeros((len(waveforms), max_length), device=device, dtype=torch.bool)
    for index, item in enumerate(waveforms):
        length = int(lengths[index].item())
        x[index, :length] = item.to(device=device, dtype=dtype)
        mask[index, :length] = True
    return AudioBatch(x=x, mask=mask)


AudioAugmentor = AugmentationPipeline

__all__ = ["AugmentationPipeline", "AudioAugmentor"]

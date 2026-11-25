"""Shared base class for waveform processors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from audiossl.config.batch_dataclasses import AudioBatch


class BaseProcessor(nn.Module):
    """Base class for raw-waveform to frame-feature processors.

    Concrete processors are constructed as ``cls(config=processor_config)``,
    accept raw mono waveform batches ``(B, S)`` with a boolean validity mask
    ``(B, S)`` (``True`` means valid), and return frame features ``(B, T, F)``
    with a recomputed ``(B, T)`` mask. ``output_mask`` and ``output_lengths``
    must agree with the mask returned by ``forward``.
    """

    config: Any

    @property
    def output_dim(self) -> int:
        return int(self.config.output_dim)

    def _validate_waveform_batch(self, batch: AudioBatch, *, method: str = "forward") -> None:
        name = f"{type(self).__name__}.{method}"
        if not isinstance(batch, AudioBatch):
            raise TypeError(f"{name} expected AudioBatch, got {type(batch).__name__}")
        if not isinstance(batch.x, torch.Tensor) or batch.x.ndim != 2:
            raise ValueError(
                f"{name} expected batch.x shape [B, S], "
                f"got {tuple(getattr(batch.x, 'shape', ()))}"
            )
        if not isinstance(batch.mask, torch.Tensor) or batch.mask.dtype != torch.bool:
            raise ValueError(
                f"{name} expected boolean batch.mask, "
                f"got {getattr(batch.mask, 'dtype', type(batch.mask).__name__)}"
            )
        if batch.mask.shape != batch.x.shape:
            raise ValueError(
                f"{name} expected batch.mask to match batch.x, "
                f"got {tuple(batch.mask.shape)} and {tuple(batch.x.shape)}"
            )

    def _validate_input_mask(
        self, input_mask: torch.Tensor, *, method: str = "output_mask"
    ) -> None:
        name = f"{type(self).__name__}.{method}"
        if not isinstance(input_mask, torch.Tensor):
            raise TypeError(f"{name} expected torch.Tensor, got {type(input_mask).__name__}")
        if input_mask.ndim != 2:
            raise ValueError(f"{name} expected rank-2 mask, got shape {tuple(input_mask.shape)}")
        if input_mask.dtype != torch.bool:
            raise ValueError(f"{name} expected boolean mask, got {input_mask.dtype}")

    def _validate_input_lengths(
        self, input_lengths: torch.Tensor, *, method: str = "output_lengths"
    ) -> None:
        name = f"{type(self).__name__}.{method}"
        if not isinstance(input_lengths, torch.Tensor):
            raise TypeError(f"{name} expected torch.Tensor, got {type(input_lengths).__name__}")
        if input_lengths.ndim != 1:
            raise ValueError(
                f"{name} expected rank-1 lengths, got shape {tuple(input_lengths.shape)}"
            )

    @staticmethod
    def _require_valid_frames(
        output_mask: torch.Tensor, input_mask: torch.Tensor, owner: str
    ) -> None:
        if bool((output_mask.sum(dim=1) <= 0).any()):
            input_lengths = input_mask.sum(dim=1)
            shortest = int(input_lengths.min().item()) if input_lengths.numel() > 0 else 0
            raise ValueError(
                f"{owner} produces no valid output frames for at least one sample; "
                f"shortest input length is {shortest}"
            )

    def output_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement output_mask")

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not implement output_lengths")

    def reconstruction_target(self, batch: AudioBatch) -> AudioBatch:
        """Return deterministic, gradient-free features used as reconstruction targets.

        Processors that declare the ``"reconstruction_target"`` config
        capability implement this; reconstruction objectives call it instead
        of reaching into processor-specific intermediates.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide reconstruction targets"
        )


__all__ = ["BaseProcessor"]

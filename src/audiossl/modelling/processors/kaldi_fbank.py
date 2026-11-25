"""Kaldi fbank processor matching the EAT Hugging Face feature recipe."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from audiossl.config import KaldiFbankProcessorConfig
from audiossl.config.batch_dataclasses import AudioBatch

from .base import BaseProcessor
from .kaldi_compliance import fbank as kaldi_fbank
from .kaldi_compliance import fbank_batch as kaldi_fbank_batch

_FRAME_LENGTH_MS = 25.0


class KaldiFbankProcessor(BaseProcessor):
    """Convert raw waveform batches to normalized Kaldi fbank features.

    Each row is processed as if it were the only clip: padded samples are
    zero-filled, fbank frames are computed with Kaldi ``snip_edges`` framing,
    and frames beyond a row's valid length are replaced by the zero-padded
    log-mel the EAT reference produces before normalization. Features are then
    padded or truncated to ``target_length`` frames.

    With ``mask_all_frames=True`` every output frame is marked valid (the EAT
    recipe attends over zero-padded frames). Otherwise a frame is valid only
    when its full analysis window lies inside the valid waveform.
    """

    def __init__(self, *, config: KaldiFbankProcessorConfig) -> None:
        super().__init__()
        if not isinstance(config, KaldiFbankProcessorConfig):
            raise TypeError(
                "KaldiFbankProcessor.__init__ expected KaldiFbankProcessorConfig, "
                f"got {type(config).__name__}"
            )
        self.config = config
        self.window_size = int(config.sr * _FRAME_LENGTH_MS / 1000.0)
        self.window_shift = int(config.sr * config.frame_shift_ms / 1000.0)
        if self.window_size <= 0 or self.window_shift <= 0:
            raise ValueError(
                "KaldiFbankProcessor requires positive window size and shift, got "
                f"{self.window_size} and {self.window_shift} samples"
            )

    def _compute_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        return kaldi_fbank(
            waveform.unsqueeze(0),
            sample_frequency=self.config.sr,
            frame_shift=self.config.frame_shift_ms,
            num_mel_bins=self.config.num_mel_bins,
            dither=self.config.dither,
            htk_compat=self.config.htk_compat,
            use_energy=self.config.use_energy,
            window_type=self.config.window_type,
        )

    def _frame_counts(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.to(dtype=torch.long)
        frames = torch.div(lengths - self.window_size, self.window_shift, rounding_mode="floor") + 1
        return frames.clamp(min=0, max=self.config.target_length)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Return valid frame counts for prefix-valid raw lengths."""
        self._validate_input_lengths(input_lengths)
        if self.config.mask_all_frames:
            return torch.full_like(input_lengths, self.config.target_length, dtype=torch.long)
        return self._frame_counts(input_lengths)

    def output_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, target_length)`` frame validity mask for a waveform mask."""
        self._validate_input_mask(input_mask)
        if self.config.mask_all_frames:
            return torch.ones(
                input_mask.shape[0],
                self.config.target_length,
                dtype=torch.bool,
                device=input_mask.device,
            )
        frame_idx = torch.arange(self.config.target_length, device=input_mask.device)
        counts = self._frame_counts(self._prefix_lengths(input_mask))
        return frame_idx.unsqueeze(0) < counts.unsqueeze(1)

    @staticmethod
    def _prefix_lengths(input_mask: torch.Tensor) -> torch.Tensor:
        """Count the leading run of valid samples per row (strict framing)."""
        return input_mask.to(torch.long).cumprod(dim=1).sum(dim=1)

    def _pad_or_truncate_batch(self, mel: torch.Tensor) -> torch.Tensor:
        target = self.config.target_length
        num_frames = mel.shape[1]
        if num_frames < target:
            mel = F.pad(mel, (0, 0, 0, target - num_frames))
        elif num_frames > target:
            mel = mel[:, :target, :]
        return mel

    def _normalize(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.config.norm_mean) / (self.config.norm_std * 2.0)

    def forward(self, batch: AudioBatch) -> AudioBatch:
        self._validate_waveform_batch(batch)

        mask = batch.mask
        lengths = self._prefix_lengths(mask)
        valid_frames = self._frame_counts(lengths)
        if bool((valid_frames <= 0).any()):
            raise ValueError(
                "KaldiFbankProcessor produces no valid output frames for at least one sample; "
                f"shortest input length is {int(lengths.min().item())} samples, the analysis "
                f"window needs {self.window_size}"
            )

        waveform = batch.x.masked_fill(~mask, 0.0)
        if self.config.remove_waveform_mean:
            mask_f = mask.to(waveform.dtype)
            row_mean = (waveform * mask_f).sum(dim=1, keepdim=True) / lengths.clamp(min=1).to(
                waveform.dtype
            ).unsqueeze(1)
            waveform = (waveform - row_mean) * mask_f

        mel = kaldi_fbank_batch(
            waveform,
            sample_frequency=self.config.sr,
            frame_shift=self.config.frame_shift_ms,
            num_mel_bins=self.config.num_mel_bins,
            dither=self.config.dither,
            htk_compat=self.config.htk_compat,
            use_energy=self.config.use_energy,
            window_type=self.config.window_type,
        )  # (B, frames, num_mel)
        frame_idx = torch.arange(mel.shape[1], device=mel.device)
        row_frames = frame_idx.unsqueeze(0) < valid_frames.to(mel.device).unsqueeze(1)
        mel = mel.masked_fill(~row_frames.unsqueeze(-1), 0.0)
        mel = self._pad_or_truncate_batch(mel)
        features = self._normalize(mel)

        return AudioBatch(x=features, mask=self.output_mask(mask).to(features.device))


__all__ = ["KaldiFbankProcessor"]

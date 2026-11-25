"""Mel-spectrogram processor and public mel intermediate APIs."""

import torch
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from audiossl.config import MelSpectrogramProcessorConfig
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.core.normalization import MaskedFeatureBatchNorm1d

from .base import BaseProcessor

_WINDOW_FNS = {
    "hann": torch.hann_window,
}


class MelSpectrogramProcessor(BaseProcessor):
    """Convert raw waveform batches to mel-spectrogram frame features.

    The processor accepts raw mono waveform `AudioBatch` inputs with `x` shape
    `(B, S)` and boolean `mask` shape `(B, S)`, where `True` means valid. It
    zero-fills invalid waveform samples, computes mel features, optionally
    applies amplitude-to-decibel conversion and masked batch normalization, and
    returns features with shape `(B, T, F)` plus a recomputed `(B, T)` mask.

    `config.sr` declares the expected input sample rate and is not
    runtime-validated. This processor emits frame features with
    `config.output_dim`; it does not tokenize, project to `d_model`, apply
    positional encoding, or compute losses.

    Parameters
    ----------
    config
        Frozen mel-spectrogram processor config.
    """

    def __init__(self, *, config: MelSpectrogramProcessorConfig) -> None:
        super().__init__()
        if not isinstance(config, MelSpectrogramProcessorConfig):
            raise TypeError(
                "MelSpectrogramProcessor.__init__ expected MelSpectrogramProcessorConfig, "
                f"got {type(config).__name__}"
            )
        self.config = config
        window_fn = _WINDOW_FNS[self.config.window_fn]
        self.melspec = MelSpectrogram(
            sample_rate=self.config.sr,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            pad=self.config.pad,
            window_fn=window_fn,
            power=self.config.power,
            normalized=self.config.normalized,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            norm=self.config.norm,
            mel_scale=self.config.mel_scale,
        )
        stype = "magnitude" if self.config.power == 1.0 else "power"
        self.amplitude_to_db = (
            AmplitudeToDB(stype=stype, top_db=self.config.log_mel_top_db)
            if self.config.log_mel
            else None
        )
        self.batch_norm = (
            MaskedFeatureBatchNorm1d(
                self.config.n_mels,
                eps=self.config.batch_norm_eps,
                momentum=self.config.batch_norm_momentum,
            )
            if self.config.batch_norm
            else None
        )

    def _apply(self, fn):
        super()._apply(fn)
        self.melspec.to(dtype=torch.float32)
        return self

    def _frame_offset(self) -> int:
        offset = self.config.pad
        if self.config.center:
            offset += self.config.n_fft // 2
        return offset

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute valid mel frame counts for prefix-valid raw lengths.

        Parameters
        ----------
        input_lengths
            Tensor with shape `(B,)` containing per-row valid raw sample counts.

        Returns
        -------
        torch.Tensor
            Tensor with shape `(B,)` containing valid mel frame counts after
            FFT framing, padding, and hop-length rules.
        """
        self._validate_input_lengths(input_lengths)

        lengths = input_lengths.to(dtype=torch.long)
        offset = self._frame_offset()
        first_frame = (offset + self.config.hop_length - 1) // self.config.hop_length
        last_frame = torch.div(
            lengths + offset - self.config.n_fft,
            self.config.hop_length,
            rounding_mode="floor",
        )
        valid_lengths = last_frame - first_frame + 1
        return torch.clamp(valid_lengths, min=0)

    def output_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Recompute mel frame validity from a raw waveform mask.

        Parameters
        ----------
        input_mask
            Boolean raw-sample mask with shape `(B, S)`, where `True` means
            valid.

        Returns
        -------
        torch.Tensor
            Boolean mel-frame mask with shape `(B, T)`. With
            `strict_window`, a frame is valid only when every sample in the FFT
            window is valid. With `overlap_any`, a frame is valid when it
            overlaps any valid input sample.
        """
        self._validate_input_mask(input_mask)

        total_padding = self._frame_offset()
        padded_mask = F.pad(input_mask, (total_padding, total_padding), value=False)
        if padded_mask.shape[1] < self.config.n_fft:
            return torch.zeros(
                input_mask.shape[0],
                0,
                device=input_mask.device,
                dtype=torch.bool,
            )
        frames = padded_mask.unfold(
            dimension=1,
            size=self.config.n_fft,
            step=self.config.hop_length,
        )
        if self.config.frame_validity == "strict_window":
            return frames.all(dim=-1)
        if self.config.frame_validity != "overlap_any":
            raise ValueError(
                "MelSpectrogramProcessorConfig.frame_validity must be "
                f"'strict_window' or 'overlap_any', got {self.config.frame_validity!r}"
            )
        # Any frame that overlaps at least one valid input sample is treated
        # as valid; STFT center padding alone does not invalidate a frame.
        return frames.any(dim=-1)

    def _validate_feature_batch(self, batch: AudioBatch, *, method: str) -> None:
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                f"MelSpectrogramProcessor.{method} expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                f"MelSpectrogramProcessor.{method} expected batch.x shape [B, T, F], "
                f"got {tuple(batch.x.shape)}"
            )
        if batch.x.shape[-1] != self.config.n_mels:
            raise ValueError(
                f"MelSpectrogramProcessor.{method} expected feature dimension "
                f"{self.config.n_mels}, got {batch.x.shape[-1]}"
            )
        if batch.mask.dtype != torch.bool:
            raise ValueError(
                f"MelSpectrogramProcessor.{method} expected boolean batch.mask, "
                f"got {batch.mask.dtype}"
            )
        if batch.mask.shape != batch.x.shape[:2]:
            raise ValueError(
                f"MelSpectrogramProcessor.{method} expected batch.mask shape to match "
                f"batch.x.shape[:2], got {tuple(batch.mask.shape)} and "
                f"{tuple(batch.x.shape[:2])}"
            )

    def compute_melspec(self, batch: AudioBatch) -> AudioBatch:
        """Compute masked mel-power features from raw waveform.

        Parameters
        ----------
        batch
            Raw waveform batch with `x` shape `(B, S)` and boolean `mask` shape
            `(B, S)`, where `True` means valid.

        Returns
        -------
        AudioBatch
            Mel-power feature batch with `x` shape `(B, T, F)` and boolean
            `mask` shape `(B, T)`, where `True` means valid.
        """
        self._validate_waveform_batch(batch, method="compute_melspec")

        melspec_device = next(self.melspec.buffers()).device
        runtime_dtype = batch.x.dtype
        mask = batch.mask.to(device=melspec_device)
        output_mask = self.output_mask(mask)
        self._require_valid_frames(
            output_mask, mask, "MelSpectrogramProcessor configured spectrogram"
        )

        waveform = batch.x.to(device=melspec_device, dtype=self.melspec.mel_scale.fb.dtype)
        waveform = waveform.masked_fill(~mask, 0.0)
        x = self.melspec(waveform)
        x = x.transpose(1, 2).to(dtype=runtime_dtype)
        return AudioBatch(
            x=x,
            mask=output_mask,
        )

    def compute_amplitude_to_db(self, batch: AudioBatch) -> AudioBatch:
        """Apply optional amplitude-to-decibel conversion to mel features.

        Parameters
        ----------
        batch
            Mel feature batch with `x` shape `(B, T, F)` and boolean `mask`
            shape `(B, T)`, where `True` means valid.

        Returns
        -------
        AudioBatch
            Feature batch with the same shapes. The input batch is returned
            unchanged when log-mel conversion is disabled.
        """
        self._validate_feature_batch(batch, method="compute_amplitude_to_db")
        if self.amplitude_to_db is not None:
            x = batch.x.transpose(1, 2)
            x = self.amplitude_to_db(x)
            x = x.transpose(1, 2)
            return AudioBatch(x=x, mask=batch.mask)
        return batch

    def compute_batch_norm(self, batch: AudioBatch) -> AudioBatch:
        """Apply optional masked batch normalization to mel features.

        Parameters
        ----------
        batch
            Mel feature batch with `x` shape `(B, T, F)` and boolean `mask`
            shape `(B, T)`, where `True` means valid.

        Returns
        -------
        AudioBatch
            Feature batch with the same shapes. The input batch is returned
            unchanged when batch normalization is disabled.
        """
        self._validate_feature_batch(batch, method="compute_batch_norm")
        if self.batch_norm is not None:
            x = batch.x.transpose(1, 2)
            x = self.batch_norm(x, batch.mask)
            x = x.transpose(1, 2)
            return AudioBatch(x=x, mask=batch.mask)
        return batch

    def reconstruction_target(self, batch: AudioBatch) -> AudioBatch:
        """Return detached (log-)mel features before masked batch normalization.

        Batch normalization is excluded so targets do not depend on running
        statistics or batch composition.
        """
        with torch.no_grad():
            return self.compute_amplitude_to_db(self.compute_melspec(batch))

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Run the full mel waveform-to-feature pipeline.

        Parameters
        ----------
        batch
            Raw waveform batch with `x` shape `(B, S)` and boolean `mask` shape
            `(B, S)`, where `True` means valid.

        Returns
        -------
        AudioBatch
            Frame feature batch with `x` shape `(B, T, F)` and boolean `mask`
            shape `(B, T)`, where `True` means valid.
        """
        batch = self.compute_melspec(batch)
        batch = self.compute_amplitude_to_db(batch)
        batch = self.compute_batch_norm(batch)
        return batch

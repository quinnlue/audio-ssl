"""Shared EAT mel preprocessing for parity tests and scripts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from audiossl.modelling.processors.kaldi_compliance import fbank as kaldi_fbank

_EAT_NORM_MEAN = -4.268
_EAT_NORM_STD = 4.569


def preprocess_eat_mel(
    waveform: torch.Tensor,
    *,
    target_length: int = 1024,
    sample_rate: int = 16000,
    remove_waveform_mean: bool = True,
) -> torch.Tensor:
    """Return normalized mel with shape ``(1, 1, T, F)`` for HF ``extract_features``."""
    if remove_waveform_mean:
        waveform = waveform - waveform.mean()
    mel = kaldi_fbank(
        waveform.unsqueeze(0),
        sample_frequency=sample_rate,
        frame_shift=10,
        num_mel_bins=128,
        dither=0.0,
        htk_compat=True,
        use_energy=False,
        window_type="hanning",
    )
    if mel.shape[0] < target_length:
        mel = F.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
    else:
        mel = mel[:target_length, :]
    mel = (mel - _EAT_NORM_MEAN) / (_EAT_NORM_STD * 2.0)
    return mel.unsqueeze(0).unsqueeze(0)


__all__ = ["preprocess_eat_mel"]

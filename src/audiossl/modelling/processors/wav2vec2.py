"""wav2vec 2.0-style convolutional feature extractor processor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import Wav2Vec2FeatureExtractorProcessorConfig
from audiossl.config.batch_dataclasses import AudioBatch

from ..core.initialization import initialize_conv1d
from ..registry import build_activation
from .base import BaseProcessor


class _TransposeLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-2, -1)


class Wav2Vec2FeatureExtractorProcessor(BaseProcessor):
    """Convert raw waveform batches to wav2vec 2.0 convolutional features.

    This processor mirrors the wav2vec 2.0 feature-extractor stack: raw mono
    waveform input `x` has shape `(B, S)`, the input mask has shape `(B, S)`,
    and `True` means valid. Invalid waveform samples are zero-filled before the
    convolutional stack, and the returned `AudioBatch` contains frame features
    `(B, T, F)` with a recomputed boolean mask `(B, T)`.

    The config `sr` declares the expected input sample rate but is not
    runtime-validated. The processor emits native frame features with
    `config.output_dim`; tokenization, projection to `d_model`, positional
    encoding, and loss computation are outside this component.

    Parameters
    ----------
    config
        Frozen wav2vec 2.0 feature extractor processor config.
    """

    def __init__(self, *, config: Wav2Vec2FeatureExtractorProcessorConfig) -> None:
        super().__init__()
        if not isinstance(config, Wav2Vec2FeatureExtractorProcessorConfig):
            raise TypeError(
                "Wav2Vec2FeatureExtractorProcessor.__init__ expected "
                "Wav2Vec2FeatureExtractorProcessorConfig, "
                f"got {type(config).__name__}"
            )

        self.config = config

        blocks: list[nn.Module] = []
        in_channels = 1
        for index, (out_channels, kernel_size, stride) in enumerate(
            zip(config.conv_dim, config.conv_kernel, config.conv_stride, strict=True)
        ):
            layers: list[nn.Module] = []
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=config.conv_bias,
            )
            initialize_conv1d(conv, config.initialization)
            layers.append(conv)
            if config.extractor_mode == "default":
                if index == 0:
                    layers.append(
                        nn.GroupNorm(
                            num_groups=config.group_norm_num_groups,
                            num_channels=out_channels,
                            affine=True,
                        )
                    )
            elif config.extractor_mode == "layer_norm":
                layers.append(_TransposeLast())
                layers.append(nn.LayerNorm(out_channels))
                layers.append(_TransposeLast())
            else:
                raise ValueError(
                    "Wav2Vec2FeatureExtractorProcessor unsupported extractor_mode "
                    f"{config.extractor_mode!r}"
                )
            layers.append(build_activation(config.activation))
            blocks.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*blocks)

    def output_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Recompute strict frame validity from a raw waveform mask.

        Parameters
        ----------
        input_mask
            Boolean raw-sample mask with shape `(B, S)`, where `True` means
            valid.

        Returns
        -------
        torch.Tensor
            Boolean frame mask with shape `(B, T)`. A frame is valid only when
            every raw sample in its convolutional receptive field is valid.
        """
        self._validate_input_mask(input_mask)

        mask = input_mask.to(dtype=torch.float32)
        for kernel_size, stride in zip(
            self.config.conv_kernel,
            self.config.conv_stride,
            strict=True,
        ):
            if mask.shape[-1] < kernel_size:
                return mask.new_zeros((mask.shape[0], 0), dtype=torch.bool)
            kernel = torch.ones(
                1,
                1,
                kernel_size,
                device=mask.device,
                dtype=mask.dtype,
            )
            conv_sum = F.conv1d(mask.unsqueeze(1), kernel, stride=stride)
            mask = (conv_sum == float(kernel_size)).to(dtype=torch.float32).squeeze(1)
        return mask.to(dtype=torch.bool)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute valid output frame counts for prefix-valid raw lengths.

        Parameters
        ----------
        input_lengths
            Tensor with shape `(B,)` containing per-row valid raw sample counts.

        Returns
        -------
        torch.Tensor
            Tensor with shape `(B,)` containing valid frame counts after the
            feature extractor stack.
        """
        self._validate_input_lengths(input_lengths)

        lengths = input_lengths.to(dtype=torch.long)
        for kernel_size, stride in zip(
            self.config.conv_kernel,
            self.config.conv_stride,
            strict=True,
        ):
            lengths = torch.clamp(
                torch.div(lengths - kernel_size, stride, rounding_mode="floor") + 1,
                min=0,
            )
        return lengths

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Run wav2vec 2.0-style waveform-to-feature processing.

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
        self._validate_waveform_batch(batch)

        runtime_device = next(self.conv_layers.parameters()).device
        runtime_dtype = next(self.conv_layers.parameters()).dtype

        mask = batch.mask.to(device=runtime_device)
        output_mask = self.output_mask(mask)
        self._require_valid_frames(
            output_mask, mask, "Wav2Vec2FeatureExtractorProcessor configured conv stack"
        )

        waveform = batch.x.to(device=runtime_device, dtype=runtime_dtype)
        waveform = waveform.masked_fill(~mask, 0.0).unsqueeze(1)
        x = self.conv_layers(waveform).transpose(1, 2)

        return AudioBatch(x=x, mask=output_mask)


__all__ = ["Wav2Vec2FeatureExtractorProcessor"]

"""Convolutional raw-waveform processor implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import ConvolutionalProcessorConfig
from audiossl.config.batch_dataclasses import AudioBatch
from ..core.initialization import initialize_conv1d
from ..registry import build_activation
from .base import BaseProcessor


class ConvolutionalProcessor(BaseProcessor):
    """Convert raw waveform batches to learned convolutional frame features.

    The processor accepts mono waveform `AudioBatch` inputs with `x` shape
    `(B, S)` and boolean `mask` shape `(B, S)`, where `True` means valid. It
    zero-fills invalid waveform samples, applies the configured Conv1d stack,
    and returns frame features with shape `(B, T, F)` plus a recomputed
    `(B, T)` validity mask.

    `config.sr`, when present on the config variant, is a declaration of the
    expected input sample rate and is not runtime-validated. The output feature
    width is `config.output_dim`; this processor does not tokenize, project to
    `d_model`, apply positional encoding, or compute losses.

    Parameters
    ----------
    config
        Frozen convolutional processor config.
    """

    def __init__(self, *, config: ConvolutionalProcessorConfig) -> None:
        super().__init__()
        if not isinstance(config, ConvolutionalProcessorConfig):
            raise TypeError(
                "ConvolutionalProcessor.__init__ expected ConvolutionalProcessorConfig, "
                f"got {type(config).__name__}"
            )
        if config.in_channels != 1:
            raise ValueError(
                "ConvolutionalProcessor currently supports raw mono waveforms only; "
                f"expected in_channels=1, got {config.in_channels}"
            )

        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.in_channels
        for kernel_size, stride in zip(config.kernel_sizes, config.strides, strict=True):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=config.padding,
                dilation=config.dilation,
                groups=config.groups,
                bias=config.bias,
            )
            initialize_conv1d(conv, config.initialization)
            layers.append(conv)
            layers.append(build_activation(config.activation))
            in_channels = config.out_channels

        self.conv_layers = nn.Sequential(*layers)

    def output_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Recompute frame validity from a raw waveform mask.

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
            self.config.kernel_sizes,
            self.config.strides,
            strict=True,
        ):
            kernel = torch.ones(
                1,
                1,
                kernel_size,
                device=mask.device,
                dtype=mask.dtype,
            )
            conv_sum = F.conv1d(
                mask.unsqueeze(1),
                kernel,
                stride=stride,
                padding=self.config.padding,
                dilation=self.config.dilation,
            )
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
            convolutional stack.
        """
        self._validate_input_lengths(input_lengths)

        lengths = input_lengths.to(dtype=torch.long)
        for kernel_size, stride in zip(
            self.config.kernel_sizes,
            self.config.strides,
            strict=True,
        ):
            receptive_width = self.config.dilation * (kernel_size - 1) + 1
            lengths = torch.clamp(
                torch.div(
                    lengths - receptive_width,
                    stride,
                    rounding_mode="floor",
                )
                + 1,
                min=0,
            )
        return lengths

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Run waveform-to-feature processing.

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
            output_mask, mask, "ConvolutionalProcessor configured conv stack"
        )

        waveform = batch.x.to(device=runtime_device, dtype=runtime_dtype)
        waveform = waveform.masked_fill(~mask, 0.0).unsqueeze(1)

        x = self.conv_layers(waveform).transpose(1, 2)

        return AudioBatch(x=x, mask=output_mask)

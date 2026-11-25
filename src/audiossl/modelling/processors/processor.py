"""Processor dispatcher for raw-waveform feature extraction."""

import torch.nn as nn

from audiossl.config import AudioModelConfig
from ..registry import get_processor_registry
from audiossl.config.batch_dataclasses import AudioBatch


class Processor(nn.Module):
    """Dispatch an audio model config to its concrete processor.

    Processors are the first stage of the modelling pipeline. They accept raw
    mono waveform batches with shape `(B, S)` and boolean masks `(B, S)`, where
    `True` means valid, and return frame-level feature batches with shape
    `(B, T, F)` and masks `(B, T)`.

    The processor config declares the expected sample rate, but sample rate is
    not runtime-validated because `AudioBatch` carries no sample-rate field.
    Tokenization, projection to `d_model`, positional encoding, and losses
    belong to later pipeline stages.

    Parameters
    ----------
    config
        Audio model config containing the processor variant. Concrete
        processors are constructed from their own config only.
    """

    def __init__(self, config: AudioModelConfig) -> None:
        super().__init__()
        self.config = config
        self.processor = get_processor_registry()[config.processor.type](config=config.processor)

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Map raw waveform `(B, S)` to processor features `(B, T, F)`."""
        return self.processor(batch)

    def output_mask(self, input_mask):
        return self.processor.output_mask(input_mask)

    def output_lengths(self, input_lengths):
        return self.processor.output_lengths(input_lengths)

    def reconstruction_target(self, batch: AudioBatch) -> AudioBatch:
        """Delegate reconstruction-target features to the concrete processor."""
        if not hasattr(self.processor, "reconstruction_target"):
            raise TypeError(
                f"{type(self.processor).__name__} does not provide reconstruction targets"
            )
        return self.processor.reconstruction_target(batch)

    def compute_melspec(self, batch: AudioBatch) -> AudioBatch:
        """Delegate mel-power feature extraction for mel processor variants."""
        if not hasattr(self.processor, "compute_melspec"):
            raise TypeError(
                f"{type(self.processor).__name__} does not expose compute_melspec"
            )
        return self.processor.compute_melspec(batch)

    def compute_amplitude_to_db(self, batch: AudioBatch) -> AudioBatch:
        """Delegate optional amplitude-to-decibel conversion for mel features."""
        if not hasattr(self.processor, "compute_amplitude_to_db"):
            raise TypeError(
                f"{type(self.processor).__name__} does not expose compute_amplitude_to_db"
            )
        return self.processor.compute_amplitude_to_db(batch)

    def compute_batch_norm(self, batch: AudioBatch) -> AudioBatch:
        """Delegate optional masked batch normalization for mel features."""
        if not hasattr(self.processor, "compute_batch_norm"):
            raise TypeError(
                f"{type(self.processor).__name__} does not expose compute_batch_norm"
            )
        return self.processor.compute_batch_norm(batch)

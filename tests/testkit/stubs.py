"""Stand-ins for trainer- and hub-owned collaborators.

They implement the objective-side protocols (``AugmentationPipeline``, HuBERT
teacher) with explicit, seedable randomness so objective tests stay hermetic
and decoupled from ``audiotrainer`` and network downloads.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from audiossl.config.batch_dataclasses import AudioBatch


class SeededGainAugmentor:
    """Random per-row gain plus small additive noise, driven by its own generator.

    Satisfies ``audiossl.modelling.objectives.protocols.AugmentationPipeline``.
    Padded samples stay zero so view masks remain truthful.
    """

    def __init__(self, seed: int = 0, *, min_gain: float = 0.5, max_gain: float = 1.5) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.calls = 0

    def __call__(self, batch: AudioBatch, generator: torch.Generator | None = None) -> AudioBatch:
        self.calls += 1
        gen = generator or self.generator
        batch_size = batch.x.shape[0]
        gain = self.min_gain + (self.max_gain - self.min_gain) * torch.rand(
            batch_size, 1, generator=gen
        )
        noise = 0.01 * torch.randn(batch.x.shape, generator=gen)
        x = (batch.x * gain.to(batch.x) + noise.to(batch.x)).masked_fill(~batch.mask, 0.0)
        return AudioBatch(x=x, mask=batch.mask.clone())


class FrameAlignedStubTeacher(nn.Module):
    """Deterministic HuBERT teacher emitting one label per student frame.

    Labels depend on frame position and on the waveform energy in that frame,
    so they are input-dependent but reproducible. ``frame_mask_fn`` must be the
    student processor's ``output_mask`` so teacher and student agree on frame
    rate by construction.
    """

    def __init__(
        self,
        frame_mask_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        num_codes: int,
    ) -> None:
        super().__init__()
        self.frame_mask_fn = frame_mask_fn
        self.num_codes = num_codes
        # A parameter (deterministically initialized) so frozen-teacher checks
        # have something to inspect without consuming global RNG.
        self.frozen_scale = nn.Parameter(torch.ones(1))

    def forward(self, batch: AudioBatch) -> torch.Tensor:
        frames = self.frame_mask_fn(batch.mask).shape[1]
        width = batch.x.shape[1]
        hop = max(1, width // max(frames, 1))
        chunks = batch.x[:, : hop * frames].reshape(batch.x.shape[0], frames, hop)
        energy_bin = (chunks.abs().mean(dim=-1) * 37.0).floor().long()
        position = torch.arange(frames, device=batch.x.device).unsqueeze(0)
        return (position + energy_bin).remainder(self.num_codes)

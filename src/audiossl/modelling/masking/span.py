"""Contiguous span masking for objective-owned training masking."""

from __future__ import annotations

import math

import torch

from audiossl.config import SpanMaskingConfig
from audiossl.rng import generator_device

from ._base import MaskingStrategy, validate_mask_inputs


class SpanMaskingStrategy(MaskingStrategy):
    """Sample contiguous masks along one rank-2 axis.

    The strategy is a training-only module invoked by objectives. It computes a
    boolean mask with exactly the requested shape and leaves tensor values and
    shapes unchanged.

    Parameters
    ----------
    config
        Span masking configuration. `mask_prob` controls the expected covered
        fraction, `mask_length` controls the nominal span length, and
        `mask_selection` selects the length distribution.
    """

    def __init__(self, *, config: SpanMaskingConfig) -> None:
        """Initialize the span masking strategy from its configuration."""
        super().__init__()
        self.config = config

    def compute_mask(
        self,
        *,
        shape: tuple[int, ...] | torch.Size,
        valid_mask: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Compute a boolean span mask.

        Parameters
        ----------
        shape
            Requested rank-2 mask shape, usually `(B, T)` for time masks or
            `(B, F)` for channel masks.
        valid_mask
            Optional boolean validity mask with shape matching `shape`. `True`
            means valid and eligible for masking.
        generator
            Optional `torch.Generator` used for all random span counts, lengths,
            and start positions when supplied.

        Returns
        -------
        torch.Tensor
            Boolean mask with exactly `shape`; `True` marks selected span
            positions.
        """
        batch, length, device = validate_mask_inputs(shape=shape, valid_mask=valid_mask)
        mask = torch.zeros((batch, length), dtype=torch.bool, device=device)
        if self.config.mask_prob <= 0.0:
            return mask

        for batch_index in range(batch):
            valid = (
                valid_mask[batch_index]
                if valid_mask is not None
                else torch.ones(length, dtype=torch.bool, device=device)
            )
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            valid_len = int(valid_indices.numel())
            if valid_len == 0:
                continue
            span_lengths = self._sample_lengths(
                valid_len,
                generator=generator,
                device=generator_device(generator),
            )
            if span_lengths.numel() == 0:
                continue
            occupied = torch.zeros(valid_len, dtype=torch.bool, device=device)
            for span_len in span_lengths.tolist():
                span_len = int(max(1, min(span_len, valid_len)))
                starts = torch.arange(0, valid_len - span_len + 1, device=device)
                if starts.numel() == 0:
                    continue
                if self.config.no_overlap:
                    keep = []
                    for start in starts.tolist():
                        left = max(0, start - self.config.min_space)
                        right = min(valid_len, start + span_len + self.config.min_space)
                        if not occupied[left:right].any():
                            keep.append(start)
                    if not keep:
                        continue
                    starts = torch.tensor(keep, device=device)
                rng_device = generator_device(generator)
                choice = starts[
                    torch.randint(starts.numel(), (1,), generator=generator, device=rng_device)
                ].item()
                occupied[choice : choice + span_len] = True
            mask[batch_index, valid_indices[occupied]] = True
        return mask

    def _sample_lengths(
        self,
        valid_len: int,
        *,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample span lengths for one batch row using the configured distribution."""
        expected = self.config.mask_prob * valid_len / max(self.config.mask_length, 1)
        num_spans = max(
            1,
            int(
                math.floor(
                    expected + torch.rand((), device=device, generator=generator).item()
                )
            ),
        )
        if self.config.mask_selection == "static":
            return torch.full((num_spans,), self.config.mask_length, dtype=torch.long)
        if self.config.mask_selection == "uniform":
            high = max(int(self.config.mask_other), self.config.mask_length)
            low = min(int(self.config.mask_other), self.config.mask_length)
            return torch.randint(
                low, high + 1, (num_spans,), device=device, generator=generator
            )
        if self.config.mask_selection == "normal":
            values = torch.normal(
                mean=float(self.config.mask_length),
                std=max(float(self.config.mask_other), 1e-6),
                size=(num_spans,),
                device=device,
                generator=generator,
            )
            return values.round().clamp_min(1).long()
        if self.config.mask_selection == "poisson":
            return torch.poisson(
                torch.full((num_spans,), float(self.config.mask_length), device=device),
                generator=generator,
            ).clamp_min(1).long()
        raise ValueError(f"unsupported mask_selection {self.config.mask_selection!r}")


__all__ = ["SpanMaskingStrategy"]

"""JEPA context and target mask sampling helpers."""

from __future__ import annotations

import math

import torch

from audiossl.config import TemporalMaskingConfig
from audiossl.rng import generator_device


def _validate_valid_mask(valid_mask: torch.Tensor) -> None:
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError(f"valid_mask must be torch.Tensor, got {type(valid_mask).__name__}")
    if valid_mask.ndim != 2:
        raise ValueError(
            f"valid_mask must be rank-2 [B, S], got shape {tuple(valid_mask.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be torch.bool, got {valid_mask.dtype}")


def _validate_masking_config(config: TemporalMaskingConfig) -> None:
    if not isinstance(config, TemporalMaskingConfig):
        raise TypeError(
            f"config must be TemporalMaskingConfig, got {type(config).__name__}"
        )
    if (
        isinstance(config.mask_ratio, bool)
        or not isinstance(config.mask_ratio, int | float)
        or not math.isfinite(float(config.mask_ratio))
        or not 0.0 < float(config.mask_ratio) < 1.0
    ):
        raise ValueError(
            "TemporalMaskingConfig.mask_ratio must be finite and in (0, 1), "
            f"got {config.mask_ratio!r}"
        )
    for name, value in (
        ("min_span_len", config.min_span_len),
        ("max_span_len", config.max_span_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"TemporalMaskingConfig.{name} must be int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"TemporalMaskingConfig.{name} must be >= 1, got {value}")
    if config.max_span_len < config.min_span_len:
        raise ValueError(
            "TemporalMaskingConfig.max_span_len must be >= min_span_len, "
            f"got {config.max_span_len} and {config.min_span_len}"
        )
    if not isinstance(config.allow_overlap, bool):
        raise TypeError(
            "TemporalMaskingConfig.allow_overlap must be bool, "
            f"got {type(config.allow_overlap).__name__}"
        )


def _sample_index(num_items: int, generator: torch.Generator | None) -> int:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    device = generator_device(generator)
    return int(torch.randint(num_items, (1,), generator=generator, device=device).item())


def _enumerate_fully_valid_spans(
    *,
    valid_prefix: torch.Tensor,
    seq_len: int,
    min_span_len: int,
    max_span_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (starts, lengths) for every (start, length) whose span is fully valid.

    Order: ascending by length, then ascending by start. This matches the
    original scalar enumeration order so that, given an identical RNG draw of
    `torch.randint(num_candidates, ...)`, the chosen span is unchanged.
    """

    starts_chunks: list[torch.Tensor] = []
    lens_chunks: list[torch.Tensor] = []
    for span_len in range(min_span_len, max_span_len + 1):
        if span_len > seq_len:
            break
        starts = torch.arange(0, seq_len - span_len + 1, dtype=torch.long)
        stops = starts + span_len
        span_valid_count = valid_prefix[stops] - valid_prefix[starts]
        keep = span_valid_count == span_len
        if not bool(keep.any()):
            continue
        kept_starts = starts[keep]
        starts_chunks.append(kept_starts)
        lens_chunks.append(torch.full_like(kept_starts, span_len))
    if not starts_chunks:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    return torch.cat(starts_chunks), torch.cat(lens_chunks)


def create_jepa_target_mask(
    valid_mask: torch.Tensor,
    *,
    config: TemporalMaskingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a token-resolution JEPA target mask.

    Parameters
    ----------
    valid_mask
        Encoder-token validity mask with shape ``(B, N)``. This follows the
        default convention: ``True`` means valid.
    config
        Temporal masking policy controlling target coverage and span lengths.
    generator
        Optional RNG supplied by the trainer for reproducible span choices.

    Returns
    -------
    torch.Tensor
        Boolean target mask with shape ``(B, N)``. This is an objective-specific
        mask: ``True`` means selected for JEPA prediction, not merely valid.
        Padding is never selected, and each sample retains at least one context
        token.

    The mask is small and index-only, so this function moves `valid_mask`
    to CPU once and does all span enumeration / selection in vectorized
    prefix-sum lookups. This avoids per-candidate `.item()` GPU-CPU syncs
    that would otherwise dominate step time.
    """

    _validate_valid_mask(valid_mask)
    _validate_masking_config(config)

    batch_size, seq_len = valid_mask.shape
    if batch_size == 0 or seq_len == 0:
        raise ValueError("valid_mask must contain at least one sample and one timestep")

    original_device = valid_mask.device
    cpu_valid_mask = valid_mask.detach().to(device="cpu", copy=False)
    target_mask = torch.zeros_like(cpu_valid_mask)

    for batch_index in range(batch_size):
        valid_row = cpu_valid_mask[batch_index]
        valid_count = int(valid_row.sum().item())
        if valid_count < 2:
            raise ValueError(
                "JEPA masking requires at least two valid tokens per sample "
                "so one target and one context token can remain"
            )

        max_span_len = min(config.max_span_len, valid_count - 1)
        if config.min_span_len > max_span_len:
            raise ValueError(
                "TemporalMaskingConfig.min_span_len is too large for a sample with "
                f"{valid_count} valid tokens; need at least one context token"
            )

        target_budget = int(round(float(config.mask_ratio) * valid_count))
        target_budget = max(1, min(target_budget, valid_count - 1))
        selected_row = target_mask[batch_index]

        # valid_prefix[i] = count of valid positions in valid_row[:i]; padded with a leading 0
        # so any span sum is a single subtraction `valid_prefix[stop] - valid_prefix[start]`.
        valid_prefix = torch.zeros(seq_len + 1, dtype=torch.long)
        torch.cumsum(valid_row.long(), dim=0, out=valid_prefix[1:])

        candidate_starts, candidate_lens = _enumerate_fully_valid_spans(
            valid_prefix=valid_prefix,
            seq_len=seq_len,
            min_span_len=config.min_span_len,
            max_span_len=max_span_len,
        )
        candidate_stops = candidate_starts + candidate_lens

        selected_sum = 0
        while selected_sum < target_budget:
            remaining_context_budget = valid_count - 1 - selected_sum

            if candidate_starts.numel() == 0:
                break

            # Selected prefix sums change after each pick; recompute once per while-iteration only.
            selected_prefix = torch.zeros(seq_len + 1, dtype=torch.long)
            torch.cumsum(selected_row.long(), dim=0, out=selected_prefix[1:])
            overlap_counts = selected_prefix[candidate_stops] - selected_prefix[candidate_starts]
            new_targets = candidate_lens - overlap_counts

            keep = (new_targets >= 1) & (new_targets <= remaining_context_budget)
            if not config.allow_overlap:
                keep &= overlap_counts == 0

            candidate_indices = torch.nonzero(keep, as_tuple=False).flatten()
            num_candidates = int(candidate_indices.numel())
            if num_candidates == 0:
                break

            choice = _sample_index(num_candidates, generator)
            chosen = int(candidate_indices[choice].item())
            start = int(candidate_starts[chosen].item())
            span_len = int(candidate_lens[chosen].item())
            selected_row[start : start + span_len] |= valid_row[start : start + span_len]
            selected_sum += int(new_targets[chosen].item())

        if not bool(selected_row.any().item()):
            raise ValueError("JEPA masking selected zero target tokens for a sample")
        context_row = valid_row & ~selected_row
        if not bool(context_row.any().item()):
            raise ValueError("JEPA masking left zero context tokens for a sample")

    return target_mask.to(original_device, copy=False)


__all__ = ["create_jepa_target_mask"]

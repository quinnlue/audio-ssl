"""Inverse block masking helpers for EAT UFO pre-training."""

from __future__ import annotations

import math

import torch


def num_visible_tokens(*, time_patches: int, frequency_patches: int, mask_ratio: float) -> int:
    """Fixed number of visible (kept) patch tokens per row for ``mask_ratio``."""
    num_tokens = time_patches * frequency_patches
    return max(1, int(round(num_tokens * (1.0 - float(mask_ratio)))))


def inverse_block_masking(
    *,
    batch_size: int,
    time_patches: int,
    frequency_patches: int,
    mask_ratio: float,
    block_size: tuple[int, int],
    device: torch.device,
    generator: torch.Generator | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample inverse block masks over a 2-D patch grid — fully vectorized.

    Every row keeps exactly ``num_visible_tokens(...)`` tokens (a rectangular
    student batch, so no per-row padding downstream). Visible tokens form
    spatially-contiguous blocks: random ``block_size`` blocks are placed in a
    random order and the cells revealed by the earliest blocks are kept until
    the fixed visible budget is filled. Implemented with batched ``randint`` +
    ``scatter_reduce`` + ``argsort`` — no Python per-row loop and no
    ``.item()`` device→host syncs (the previous implementation issued tens of
    thousands of syncs per step).

    With a boolean ``(B, N)`` ``valid_mask``, padding cells rank after every
    valid cell, so they are kept only when a row has fewer valid cells than
    the visible budget. Callers must still mask kept padding for the encoder.

    Returns
    -------
    visible_mask : (B, N) bool
        ``True`` where a token is kept for the student encoder.
    removal_mask : (B, N) bool
        ``True`` where a token is masked (included in the frame loss).
    ids_restore : (B, N) long
        Permutation restoring shuffled (kept-then-masked) tokens to patch order.
    ids_keep : (B, K) long
        Indices of the kept tokens in original patch order (for a single gather).
    """
    if not (0.0 < float(mask_ratio) < 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio!r}")

    num_tokens = time_patches * frequency_patches
    target_visible = num_visible_tokens(
        time_patches=time_patches,
        frequency_patches=frequency_patches,
        mask_ratio=mask_ratio,
    )
    block_t = min(int(block_size[0]), time_patches)
    block_f = min(int(block_size[1]), frequency_patches)
    cells_per_block = max(1, block_t * block_f)

    # Enough block placements to cover well past the visible budget after overlaps.
    num_blocks = 4 * math.ceil(target_visible / cells_per_block) + 16

    # Batched block top-left corners: (B, M).
    start_t = torch.randint(
        0, max(time_patches - block_t + 1, 1), (batch_size, num_blocks),
        device=device, generator=generator,
    )
    start_f = torch.randint(
        0, max(frequency_patches - block_f + 1, 1), (batch_size, num_blocks),
        device=device, generator=generator,
    )

    # Cell offsets within a block: (cells_per_block,).
    off_i, off_j = torch.meshgrid(
        torch.arange(block_t, device=device),
        torch.arange(block_f, device=device),
        indexing="ij",
    )
    off_i = off_i.reshape(1, 1, -1)
    off_j = off_j.reshape(1, 1, -1)

    # Absolute flat cell index for every (row, block, offset): (B, M*cells_per_block).
    cell_t = start_t.unsqueeze(-1) + off_i
    cell_f = start_f.unsqueeze(-1) + off_j
    cell_idx = (cell_t * frequency_patches + cell_f).reshape(batch_size, -1)

    # Reveal order: block m reveals with priority m (earlier blocks win).
    order = (
        torch.arange(num_blocks, device=device, dtype=torch.float32)
        .view(1, num_blocks, 1)
        .expand(batch_size, num_blocks, cells_per_block)
        .reshape(batch_size, -1)
    )

    # Per-cell rank = min reveal order among covering blocks; uncovered stay high.
    rank = torch.full(
        (batch_size, num_tokens), float(num_blocks) + 1.0, device=device
    )
    rank.scatter_reduce_(1, cell_idx, order, reduce="amin", include_self=True)

    # Random tiebreak within equal ranks (and among uncovered cells).
    noise = torch.rand((batch_size, num_tokens), device=device, generator=generator)
    score = rank + noise
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != (batch_size, num_tokens):
            raise ValueError(
                "inverse_block_masking expected boolean valid_mask of shape "
                f"{(batch_size, num_tokens)}, got {valid_mask.dtype} {tuple(valid_mask.shape)}"
            )
        score = score + (~valid_mask).to(score.dtype) * (float(num_blocks) + 2.0)

    ids_shuffle = torch.argsort(score, dim=1)  # first target_visible -> kept
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :target_visible].contiguous()

    visible_mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
    visible_mask.scatter_(1, ids_keep, True)
    removal_mask = ~visible_mask

    return visible_mask, removal_mask, ids_restore, ids_keep


def restore_patch_tokens(
    visible_tokens: torch.Tensor,
    *,
    mask_token: torch.Tensor,
    ids_restore: torch.Tensor,
) -> torch.Tensor:
    """Restore full patch sequences by inserting mask tokens."""
    batch_size, num_tokens = ids_restore.shape
    dim = visible_tokens.shape[-1]
    num_visible = visible_tokens.shape[1]
    num_masked = num_tokens - num_visible
    if num_masked < 0:
        raise ValueError("restore_patch_tokens received more visible tokens than total tokens")

    expanded_mask = mask_token.reshape(1, 1, dim).expand(batch_size, num_masked, dim)
    full = torch.cat([visible_tokens, expanded_mask], dim=1)
    return torch.gather(
        full,
        dim=1,
        index=ids_restore.unsqueeze(-1).expand(-1, -1, dim),
    )


__all__ = [
    "inverse_block_masking",
    "num_visible_tokens",
    "restore_patch_tokens",
]

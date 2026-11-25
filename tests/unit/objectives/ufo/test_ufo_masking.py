"""Tests for EAT UFO masking helpers."""

from __future__ import annotations

import torch

from audiossl.modelling.objectives.ufo.masking import inverse_block_masking, restore_patch_tokens


def test_inverse_block_masking_visible_fraction():
    visible, removal, ids_restore, ids_keep = inverse_block_masking(
        batch_size=2,
        time_patches=8,
        frequency_patches=8,
        mask_ratio=0.8,
        block_size=(2, 2),
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )
    assert visible.shape == (2, 64)
    assert removal.shape == (2, 64)
    assert ids_restore.shape == (2, 64)
    assert visible.dtype == torch.bool
    assert torch.all(visible | removal)
    assert torch.all(~(visible & removal))
    visible_fraction = visible.float().mean(dim=1)
    assert torch.all(visible_fraction <= 0.35)


def test_inverse_block_masking_fixed_count_and_consistency():
    # Every row keeps exactly the fixed visible budget, and ids_keep / ids_restore
    # are mutually consistent with the boolean visible mask.
    b, t, f, ratio = 4, 8, 8, 0.8
    visible, removal, ids_restore, ids_keep = inverse_block_masking(
        batch_size=b,
        time_patches=t,
        frequency_patches=f,
        mask_ratio=ratio,
        block_size=(2, 2),
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )
    n = t * f
    k = max(1, round(n * (1.0 - ratio)))
    # Fixed count of kept tokens per row.
    assert ids_keep.shape == (b, k)
    assert torch.all(visible.sum(dim=1) == k)
    # ids_keep indexes exactly the visible positions (no duplicates).
    gathered = torch.zeros_like(visible)
    gathered.scatter_(1, ids_keep, True)
    assert torch.equal(gathered, visible)
    # ids_restore is a valid permutation of every row.
    expected = torch.arange(n).expand(b, n)
    assert torch.equal(ids_restore.sort(dim=1).values, expected)


def _sample(valid_mask=None, seed=2):
    return inverse_block_masking(
        batch_size=3,
        time_patches=8,
        frequency_patches=4,
        mask_ratio=0.75,
        block_size=(2, 2),
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(seed),
        valid_mask=valid_mask,
    )


def test_inverse_block_masking_all_valid_mask_is_bit_identical():
    plain = _sample()
    masked = _sample(valid_mask=torch.ones(3, 32, dtype=torch.bool))
    for a, b in zip(plain, masked, strict=True):
        assert torch.equal(a, b)


def test_inverse_block_masking_prefers_valid_cells():
    # Row 0 all valid; row 1 has more valid cells than the visible budget (8),
    # so no padding is kept; row 2 has fewer, so padding fills the rest.
    valid = torch.ones(3, 32, dtype=torch.bool)
    valid[1, 16:] = False
    valid[2, 4:] = False
    visible, removal, _, ids_keep = _sample(valid_mask=valid)
    assert ids_keep.shape[1] == 8
    assert torch.all(visible.sum(dim=1) == 8)
    assert not torch.any(visible[1] & ~valid[1])
    assert torch.all(visible[2, :4])
    assert torch.equal(visible | removal, torch.ones_like(visible))
    visible = torch.randn(1, 3, 8)
    mask_token = torch.randn(8)
    ids_restore = torch.tensor([[2, 0, 1, 3, 4]])
    restored = restore_patch_tokens(
        visible,
        mask_token=mask_token,
        ids_restore=ids_restore,
    )
    assert restored.shape == (1, 5, 8)

from __future__ import annotations

import pytest
import torch

from audiossl.modelling.objectives.mae.masking import (
    kept_token_mask,
    random_masking,
    random_masking_group,
)


def _call(fn, x, ratio, seed, **kwargs):
    return fn(x, ratio, generator=torch.Generator().manual_seed(seed), **kwargs)


class TestValidMask:
    @pytest.mark.parametrize(
        ("fn", "kwargs"),
        [(random_masking, {}), (random_masking_group, {"group_factor": 4})],
    )
    def test_all_valid_mask_is_bit_identical_to_unmasked_call(self, fn, kwargs) -> None:
        x = torch.randn(3, 24, 5)
        valid = torch.ones(3, 24, dtype=torch.bool)

        plain = _call(fn, x, 0.75, 3, **kwargs)
        masked = _call(fn, x, 0.75, 3, valid_mask=valid, **kwargs)

        for a, b in zip(plain, masked, strict=True):
            assert torch.equal(a, b)

    @pytest.mark.parametrize(
        ("fn", "kwargs"),
        [(random_masking, {}), (random_masking_group, {"group_factor": 4})],
    )
    def test_padding_is_never_kept_and_rows_keep_their_ratio(self, fn, kwargs) -> None:
        x = torch.randn(3, 24, 5)
        valid = torch.ones(3, 24, dtype=torch.bool)
        valid[1, 16:] = False
        valid[2, 8:] = False

        x_kept, mask, ids_restore = _call(fn, x, 0.5, 0, valid_mask=valid, **kwargs)
        kept = mask < 0.5

        assert not (kept & ~valid).any()
        assert kept.sum(dim=1).tolist() == [12, 8, 4]
        slot_valid = kept_token_mask(mask, ids_restore, x_kept.shape[1])
        assert slot_valid.sum(dim=1).tolist() == [12, 8, 4]

    def test_rejects_malformed_valid_mask(self) -> None:
        x = torch.randn(2, 10, 4)
        with pytest.raises(TypeError):
            random_masking(x, 0.5, valid_mask=torch.ones(2, 10))
        with pytest.raises(ValueError):
            random_masking(x, 0.5, valid_mask=torch.ones(2, 9, dtype=torch.bool))


class TestRandomMasking:
    def test_shapes(self) -> None:
        x = torch.randn(2, 10, 4)

        x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)

        assert x_masked.shape == (2, 2, 4)
        assert mask.shape == (2, 10)
        assert ids_restore.shape == (2, 10)
        assert ids_restore.dtype == torch.int64

    def test_mask_count(self) -> None:
        x = torch.randn(3, 100, 8)

        _, mask, _ = random_masking(x, mask_ratio=0.6)

        assert torch.equal(mask.sum(dim=1), torch.full((3,), 60.0))

    def test_ids_restore_is_permutation(self) -> None:
        x = torch.randn(2, 12, 4)

        _, _, ids_restore = random_masking(x, mask_ratio=0.5)

        for row in ids_restore:
            assert torch.equal(torch.sort(row).values, torch.arange(12))

    def test_reproducibility(self) -> None:
        x = torch.randn(2, 10, 4)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)

        first = random_masking(x, 0.5, generator=g1)
        second = random_masking(x, 0.5, generator=g2)

        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])
        assert torch.equal(first[2], second[2])

    @pytest.mark.parametrize("bad_x", [torch.randn(10, 4), "not a tensor"])
    def test_invalid_x_raises(self, bad_x: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            random_masking(bad_x, 0.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_mask_ratio_raises(self, ratio: float) -> None:
        x = torch.randn(1, 10, 4)
        with pytest.raises(ValueError):
            random_masking(x, ratio)


class TestRandomMaskingGroup:
    def test_groups_contiguous_when_boundary_is_aligned(self) -> None:
        x = torch.randn(1, 12, 4)

        _, mask, _ = random_masking_group(x, mask_ratio=0.5, group_factor=2)

        kept = (mask[0] == 0).nonzero().flatten().tolist()
        kept_set = set(kept)
        for pos in kept:
            assert (pos ^ 1) in kept_set

    def test_rejects_non_multiple_n(self) -> None:
        x = torch.randn(1, 11, 4)
        with pytest.raises(ValueError, match="multiple of group_factor"):
            random_masking_group(x, mask_ratio=0.5, group_factor=2)

    def test_group_factor_1_matches_random_masking(self) -> None:
        x = torch.randn(2, 10, 4)
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)

        per_token = random_masking(x, 0.5, generator=g1)
        grouped = random_masking_group(x, 0.5, group_factor=1, generator=g2)

        assert torch.equal(per_token[1], grouped[1])
        assert torch.equal(per_token[2], grouped[2])

    @pytest.mark.parametrize("group_factor", [0, -1, 1.5, True])
    def test_invalid_group_factor_raises(self, group_factor: object) -> None:
        x = torch.randn(1, 10, 4)
        with pytest.raises((TypeError, ValueError)):
            random_masking_group(
                x,
                0.5,
                group_factor=group_factor,  # type: ignore[arg-type]
            )

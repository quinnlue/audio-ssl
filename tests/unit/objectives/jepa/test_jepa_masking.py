from __future__ import annotations

import unittest

import torch

from audiossl.config import TemporalMaskingConfig
from audiossl.modelling.objectives.jepa.masking import create_jepa_target_mask


def _sample_index_scalar_reference(
    num_items: int, generator: torch.Generator | None
) -> int:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    device = (
        getattr(generator, "device", torch.device("cpu"))
        if generator is not None
        else "cpu"
    )
    return int(torch.randint(num_items, (1,), generator=generator, device=device).item())


def _candidate_spans_scalar_reference(
    *,
    valid_row: torch.Tensor,
    selected_row: torch.Tensor,
    min_span_len: int,
    max_span_len: int,
    allow_overlap: bool,
    max_new_targets: int,
) -> list[tuple[int, int]]:
    seq_len = valid_row.shape[0]
    candidates: list[tuple[int, int]] = []
    for span_len in range(min_span_len, max_span_len + 1):
        if span_len > seq_len:
            break
        for start in range(0, seq_len - span_len + 1):
            stop = start + span_len
            valid_span = valid_row[start:stop]
            if not bool(valid_span.all().item()):
                continue
            selected_span = selected_row[start:stop]
            if not allow_overlap and bool(selected_span.any().item()):
                continue
            new_targets = int((~selected_span).sum().item())
            if new_targets == 0 or new_targets > max_new_targets:
                continue
            candidates.append((start, span_len))
    return candidates


def _create_jepa_target_mask_scalar_reference(
    valid_mask: torch.Tensor,
    *,
    config: TemporalMaskingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Frozen copy of the pre-vectorization implementation, retained here for
    regression testing only. Do not import from production code.
    """

    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be torch.Tensor")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be rank-2")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool")

    batch_size, seq_len = valid_mask.shape
    target_mask = torch.zeros_like(valid_mask)
    if batch_size == 0 or seq_len == 0:
        raise ValueError("valid_mask must contain at least one sample and one timestep")

    for batch_index in range(batch_size):
        valid_row = valid_mask[batch_index]
        valid_count = int(valid_row.sum().item())
        if valid_count < 2:
            raise ValueError("requires at least two valid tokens per sample")

        max_span_len = min(config.max_span_len, valid_count - 1)
        if config.min_span_len > max_span_len:
            raise ValueError("min_span_len too large")

        target_budget = int(round(float(config.mask_ratio) * valid_count))
        target_budget = max(1, min(target_budget, valid_count - 1))
        selected_row = target_mask[batch_index]

        while int(selected_row.sum().item()) < target_budget:
            remaining_context_budget = valid_count - 1 - int(selected_row.sum().item())
            candidates = _candidate_spans_scalar_reference(
                valid_row=valid_row,
                selected_row=selected_row,
                min_span_len=config.min_span_len,
                max_span_len=max_span_len,
                allow_overlap=config.allow_overlap,
                max_new_targets=remaining_context_budget,
            )
            if not candidates:
                break
            start, span_len = candidates[
                _sample_index_scalar_reference(len(candidates), generator)
            ]
            selected_row[start : start + span_len] |= valid_row[start : start + span_len]

        if not bool(selected_row.any().item()):
            raise ValueError("zero target tokens")
        context_row = valid_row & ~selected_row
        if not bool(context_row.any().item()):
            raise ValueError("zero context tokens")

    return target_mask


class JEPAMaskingTests(unittest.TestCase):
    def test_returns_bool_mask_with_padding_excluded(self) -> None:
        valid_mask = torch.tensor(
            [
                [True, True, True, True, False, False],
                [True, True, True, True, True, False],
            ]
        )

        target_mask = create_jepa_target_mask(
            valid_mask,
            config=TemporalMaskingConfig(mask_ratio=0.5, min_span_len=1, max_span_len=2),
            generator=torch.Generator().manual_seed(0),
        )

        self.assertEqual(target_mask.shape, valid_mask.shape)
        self.assertEqual(target_mask.dtype, torch.bool)
        self.assertFalse(bool((target_mask & ~valid_mask).any().item()))
        self.assertTrue(bool(target_mask.any(dim=1).all().item()))
        self.assertTrue(bool((valid_mask & ~target_mask).any(dim=1).all().item()))

    def test_non_overlap_mode_keeps_target_count_below_full_context(self) -> None:
        valid_mask = torch.ones(1, 6, dtype=torch.bool)

        target_mask = create_jepa_target_mask(
            valid_mask,
            config=TemporalMaskingConfig(
                mask_ratio=0.67,
                min_span_len=2,
                max_span_len=2,
                allow_overlap=False,
            ),
            generator=torch.Generator().manual_seed(1),
        )

        self.assertEqual(int(target_mask.sum().item()), 4)
        self.assertTrue(bool((valid_mask & ~target_mask).any().item()))

    def test_overlap_mode_smoke(self) -> None:
        valid_mask = torch.ones(2, 8, dtype=torch.bool)

        target_mask = create_jepa_target_mask(
            valid_mask,
            config=TemporalMaskingConfig(
                mask_ratio=0.5,
                min_span_len=2,
                max_span_len=3,
                allow_overlap=True,
            ),
            generator=torch.Generator().manual_seed(2),
        )

        self.assertEqual(target_mask.shape, valid_mask.shape)
        self.assertTrue(bool(target_mask.any().item()))

    def test_rejects_short_sequences_without_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two valid tokens"):
            create_jepa_target_mask(
                torch.ones(1, 1, dtype=torch.bool),
                config=TemporalMaskingConfig(mask_ratio=0.5),
            )

    def test_rejects_min_span_that_cannot_leave_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_span_len"):
            create_jepa_target_mask(
                torch.ones(1, 2, dtype=torch.bool),
                config=TemporalMaskingConfig(
                    mask_ratio=0.5,
                    min_span_len=2,
                    max_span_len=2,
                ),
            )

    def test_vectorized_matches_scalar_reference(self) -> None:
        # Locks the vectorized rewrite to the original scalar semantics across
        # several shapes, validity patterns, overlap modes, and RNG seeds.
        cases = []
        for seed in (0, 1, 2):
            cases.extend(
                [
                    {
                        "valid_mask": torch.ones(4, 16, dtype=torch.bool),
                        "config": TemporalMaskingConfig(
                            mask_ratio=0.5,
                            min_span_len=1,
                            max_span_len=3,
                            allow_overlap=True,
                        ),
                        "seed": seed,
                    },
                    {
                        "valid_mask": torch.ones(4, 16, dtype=torch.bool),
                        "config": TemporalMaskingConfig(
                            mask_ratio=0.5,
                            min_span_len=2,
                            max_span_len=4,
                            allow_overlap=False,
                        ),
                        "seed": seed,
                    },
                    {
                        # Mixed validity: trailing padding on each row.
                        "valid_mask": torch.tensor(
                            [
                                [True] * 12 + [False] * 4,
                                [True] * 10 + [False] * 6,
                                [True] * 14 + [False] * 2,
                                [True] * 8 + [False] * 8,
                            ],
                            dtype=torch.bool,
                        ),
                        "config": TemporalMaskingConfig(
                            mask_ratio=0.5,
                            min_span_len=1,
                            max_span_len=3,
                            allow_overlap=True,
                        ),
                        "seed": seed,
                    },
                    {
                        "valid_mask": torch.tensor(
                            [
                                [True] * 12 + [False] * 4,
                                [True] * 10 + [False] * 6,
                                [True] * 14 + [False] * 2,
                                [True] * 8 + [False] * 8,
                            ],
                            dtype=torch.bool,
                        ),
                        "config": TemporalMaskingConfig(
                            mask_ratio=0.5,
                            min_span_len=2,
                            max_span_len=3,
                            allow_overlap=False,
                        ),
                        "seed": seed,
                    },
                ]
            )

        for case in cases:
            with self.subTest(
                shape=tuple(case["valid_mask"].shape),
                allow_overlap=case["config"].allow_overlap,
                min_span=case["config"].min_span_len,
                max_span=case["config"].max_span_len,
                seed=case["seed"],
            ):
                vec_out = create_jepa_target_mask(
                    case["valid_mask"],
                    config=case["config"],
                    generator=torch.Generator().manual_seed(case["seed"]),
                )
                ref_out = _create_jepa_target_mask_scalar_reference(
                    case["valid_mask"],
                    config=case["config"],
                    generator=torch.Generator().manual_seed(case["seed"]),
                )
                self.assertTrue(
                    torch.equal(vec_out, ref_out),
                    msg=f"vectorized vs scalar mismatch for case {case}",
                )

    def test_rejects_bad_valid_mask_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-2"):
            create_jepa_target_mask(
                torch.ones(1, 2, 3, dtype=torch.bool),
                config=TemporalMaskingConfig(mask_ratio=0.5),
            )

        with self.assertRaisesRegex(ValueError, "torch.bool"):
            create_jepa_target_mask(
                torch.ones(1, 2),
                config=TemporalMaskingConfig(mask_ratio=0.5),
            )


if __name__ == "__main__":
    unittest.main()

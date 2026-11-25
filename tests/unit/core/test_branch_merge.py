from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from audiossl.config import BranchMergeConfig
from audiossl.modelling.core.branch_merge import BranchMerger


class BranchMergerTests(unittest.TestCase):
    def test_concat_preserves_shape(self) -> None:
        merger = BranchMerger(d_model=8, config=BranchMergeConfig(method="concat"))
        g = torch.randn(2, 5, 8)
        l = torch.randn(2, 5, 8)

        output = merger(g, l)

        self.assertEqual(tuple(output.shape), (2, 5, 8))

    def test_concat_has_trainable_projection(self) -> None:
        merger = BranchMerger(d_model=8, config=BranchMergeConfig(method="concat"))
        trainable_names = {name for name, p in merger.named_parameters() if p.requires_grad}

        self.assertIn("projection.weight", trainable_names)
        self.assertIn("projection.bias", trainable_names)

    def test_learned_ave_weights_softmax_sum_to_one(self) -> None:
        merger = BranchMerger(d_model=8, config=BranchMergeConfig(method="learned_ave"))

        weights = F.softmax(merger.merge_weights, dim=0)
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))

    def test_learned_ave_initial_weights_are_uniform(self) -> None:
        merger = BranchMerger(d_model=8, config=BranchMergeConfig(method="learned_ave"))
        weights = F.softmax(merger.merge_weights, dim=0)

        torch.testing.assert_close(weights, torch.tensor([0.5, 0.5]))

    def test_learned_ave_preserves_shape_and_is_a_weighted_sum(self) -> None:
        merger = BranchMerger(d_model=8, config=BranchMergeConfig(method="learned_ave"))
        g = torch.randn(2, 5, 8)
        l = torch.randn(2, 5, 8)

        output = merger(g, l)
        weights = F.softmax(merger.merge_weights, dim=0)
        expected = weights[0] * g + weights[1] * l

        torch.testing.assert_close(output, expected)
        self.assertEqual(tuple(output.shape), (2, 5, 8))

    def test_fixed_ave_uses_configured_weight(self) -> None:
        merger = BranchMerger(
            d_model=8,
            config=BranchMergeConfig(method="fixed_ave", fixed_weight=0.25),
        )
        g = torch.ones(1, 1, 8)
        l = torch.zeros(1, 1, 8)

        output = merger(g, l)

        torch.testing.assert_close(output, 0.25 * torch.ones(1, 1, 8))

    def test_fixed_ave_weight_is_not_trainable(self) -> None:
        merger = BranchMerger(
            d_model=8,
            config=BranchMergeConfig(method="fixed_ave", fixed_weight=0.5),
        )

        self.assertFalse(any(p.requires_grad for p in merger.parameters()))
        self.assertIn("fixed_weight", dict(merger.named_buffers()))

    def test_invalid_method_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "method"):
            BranchMerger(d_model=8, config=BranchMergeConfig(method="not_a_method"))  # type: ignore[arg-type]

    def test_mismatched_branch_shapes_fail(self) -> None:
        merger = BranchMerger(d_model=8, config=BranchMergeConfig(method="concat"))
        g = torch.randn(2, 5, 8)
        l = torch.randn(2, 4, 8)

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            merger(g, l)


if __name__ == "__main__":
    unittest.main()

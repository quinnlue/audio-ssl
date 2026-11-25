from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from audiossl.modelling.objectives.contrastive.loss import (
    InfoNCELossOutput,
    contrastive_info_nce_loss,
)


def _hand_computed_info_nce(
    zi: torch.Tensor,
    zj: torch.Tensor,
    temperature: float,
    *,
    symmetric: bool,
) -> torch.Tensor:
    zi_norm = F.normalize(zi, dim=-1)
    zj_norm = F.normalize(zj, dim=-1)
    logits = (zi_norm @ zj_norm.t()) / temperature
    targets = torch.arange(zi.shape[0])
    loss_ij = F.cross_entropy(logits, targets)
    if not symmetric:
        return loss_ij
    loss_ji = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_ij + loss_ji)


class ContrastiveInfoNCELossTests(unittest.TestCase):
    def test_returns_dataclass_with_expected_shapes(self) -> None:
        torch.manual_seed(0)
        zi = torch.randn(4, 6, requires_grad=True)
        zj = torch.randn(4, 6)

        out = contrastive_info_nce_loss(zi, zj, temperature=0.1, symmetric=True)

        self.assertIsInstance(out, InfoNCELossOutput)
        self.assertEqual(out.loss.ndim, 0)
        self.assertEqual(out.positive_loss.ndim, 0)
        self.assertEqual(out.negative_loss.ndim, 0)
        self.assertEqual(tuple(out.logits.shape), (4, 4))
        self.assertTrue(torch.isfinite(out.loss))
        self.assertTrue(out.loss.requires_grad)

    def test_loss_equals_positive_loss_when_asymmetric(self) -> None:
        torch.manual_seed(1)
        zi = torch.randn(5, 7)
        zj = torch.randn(5, 7)

        out = contrastive_info_nce_loss(zi, zj, temperature=0.2, symmetric=False)

        torch.testing.assert_close(out.loss, out.positive_loss)

    def test_symmetric_loss_matches_mean_of_directions(self) -> None:
        torch.manual_seed(2)
        zi = torch.randn(3, 4)
        zj = torch.randn(3, 4)

        zi_norm = F.normalize(zi, dim=-1)
        zj_norm = F.normalize(zj, dim=-1)
        logits = (zi_norm @ zj_norm.t()) / 0.3
        targets = torch.arange(3)
        expected = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
        )

        out = contrastive_info_nce_loss(zi, zj, temperature=0.3, symmetric=True)

        torch.testing.assert_close(out.loss, expected)

    def test_numeric_reference_matches_hand_computed(self) -> None:
        zi = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        zj = torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.1],
                [0.4, 0.6, 0.0],
            ]
        )

        for symmetric in (True, False):
            with self.subTest(symmetric=symmetric):
                expected = _hand_computed_info_nce(
                    zi, zj, temperature=0.5, symmetric=symmetric
                )
                out = contrastive_info_nce_loss(
                    zi, zj, temperature=0.5, symmetric=symmetric
                )
                torch.testing.assert_close(out.loss, expected)

    def test_logits_are_reusable_for_top1_metric(self) -> None:
        torch.manual_seed(3)
        zi = torch.randn(4, 5)
        zj = torch.randn(4, 5)

        out = contrastive_info_nce_loss(zi, zj, temperature=0.1, symmetric=True)

        predicted = out.logits.argmax(dim=-1)
        self.assertEqual(tuple(predicted.shape), (4,))
        top1 = (predicted == torch.arange(4)).float().mean()
        self.assertGreaterEqual(top1.item(), 0.0)
        self.assertLessEqual(top1.item(), 1.0)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shape"):
            contrastive_info_nce_loss(
                torch.randn(4, 6),
                torch.randn(4, 7),
                temperature=0.1,
                symmetric=True,
            )

    def test_rejects_batch_without_negatives(self) -> None:
        for batch_size in (0, 1):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "batch size"):
                    contrastive_info_nce_loss(
                        torch.zeros((batch_size, 3)),
                        torch.zeros((batch_size, 3)),
                        temperature=0.1,
                        symmetric=True,
                    )

    def test_rejects_non_finite_or_non_positive_temperature(self) -> None:
        zi = torch.randn(3, 4)
        zj = torch.randn(3, 4)
        for bad in (0.0, -0.5, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "temperature"):
                    contrastive_info_nce_loss(zi, zj, temperature=bad, symmetric=True)

    def test_rejects_non_rank_two_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-2"):
            contrastive_info_nce_loss(
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                temperature=0.1,
                symmetric=True,
            )

    def test_aligned_pairs_have_lower_loss_than_misaligned(self) -> None:
        x = torch.eye(4)
        aligned = contrastive_info_nce_loss(x, x, temperature=0.1, symmetric=True)
        misaligned = contrastive_info_nce_loss(
            x, torch.roll(x, shifts=1, dims=0), temperature=0.1, symmetric=True
        )
        self.assertLess(aligned.loss.item(), misaligned.loss.item())
        self.assertTrue(math.isfinite(aligned.loss.item()))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from audiossl.modelling.objectives.jepa.loss import JEPALossOutput, jepa_prediction_loss


class JEPALossTests(unittest.TestCase):
    def test_mse_matches_masked_manual_reference(self) -> None:
        prediction = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [2.0, 3.0]]]
        )
        target = torch.zeros_like(prediction)
        target_mask = torch.tensor([[True, False], [False, True]])

        out = jepa_prediction_loss(
            prediction,
            target,
            target_mask,
            loss_style="mse",
        )

        expected = torch.tensor([2.5, 6.5]).mean()
        self.assertIsInstance(out, JEPALossOutput)
        torch.testing.assert_close(out.loss, expected)
        self.assertEqual(tuple(out.per_position_loss.shape), (2, 2))

    def test_mae_matches_masked_manual_reference(self) -> None:
        prediction = torch.tensor([[[1.0, -3.0], [2.0, 4.0]]])
        target = torch.zeros_like(prediction)
        target_mask = torch.tensor([[False, True]])

        out = jepa_prediction_loss(
            prediction,
            target,
            target_mask,
            loss_style="mae",
        )

        torch.testing.assert_close(out.loss, torch.tensor(3.0))

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shape"):
            jepa_prediction_loss(
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, 5),
                torch.ones(2, 3, dtype=torch.bool),
                loss_style="mse",
            )

    def test_rejects_bad_mask_contract(self) -> None:
        prediction = torch.zeros(2, 3, 4)
        target = torch.zeros_like(prediction)

        with self.assertRaisesRegex(ValueError, "torch.bool"):
            jepa_prediction_loss(
                prediction,
                target,
                torch.ones(2, 3),
                loss_style="mse",
            )

        with self.assertRaisesRegex(ValueError, "batch/time"):
            jepa_prediction_loss(
                prediction,
                target,
                torch.ones(2, 2, dtype=torch.bool),
                loss_style="mse",
            )

    def test_rejects_zero_selected_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero positions"):
            jepa_prediction_loss(
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, dtype=torch.bool),
                loss_style="mse",
            )

    def test_rejects_unknown_loss_style(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss_style"):
            jepa_prediction_loss(
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, 4),
                torch.ones(2, 3, dtype=torch.bool),
                loss_style="huber",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()

"""BYOL-specific behaviour beyond the generic objective contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from testkit.cases import objective_case


def test_predictor_is_student_only():
    objective = objective_case("byol_ema").build()
    assert not hasattr(objective, "teacher_predictor")
    assert not any(
        name.startswith("teacher_") and "predictor" in name
        for name, _ in objective.named_parameters()
    )


def test_loss_is_symmetric_cross_view_cosine_regression():
    case = objective_case("byol_ema")
    out = case.build().train()(case.batch())

    def regression(prediction, target):
        return (2.0 - 2.0 * F.cosine_similarity(prediction, target, dim=-1)).mean()

    expected = 0.5 * (
        regression(out.prediction_i, out.target_j) + regression(out.prediction_j, out.target_i)
    )
    torch.testing.assert_close(out.loss, expected)
    assert not out.target_i.requires_grad and not out.target_j.requires_grad

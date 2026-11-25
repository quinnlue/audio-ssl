"""DINO-specific behaviour beyond the generic objective contract."""

from __future__ import annotations

import pytest
import torch

from audiossl.config.batch_dataclasses import ObjectiveOutput
from audiotrainer.step import StepContext
from testkit.cases import objective_case


def _ctx() -> StepContext:
    return StepContext(
        optimizer_step=0,
        total_optimizer_steps=1,
        epoch=0,
        global_step=0,
        step_was_skipped=False,
    )


@pytest.mark.parametrize("case_id", ["dino_ema", "dino_stop_grad"])
def test_on_after_forward_updates_center_by_ema_of_uncentered_teacher_logits(case_id):
    case = objective_case(case_id)
    objective = case.build().train()
    out = objective(case.batch())
    momentum = case.objective_config().center_momentum
    before = objective.center.detach().clone()

    objective.on_after_forward(out, _ctx())

    expected = momentum * before + (1.0 - momentum) * out.teacher_logits_uncentered.mean(dim=0)
    torch.testing.assert_close(objective.center, expected)
    assert out.teacher_logits_uncentered.shape[0] == 2 * case.batch().x.shape[0]


def test_on_after_forward_rejects_foreign_output():
    objective = objective_case("dino_ema").build()
    with pytest.raises(TypeError):
        objective.on_after_forward(ObjectiveOutput(loss=torch.tensor(0.0)), _ctx())

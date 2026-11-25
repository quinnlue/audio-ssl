"""Contrastive-specific behaviour beyond the generic objective contract."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from audiotrainer.step import StepContext

from testkit.cases import objective_case

STATS = ("top1", "pos_sim", "neg_sim", "align", "uniformity")


def _run(compute_stats: bool):
    base = objective_case("contrastive_augmentation")
    case = replace(base, objective=lambda: replace(base.objective(), compute_stats=compute_stats))
    objective = case.build().train()
    return objective, objective(case.batch())


@pytest.mark.parametrize("compute_stats", [True, False])
def test_compute_stats_controls_diagnostic_outputs_and_metrics(compute_stats):
    objective, out = _run(compute_stats)
    metrics = objective.metrics(
        out, StepContext(epoch=0, global_step=0, optimizer_step=0, total_optimizer_steps=1)
    )
    for field in STATS:
        value = getattr(out, field)
        if compute_stats:
            assert isinstance(value, torch.Tensor) and value.ndim == 0
            assert torch.isfinite(value)
            assert field in metrics
        else:
            assert value is None
            assert field not in metrics
    assert {"positive_loss", "negative_loss"} <= set(metrics)


def test_cls_fields_are_pooled_encoder_states_not_projections():
    objective, out = _run(True)
    d_model = objective.model.config.d_model
    projection_dim = objective.config.projection.output_dim
    assert out.cls_i.shape[-1] == out.cls_j.shape[-1] == d_model
    assert out.zi.shape[-1] == out.zj.shape[-1] == projection_dim
    torch.testing.assert_close(objective.projection(out.cls_i), out.zi)

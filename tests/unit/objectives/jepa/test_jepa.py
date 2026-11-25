"""JEPA-specific behaviour beyond the generic objective contract."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from audiossl.config import JEPAObjectiveConfig, MLPConfig
from audiossl.modelling.model import AudioModel
from testkit import configs
from testkit.cases import objective_case


def _with_objective(case_id: str, **overrides):
    base = objective_case(case_id)
    return replace(base, objective=lambda: replace(base.objective(), **overrides))


def _with_model(case_id: str, model):
    return replace(objective_case(case_id), model=model)


def _subsampled_model():
    return configs.model(
        processor=configs.conv_processor(),
        tokenizer=configs.noop_tokenizer(),
        embedding=configs.linear_embedding(subsampling=configs.subsampling(2)),
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(objective_case("jepa_ema"), id="ema"),
        pytest.param(_with_objective("jepa_stop_grad", output_norm_type="layernorm"), id="ln"),
        pytest.param(_with_model("jepa_ema", _subsampled_model), id="subsampled"),
    ],
)
def test_output_masks_are_consistent_and_nonempty(case):
    out = case.build().train()(case.batch())
    tokens = out.prediction.shape[:2]
    assert out.prediction.shape == out.target.shape
    assert out.z is out.prediction and out.mask is out.encoder_mask
    for mask in (out.encoder_mask, out.target_mask, out.context_mask):
        assert mask.dtype == torch.bool and mask.shape == tokens
    assert not (out.target_mask & ~out.encoder_mask).any(), "targets outside valid tokens"
    assert not (out.context_mask & ~out.encoder_mask).any(), "context outside valid tokens"
    assert out.target_mask.any(dim=1).all() and out.context_mask.any(dim=1).all()
    assert not out.target.requires_grad and out.prediction.requires_grad


def test_mask_token_respects_configured_init_scale():
    scale = 0.0125
    objective = _with_objective("jepa_ema", mask_token_init_scale=scale).build()
    magnitude = objective.mask_token.detach().abs()
    assert 0.0 < float(magnitude.max()) <= scale


def test_config_requires_explicit_mask_token_init_scale():
    with pytest.raises(TypeError):
        JEPAObjectiveConfig(loss_style="mse", predictor_config=MLPConfig(d_ff=16))  # type: ignore[call-arg]


class _PersistentFloatBuffer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running_stat", torch.zeros(3))


class _NonPersistentFloatBuffer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scratch", torch.zeros(3), persistent=False)


def _objective_with_extra_module(factory):
    case = objective_case("jepa_ema")
    model = AudioModel(case.model())
    model.add_module("extra", factory())
    return case.build(model=model)


@pytest.mark.parametrize(
    "factory",
    [lambda: torch.nn.BatchNorm1d(4), _PersistentFloatBuffer, _NonPersistentFloatBuffer],
)
def test_ema_update_hard_copies_float_buffers(factory):
    objective = _objective_with_extra_module(factory)
    with torch.no_grad():
        for buffer in objective.model.extra.buffers():
            if buffer.is_floating_point():
                buffer.add_(1.5)
    objective.update_teacher()
    student = dict(objective.model.extra.named_buffers())
    for name, buffer in objective.teacher_model.extra.named_buffers():
        assert torch.equal(buffer, student[name]), name

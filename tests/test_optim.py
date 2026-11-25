"""Optimizer and schedule builders, pinned against closed-form values."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from audiossl.config import (
    AdamWOptimizerConfig,
    LARSOptimizerConfig,
)

from audiotrainer.optim import (
    LARS,
    ConstantScale,
    CosineAnnealingScale,
    StepLRScale,
    WarmupCosineScale,
    build_optimizer,
    build_param_groups,
    build_scheduler,
)


def test_constant_scale() -> None:
    assert [ConstantScale()(step) for step in range(3)] == [1.0, 1.0, 1.0]


def test_warmup_cosine_scale() -> None:
    scale = WarmupCosineScale(warmup_steps=2, total_steps=6, min_lr_scale=0.1)

    assert [scale(step) for step in range(6)] == pytest.approx(
        [
            0.5,  # warmup (0 + 1) / 2
            1.0,  # warmup (1 + 1) / 2
            1.0,  # cosine progress 0 / 3
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 3)),
            0.1 + 0.9 * 0.5 * (1 + math.cos(2 * math.pi / 3)),
            0.1,  # cosine progress 3 / 3
        ]
    )
    assert scale(100) == pytest.approx(0.1)


def test_cosine_annealing_scale() -> None:
    scale = CosineAnnealingScale(total_steps=5, min_lr_scale=0.0)

    assert [scale(step) for step in range(5)] == pytest.approx(
        [1.0, 0.5 + 0.5 * math.cos(math.pi / 4), 0.5, 0.5 - 0.5 * math.cos(math.pi / 4), 0.0]
    )


def test_step_lr_scale() -> None:
    scale = StepLRScale(step_size=2, gamma=0.5)

    assert [scale(step) for step in range(5)] == [1.0, 1.0, 0.5, 0.5, 0.25]


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2)
        self.norm = nn.LayerNorm(2)
        self.frozen = nn.Linear(2, 2)
        self.frozen.requires_grad_(False)


def test_param_groups_split_decay_by_substring_and_drop_frozen() -> None:
    net = _Net()

    decay, no_decay = build_param_groups(
        net, weight_decay=0.1, no_weight_decay_patterns=("bias", "norm")
    )

    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert {id(p) for p in decay["params"]} == {id(net.proj.weight)}
    assert {id(p) for p in no_decay["params"]} == {
        id(net.proj.bias),
        id(net.norm.weight),
        id(net.norm.bias),
    }


def test_param_groups_reject_typo_patterns_and_fully_frozen_modules() -> None:
    with pytest.raises(ValueError, match="matched zero parameters"):
        build_param_groups(_Net(), weight_decay=0.1, no_weight_decay_patterns=("nrom",))

    frozen = _Net().requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_param_groups(frozen, weight_decay=0.1, no_weight_decay_patterns=())


def test_build_optimizer_applies_config_to_every_group() -> None:
    optimizer = build_optimizer(
        _Net(),
        AdamWOptimizerConfig(
            lr=3e-4,
            weight_decay=0.2,
            betas=(0.8, 0.9),
            eps=1e-6,
            no_weight_decay_patterns=("bias",),
        ),
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.2, 0.0]
    for group in optimizer.param_groups:
        assert (group["lr"], group["betas"], group["eps"]) == (3e-4, (0.8, 0.9), 1e-6)


def test_lars_step_matches_trust_ratio_formula() -> None:
    param = nn.Parameter(torch.tensor([3.0, 4.0]))
    param.grad = torch.tensor([0.6, 0.8])
    lr, eta, weight_decay = 0.5, 0.01, 0.1
    optimizer = LARS([param], lr=lr, momentum=0.0, weight_decay=weight_decay, eta=eta)
    expected_grad = param.grad + weight_decay * param.detach()
    trust_ratio = eta * param.detach().norm() / expected_grad.norm()
    expected = param.detach() - lr * trust_ratio * expected_grad

    optimizer.step()

    assert torch.allclose(param.detach(), expected)


def test_builders_reject_unknown_configs() -> None:
    with pytest.raises(TypeError):
        build_optimizer(object(), LARSOptimizerConfig(lr=1.0))
    optimizer = build_optimizer(_Net(), LARSOptimizerConfig(lr=1.0, no_weight_decay_patterns=()))
    with pytest.raises(TypeError):
        build_scheduler(optimizer, object(), total_steps=1)

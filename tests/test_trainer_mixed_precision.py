"""Autocast and gradient scaling policy."""

from __future__ import annotations

import math

import pytest
import torch
from audiossl.config import MixedPrecisionConfig, SGDOptimizerConfig

from harness import DEFAULT_CASE, assert_trees_equal, snapshot, with_device

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _record_autocast(run) -> list[tuple[bool, torch.dtype]]:
    device_type = run.model.applied_runtime.device.type
    observed: list[tuple[bool, torch.dtype]] = []
    forward = run.objective.forward

    def recording_forward(batch):
        observed.append(
            (
                torch.is_autocast_enabled(device_type),
                torch.get_autocast_dtype(device_type),
            )
        )
        return forward(batch)

    run.objective.forward = recording_forward
    return observed


def test_bfloat16_autocast_wraps_every_training_forward() -> None:
    run = DEFAULT_CASE.run(num_batches=2, mixed_precision=MixedPrecisionConfig(dtype="bfloat16"))
    observed = _record_autocast(run)

    run.trainer().fit()

    assert observed == [(True, torch.bfloat16)] * 2


def test_no_mixed_precision_means_no_autocast() -> None:
    run = DEFAULT_CASE.run(num_batches=2)
    observed = _record_autocast(run)

    run.trainer().fit()

    assert [enabled for enabled, _ in observed] == [False, False]


def test_bfloat16_training_is_deterministic_and_finite() -> None:
    def train():
        run = DEFAULT_CASE.run(
            num_batches=4, max_epochs=2, mixed_precision=MixedPrecisionConfig(dtype="bfloat16")
        )
        return run, run.fit()

    first, first_trace = train()
    second, second_trace = train()

    assert all(math.isfinite(loss) for loss in first_trace.losses.values())
    assert first_trace.losses == second_trace.losses
    assert_trees_equal(snapshot(second.objective), snapshot(first.objective))


@requires_cuda
def test_float16_grad_scaler_reports_unscaled_grad_norm() -> None:
    def first_step_grad_norm(mixed_precision) -> float:
        config = with_device(
            DEFAULT_CASE.training_config(
                optimizer=SGDOptimizerConfig(lr=0.0), mixed_precision=mixed_precision
            ),
            "cuda",
        )
        run = DEFAULT_CASE.run(config=config, num_batches=1)
        return run.fit().grad_norms[1]

    fp32 = first_step_grad_norm(None)
    fp16 = first_step_grad_norm(MixedPrecisionConfig(dtype="float16", grad_scaler=True))

    assert fp16 == pytest.approx(fp32, rel=0.05)

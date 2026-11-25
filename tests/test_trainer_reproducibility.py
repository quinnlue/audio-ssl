"""A training run is a pure function of its config.

Same config gives a bit-identical model regardless of ambient RNG state, and
attaching observers (checkpointing, logging, recorders, validation) must not
change the model that comes out.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from audiossl.config import CheckpointConfig, EvaluationConfig, WandBConfig
from harness.configs import run_config

from audiotrainer import ObjectiveLossEvaluator, ValidationCallback

from harness import (
    DEFAULT_CASE,
    FakeEvaluator,
    HookRecorder,
    ObjectiveCase,
    assert_trees_equal,
    over_objectives,
    snapshot,
    waveform_loader,
    with_seed,
)


@over_objectives
def test_same_config_trains_bit_identical_model_regardless_of_ambient_rng(
    case: ObjectiveCase,
) -> None:
    if case.name == "mae":
        pytest.importorskip("torchaudio")
    torch.manual_seed(999)
    random.seed(999)
    first = case.run(num_batches=4, max_epochs=2)
    first_trace = first.fit()

    torch.manual_seed(12345)
    random.seed(12345)
    torch.rand(1000)
    second = case.run(num_batches=4, max_epochs=2)
    second_trace = second.fit()

    assert_trees_equal(snapshot(second.objective), snapshot(first.objective))
    assert second_trace.losses == first_trace.losses


def test_changing_the_seed_changes_initialization_and_trained_model() -> None:
    config = DEFAULT_CASE.training_config(max_epochs=1)
    seed_0 = DEFAULT_CASE.run(config=config)
    seed_1 = DEFAULT_CASE.run(config=with_seed(config, 1))

    with pytest.raises(AssertionError):
        assert_trees_equal(snapshot(seed_1.objective), snapshot(seed_0.objective))
    trace_0, trace_1 = seed_0.fit(), seed_1.fit()
    assert trace_0.losses != trace_1.losses


def _checkpoint_everything(tmp_path: Path) -> dict:
    return dict(
        overrides=dict(
            checkpoint=CheckpointConfig(
                save_dir=str(tmp_path),
                save_every_n_epochs=1,
                save_every_n_steps=1,
                keep_last_n=2,
                save_best=True,
                monitor="train/loss",
            ),
        )
    )


def _stdout_logger(tmp_path: Path) -> dict:
    return dict(overrides=dict(stdout_logging=True))


def _hook_recorder(tmp_path: Path) -> dict:
    return dict(callbacks=[HookRecorder()])


def _wandb_logger(tmp_path: Path) -> dict:
    return dict(
        run=run_config(loggers=(WandBConfig(project="unit-test", mode="disabled"),))
    )


def _validation_without_rng(tmp_path: Path) -> dict:
    return dict(
        overrides=dict(evaluation=EvaluationConfig(every_n_steps=1, at_train_end=True)),
        callbacks=[ValidationCallback(FakeEvaluator())],
    )


OBSERVERS = {
    "checkpoint_every_step_epoch_and_best": _checkpoint_everything,
    "stdout_logger": _stdout_logger,
    "hook_recorder": _hook_recorder,
    "wandb_logger": _wandb_logger,
    "validation_with_rng_free_evaluator": _validation_without_rng,
}


@pytest.mark.parametrize("observer", OBSERVERS.values(), ids=OBSERVERS.keys())
def test_observers_do_not_change_the_trained_model(
    observer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import wandb as wandb_module

    for name in ("init", "log", "finish"):
        monkeypatch.setattr(wandb_module, name, lambda *a, **k: None)
    setup = observer(tmp_path)
    bare = DEFAULT_CASE.run(num_batches=4, max_epochs=2)
    bare_trace = bare.fit()

    observed = DEFAULT_CASE.run(
        num_batches=4,
        max_epochs=2,
        run=setup.get("run"),
        **setup.get("overrides", {}),
    )
    observed_trace = observed.fit(*setup.get("callbacks", ()))

    assert_trees_equal(snapshot(observed.objective), snapshot(bare.objective))
    assert observed_trace.losses == bare_trace.losses


def test_mid_training_validation_does_not_change_the_trained_model() -> None:
    def train(evaluation: EvaluationConfig) -> dict[str, torch.Tensor]:
        run = DEFAULT_CASE.run(num_batches=4, max_epochs=2, evaluation=evaluation)
        val_loader = waveform_loader(
            num_batches=2, batch_size=2, length=DEFAULT_CASE.waveform_length
        )
        run.fit(ValidationCallback(ObjectiveLossEvaluator(run.objective, val_loader)))
        return snapshot(run.objective)

    every_epoch = train(EvaluationConfig(every_n_epochs=1, at_train_end=False))
    after_training_only = train(EvaluationConfig(at_train_end=True))

    assert_trees_equal(every_epoch, after_training_only)

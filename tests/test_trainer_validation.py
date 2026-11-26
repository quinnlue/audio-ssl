"""Validation cadence and how its metrics flow into state, checkpoints, and loggers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from audiossl.config import (
    CheckpointConfig,
    EvaluationConfig,
    RunConfig,
    WandBConfig,
    validate_config,
)
from audiotrainer import ValidationCallback

from harness import DEFAULT_CASE, FakeEvaluator

CADENCES = {
    "every_epoch": (EvaluationConfig(every_n_epochs=1, at_train_end=False), 3, [2, 4, 6]),
    "every_two_epochs": (EvaluationConfig(every_n_epochs=2, at_train_end=False), 3, [4]),
    "every_two_steps": (EvaluationConfig(every_n_steps=2, at_train_end=False), 2, [2, 4]),
    "before_train": (EvaluationConfig(before_train=True, at_train_end=False), 2, [0]),
    "at_train_end": (EvaluationConfig(at_train_end=True), 2, [4]),
    "steps_then_end_without_duplicate": (
        EvaluationConfig(every_n_steps=2, at_train_end=True),
        2,
        [2, 4],
    ),
}


@pytest.mark.parametrize(
    ("evaluation", "max_epochs", "expected_steps"), CADENCES.values(), ids=CADENCES.keys()
)
def test_validation_cadence(evaluation, max_epochs, expected_steps, tmp_path: Path) -> None:
    evaluator = FakeEvaluator()
    DEFAULT_CASE.run(
        num_batches=2,
        max_epochs=max_epochs,
        evaluation=evaluation,
        checkpoint=CheckpointConfig(save_dir=str(tmp_path)),
    ).trainer(ValidationCallback(evaluator)).fit()

    assert evaluator.steps == expected_steps


def test_step_cadence_validates_once_per_optimizer_step_under_accumulation() -> None:
    evaluator = FakeEvaluator()
    DEFAULT_CASE.run(
        num_batches=4,
        grad_accum=2,
        evaluation=EvaluationConfig(every_n_steps=1, at_train_end=False),
    ).trainer(ValidationCallback(evaluator)).fit()

    assert evaluator.steps == [1, 2]


def test_val_metrics_are_prefixed_into_state_and_logged_to_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged: list[tuple[dict, int | None]] = []
    import wandb as wandb_module

    handle = SimpleNamespace(
        log=lambda payload, step=None: logged.append((dict(payload), step)),
        finish=lambda **kwargs: None,
    )
    monkeypatch.setattr(wandb_module, "init", lambda **kwargs: handle)
    run = DEFAULT_CASE.run(
        num_batches=2,
        checkpoint=CheckpointConfig(save_dir=str(tmp_path)),
        evaluation=EvaluationConfig(at_train_end=True),
    )

    run.run_config = replace(
        run.run_config,
        loggers=(WandBConfig(project="unit-test", mode="disabled"),),
    )
    state = run.trainer(ValidationCallback(FakeEvaluator(values=[3.25]))).fit()

    assert state.last_val_metrics == {"val/loss": 3.25, "val/custom": 4.25}
    assert ({"val/loss": 3.25, "val/custom": 4.25}, state.optimizer_step) in logged


@pytest.mark.parametrize(
    "validation_first", [False, True], ids=["default_order", "validation_first"]
)
def test_save_best_uses_the_current_epoch_metric(validation_first: bool, tmp_path: Path) -> None:
    validation = ValidationCallback(FakeEvaluator(values=[10.0, 1.0]))
    trainer = DEFAULT_CASE.run(
        num_batches=2,
        max_epochs=2,
        checkpoint=CheckpointConfig(
            save_dir=str(tmp_path), save_every_n_epochs=1, save_best=True, monitor="val/loss"
        ),
        evaluation=EvaluationConfig(every_n_epochs=1, at_train_end=False),
    ).trainer(validation)
    if validation_first:
        trainer.callbacks.remove(validation)
        trainer.callbacks.insert(0, validation)

    state = trainer.fit()

    assert state.best_metric == 1.0
    assert state.best_optimizer_step == 4
    assert state.last_val_metrics["val/loss"] == 1.0


def test_unwired_validation_callback_refuses_to_run() -> None:
    run = DEFAULT_CASE.run(
        num_batches=1, evaluation=EvaluationConfig(before_train=True, at_train_end=False)
    )
    callback = ValidationCallback(FakeEvaluator())
    state = run.trainer(ValidationCallback(FakeEvaluator())).state

    with pytest.raises(RuntimeError, match="before attach"):
        callback.on_train_start(state)


def test_validation_callback_requires_an_evaluator() -> None:
    with pytest.raises(TypeError, match="evaluate"):
        ValidationCallback(object())


def test_evaluation_config_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot both be set"):
        validate_config(EvaluationConfig(every_n_epochs=1, every_n_steps=1))
    with pytest.raises(ValueError, match="at least one validation trigger"):
        validate_config(EvaluationConfig(at_train_end=False))
    with pytest.raises(ValueError, match="RunConfig.evaluation"):
        validate_config(
            RunConfig(
                checkpoint=CheckpointConfig(
                    save_dir=str(tmp_path), save_best=True, monitor="val/loss"
                )
            )
        )

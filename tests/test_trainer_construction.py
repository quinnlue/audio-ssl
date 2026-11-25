"""``Trainer.__init__`` refuses setups whose run identity would be wrong or ambiguous."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from audiossl.config import (
    AdamWOptimizerConfig,
    CheckpointConfig,
    EvaluationConfig,
    MixedPrecisionConfig,
    RunConfig,
    WandBConfig,
)
from torch.utils.data import DataLoader

from audiotrainer import Checkpoint, MetricsJsonlLogger, StdoutLogger, Trainer, WandBLogger, config_fingerprint

from harness import DEFAULT_CASE, HookRecorder, collate_waveforms, with_seed


def _trainer_args(run) -> dict:
    return dict(
        model=run.model,
        objective=run.objective,
        train_loader=run.loader,
        train_config=run.config,
        data_fingerprint=run.data_fingerprint,
        run_config=run.run_config,
    )


BAD_ARGUMENTS = {
    "model": dict(model=object()),
    "objective": dict(objective=object()),
    "train_loader": dict(train_loader=[1, 2]),
    "train_config": dict(train_config=object()),
    "callbacks_str": dict(callbacks="callback"),
    "callbacks_item": dict(callbacks=[object()]),
    "run_config": dict(run_config=object()),
}


@pytest.mark.parametrize("bad", BAD_ARGUMENTS.values(), ids=BAD_ARGUMENTS.keys())
def test_rejects_wrong_argument_types(bad: dict) -> None:
    args = _trainer_args(DEFAULT_CASE.run(num_batches=1))
    args.update(bad)

    with pytest.raises(TypeError):
        Trainer(**args)


def test_rejects_model_that_does_not_match_config() -> None:
    run = DEFAULT_CASE.run(num_batches=1)
    other_mlp = replace(run.config.model.encoder.mlp, d_ff=run.config.model.encoder.mlp.d_ff + 1)
    other_model = replace(
        run.config.model, encoder=replace(run.config.model.encoder, mlp=other_mlp)
    )

    with pytest.raises(ValueError, match="model.config does not match"):
        Trainer(
            run.model,
            run.objective,
            run.loader,
            replace(run.config, model=other_model),
            data_fingerprint=run.data_fingerprint,
            run_config=run.run_config,
        )


def test_rejects_model_built_with_a_different_seed() -> None:
    run = DEFAULT_CASE.run(num_batches=1)

    with pytest.raises(ValueError, match="model.config does not match"):
        Trainer(
            run.model,
            run.objective,
            run.loader,
            with_seed(run.config, 7),
            data_fingerprint=run.data_fingerprint,
            run_config=run.run_config,
        )


def test_rejects_objective_that_does_not_match_config() -> None:
    run = DEFAULT_CASE.run(num_batches=1)
    other_objective = replace(run.config.objective, teacher_ema_decay=0.5)

    with pytest.raises(ValueError, match="objective.config does not match"):
        Trainer(
            run.model,
            run.objective,
            run.loader,
            replace(run.config, objective=other_objective),
            data_fingerprint=run.data_fingerprint,
            run_config=run.run_config,
        )


def test_rejects_empty_loader() -> None:
    run = DEFAULT_CASE.run(num_batches=1)
    run.loader = DataLoader(
        [],
        batch_size=run.config.trainer.batch.per_device_batch_size,
        drop_last=run.config.trainer.batch.drop_last,
        collate_fn=collate_waveforms,
    )

    with pytest.raises(ValueError, match="zero batches"):
        run.trainer()


def test_evaluation_requires_a_validation_callback() -> None:
    run = DEFAULT_CASE.run(num_batches=1, evaluation=EvaluationConfig(at_train_end=True))

    with pytest.raises(ValueError, match="ValidationCallback"):
        run.trainer()


@pytest.mark.parametrize(
    "mixed_precision",
    [
        MixedPrecisionConfig(dtype="float16"),
        MixedPrecisionConfig(dtype="float16", grad_scaler=True),
    ],
    ids=["float16", "float16_grad_scaler"],
)
def test_cuda_only_precision_settings_are_rejected_on_cpu(mixed_precision) -> None:
    run = DEFAULT_CASE.run(num_batches=1, mixed_precision=mixed_precision)
    assert run.model.applied_runtime.device.type == "cpu"

    with pytest.raises(RuntimeError, match="CUDA"):
        run.trainer()


def test_rejects_weight_decay_patterns_that_match_nothing() -> None:
    run = DEFAULT_CASE.run(
        num_batches=1,
        optimizer=AdamWOptimizerConfig(lr=1e-3, no_weight_decay_patterns=("nonexistent",)),
    )

    with pytest.raises(ValueError, match="matched zero parameters"):
        run.trainer()


def test_rejects_invalid_run_config() -> None:
    run = DEFAULT_CASE.run(num_batches=1)
    run.run_config = RunConfig(loggers=(WandBConfig(project="x", mode="bad_mode"),))

    with pytest.raises(ValueError):
        run.trainer()


def test_auto_attached_callbacks_precede_user_callbacks_in_contract_order(tmp_path: Path) -> None:
    run = DEFAULT_CASE.run(
        num_batches=1, stdout_logging=True, checkpoint=CheckpointConfig(save_dir=str(tmp_path))
    )
    user = HookRecorder()

    run.run_config = replace(
        run.run_config, loggers=(WandBConfig(project="x", mode="disabled"),)
    )
    trainer = run.trainer(user)

    assert [type(callback) for callback in trainer.callbacks] == [
        Checkpoint,
        MetricsJsonlLogger,
        StdoutLogger,
        WandBLogger,
        HookRecorder,
    ]
    assert trainer.callbacks[-1] is user


def test_fingerprint_identifies_the_research_config() -> None:
    config = DEFAULT_CASE.training_config()
    run = DEFAULT_CASE.run(config=config)
    run.run_config = replace(
        run.run_config, loggers=(WandBConfig(project="x", mode="disabled"),)
    )
    trainer = run.trainer()
    lr_changed = replace(
        config,
        trainer=replace(config.trainer, optimizer=replace(config.trainer.optimizer, lr=2e-3)),
    )

    fp = config_fingerprint(config, data_fingerprint=run.data_fingerprint)
    assert trainer.state.config_fingerprint == fp
    assert config_fingerprint(
        DEFAULT_CASE.training_config(), data_fingerprint=run.data_fingerprint
    ) == fp
    assert (
        config_fingerprint(lr_changed, data_fingerprint=run.data_fingerprint)
        != fp
    )
    assert (
        config_fingerprint(with_seed(config, 1), data_fingerprint=run.data_fingerprint)
        != fp
    )

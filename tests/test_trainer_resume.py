"""Preemption is invisible: a resumed run trains the same model as an uninterrupted one."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from audiossl.config import (
    AdamWOptimizerConfig,
    CheckpointConfig,
    EvaluationConfig,
    WarmupCosineSchedulerConfig,
)
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives.objective import build_objective

from audiotrainer import AugmentationPipeline, Callback, ValidationCallback

from harness import (
    BYOL,
    DEFAULT_CASE,
    FakeEvaluator,
    JEPA,
    ObjectiveCase,
    Run,
    TrainingTrace,
    assert_trees_equal,
    over_objectives,
    snapshot,
    waveform_loader,
    with_seed,
)


def epoch_checkpoints(save_dir: Path, **overrides) -> CheckpointConfig:
    values = dict(save_dir=str(save_dir), save_every_n_epochs=1, keep_last_n=10)
    values.update(overrides)
    return CheckpointConfig(**values)


@over_objectives
def test_resume_from_epoch_checkpoint_reproduces_uninterrupted_run(
    case: ObjectiveCase, tmp_path: Path
) -> None:
    if case.name == "mae":
        pytest.importorskip("torchaudio")
    checkpoint = epoch_checkpoints(tmp_path)
    config = case.training_config(
        max_epochs=2,
        grad_clip=1.0,
        optimizer=AdamWOptimizerConfig(lr=1e-2),
        scheduler=WarmupCosineSchedulerConfig(warmup_steps=1, min_lr_scale=0.1),
    )
    straight = case.run(config=config, checkpoint=checkpoint)
    straight_trace = TrainingTrace()
    straight_trainer = straight.trainer(straight_trace)
    straight_trainer.fit()

    resumed = case.run(config=config, checkpoint=checkpoint)
    resumed_trace = TrainingTrace()
    resumed_trainer = resumed.resume(tmp_path / "epoch_0001.pt", resumed_trace)
    resumed_trainer.fit()

    assert_trees_equal(snapshot(resumed.objective), snapshot(straight.objective), label="objective")
    assert_trees_equal(
        resumed_trainer.optimizer.state_dict(),
        straight_trainer.optimizer.state_dict(),
        label="optimizer",
    )
    assert_trees_equal(
        resumed_trainer.scheduler.state_dict(),
        straight_trainer.scheduler.state_dict(),
        label="scheduler",
    )
    for counter in ("epoch", "global_step", "optimizer_step"):
        assert getattr(resumed_trainer.state, counter) == getattr(straight_trainer.state, counter)
    second_epoch = {step: straight_trace.losses[step] for step in range(5, 9)}
    assert resumed_trace.losses == second_epoch
    assert resumed_trace.applied_lrs == {
        step: straight_trace.applied_lrs[step] for step in range(5, 9)
    }


def _val_best_setup(tmp_path: Path, *, max_epochs: int) -> tuple:
    checkpoint = epoch_checkpoints(tmp_path, save_best=True, monitor="val/loss")
    evaluation = EvaluationConfig(every_n_epochs=1, at_train_end=False)
    config = DEFAULT_CASE.training_config(max_epochs=max_epochs)
    return config, checkpoint, evaluation


def test_resume_restores_counters_and_tracking_fields_from_payload(tmp_path: Path) -> None:
    config, checkpoint, evaluation = _val_best_setup(tmp_path, max_epochs=3)
    DEFAULT_CASE.run(config=config, checkpoint=checkpoint, evaluation=evaluation).trainer(
        ValidationCallback(FakeEvaluator(values=[4.0, 9.0, 9.0]))
    ).fit()
    payload = torch.load(tmp_path / "epoch_0002.pt", weights_only=False)
    assert payload["best_metric"] is not None, "test needs a payload with tracking state"

    resumed = DEFAULT_CASE.run(config=config, checkpoint=checkpoint, evaluation=evaluation).resume(
        tmp_path / "epoch_0002.pt", ValidationCallback(FakeEvaluator())
    )

    assert resumed.state.last_val_metrics == payload["last_val_metrics"]
    assert resumed.state.best_metric == payload["best_metric"]
    assert resumed.state.best_optimizer_step == payload["best_optimizer_step"]
    assert resumed.state.optimizer_step == payload["optimizer_step"] == 8
    assert resumed.state.global_step == payload["global_step"] == 8
    assert resumed.state.epoch == payload["epoch"] + 1 == 2


def test_best_checkpoint_is_not_replaced_by_a_worse_model_after_resume(tmp_path: Path) -> None:
    config, checkpoint, evaluation = _val_best_setup(tmp_path, max_epochs=3)
    DEFAULT_CASE.run(config=config, checkpoint=checkpoint, evaluation=evaluation).trainer(
        ValidationCallback(FakeEvaluator(values=[5.0, 1.0, 3.0]))
    ).fit()
    assert torch.load(tmp_path / "best.pt", weights_only=False)["best_metric"] == 1.0

    # Preempted during the third epoch: resume from the second epoch's snapshot.
    DEFAULT_CASE.run(config=config, checkpoint=checkpoint, evaluation=evaluation).resume(
        tmp_path / "epoch_0002.pt", ValidationCallback(FakeEvaluator(values=[3.0]))
    ).fit()

    assert torch.load(tmp_path / "best.pt", weights_only=False)["best_metric"] == 1.0


def test_extending_max_epochs_on_resume_matches_longer_uninterrupted_run(tmp_path: Path) -> None:
    longer = DEFAULT_CASE.run(max_epochs=3)
    longer_trace = longer.fit()

    ckpt = epoch_checkpoints(tmp_path)
    DEFAULT_CASE.run(max_epochs=2, checkpoint=ckpt).fit()
    extended = DEFAULT_CASE.run(max_epochs=3, checkpoint=ckpt)
    extended_trace = TrainingTrace()
    trainer = extended.resume(tmp_path / "last.pt", extended_trace)
    assert (trainer.state.epoch, trainer.state.optimizer_step) == (2, 8)
    state = trainer.fit()

    assert (state.epoch, state.optimizer_step) == (2, 12)
    assert_trees_equal(snapshot(extended.objective), snapshot(longer.objective))
    assert extended_trace.losses == {step: longer_trace.losses[step] for step in range(9, 13)}


def test_resuming_a_finished_run_with_same_config_is_a_no_op(tmp_path: Path) -> None:
    checkpoint = epoch_checkpoints(tmp_path)
    config = DEFAULT_CASE.training_config(max_epochs=2)
    finished = DEFAULT_CASE.run(config=config, checkpoint=checkpoint)
    finished.fit()

    resumed = DEFAULT_CASE.run(config=config, checkpoint=checkpoint)
    state = resumed.resume(tmp_path / "last.pt", TrainingTrace()).fit()

    assert (state.epoch, state.optimizer_step) == (2, 8)
    assert_trees_equal(snapshot(resumed.objective), snapshot(finished.objective))


def _drift_lr(config):
    optimizer = replace(config.trainer.optimizer, lr=config.trainer.optimizer.lr * 2)
    return replace(config, trainer=replace(config.trainer, optimizer=optimizer))


def _drift_seed(config):
    return with_seed(config, config.model.runtime.seed + 1)


def _drift_grad_accum(config):
    batch = replace(config.trainer.batch, gradient_accumulation_steps=2)
    return replace(config, trainer=replace(config.trainer, batch=batch))


def _drift_objective(config):
    objective = replace(config.objective, teacher_ema_decay=0.5)
    return replace(config, objective=objective)


CONFIG_DRIFTS = {
    "optimizer_lr": _drift_lr,
    "model_seed": _drift_seed,
    "gradient_accumulation": _drift_grad_accum,
    "objective_hyperparameter": _drift_objective,
}


@pytest.mark.parametrize("drift", CONFIG_DRIFTS.values(), ids=CONFIG_DRIFTS.keys())
def test_resume_rejects_any_research_config_drift(drift, tmp_path: Path) -> None:
    checkpoint = epoch_checkpoints(tmp_path)
    config = DEFAULT_CASE.training_config(max_epochs=1)
    DEFAULT_CASE.run(config=config, checkpoint=checkpoint).fit()
    extended_loop = replace(config.trainer.loop, max_epochs=2)
    drifted = drift(replace(config, trainer=replace(config.trainer, loop=extended_loop)))

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        DEFAULT_CASE.run(config=drifted, checkpoint=checkpoint).resume(
            tmp_path / "last.pt", TrainingTrace()
        )


def test_resume_rejects_max_epochs_before_next_epoch(tmp_path: Path) -> None:
    ckpt = epoch_checkpoints(tmp_path)
    DEFAULT_CASE.run(max_epochs=2, checkpoint=ckpt).fit()

    with pytest.raises(RuntimeError, match="Increase max_epochs"):
        DEFAULT_CASE.run(max_epochs=1, checkpoint=ckpt).resume(tmp_path / "last.pt")


def test_resume_from_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DEFAULT_CASE.run().resume(tmp_path / "missing.pt")


class _BatchTracker(Callback):
    def __init__(self, batches: list[torch.Tensor]) -> None:
        self._batches = batches
        self.seen: list[tuple[int, int]] = []

    def on_train_batch_start(self, state, batch) -> None:
        index = next(i for i, x in enumerate(self._batches) if torch.equal(x, batch.x))
        self.seen.append((state.epoch, index))


def test_resume_from_mid_epoch_step_checkpoint_does_not_skip_batches(tmp_path: Path) -> None:
    checkpoint = CheckpointConfig(
        save_dir=str(tmp_path), save_every_n_epochs=None, save_every_n_steps=3, keep_last_n=10
    )
    config = DEFAULT_CASE.training_config(max_epochs=2)
    DEFAULT_CASE.run(config=config, checkpoint=checkpoint).fit()
    batches = [
        batch.x
        for batch in waveform_loader(
            num_batches=4, batch_size=2, length=DEFAULT_CASE.waveform_length
        )
    ]

    tracker = _BatchTracker(batches)
    DEFAULT_CASE.run(config=config, checkpoint=checkpoint).resume(
        tmp_path / "step_00000003.pt", TrainingTrace(), tracker
    ).fit()

    assert (0, 3) in tracker.seen, f"epoch 0 batch 3 never trained; resumed run saw {tracker.seen}"


def test_extending_max_epochs_does_not_train_on_an_exhausted_schedule(tmp_path: Path) -> None:
    schedule = WarmupCosineSchedulerConfig(warmup_steps=1, min_lr_scale=0.0)
    ckpt = epoch_checkpoints(tmp_path)
    DEFAULT_CASE.run(max_epochs=2, scheduler=schedule, checkpoint=ckpt).fit()
    extended = DEFAULT_CASE.run(
        max_epochs=4, scheduler=schedule, checkpoint=ckpt
    )

    try:
        trainer = extended.resume(tmp_path / "last.pt", trace := TrainingTrace())
    except (RuntimeError, ValueError):
        return
    trainer.fit()

    first_extension_step = min(trace.applied_lrs)
    assert trace.applied_lrs[first_extension_step] > 0.0


def _byol_run_with_seeded_augmentor(config, *, checkpoint: CheckpointConfig | None = None) -> Run:
    model = AudioModel(config.model)
    augmentor = AugmentationPipeline(
        config.trainer.augmentor, generator=torch.Generator().manual_seed(0)
    )
    run = Run(
        case=BYOL,
        config=config,
        model=model,
        objective=build_objective(model, config.objective, augmentor=augmentor),
        loader=waveform_loader(num_batches=4, batch_size=2, length=BYOL.waveform_length),
    )
    if checkpoint is not None:
        run.run_config = replace(run.run_config, checkpoint=checkpoint)
    return run


def test_augmentor_generator_state_survives_resume(tmp_path: Path) -> None:
    checkpoint = epoch_checkpoints(tmp_path)
    config = BYOL.training_config(max_epochs=2)
    straight = _byol_run_with_seeded_augmentor(config, checkpoint=checkpoint)
    straight.fit()

    resumed = _byol_run_with_seeded_augmentor(config, checkpoint=checkpoint)
    resumed.resume(tmp_path / "epoch_0001.pt", TrainingTrace()).fit()

    assert_trees_equal(snapshot(resumed.objective), snapshot(straight.objective))


def test_masking_generator_state_survives_resume(tmp_path: Path) -> None:
    checkpoint = epoch_checkpoints(tmp_path)
    config = JEPA.training_config(max_epochs=2)
    straight = JEPA.run(config=config, checkpoint=checkpoint)
    straight_trace = straight.fit()

    resumed = JEPA.run(config=config, checkpoint=checkpoint)
    resumed_trace = TrainingTrace()
    resumed.resume(tmp_path / "epoch_0001.pt", resumed_trace).fit()

    assert resumed_trace.losses == {step: straight_trace.losses[step] for step in range(5, 9)}
    assert_trees_equal(snapshot(resumed.objective), snapshot(straight.objective))


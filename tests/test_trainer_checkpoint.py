"""Checkpoint artifacts hold exactly the state they claim to, written where and when configured."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
from audiossl.config import CheckpointConfig, EvaluationConfig

from audiotrainer import ObjectiveLossEvaluator, ValidationCallback
from audiotrainer.fingerprint import _to_plain

from harness import (
    DEFAULT_CASE,
    FakeEvaluator,
    HookRecorder,
    assert_trees_equal,
    waveform_loader,
)


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def test_last_checkpoint_payload_matches_live_training_state(tmp_path: Path) -> None:
    run = DEFAULT_CASE.run(
        num_batches=4,
        max_epochs=2,
        checkpoint=CheckpointConfig(save_dir=str(tmp_path), save_every_n_epochs=1),
    )
    trainer = run.trainer()
    state = trainer.fit()

    payload = _load(tmp_path / "last.pt")

    assert_trees_equal(payload["model_state"], run.model.state_dict(), label="model_state")
    assert_trees_equal(payload["objective_state"], run.objective.state_dict(), label="objective")
    assert_trees_equal(payload["optimizer_state"], trainer.optimizer.state_dict(), label="opt")
    assert_trees_equal(payload["scheduler_state"], trainer.scheduler.state_dict(), label="sched")
    assert (payload["epoch"], payload["optimizer_step"], payload["global_step"]) == (1, 8, 8)
    assert payload["config_fingerprint"] == state.config_fingerprint
    assert payload["train_config"] == _to_plain(asdict(run.config))
    assert set(payload["rng"]) == {"python", "torch", "cuda", "objective_generators"}
    assert payload["provenance"] == state.provenance
    assert {"torch", "python", "device", "git"} <= set(payload["provenance"])
    assert list(tmp_path.glob("*.tmp")) == []


def test_epoch_snapshots_rotate_to_keep_last_n(tmp_path: Path) -> None:
    DEFAULT_CASE.run(
        num_batches=2,
        max_epochs=4,
        checkpoint=CheckpointConfig(save_dir=str(tmp_path), save_every_n_epochs=1, keep_last_n=2),
    ).trainer().fit()

    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        "epoch_0003.pt",
        "epoch_0004.pt",
        "last.pt",
    ]


def test_epoch_snapshot_cadence(tmp_path: Path) -> None:
    DEFAULT_CASE.run(
        num_batches=1,
        max_epochs=5,
        checkpoint=CheckpointConfig(save_dir=str(tmp_path), save_every_n_epochs=2, keep_last_n=5),
    ).trainer().fit()

    assert sorted(p.name for p in tmp_path.glob("epoch_*.pt")) == ["epoch_0002.pt", "epoch_0004.pt"]
    assert _load(tmp_path / "last.pt")["epoch"] == 3


def test_step_snapshots_cadence_and_rotation(tmp_path: Path) -> None:
    run = DEFAULT_CASE.run(
        num_batches=2,
        max_epochs=3,
        checkpoint=CheckpointConfig(
            save_dir=str(tmp_path), save_every_n_epochs=None, save_every_n_steps=2, keep_last_n=2
        ),
    )
    state = run.trainer().fit()

    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        "last.pt",
        "step_00000004.pt",
        "step_00000006.pt",
    ]
    assert _load(tmp_path / "last.pt")["optimizer_step"] == state.optimizer_step


def test_step_snapshots_fire_once_per_optimizer_step_under_accumulation(tmp_path: Path) -> None:
    recorder = HookRecorder()
    DEFAULT_CASE.run(
        num_batches=4,
        grad_accum=2,
        checkpoint=CheckpointConfig(
            save_dir=str(tmp_path), save_every_n_epochs=None, save_every_n_steps=1, keep_last_n=5
        ),
    ).trainer(recorder).fit()

    written = [event[1] for event in recorder.events if event[0] == "on_checkpoint_written"]
    assert written == ["last.pt", "step_00000001.pt", "last.pt", "step_00000002.pt"]


def test_best_checkpoint_holds_the_weights_of_the_best_validated_epoch(tmp_path: Path) -> None:
    run = DEFAULT_CASE.run(
        num_batches=2,
        max_epochs=3,
        checkpoint=CheckpointConfig(
            save_dir=str(tmp_path),
            save_every_n_epochs=1,
            keep_last_n=3,
            save_best=True,
            monitor="val/loss",
        ),
        evaluation=EvaluationConfig(every_n_epochs=1, at_train_end=False),
    )
    state = run.trainer(ValidationCallback(FakeEvaluator(values=[5.0, 1.0, 3.0]))).fit()

    best = _load(tmp_path / "best.pt")
    assert (state.best_metric, state.best_optimizer_step) == (1.0, 4)
    assert (best["best_metric"], best["optimizer_step"]) == (1.0, 4)
    assert_trees_equal(
        best["objective_state"], _load(tmp_path / "epoch_0002.pt")["objective_state"]
    )


def test_best_checkpoint_with_real_objective_loss_evaluator(tmp_path: Path) -> None:
    run = DEFAULT_CASE.run(
        num_batches=2,
        max_epochs=2,
        checkpoint=CheckpointConfig(
            save_dir=str(tmp_path), save_every_n_epochs=1, save_best=True, monitor="val/loss"
        ),
        evaluation=EvaluationConfig(every_n_epochs=1, at_train_end=False),
    )
    val_loader = waveform_loader(num_batches=1, batch_size=2, length=DEFAULT_CASE.waveform_length)

    state = run.trainer(ValidationCallback(ObjectiveLossEvaluator(run.objective, val_loader))).fit()

    assert math.isfinite(state.best_metric)
    assert _load(tmp_path / "best.pt")["best_metric"] == state.best_metric


def test_best_checkpoint_on_train_loss_tracks_lowest_epoch_end_loss(tmp_path: Path) -> None:
    run = DEFAULT_CASE.run(
        num_batches=2,
        max_epochs=4,
        checkpoint=CheckpointConfig(
            save_dir=str(tmp_path),
            save_every_n_epochs=1,
            keep_last_n=4,
            save_best=True,
            monitor="train/loss",
            mode="min",
        ),
    )
    trace = run.fit()

    best_epoch = min(trace.epoch_end_losses, key=trace.epoch_end_losses.get)
    best = _load(tmp_path / "best.pt")
    assert best["best_metric"] == trace.epoch_end_losses[best_epoch]
    assert_trees_equal(
        best["objective_state"],
        _load(tmp_path / f"epoch_{best_epoch + 1:04d}.pt")["objective_state"],
    )


def test_checkpoint_written_hook_fans_out_every_artifact_in_order(tmp_path: Path) -> None:
    recorder = HookRecorder()
    DEFAULT_CASE.run(
        num_batches=1,
        max_epochs=2,
        checkpoint=CheckpointConfig(
            save_dir=str(tmp_path), save_every_n_epochs=1, save_best=True, monitor="train/loss"
        ),
    ).trainer(recorder).fit()

    written = [event[1] for event in recorder.events if event[0] == "on_checkpoint_written"]
    assert written[:3] == ["last.pt", "epoch_0001.pt", "best.pt"]
    assert written[3:5] == ["last.pt", "epoch_0002.pt"]

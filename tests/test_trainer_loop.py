"""Loop mechanics and the hook contract in callbacks ``CONTRACT.md`` section 3, 5, and 8."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch
from audiossl.config import EvaluationConfig
from torch.utils.data import DataLoader

from audiotrainer import Callback, ObjectiveLossEvaluator, ValidationCallback

from harness import (
    CONTRASTIVE,
    DEFAULT_CASE,
    FakeEvaluator,
    HookRecorder,
    install_objective_hook,
    spy_objective_hooks,
    waveform_loader,
)


@pytest.mark.parametrize(
    ("num_batches", "accum", "epochs", "expected_optimizer_steps"),
    [
        (4, 1, 1, 4),
        (4, 2, 1, 2),
        (5, 2, 1, 3),
        (5, 2, 2, 6),
        (6, 3, 2, 4),
    ],
)
def test_step_counters(num_batches, accum, epochs, expected_optimizer_steps) -> None:
    run = DEFAULT_CASE.run(num_batches=num_batches, grad_accum=accum, max_epochs=epochs)

    state = run.trainer().fit()

    assert state.optimizer_step == expected_optimizer_steps
    assert state.global_step == num_batches * epochs
    assert state.epoch == epochs - 1


def test_trainer_and_objective_hooks_fire_in_contract_order() -> None:
    run = DEFAULT_CASE.run(num_batches=3, grad_accum=2)
    events: list[tuple] = []
    spy_objective_hooks(run.objective, events)

    run.trainer(HookRecorder(events)).fit()

    # (hook, epoch, global_step, optimizer_step); the 3rd batch is a short trailing window.
    assert events == [
        ("on_train_start", 0, 0, 0),
        ("objective:on_train_start",),
        ("on_train_epoch_start", 0, 0, 0),
        ("objective:on_epoch_start",),
        ("on_train_batch_start", 0, 0, 0),
        ("objective:on_after_forward",),
        ("on_train_batch_end", 0, 1, 0),
        ("on_train_batch_start", 0, 1, 0),
        ("objective:on_after_forward",),
        ("objective:on_before_optimizer_step",),
        ("objective:on_after_optimizer_step",),
        ("on_train_batch_end", 0, 2, 1),
        ("on_train_batch_start", 0, 2, 1),
        ("objective:on_after_forward",),
        ("objective:on_before_optimizer_step",),
        ("objective:on_after_optimizer_step",),
        ("on_train_batch_end", 0, 3, 2),
        ("objective:on_epoch_end",),
        ("on_train_epoch_end", 0, 3, 2),
        ("objective:on_train_end",),
        ("on_train_end", 0, 3, 2),
    ]


def test_hook_counts_over_multiple_epochs() -> None:
    recorder = HookRecorder()
    DEFAULT_CASE.run(num_batches=2, max_epochs=3).trainer(recorder).fit()

    names = recorder.names()
    assert names.count("on_train_start") == names.count("on_train_end") == 1
    assert names.count("on_train_epoch_start") == names.count("on_train_epoch_end") == 3
    assert names.count("on_train_batch_start") == names.count("on_train_batch_end") == 6
    assert [epoch for name, epoch, *_ in recorder.events if name == "on_train_epoch_start"] == [
        0,
        1,
        2,
    ]


def test_batch_hooks_see_device_batches_and_unscaled_loss() -> None:
    class Check(Callback):
        def __init__(self) -> None:
            self.losses: list[tuple[float, float]] = []

        def on_train_batch_start(self, state, batch) -> None:
            assert batch.x.device == state.applied_runtime.device
            assert batch.mask.device == state.applied_runtime.device

        def on_train_batch_end(self, state, batch, output) -> None:
            self.losses.append((state.last_loss, float(output.loss.detach())))

    check = Check()
    DEFAULT_CASE.run(num_batches=4, grad_accum=1).trainer(check).fit()

    assert [state_loss for state_loss, _ in check.losses] == [loss for _, loss in check.losses]


def test_objective_hooks_default_to_noops() -> None:
    run = CONTRASTIVE.run(num_batches=2)
    assert not hasattr(run.objective, "callbacks")
    run.trainer().fit()


def test_legacy_objective_callbacks_dict_is_rejected() -> None:
    run = CONTRASTIVE.run(num_batches=1)
    run.objective.callbacks = {"post_step": ()}
    with pytest.raises(TypeError, match="lifecycle methods"):
        run.trainer()


def test_on_after_forward_hook_receives_output_before_backward() -> None:
    run = CONTRASTIVE.run(num_batches=1)
    observed: list[tuple[bool, bool]] = []

    def hook(output, ctx=None) -> None:
        del ctx
        has_grad = any(p.grad is not None for p in run.objective.parameters())
        observed.append((output.loss.requires_grad, has_grad))

    install_objective_hook(run.objective, "on_after_forward", hook)
    run.trainer().fit()

    assert observed == [(True, False)]


def test_training_forwards_run_in_train_mode_even_after_validation() -> None:
    run = DEFAULT_CASE.run(
        num_batches=4, evaluation=EvaluationConfig(every_n_steps=1, at_train_end=False)
    )
    val_loader = waveform_loader(num_batches=1, batch_size=2, length=DEFAULT_CASE.waveform_length)
    modes: list[tuple[bool, bool, bool]] = []
    forward = run.objective.forward

    def recording_forward(batch):
        modes.append((torch.is_grad_enabled(), run.objective.training, run.model.training))
        return forward(batch)

    run.objective.forward = recording_forward
    run.fit(ValidationCallback(ObjectiveLossEvaluator(run.objective, val_loader)))

    assert [mode for mode in modes if mode[0]] == [(True, True, True)] * 4
    assert [mode for mode in modes if not mode[0]] == [(False, False, False)] * 4


def test_every_epoch_trains_in_train_mode_even_if_left_in_eval_mode() -> None:
    class LeavesEvalMode(Callback):
        def on_train_epoch_end(self, state) -> None:
            state.objective.eval()

    run = DEFAULT_CASE.run(num_batches=2, max_epochs=2)
    run.objective.eval()
    modes: list[bool] = []
    forward = run.objective.forward

    def recording_forward(batch):
        modes.append(run.objective.training and run.model.training)
        return forward(batch)

    run.objective.forward = recording_forward
    run.fit(LeavesEvalMode())

    assert modes == [True] * 4


def test_validation_runs_before_other_callbacks_at_train_end() -> None:
    run = DEFAULT_CASE.run(num_batches=2, evaluation=EvaluationConfig(at_train_end=True))
    recorder = HookRecorder()

    run.trainer(recorder, ValidationCallback(FakeEvaluator())).fit()

    assert recorder.names()[-3:] == ["on_validation_start", "on_validation_end", "on_train_end"]


def test_callback_exceptions_abort_training() -> None:
    class Boom(Callback):
        def on_train_batch_end(self, state, batch, output) -> None:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        DEFAULT_CASE.run(num_batches=2).trainer(Boom()).fit()


def test_train_loader_must_yield_audio_batch() -> None:
    run = DEFAULT_CASE.run(num_batches=1)
    run.loader = DataLoader(
        [torch.zeros(64), torch.zeros(64)],
        batch_size=run.config.trainer.batch.per_device_batch_size,
        drop_last=run.config.trainer.batch.drop_last,
    )

    with pytest.raises(TypeError, match="AudioBatch"):
        run.trainer().fit()


def test_input_seconds_and_grad_norm_telemetry() -> None:
    state = DEFAULT_CASE.run(num_batches=2).trainer().fit()

    assert state.last_input_seconds == 2 * 64 / 16_000
    assert state.last_grad_norm is not None and state.last_grad_norm > 0.0
    assert state.current_lr == pytest.approx(1e-3)


def test_throughput_counts_all_accumulated_micro_batches() -> None:
    run = DEFAULT_CASE.run(num_batches=4, grad_accum=2)

    with patch("audiotrainer.trainer.time.perf_counter", side_effect=[0.0, 1.0]):
        state = run.trainer().fit()

    # The second optimizer step covers two micro-batches of 2 x 64 samples in 1 s.
    assert state.last_throughput_audio_seconds_per_sec == pytest.approx(2 * (2 * 64 / 16_000))


def test_stdout_logger_prints_header_and_step_lines(caplog: pytest.LogCaptureFixture) -> None:
    run = DEFAULT_CASE.run(num_batches=4, stdout_logging=True, log_every_n_steps=2)

    with caplog.at_level(logging.INFO, logger="audiotrainer"):
        state = run.trainer().fit()

    lines = [record.getMessage() for record in caplog.records if record.name == "audiotrainer"]
    start_lines = [line for line in lines if line.startswith("start fingerprint=")]
    assert len(start_lines) == 1
    assert start_lines[0].startswith(f"start fingerprint={state.config_fingerprint[:8]}")
    step_prefixes = [line.split(" loss=")[0] for line in lines if line.startswith("epoch=")]
    assert step_prefixes == [
        "epoch=0 step=2",
        "epoch=0 step=4",
    ]


def test_stdout_logging_disabled_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="audiotrainer"):
        DEFAULT_CASE.run(num_batches=2, stdout_logging=False).trainer().fit()

    assert [r for r in caplog.records if r.name == "audiotrainer"] == []

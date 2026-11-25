"""What ``Trainer.fit`` does to a model is exactly the reference algorithm.

Each test trains two independently-constructed copies of one setup, one with
``Trainer.fit`` and one with :func:`harness.reference_fit`, and requires the
final objective state (encoder, heads, EMA teachers, buffers) and per-step
losses / learning rates to match bit-for-bit on CPU. Any change to the
optimization the trainer performs fails here.
"""

from __future__ import annotations

import pytest
import torch
from audiossl.config import (
    AdamWOptimizerConfig,
    CosineAnnealingSchedulerConfig,
    LARSOptimizerConfig,
    SGDOptimizerConfig,
    StepLRSchedulerConfig,
    WarmupCosineSchedulerConfig,
)

from harness import (
    DEFAULT_CASE,
    ObjectiveCase,
    TrainingTrace,
    accumulation_windows,
    assert_trees_equal,
    install_objective_hook,
    over_objectives,
    reference_fit,
    snapshot,
)


def fit_against_reference(
    case: ObjectiveCase, *, num_batches: int, **overrides
) -> tuple[TrainingTrace, TrainingTrace]:
    trained = case.run(num_batches=num_batches, **overrides)
    reference = case.run(num_batches=num_batches, **overrides)
    initial = snapshot(trained.objective)

    trace = trained.fit()
    expected = reference_fit(reference)

    with pytest.raises(AssertionError):
        assert_trees_equal(snapshot(trained.objective), initial)  # training must move weights
    assert_trees_equal(
        snapshot(trained.objective), snapshot(reference.objective), label="objective"
    )
    assert trace.losses == expected.losses
    assert trace.applied_lrs == expected.applied_lrs
    assert_trees_equal(trace.grad_norms, expected.grad_norms, rtol=1e-6, label="grad_norms")
    return trace, expected


@over_objectives
def test_trainer_matches_reference_for_every_objective(case: ObjectiveCase) -> None:
    if case.name == "mae":
        pytest.importorskip("torchaudio")
    fit_against_reference(
        case,
        num_batches=4,
        max_epochs=2,
        grad_accum=2,
        grad_clip=1.0,
        optimizer=AdamWOptimizerConfig(
            lr=1e-2, weight_decay=0.05, no_weight_decay_patterns=("bias", "norm")
        ),
        scheduler=WarmupCosineSchedulerConfig(warmup_steps=1, min_lr_scale=0.1),
    )


LOOP_VARIANTS = {
    "sgd_constant": dict(num_batches=4, optimizer=SGDOptimizerConfig(lr=0.05)),
    "sgd_nesterov_two_epochs": dict(
        num_batches=4,
        max_epochs=2,
        optimizer=SGDOptimizerConfig(lr=0.05, momentum=0.9, nesterov=True),
    ),
    "adamw_weight_decay_accum3": dict(
        num_batches=6,
        max_epochs=2,
        grad_accum=3,
        optimizer=AdamWOptimizerConfig(lr=1e-2, weight_decay=0.1),
    ),
    "lars_step_lr": dict(
        num_batches=4,
        max_epochs=2,
        optimizer=LARSOptimizerConfig(lr=0.5, momentum=0.9, weight_decay=1e-4, eta=0.02),
        scheduler=StepLRSchedulerConfig(step_size=2, gamma=0.5),
    ),
    "cosine_annealing_accum2": dict(
        num_batches=8,
        grad_accum=2,
        scheduler=CosineAnnealingSchedulerConfig(min_lr_scale=0.1),
    ),
    "warmup_cosine_two_epochs": dict(
        num_batches=4,
        max_epochs=2,
        scheduler=WarmupCosineSchedulerConfig(warmup_steps=3, min_lr_scale=0.0),
    ),
}


@pytest.mark.parametrize("variant", LOOP_VARIANTS.values(), ids=LOOP_VARIANTS.keys())
def test_trainer_matches_reference_across_loop_variants(variant: dict) -> None:
    fit_against_reference(DEFAULT_CASE, **variant)


def test_gradient_clipping_bounds_the_update_and_reports_pre_clip_norm() -> None:
    lr, clip = 0.1, 1e-3
    run = DEFAULT_CASE.run(num_batches=4, optimizer=SGDOptimizerConfig(lr=lr), grad_clip=clip)
    trainable = [p for p in run.objective.parameters() if p.requires_grad]
    weights_before_step: list[torch.Tensor] = []
    update_norms: list[float] = []

    def flat_weights() -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in trainable])

    def before_step(ctx=None) -> None:
        del ctx
        weights_before_step.append(flat_weights())

    def after_step(ctx=None) -> None:
        del ctx
        delta = flat_weights() - weights_before_step[-1]
        update_norms.append(float(torch.linalg.vector_norm(delta)))

    install_objective_hook(run.objective, "on_before_optimizer_step", before_step)
    install_objective_hook(run.objective, "on_after_optimizer_step", after_step)
    trace = run.fit()

    assert min(trace.grad_norms.values()) > 10 * clip, "clip must actually engage"
    # Plain SGD: |delta theta| = lr * min(1, clip / |g|) * |g| = lr * clip.
    assert update_norms == pytest.approx([lr * clip] * len(update_norms), rel=1e-3)
    fit_against_reference(
        DEFAULT_CASE, num_batches=4, optimizer=SGDOptimizerConfig(lr=lr), grad_clip=clip
    )


def test_applied_learning_rate_follows_schedule_over_whole_run() -> None:
    base_lr = 1e-2
    run = DEFAULT_CASE.run(
        num_batches=6,
        max_epochs=3,
        grad_accum=2,
        optimizer=AdamWOptimizerConfig(lr=base_lr),
        scheduler=WarmupCosineSchedulerConfig(warmup_steps=2, min_lr_scale=0.1),
    )
    trace = run.fit()

    # 6 micro-batches / accum 2 = 3 steps per epoch, 9 in total: one LR per optimizer step.
    assert sorted(trace.applied_lrs) == list(range(1, 10))
    assert trace.applied_lrs[1] == pytest.approx(base_lr * 0.5)  # warmup 1/2
    assert trace.applied_lrs[2] == pytest.approx(base_lr)  # warmup 2/2
    assert trace.applied_lrs[9] == pytest.approx(base_lr * 0.1)  # horizon reached exactly
    decay = [trace.applied_lrs[step] for step in range(3, 10)]
    assert decay == sorted(decay, reverse=True)


@over_objectives
def test_optimizer_never_updates_frozen_parameters(case: ObjectiveCase) -> None:
    if case.name == "mae":
        pytest.importorskip("torchaudio")
    run = case.run(num_batches=2, optimizer=SGDOptimizerConfig(lr=0.5))
    frozen = {name: p for name, p in run.objective.named_parameters() if not p.requires_grad}
    trainer = run.trainer()
    optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert not optimized & {id(p) for p in frozen.values()}

    before: dict[str, torch.Tensor] = {}
    changed_by_optimizer: list[str] = []

    def snapshot_frozen(ctx=None) -> None:
        del ctx
        before.update({name: p.detach().clone() for name, p in frozen.items()})

    def compare_frozen_before_objective_hooks(ctx=None) -> None:
        del ctx
        changed_by_optimizer.extend(
            name for name, p in frozen.items() if not torch.equal(p, before[name])
        )

    install_objective_hook(run.objective, "on_before_optimizer_step", snapshot_frozen, first=False)
    install_objective_hook(
        run.objective, "on_after_optimizer_step", compare_frozen_before_objective_hooks
    )
    trainer.fit()

    assert changed_by_optimizer == []


def test_optimizer_step_hooks_bracket_the_step() -> None:
    run = DEFAULT_CASE.run(num_batches=4, grad_accum=2)
    trainable = [p for p in run.objective.parameters() if p.requires_grad]
    observations: list[tuple[str, bool, torch.Tensor]] = []

    def observe(name: str):
        def hook(ctx=None) -> None:
            del ctx
            has_grads = any(p.grad is not None for p in trainable)
            flat = torch.cat([p.detach().flatten() for p in trainable]).clone()
            observations.append((name, has_grads, flat))

        return hook

    install_objective_hook(run.objective, "on_before_optimizer_step", observe("before"))
    install_objective_hook(run.objective, "on_after_optimizer_step", observe("after"))
    run.fit()

    assert [name for name, _, _ in observations] == ["before", "after"] * 2
    for (_, pre_grads, pre_weights), (_, post_grads, post_weights) in zip(
        observations[::2], observations[1::2], strict=True
    ):
        assert pre_grads, "on_before_optimizer_step must run after backward, before the step"
        assert not post_grads, "on_after_optimizer_step must run after zero_grad"
        assert not torch.equal(pre_weights, post_weights), (
            "on_after_optimizer_step must run after the step"
        )


@over_objectives
def test_every_trainable_parameter_receives_a_gradient(case: ObjectiveCase) -> None:
    if case.name == "mae":
        pytest.importorskip("torchaudio")
    run = case.run(num_batches=2, max_epochs=1)
    named = [(name, p) for name, p in run.objective.named_parameters() if p.requires_grad]
    received: set[str] = set()

    def collect(ctx=None) -> None:
        del ctx
        received.update(name for name, p in named if p.grad is not None)

    install_objective_hook(run.objective, "on_before_optimizer_step", collect)
    run.fit()

    assert sorted(name for name, _ in named if name not in received) == []


def test_short_trailing_accumulation_window_averages_over_its_own_micro_batches() -> None:
    fit_against_reference(
        DEFAULT_CASE,
        num_batches=5,
        grad_accum=2,
        optimizer=SGDOptimizerConfig(lr=0.1),
    )


def test_reference_windows_cover_every_batch_once_with_a_short_tail() -> None:
    windows = accumulation_windows(5, 2)

    assert [list(window) for window in windows] == [[0, 1], [2, 3], [4]]

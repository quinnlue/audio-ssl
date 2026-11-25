"""Reproducibility guarantees: what a run's numbers are a function of.

A result is reproducible when it is a function of (config, runtime seed,
injected generator/augmentor seeds, data) and nothing else. These tests pin
that down on CPU, bitwise:

- construction is a function of ``RuntimeConfig.seed`` only, and does not
  consume or depend on the caller's global RNG;
- training-time randomness flows only through injected generators;
- ``state_dict`` captures everything needed to reproduce outputs;
- a checkpoint/resume cycle is bitwise identical to an uninterrupted run.
"""

from __future__ import annotations

import pytest
import torch

from audiotrainer.rng import objective_generator_states, restore_objective_generators

from testkit import checks, training
from testkit.cases import OBJECTIVE_CASES, model_params, objective_params


def _digest(module: torch.nn.Module) -> str:
    return checks.digest(module.state_dict())


# ----------------------------------------------------------------- models ---


@pytest.mark.parametrize("case", model_params("seeded_construction"))
def test_model_construction_is_a_function_of_runtime_seed_only(case):
    torch.manual_seed(1)
    first = case.build(seed=0)
    torch.manual_seed(2)
    torch.rand(17)
    second = case.build(seed=0)
    assert _digest(first) == _digest(second)
    assert _digest(case.build(seed=1)) != _digest(first)


@pytest.mark.parametrize("case", model_params("rng_isolation"))
def test_model_construction_does_not_consume_global_rng(case):
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    case.build(seed=0)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("case", model_params("state_dict_roundtrip"))
def test_model_state_dict_reproduces_outputs(case):
    source = case.build(seed=0).eval()
    target = case.build(seed=1).eval()
    target.load_state_dict(source.state_dict(), strict=True)
    batch = case.batch()
    with torch.no_grad():
        checks.assert_identical(target(batch), source(batch), what="reloaded model")


# ------------------------------------------------------------- objectives ---


@pytest.mark.parametrize("case", objective_params("seeded_construction"))
def test_objective_construction_is_a_function_of_runtime_seed_only(case):
    torch.manual_seed(1)
    first = case.build(seed=0)
    torch.manual_seed(2)
    torch.rand(17)
    second = case.build(seed=0)
    assert _digest(first) == _digest(second)


@pytest.mark.parametrize("case", objective_params("rng_isolation"))
def test_objective_construction_does_not_consume_global_rng(case):
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    case.build(seed=0)
    assert torch.equal(torch.random.get_rng_state(), before)


def _loss_and_grads(objective, batch):
    out = objective(batch)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in objective.named_parameters() if p.grad is not None}
    return out.loss.detach(), grads


@pytest.mark.parametrize("case", objective_params("injected_rng_determinism"))
def test_training_step_is_a_function_of_injected_seeds(case):
    """Same seeds give bitwise-identical loss and gradients regardless of global RNG."""
    batch = case.batch()
    torch.manual_seed(1)
    first = _loss_and_grads(case.build().train(), batch)
    torch.manual_seed(2)
    torch.rand(5)
    second = _loss_and_grads(case.build().train(), batch)
    checks.assert_identical(second, first, what="step under different global RNG")


@pytest.mark.parametrize(
    "case",
    objective_params(
        "injected_rng_is_used",
        [c for c in OBJECTIVE_CASES if c.uses_generator or c.uses_augmentor],
    ),
)
def test_injected_seeds_actually_drive_randomness(case):
    batch = case.batch()
    base = case.build(generator_seed=0, augmentor_seed=0).train()(batch).loss
    changed = case.build(generator_seed=1, augmentor_seed=1).train()(batch).loss
    assert not torch.equal(base, changed), "objective ignores its injected RNG"


@pytest.mark.parametrize("case", objective_params("resume"))
def test_checkpoint_resume_is_bitwise_identical_to_uninterrupted_training(case, tmp_path):
    batches = [case.batch(seed=s) for s in range(4)]

    # Trainers seed the global RNG too (dropout, Gumbel noise); resume must restore it.
    straight = case.build(seed=0).train()
    straight_opt = training.optimizer_for(straight)
    torch.manual_seed(0)
    straight_losses = [training.train_step(straight, b, straight_opt) for b in batches]
    straight_rng = torch.random.get_rng_state()

    first_half = case.build(seed=0).train()
    first_opt = training.optimizer_for(first_half)
    torch.manual_seed(0)
    losses = [training.train_step(first_half, b, first_opt) for b in batches[:2]]
    rng = objective_generator_states(first_half)
    expected_rngs = {}
    if case.uses_generator:
        expected_rngs["generator"] = first_half.generator.get_state()
    if case.uses_augmentor:
        expected_rngs["augmentor.generator"] = first_half.augmentor.generator.get_state()
    checks.assert_identical(rng, expected_rngs, what="registered generators")
    path = tmp_path / "objective.pt"
    torch.save({
        "objective": first_half.state_dict(),
        "optimizer": first_opt.state_dict(),
        "generators": rng,
        "torch_rng": torch.random.get_rng_state(),
    }, path)

    checkpoint = torch.load(path, weights_only=True)
    resumed = case.build(seed=3, generator_seed=7, augmentor_seed=7).train()
    resumed.load_state_dict(checkpoint["objective"], strict=True)
    resumed_opt = training.optimizer_for(resumed)
    resumed_opt.load_state_dict(checkpoint["optimizer"])
    restore_objective_generators(resumed, checkpoint["generators"])
    torch.random.set_rng_state(checkpoint["torch_rng"])
    losses += [training.train_step(resumed, b, resumed_opt) for b in batches[2:]]

    checks.assert_identical(losses, straight_losses, what="loss trajectory")
    checks.assert_identical(resumed.state_dict(), straight.state_dict(), what="final state")
    checks.assert_identical(resumed_opt.state_dict(), straight_opt.state_dict(), what="optimizer")
    checks.assert_identical(torch.random.get_rng_state(), straight_rng, what="global RNG")
    checks.assert_identical(
        objective_generator_states(resumed), objective_generator_states(straight), what="generators"
    )

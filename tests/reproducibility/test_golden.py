"""Numerical fingerprints for every case, compared against ``tests/golden/*.json``.

A golden mismatch means the numbers a given config + seed produces have
changed: initialization order, RNG consumption, or modelling math moved.
Sometimes that is intended; then re-baseline with ``--update-golden`` and
commit the JSON diff so the change is reviewed and attributable.
"""

from __future__ import annotations

import pytest
import torch

from testkit import checks, training
from testkit.cases import model_params, objective_params
from testkit.golden import GoldenStore

TRAIN_STEPS = 3


@pytest.fixture(scope="module", autouse=True)
def _golden_cpu_threads():
    """Use the reduction order of the checked-in CPU baseline."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def model_goldens(request) -> GoldenStore:
    return GoldenStore("models", update=request.config.getoption("--update-golden"))


@pytest.fixture(scope="module")
def objective_goldens(request) -> GoldenStore:
    return GoldenStore("objectives", update=request.config.getoption("--update-golden"))


def _param_stats(module: torch.nn.Module) -> dict[str, float]:
    flat = torch.cat([p.detach().flatten() for p in module.parameters()])
    return checks.stats(flat)


@pytest.mark.parametrize("case", model_params("golden_forward"))
def test_model_fingerprint(case, model_goldens):
    model = case.build(seed=0).eval()
    with torch.no_grad():
        out = model(case.batch(seed=0))
    model_goldens.check(
        case.id,
        {
            "n_params": model.n_params,
            "params": _param_stats(model),
            "z_valid": checks.stats(out.z[out.mask]),
            "cls": checks.stats(out.cls),
            "lengths": out.lengths.tolist(),
        },
    )


@pytest.mark.parametrize("case", objective_params("golden_training"))
def test_objective_fingerprint(case, objective_goldens):
    objective = case.build(seed=0).train()
    optimizer = training.optimizer_for(objective)
    torch.manual_seed(0)
    losses = [
        float(training.train_step(objective, case.batch(seed=step), optimizer))
        for step in range(TRAIN_STEPS)
    ]
    objective_goldens.check(
        case.id,
        {
            "n_params": sum(p.numel() for p in objective.parameters()),
            "init_params": _param_stats(case.build(seed=0)),
            "losses": losses,
        },
    )

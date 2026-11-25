"""Suite-wide hygiene: every test starts from the same global state and leaves it intact.

Modelling code legitimately touches process-global state (RNG streams,
deterministic-algorithm flags, default dtype, mutable registries). Without
isolation, test outcomes depend on execution order, which silently breaks
reproducibility claims. The autouse fixture below snapshots that state,
seeds every RNG to 0, and restores the snapshot afterwards.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from audiossl.config import registry as config_registry
from audiossl.modelling import registry as modelling_registry

from testkit.precision import FP32, FP64, Precision

_INTEGRATIONS = Path(__file__).resolve().parent / "integrations"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/golden/*.json entries exercised by this run.",
    )
    group = parser.getgroup("training parity")
    for field in ("rtol", "atol", "eps"):
        group.addoption(f"--parity-{field}", type=float, help=f"Override parity {field}.")


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def parity_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.fixture
def parity_precision(request: pytest.FixtureRequest) -> Precision:
    precision = FP32
    if "parity_dtype" in request.fixturenames:
        if request.getfixturevalue("parity_dtype") == torch.float64:
            precision = FP64
    overrides = {
        name: value
        for name in ("rtol", "atol", "eps")
        if (value := request.config.getoption(f"--parity-{name}")) is not None
    }
    for name, value in overrides.items():
        if not 0 <= value < float("inf") or (name == "eps" and value == 0):
            bound = "positive" if name == "eps" else "nonnegative"
            raise pytest.UsageError(f"--parity-{name} must be finite and {bound}")
    return replace(precision, **overrides)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if _INTEGRATIONS in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def _isolated_global_state() -> Iterator[None]:
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    default_dtype = torch.get_default_dtype()
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    activations = dict(modelling_registry.ACTIVATION_REGISTRY)
    initializations = dict(modelling_registry.INITIALIZATION_REGISTRY)
    components = [
        (registry, registry.snapshot()) for registry in modelling_registry.ALL_COMPONENT_REGISTRIES
    ]
    configs = config_registry.snapshot()

    random.seed(0)
    torch.manual_seed(0)
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.set_default_dtype(default_dtype)
        if cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas
        modelling_registry.ACTIVATION_REGISTRY.clear()
        modelling_registry.ACTIVATION_REGISTRY.update(activations)
        modelling_registry.INITIALIZATION_REGISTRY.clear()
        modelling_registry.INITIALIZATION_REGISTRY.update(initializations)
        for registry, state in components:
            registry.restore(state)
        config_registry.restore(configs)


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))

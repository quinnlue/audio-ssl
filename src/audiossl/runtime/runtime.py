"""Resolve and apply ``RuntimeConfig`` to devices, dtypes, and global RNG state."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from audiossl.config import RuntimeConfig


_SUPPORTED_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})


@dataclass(frozen=True)
class AppliedRuntimeConfig:
    """Resolved runtime settings after validating a ``RuntimeConfig``.

    Captures both the requested string values and the resolved ``torch.device``
    and ``torch.dtype`` objects. The ``global_*`` and ``deterministic_*`` flags
    record whether global process state was modified during configuration.
    """

    seed: int
    requested_device: str
    device: torch.device
    requested_dtype: str
    dtype: torch.dtype
    deterministic: bool
    deterministic_algorithms_enabled: bool
    cudnn_deterministic: bool | None
    cudnn_benchmark: bool | None
    cublas_workspace_config: str | None
    global_runtime_applied: bool = False
    global_rng_seeded: bool = False
    deterministic_settings_applied: bool = False


@dataclass(frozen=True)
class _ResolvedRuntimeConfig:
    seed: int
    device: torch.device
    dtype: torch.dtype
    deterministic: bool


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            "RuntimeConfig.seed must be a non-negative integer, "
            f"got {seed!r} ({type(seed).__name__})"
        )
    if seed < 0 or seed > 2**63 - 1:
        raise ValueError(
            "RuntimeConfig.seed must be in the range [0, 2**63 - 1], "
            f"got {seed}"
        )
    return seed


def resolve_runtime_dtype(dtype: str) -> torch.dtype:
    """Map a supported ``RuntimeConfig.dtype`` string to a ``torch.dtype``."""
    if not isinstance(dtype, str):
        raise ValueError(
            "RuntimeConfig.dtype must be one of "
            f"{sorted(_SUPPORTED_DTYPES)}, got {type(dtype).__name__}"
        )
    try:
        return _SUPPORTED_DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(
            "Unsupported RuntimeConfig.dtype "
            f"{dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        ) from exc


def resolve_runtime_device(device: str) -> torch.device:
    """Resolve and validate a ``RuntimeConfig.device`` string."""
    if not isinstance(device, str):
        raise ValueError(
            "RuntimeConfig.device must be a torch device string, "
            f"got {type(device).__name__}"
        )

    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid RuntimeConfig.device {device!r}; expected values like "
            "'cpu', 'cuda', or 'cuda:0'"
        ) from exc

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"RuntimeConfig.device={device!r} requested CUDA, but CUDA is not available"
            )
        device_count = torch.cuda.device_count()
        if resolved.index is not None and resolved.index >= device_count:
            raise RuntimeError(
                f"RuntimeConfig.device={device!r} requested CUDA device index "
                f"{resolved.index}, but only {device_count} CUDA device(s) are available"
            )
    elif resolved.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                f"RuntimeConfig.device={device!r} requested MPS, but MPS is not available"
            )
    elif resolved.type != "cpu":
        raise RuntimeError(
            f"Unsupported RuntimeConfig.device type {resolved.type!r}; "
            "supported device types are 'cpu', 'cuda', and 'mps'"
        )

    return resolved


def _validate_deterministic(deterministic: bool) -> bool:
    if not isinstance(deterministic, bool):
        raise ValueError(
            "RuntimeConfig.deterministic must be a bool, "
            f"got {deterministic!r} ({type(deterministic).__name__})"
        )
    return deterministic


def _resolve_runtime_config(runtime: RuntimeConfig) -> _ResolvedRuntimeConfig:
    if not isinstance(runtime, RuntimeConfig):
        raise TypeError(
            "AudioModelConfig.runtime must be a RuntimeConfig, "
            f"got {type(runtime).__name__}"
        )
    return _ResolvedRuntimeConfig(
        seed=_validate_seed(runtime.seed),
        device=resolve_runtime_device(runtime.device),
        dtype=resolve_runtime_dtype(runtime.dtype),
        deterministic=_validate_deterministic(runtime.deterministic),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_tf32(allow_tf32: bool) -> None:
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
        cuda_backend.matmul.allow_tf32 = allow_tf32
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = allow_tf32
    setter = getattr(torch, "set_float32_matmul_precision", None)
    if callable(setter):
        setter("high" if allow_tf32 else "highest")


def _configure_determinism(deterministic: bool, device: torch.device) -> bool:
    """Enable PyTorch deterministic algorithms and required cuBLAS workspace settings."""
    if not deterministic:
        return False

    if device.type == "cuda":
        existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if existing is not None and existing not in _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
            raise RuntimeError(
                "RuntimeConfig.deterministic=True requires CUBLAS_WORKSPACE_CONFIG "
                f"to be one of {sorted(_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS)}, "
                f"got {existing!r}"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Deterministic runs must not use TF32.
        _configure_tf32(False)

    torch.use_deterministic_algorithms(True)

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.deterministic = True
        cudnn_backend.benchmark = False
    return True


def _applied_runtime_config(
    runtime: RuntimeConfig,
    resolved: _ResolvedRuntimeConfig,
    *,
    global_runtime_applied: bool,
    global_rng_seeded: bool,
    deterministic_settings_applied: bool,
) -> AppliedRuntimeConfig:
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    return AppliedRuntimeConfig(
        seed=resolved.seed,
        requested_device=runtime.device,
        device=resolved.device,
        requested_dtype=runtime.dtype,
        dtype=resolved.dtype,
        deterministic=resolved.deterministic,
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=(
            bool(cudnn_backend.deterministic) if cudnn_backend is not None else None
        ),
        cudnn_benchmark=bool(cudnn_backend.benchmark) if cudnn_backend is not None else None,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        global_runtime_applied=global_runtime_applied,
        global_rng_seeded=global_rng_seeded,
        deterministic_settings_applied=deterministic_settings_applied,
    )


def configure_runtime(
    runtime: RuntimeConfig,
    *,
    apply_global: bool = False,
) -> AppliedRuntimeConfig:
    """Validate ``runtime`` and optionally apply global RNG and determinism settings.

    Parameters
    ----------
    runtime
        Requested device, dtype, seed, and determinism policy.
    apply_global
        When true, seed global RNG state and configure deterministic algorithms.

    Returns
    -------
    AppliedRuntimeConfig
        Resolved runtime settings plus flags describing any global side effects.
    """
    resolved = _resolve_runtime_config(runtime)
    deterministic_settings_applied = False
    global_rng_seeded = False

    if apply_global:
        if not resolved.deterministic:
            _configure_tf32(bool(runtime.allow_tf32))
        deterministic_settings_applied = _configure_determinism(
            resolved.deterministic,
            resolved.device,
        )
        _seed_everything(resolved.seed)
        global_rng_seeded = True

    return _applied_runtime_config(
        runtime,
        resolved,
        global_runtime_applied=apply_global,
        global_rng_seeded=global_rng_seeded,
        deterministic_settings_applied=deterministic_settings_applied,
    )


def apply_global_runtime(runtime: RuntimeConfig) -> AppliedRuntimeConfig:
    """Configure runtime settings and apply global RNG and determinism state."""
    return configure_runtime(runtime, apply_global=True)


def _runtime_seed(runtime: RuntimeConfig | AppliedRuntimeConfig) -> int:
    if isinstance(runtime, AppliedRuntimeConfig):
        return runtime.seed
    if isinstance(runtime, RuntimeConfig):
        return _resolve_runtime_config(runtime).seed
    raise TypeError(
        "runtime_seed_scope expected RuntimeConfig or AppliedRuntimeConfig, "
        f"got {type(runtime).__name__}"
    )


def _cuda_rng_devices() -> list[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def derive_seed(seed: int, component: str) -> int:
    """Stable per-component seed: ``sha256(f"{seed}/{component}")`` folded to 63 bits."""
    if not isinstance(component, str) or not component:
        raise ValueError(f"derive_seed expected a non-empty component name, got {component!r}")
    digest = hashlib.sha256(f"{seed}/{component}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


@contextmanager
def runtime_seed_scope(
    runtime: RuntimeConfig | AppliedRuntimeConfig,
    component: str | None = None,
) -> Iterator[None]:
    """Temporarily fork RNG state and re-seed from ``runtime`` inside the block.

    Parameters
    ----------
    runtime
        Runtime config whose validated seed is applied for the scope duration.
    component
        Optional component name. When given, the scope is seeded with
        :func:`derive_seed` so each component's initialization is independent
        of every other component's parameter count and construction order.
    """
    seed = _runtime_seed(runtime)
    if component is not None:
        seed = derive_seed(seed, component)
    python_rng_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=_cuda_rng_devices()):
            random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_rng_state)


def apply_runtime_to_module(
    module: nn.Module,
    applied_runtime: AppliedRuntimeConfig,
) -> nn.Module:
    """Move ``module`` to the resolved runtime device and dtype."""
    if not isinstance(module, nn.Module):
        raise TypeError(
            "apply_runtime_to_module expected torch.nn.Module, "
            f"got {type(module).__name__}"
        )
    return module.to(device=applied_runtime.device, dtype=applied_runtime.dtype)

"""Assertion and fingerprint helpers shared by the contract suites."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn


def assert_identical(actual: Any, expected: Any, *, what: str = "value") -> None:
    """Compare every leaf and the structure of nested training state."""
    if not (isinstance(actual, Mapping) and isinstance(expected, Mapping)):
        assert type(actual) is type(expected), f"{what}: {type(actual)} != {type(expected)}"
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} != {expected.dtype}"
        assert actual.shape == expected.shape, f"{what}: {actual.shape} != {expected.shape}"
        assert actual.device == expected.device, f"{what}: {actual.device} != {expected.device}"
        assert torch.equal(actual, expected), f"{what}: tensor values differ"
    elif dataclasses.is_dataclass(actual):
        for field in dataclasses.fields(actual):
            assert_identical(
                getattr(actual, field.name), getattr(expected, field.name),
                what=f"{what}.{field.name}",
            )
    elif isinstance(actual, Mapping):
        assert actual.keys() == expected.keys(), f"{what}: {actual.keys() ^ expected.keys()}"
        for key, item in actual.items():
            assert_identical(item, expected[key], what=f"{what}[{key!r}]")
    elif isinstance(actual, list | tuple):
        assert len(actual) == len(expected), f"{what}: {len(actual)} != {len(expected)}"
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_identical(a, e, what=f"{what}[{index}]")
    else:
        assert actual == expected, f"{what}: {actual!r} != {expected!r}"


def trainable(module: nn.Module) -> dict[str, nn.Parameter]:
    return {name: p for name, p in module.named_parameters() if p.requires_grad}


def frozen(module: nn.Module) -> dict[str, nn.Parameter]:
    return {name: p for name, p in module.named_parameters() if not p.requires_grad}


def missing_grads(params: Mapping[str, nn.Parameter]) -> list[str]:
    return sorted(name for name, p in params.items() if p.grad is None)


def nonfinite_grads(params: Mapping[str, nn.Parameter]) -> list[str]:
    return sorted(
        name
        for name, p in params.items()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    )


def zero_grads(params: Mapping[str, nn.Parameter]) -> list[str]:
    return sorted(
        name for name, p in params.items() if p.grad is not None and not p.grad.abs().sum() > 0
    )


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """Detached clone of every parameter and buffer, keyed by ``state_dict`` name."""
    return {name: t.detach().clone() for name, t in module.state_dict(keep_vars=True).items()}


def changed(before: Mapping[str, torch.Tensor], module: nn.Module) -> list[str]:
    after = module.state_dict(keep_vars=True)
    return sorted(name for name, t in before.items() if not torch.equal(t, after[name].detach()))


def digest(tensors: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]]) -> str:
    """Stable SHA-256 over names, dtypes, shapes, and raw bytes."""
    items = tensors.items() if isinstance(tensors, Mapping) else tensors
    h = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda kv: kv[0]):
        t = tensor.detach().cpu().contiguous().reshape(-1)
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def stats(tensor: torch.Tensor) -> dict[str, float]:
    """Summary statistics robust enough to compare across BLAS builds with tolerance."""
    flat = tensor.detach().double().cpu().flatten()
    return {
        "sum": float(flat.sum()),
        "abs_sum": float(flat.abs().sum()),
        "sq_sum": float((flat * flat).sum()),
        "first": float(flat[0]) if flat.numel() else 0.0,
        "last": float(flat[-1]) if flat.numel() else 0.0,
    }

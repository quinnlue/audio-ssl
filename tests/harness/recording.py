"""Observers and comparison helpers for trainer tests.

The observers here are read-only ``Callback`` subclasses: they must never
mutate ``TrainerState`` or consume RNG, so attaching them cannot change what is
being measured (``test_trainer_reproducibility`` checks this).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
import torch
import torch.nn as nn
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives import OBJECTIVE_HOOKS

from audiotrainer import Callback, TrainerState

OBJECTIVE_HOOK_NAMES = OBJECTIVE_HOOKS


def known_bug(reason: str) -> pytest.MarkDecorator:
    """Mark a test that encodes correct behavior the trainer does not yet have.

    Strict, and limited to ``AssertionError`` so that an unrelated crash is not
    mistaken for the documented failure. When the bug is fixed the test starts
    passing, strict mode fails the suite, and the marker must be removed.
    """

    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"KNOWN BUG: {reason}")


class TrainingTrace(Callback):
    """Record what the trainer did, keyed by step number so traces from a
    straight run and a resumed run can be compared on their overlap.

    * ``losses[global_step]`` is the unscaled micro-batch loss.
    * ``applied_lrs[k]`` is the param-group-0 learning rate used by the k-th
      optimizer step (1-based).
    * ``grad_norms[k]`` is the pre-clip global gradient norm at the k-th step.
    * ``epoch_end_losses[epoch]`` is ``state.last_loss`` at epoch end.
    """

    def __init__(self) -> None:
        self.losses: dict[int, float] = {}
        self.applied_lrs: dict[int, float] = {}
        self.grad_norms: dict[int, float] = {}
        self.epoch_end_losses: dict[int, float] = {}

    def on_train_batch_start(self, state: TrainerState, batch: Any) -> None:
        self.applied_lrs.setdefault(
            state.optimizer_step + 1, float(state.optimizer.param_groups[0]["lr"])
        )

    def on_train_batch_end(self, state: TrainerState, batch: Any, output: Any) -> None:
        if output is not None:
            self.losses[state.global_step] = float(output.loss.detach().item())

    def on_optimizer_step(self, state: TrainerState, ctx) -> None:
        del ctx
        step = state.optimizer_step
        if state.last_grad_norm is not None:
            self.grad_norms[step] = state.last_grad_norm

    def on_train_epoch_end(self, state: TrainerState) -> None:
        self.epoch_end_losses[state.epoch] = state.last_loss


class HookRecorder(Callback):
    """Append ``(hook, epoch, global_step, optimizer_step)`` for every trainer hook.

    Pass a shared ``events`` list to interleave with :func:`spy_objective_hooks`.
    """

    def __init__(self, events: list[tuple] | None = None) -> None:
        self.events: list[tuple] = events if events is not None else []

    def _push(self, name: str, state: TrainerState) -> None:
        self.events.append((name, state.epoch, state.global_step, state.optimizer_step))

    def on_train_start(self, state):
        self._push("on_train_start", state)

    def on_train_epoch_start(self, state):
        self._push("on_train_epoch_start", state)

    def on_train_batch_start(self, state, batch):
        self._push("on_train_batch_start", state)

    def on_train_batch_end(self, state, batch, output):
        self._push("on_train_batch_end", state)

    def on_train_epoch_end(self, state):
        self._push("on_train_epoch_end", state)

    def on_train_end(self, state):
        self._push("on_train_end", state)

    def on_validation_start(self, state):
        self._push("on_validation_start", state)

    def on_validation_end(self, state):
        self._push("on_validation_end", state)

    def on_checkpoint_written(self, state, path):
        self.events.append(("on_checkpoint_written", path.name))

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


class FakeEvaluator:
    """Evaluator returning a scripted sequence of losses; consumes no RNG."""

    def __init__(self, values: Sequence[float] = (1.0,)) -> None:
        self.values = list(values)
        self.steps: list[int] = []

    @property
    def calls(self) -> int:
        return len(self.steps)

    def evaluate(self, model: AudioModel, state: TrainerState) -> dict[str, float]:
        self.steps.append(state.optimizer_step)
        value = self.values[min(self.calls - 1, len(self.values) - 1)]
        return {"loss": value, "custom": value + 1.0}


def install_objective_hook(
    objective: nn.Module,
    name: str,
    hook: Callable[..., None],
    *,
    first: bool = True,
) -> None:
    """Wrap the objective's lifecycle method ``name`` so ``hook`` also runs.

    The original method still runs; ``first`` controls whether ``hook``
    observes the call before or after it.
    """

    if name not in OBJECTIVE_HOOKS:
        raise ValueError(f"unknown objective hook {name!r}")
    original = getattr(objective, name)

    def wrapped(*args: Any, **kwargs: Any) -> None:
        if first:
            hook(*args, **kwargs)
        original(*args, **kwargs)
        if not first:
            hook(*args, **kwargs)

    setattr(objective, name, wrapped)


def spy_objective_hooks(objective: nn.Module, events: list[tuple]) -> None:
    """Record every objective-side hook invocation as ``("objective:<name>",)``."""

    for name in OBJECTIVE_HOOK_NAMES:

        def spy(*_args, _name=name):
            events.append((f"objective:{_name}",))

        install_objective_hook(objective, name, spy)


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_trees_equal(
    actual: Any,
    expected: Any,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    label: str = "value",
) -> None:
    """Assert two nested dict/list/tensor/scalar trees are equal.

    Exact (bitwise for tensors) by default. Reports every mismatching path with
    its max absolute difference rather than stopping at the first.
    """

    mismatches: list[str] = []
    _compare(actual, expected, label, atol, rtol, mismatches)
    if mismatches:
        shown = "\n  ".join(mismatches[:15])
        more = f"\n  ... and {len(mismatches) - 15} more" if len(mismatches) > 15 else ""
        raise AssertionError(f"{len(mismatches)} mismatch(es):\n  {shown}{more}")


def _compare(actual, expected, path, atol, rtol, out: list[str]) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            out.append(f"{path}: expected tensor, got {type(actual).__name__}")
            return
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            out.append(
                f"{path}: shape/dtype {tuple(actual.shape)}/{actual.dtype} "
                f"!= {tuple(expected.shape)}/{expected.dtype}"
            )
            return
        a = actual.detach().cpu()
        e = expected.detach().cpu()
        exact = atol == 0.0 and rtol == 0.0
        equal = torch.equal(a, e) if exact else torch.allclose(a, e, atol=atol, rtol=rtol)
        if not equal:
            diff = (a.double() - e.double()).abs().max().item() if a.numel() else 0.0
            out.append(f"{path}: max |diff| = {diff:.3e}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            out.append(f"{path}: expected mapping, got {type(actual).__name__}")
            return
        if set(actual) != set(expected):
            out.append(
                f"{path}: keys differ; missing={sorted(map(str, set(expected) - set(actual)))} "
                f"extra={sorted(map(str, set(actual) - set(expected)))}"
            )
        for key in expected:
            if key in actual:
                _compare(actual[key], expected[key], f"{path}[{key!r}]", atol, rtol, out)
        return
    if isinstance(expected, list | tuple):
        if not isinstance(actual, list | tuple) or len(actual) != len(expected):
            out.append(f"{path}: sequence length/type differs")
            return
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, e, f"{path}[{index}]", atol, rtol, out)
        return
    if isinstance(expected, float) and isinstance(actual, float) and (atol or rtol):
        if abs(actual - expected) > atol + rtol * abs(expected):
            out.append(f"{path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        out.append(f"{path}: {actual!r} != {expected!r}")


__all__ = [
    "FakeEvaluator",
    "HookRecorder",
    "OBJECTIVE_HOOK_NAMES",
    "TrainingTrace",
    "assert_trees_equal",
    "install_objective_hook",
    "known_bug",
    "snapshot",
    "spy_objective_hooks",
]

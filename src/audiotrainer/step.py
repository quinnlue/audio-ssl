"""Read-only step context passed to objective lifecycle hooks and metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepContext:
    """Counters visible to objective hooks at a single lifecycle call site.

    ``step_was_skipped`` is True when a GradScaler refused the optimizer step
    (or a nonfinite policy skipped it); EMA teachers and horizon-dependent
    schedules must not advance in that case.
    """

    epoch: int
    global_step: int
    optimizer_step: int
    total_optimizer_steps: int
    step_was_skipped: bool = False


__all__ = ["StepContext"]

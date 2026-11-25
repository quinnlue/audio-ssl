"""wav2vec 2.0-specific behaviour beyond the generic objective contract."""

from __future__ import annotations

from audiotrainer.step import StepContext
from testkit.cases import objective_case


def _ctx(step: int = 0) -> StepContext:
    return StepContext(
        optimizer_step=step,
        total_optimizer_steps=200,
        epoch=0,
        global_step=step,
        step_was_skipped=False,
    )


def test_after_optimizer_step_advances_quantizer_temperature_schedule():
    objective = objective_case("wav2vec2").build()
    scheduler = objective.quantizer.temperature_scheduler
    assert scheduler.current_value() == 2.0
    objective.on_after_optimizer_step(_ctx(0))
    assert scheduler.current_value() == 2.0 * 0.8
    for i in range(100):
        objective.on_after_optimizer_step(_ctx(i + 1))
    assert scheduler.current_value() == 0.5, "temperature must clamp at the configured end"

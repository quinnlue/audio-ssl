"""State changes at the objective's optimizer-step boundary."""

from dataclasses import replace

import pytest

from audiotrainer.step import StepContext

from testkit import checks, training
from testkit.cases import objective_params


@pytest.mark.parametrize("case", objective_params("optimizer_step_hook_scope"))
def test_after_optimizer_step_hook_never_touches_student_parameters(case):
    objective = case.build().train()
    optimizer = training.optimizer_for(objective)
    training.reseed(objective, 0)
    training.train_step(objective, case.batch(), optimizer)

    before = checks.snapshot(objective)
    context = StepContext(
        epoch=0, global_step=2, optimizer_step=1, total_optimizer_steps=3,
        step_was_skipped=True,
    )
    objective.on_after_optimizer_step(context)
    assert checks.changed(before, objective) == [], "skipped step advanced objective state"
    objective.on_after_optimizer_step(replace(context, step_was_skipped=False))
    changed = set(checks.changed(before, objective))
    assert not changed & set(checks.trainable(objective)), (
        "on_after_optimizer_step modified student params"
    )
    if case.teacher == "ema":
        teacher_roots = tuple(f"{t}." for t, _ in case.teacher_pairs)
        assert changed, "EMA on_after_optimizer_step did not move the teacher"
        assert all(name.startswith(teacher_roots) for name in changed), sorted(changed)

"""The objective contract (``docs/contracts/.../objectives/CONTRACT.md``) as executable checks.

Every property runs over ``testkit.cases.OBJECTIVE_CASES``. Teacher and hook
expectations come from each case's declared traits, so a new objective only
needs a case entry to inherit the full contract.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import audiossl.modelling.objectives as objectives_pkg
from audiossl.config import (
    AudioAugmentorConfig,
    LoudnessAugmentationConfig,
    validate_config,
)
from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives._base import OBJECTIVE_HOOKS, Objective
from audiossl.modelling.objectives.objective import build_objective
from audiossl.modelling.registry import get_objective_registry
from audiotrainer.step import StepContext

from testkit import checks, configs
from testkit.cases import OBJECTIVE_CASES, objective_params
from testkit.stubs import SeededGainAugmentor

HOOKS = frozenset({"metrics", "update_teacher", *OBJECTIVE_HOOKS})


def _teacher_params(objective, case) -> dict[str, torch.nn.Parameter]:
    if case.teacher == "ema":
        prefixes = tuple(f"{teacher}." for teacher, _ in case.teacher_pairs)
    elif case.teacher == "frozen":
        prefixes = ("teacher.",)
    else:
        return {}
    return {n: p for n, p in objective.named_parameters() if n.startswith(prefixes)}


def _training_flags(module: torch.nn.Module) -> dict[str, bool]:
    return {name: m.training for name, m in module.named_modules()}


# ------------------------------------------------------------ construction ---


@pytest.mark.parametrize("case", objective_params("factory"))
def test_factory_builds_registered_objective_subclass(case):
    objective = case.build()
    registered = get_objective_registry()[case.type]
    assert type(objective) is registered
    assert isinstance(objective, Objective)


@pytest.mark.parametrize("case", objective_params("config_validates"))
def test_training_config_validates(case):
    assert validate_config(configs.training(case.objective_config(), case.model()))


@pytest.mark.parametrize("case", objective_params("rejects_wrong_types"))
def test_constructor_rejects_wrong_model_and_config_types(case):
    cls = get_objective_registry()[case.type]
    config = case.objective_config()
    model = AudioModel(case.model())
    with pytest.raises(TypeError):
        cls(object(), config)
    with pytest.raises(TypeError):
        cls(model, object())
    with pytest.raises(TypeError):
        build_objective(model, object())


@pytest.mark.parametrize("case", objective_params("trainer_kwargs"))
def test_augmentor_injection_policy(case):
    model = AudioModel(case.model())
    config = case.objective_config()
    if case.uses_augmentor:
        with pytest.raises(ValueError):
            build_objective(model, config, augmentor=None)
    else:
        with pytest.raises(ValueError):
            build_objective(model, config, augmentor=SeededGainAugmentor())


@pytest.mark.parametrize("case", objective_params("model_untouched"))
def test_construction_and_forward_do_not_mutate_model_config_or_runtime(case):
    model = AudioModel(case.model())
    config_before = repr(model.config)
    runtime_before = model.applied_runtime
    objective = case.build(model=model).train()
    objective(case.batch())
    assert objective.model is model
    assert repr(model.config) == config_before
    assert model.applied_runtime == runtime_before


# ---------------------------------------------------------------- boundary ---


@pytest.mark.parametrize("case", objective_params("boundary_validation"))
def test_forward_boundary_validation(case):
    objective = case.build().train()
    batch = case.batch()
    with pytest.raises(TypeError):
        objective(batch.x)
    with pytest.raises(ValueError):
        objective(AudioBatch(x=batch.x[..., None], mask=batch.mask[..., None]))
    with pytest.raises(ValueError):
        objective(AudioBatch(x=batch.x, mask=batch.mask.float()))
    with pytest.raises(ValueError):
        objective(AudioBatch(x=batch.x, mask=batch.mask[:, :-1]))


# --------------------------------------------------------- loss and grads ---


@pytest.mark.parametrize("case", objective_params("loss_scalar"))
def test_training_loss_is_finite_differentiable_scalar(case):
    objective = case.build().train()
    out = objective(case.batch())
    assert isinstance(out, ObjectiveOutput)
    assert out.loss.ndim == 0
    assert out.loss.dtype == torch.float32
    assert out.loss.requires_grad
    assert torch.isfinite(out.loss)


@pytest.mark.parametrize("case", objective_params("gradient_routing"))
def test_gradients_reach_every_student_parameter_and_no_teacher_parameter(case):
    objective = case.build().train()
    teacher = _teacher_params(objective, case)
    assert all(not p.requires_grad for p in teacher.values()), "teacher must be frozen"
    if case.teacher in {"ema", "frozen"}:
        assert teacher, f"{case.id}: expected teacher parameters"

    objective(case.batch()).loss.backward()

    student = checks.trainable(objective)
    assert set(student).isdisjoint(teacher)
    assert set(checks.frozen(objective)) == set(teacher) | case.frozen_student, (
        "only teacher params and declared frozen_student params may be frozen"
    )
    assert checks.missing_grads(student) == []
    assert checks.nonfinite_grads(student) == []
    assert all(p.grad is None for p in teacher.values())


@pytest.mark.parametrize("case", objective_params("forward_is_pure"))
def test_forward_and_backward_do_not_mutate_state(case):
    """Parameters and buffers only change via the optimizer, declared hooks, or BN statistics."""
    objective = case.build().train()
    batch_norm_buffers = {
        f"{module_name}.{buffer_name}" if module_name else buffer_name
        for module_name, module in objective.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for buffer_name, _ in module.named_buffers(recurse=False)
    }
    before = checks.snapshot(objective)
    objective(case.batch()).loss.backward()
    assert sorted(set(checks.changed(before, objective)) - batch_norm_buffers) == []


# ----------------------------------------------------------------- teacher ---


@pytest.mark.parametrize(
    "case", objective_params("teacher_init", [c for c in OBJECTIVE_CASES if c.teacher == "ema"])
)
def test_ema_teacher_starts_as_exact_copy_of_student(case):
    objective = case.build()
    for teacher_name, student_name in case.teacher_pairs:
        teacher = getattr(objective, teacher_name)
        student = getattr(objective, student_name)
        teacher_module = getattr(teacher, "module", teacher)
        checks.assert_identical(
            teacher_module.state_dict(), student.state_dict(), what=teacher_name
        )
        assert not teacher.training, f"{teacher_name} must be in eval mode"


@pytest.mark.parametrize(
    "case", objective_params("ema_update", [c for c in OBJECTIVE_CASES if c.teacher == "ema"])
)
def test_update_teacher_applies_exact_ema_to_every_parameter(case):
    objective = case.build().train()
    decay = float(case.objective_config().teacher_ema_decay)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in checks.trainable(objective).values():
            p.add_(0.1 * torch.randn(p.shape, generator=generator))

    pairs = [(getattr(objective, t), getattr(objective, s)) for t, s in case.teacher_pairs]
    pairs = [(getattr(t, "module", t), s) for t, s in pairs]
    teacher_before = [checks.snapshot(t) for t, _ in pairs]
    student_before = checks.snapshot(objective)
    student_param_names = set(checks.trainable(objective))

    objective.update_teacher()

    for (teacher, student), old in zip(pairs, teacher_before, strict=True):
        student_state = dict(student.named_parameters())
        for name, param in teacher.named_parameters():
            expected = decay * old[name] + (1.0 - decay) * student_state[name].detach()
            torch.testing.assert_close(param.detach(), expected, rtol=0, atol=1e-6)
        student_buffers = dict(student.named_buffers())
        for name, buf in teacher.named_buffers():
            assert torch.equal(buf, student_buffers[name]), f"buffer {name} not hard-copied"
        assert not teacher.training

    after = objective.state_dict()
    for name in student_param_names:
        assert torch.equal(after[name], student_before[name]), f"student {name} changed"


@pytest.mark.parametrize(
    "case",
    objective_params(
        "stop_grad_noop", [c for c in OBJECTIVE_CASES if c.teacher == "stop_grad"]
    ),
)
def test_update_teacher_is_noop_without_ema_teacher(case):
    objective = case.build().train()
    before = checks.snapshot(objective)
    objective.update_teacher()
    assert checks.changed(before, objective) == []


@pytest.mark.parametrize(
    "case",
    objective_params(
        "ema_matches_stop_grad_at_init", [c for c in OBJECTIVE_CASES if c.teacher == "ema"]
    ),
)
def test_ema_and_stop_grad_targets_coincide_at_initialization(case):
    """At step 0 the EMA teacher equals the student, so both teacher modes give the same loss."""
    stop_grad_case = replace(
        case,
        objective=lambda: replace(case.objective(), teacher_mode="stop_grad"),
        teacher="stop_grad",
        teacher_pairs=(),
    )
    ema = case.build().train()
    stop_grad = stop_grad_case.build().train()
    student_state = {k: v for k, v in ema.state_dict().items() if not k.startswith("teacher_")}
    checks.assert_identical(stop_grad.state_dict(), student_state, what="student init")

    batch = case.batch()
    torch.testing.assert_close(stop_grad(batch).loss, ema(batch).loss, rtol=1e-6, atol=1e-7)


# ------------------------------------------------------------------- modes ---


def _is_teacher(name: str, case) -> bool:
    roots = {teacher for teacher, _ in case.teacher_pairs}
    if case.teacher == "frozen":
        roots.add("teacher")
    return name.split(".", 1)[0] in roots


@pytest.mark.parametrize("case", objective_params("train_mode_preserved"))
def test_train_forward_preserves_student_modes_and_runs_teacher_in_eval(case):
    """Stop-grad paths may toggle eval temporarily but must restore the student's modes."""
    objective = case.build().train()
    teacher_modes_during_forward: dict[str, bool] = {}

    def record(name: str):
        def hook(module, _args) -> None:
            teacher_modes_during_forward.setdefault(name, module.training)

        return hook

    for name, module in objective.named_modules():
        if name and _is_teacher(name, case):
            module.register_forward_pre_hook(record(name))
    before = _training_flags(objective)
    objective(case.batch())
    after = _training_flags(objective)

    student = {n: f for n, f in before.items() if not _is_teacher(n, case)}
    assert {n: after[n] for n in student} == student
    if case.teacher in {"ema", "frozen"}:
        assert teacher_modes_during_forward, "teacher never ran"
    running_in_train = sorted(n for n, f in teacher_modes_during_forward.items() if f)
    assert running_in_train == [], "teacher modules ran in train mode"


@pytest.mark.parametrize("case", objective_params("eval_forward"))
def test_eval_forward_policy_and_mode_preservation(case):
    objective = case.build().eval()
    before = _training_flags(objective)
    if not case.eval_forward:
        with pytest.raises(RuntimeError):
            objective(case.batch())
        return
    with torch.no_grad():
        out = objective(case.batch())
    assert torch.isfinite(out.loss)
    assert _training_flags(objective) == before


@pytest.mark.parametrize(
    "case",
    objective_params(
        "ema_melspec_front_end",
        [c for c in OBJECTIVE_CASES if c.teacher == "ema" and c.type in {"byol", "dino", "jepa"}],
    ),
)
def test_ema_teacher_update_supports_melspec_front_end(case):
    objective = replace(case, model=configs.model).build().train()
    objective(case.batch()).loss.backward()
    objective.update_teacher()


# ------------------------------------------------------------------- hooks ---


@pytest.mark.parametrize("case", objective_params("closed_hooks"))
def test_hook_surface_is_closed_and_matches_declared_traits(case):
    objective = case.build()
    cls = type(objective)
    defined = {
        name
        for name in HOOKS
        if callable(getattr(cls, name, None))
        and getattr(cls, name) is not getattr(Objective, name, None)
    }
    assert defined == set(case.hooks)
    hook_like = {
        name
        for name, _ in inspect.getmembers(cls, callable)
        if re.fullmatch(r"(pre|post|on)_\w+", name)
    }
    assert hook_like <= HOOKS, f"unrecognized hook-like methods: {sorted(hook_like - HOOKS)}"
    if "metrics" in defined:
        assert cls.metrics is not Objective.metrics, "objectives must extend metrics()"


@pytest.mark.parametrize("case", objective_params("lifecycle_hooks"))
def test_lifecycle_hooks_accept_step_context(case):
    """The trainer calls every lifecycle hook with the same public context."""
    objective = case.build().train()
    assert not hasattr(objective, "callbacks"), "use lifecycle methods, not a callbacks dict"
    output = objective(case.batch())
    context = StepContext(epoch=0, global_step=0, optimizer_step=0, total_optimizer_steps=3)
    for name in OBJECTIVE_HOOKS:
        hook = getattr(objective, name)
        if name == "on_after_forward":
            hook(output, context)
        else:
            hook(context)


@pytest.mark.parametrize("case", objective_params("metrics_contract"))
def test_metrics_are_detached_finite_scalars(case):
    objective = case.build().train()
    out = objective(case.batch())
    context = StepContext(epoch=1, global_step=7, optimizer_step=3, total_optimizer_steps=10)
    metrics = objective.metrics(out, context)
    assert metrics, "no objective diagnostics"
    for key, value in metrics.items():
        assert key and "/" not in key, key
        assert isinstance(value, torch.Tensor), key
        assert value.ndim == 0, key
        assert not value.requires_grad, key
        assert torch.isfinite(value.float()), key


@pytest.mark.parametrize("case", objective_params("metrics_type_check"))
def test_metrics_rejects_foreign_output_type(case):
    objective = case.build()
    with pytest.raises(TypeError):
        objective.metrics(
            ObjectiveOutput(loss=torch.tensor(0.0)),
            StepContext(epoch=0, global_step=0, optimizer_step=0, total_optimizer_steps=1),
        )


@pytest.mark.parametrize("case", objective_params("output_exported"))
def test_output_is_objective_output_subclass(case):
    objective = case.build().train()
    out = objective(case.batch())
    assert isinstance(out, ObjectiveOutput)
    assert type(out) is not ObjectiveOutput


# ----------------------------------------------------------- package-level ---


def test_objective_sources_never_construct_optimizers():
    root = Path(objectives_pkg.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "torch.optim" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c.id) for c in OBJECTIVE_CASES if c.uses_augmentor]
)
def test_real_trainer_augmentation_pipeline_drives_view_objectives(case):
    """The stub augmentor must not hide an incompatibility with the trainer's real pipeline."""
    from audiotrainer.augment.augmentor import AugmentationPipeline

    pipeline = AugmentationPipeline(
        AudioAugmentorConfig(
            augmentations=(LoudnessAugmentationConfig(p=1.0, min_gain_db=-6.0, max_gain_db=6.0),)
        ),
        generator=torch.Generator().manual_seed(0),
    )
    batch = case.batch()
    view = pipeline(batch)
    assert torch.equal(view.mask, batch.mask), "augmentation must preserve validity masks"

    model = AudioModel(case.model())
    objective = build_objective(model, case.objective_config(), augmentor=pipeline).train()
    out = objective(batch)
    out.loss.backward()
    assert torch.isfinite(out.loss)

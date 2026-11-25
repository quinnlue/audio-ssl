"""Focused coverage for fingerprint split, registries, RNG, and callback state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from audiossl.config import (
    AdamWOptimizerConfig,
    AudioAugmentorConfig,
    LoudnessAugmentationConfig,
    WandBConfig,
    register_config,
    register_validator,
    validate_config,
)
from audiossl.modelling.objectives._base import Objective
from audiotrainer.augment.augmentor import AugmentationPipeline
from audiotrainer.callbacks.wandb_logger import WandBLogger
from audiotrainer.fingerprint import config_fingerprint, experiment_payload
from audiotrainer.optim import build_optimizer
from audiotrainer.registry import AUGMENTATIONS, OPTIMIZERS

from audiotrainer.evaluation import ObjectiveLossEvaluator
from harness import DEFAULT_CASE, waveform_loader


def test_validation_rng_isolation_restores_objective_generators():
    run = DEFAULT_CASE.run(num_batches=2, max_epochs=1)
    objective = run.objective
    generators = objective.named_generators()
    if not generators:
        pytest.skip("objective has no registered generators")
    # Construct the trainer before capturing RNG; Trainer.__init__ reseeds globals.
    state = run.trainer().state
    before = {name: gen.get_state().clone() for name, gen in generators.items()}
    torch.manual_seed(999)
    global_before = torch.random.get_rng_state().clone()

    loader = waveform_loader(num_batches=1, batch_size=2, length=run.case.waveform_length)
    evaluator = ObjectiveLossEvaluator(objective, loader, eval_seed=123)
    metrics = evaluator.evaluate(run.model, state)
    assert "loss" in metrics

    after = {name: gen.get_state() for name, gen in objective.named_generators().items()}
    for name, prior in before.items():
        assert torch.equal(prior, after[name]), name
    assert torch.equal(global_before, torch.random.get_rng_state())


def test_experiment_payload_excludes_device_but_keeps_seed():
    config = DEFAULT_CASE.training_config()
    payload = experiment_payload(config)
    assert "device" not in payload["model"]["runtime"]
    assert payload["model"]["runtime"]["seed"] == config.model.runtime.seed
    assert "data_fingerprint" not in payload

    other_device = replace(
        config,
        model=replace(config.model, runtime=replace(config.model.runtime, device="cuda")),
    )
    assert config_fingerprint(config, data_fingerprint="a") == config_fingerprint(
        other_device, data_fingerprint="a"
    )
    assert config_fingerprint(config, data_fingerprint="a") != config_fingerprint(
        config, data_fingerprint="b"
    )


def test_register_generator_is_required_and_checkpointed_by_name():
    class Toy(Objective):
        uses_generator = True

        def __init__(self, generator: torch.Generator | None = None) -> None:
            super().__init__()
            if generator is not None:
                self.register_generator("generator", generator)

        def forward(self, batch):  # pragma: no cover - not used
            raise NotImplementedError

    with pytest.raises(ValueError, match="no generator"):
        Toy()._require_generator()

    gen = torch.Generator().manual_seed(7)
    objective = Toy(generator=gen)
    objective._require_generator()
    named = objective.named_generators()
    assert set(named) == {"generator"}
    assert named["generator"] is gen


def test_wandb_logger_restores_run_id_from_state_dict():
    logger = WandBLogger(
        config=WandBConfig(project="x", mode="disabled"),
        train_config=DEFAULT_CASE.training_config(),
    )
    logger.load_state_dict({"run_id": "deadbeefcafebabe0123456789abcdef"})
    assert logger.state_dict() == {"run_id": "deadbeefcafebabe0123456789abcdef"}

    state = type("S", (), {"config_fingerprint": "fp", "data_fingerprint": "data"})()
    with patch("wandb.init") as init:
        logger.on_train_start(state)
    assert init.call_args.kwargs["id"] == "deadbeefcafebabe0123456789abcdef"
    assert init.call_args.kwargs["resume"] == "allow"


def test_nonfinite_policy_skip_counts_without_advancing_optimizer():
    run = DEFAULT_CASE.run(num_batches=2, max_epochs=1)
    run.config = replace(
        run.config,
        trainer=replace(
            run.config.trainer,
            loop=replace(run.config.trainer.loop, nonfinite_policy="skip"),
        ),
    )
    trainer = run.trainer()
    original_step = trainer.optimizer.step
    steps = {"n": 0}

    def counting_step(*args, **kwargs):
        steps["n"] += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step  # type: ignore[method-assign]

    def exploding_forward(batch):
        out = type(trainer.objective).forward(trainer.objective, batch)
        out.loss = torch.tensor(float("nan"), requires_grad=True)
        return out

    trainer.objective.forward = exploding_forward  # type: ignore[method-assign]
    trainer.fit()
    assert steps["n"] == 0
    assert trainer.state.skipped_optimizer_steps >= 1
    assert trainer.state.optimizer_step == 0


@dataclass(kw_only=True, frozen=True)
class IdentityAugmentationConfig:
    type: Literal["test_identity"] = "test_identity"
    scale: float = 1.0


class IdentityAugmentation:
    def __init__(self, config: IdentityAugmentationConfig) -> None:
        self.scale = float(config.scale)

    def __call__(self, x, generator=None):
        del generator
        return [item * self.scale for item in x]


@dataclass(kw_only=True, frozen=True)
class ScaleAdamWConfig:
    lr: float
    type: Literal["test_scale_adamw"] = "test_scale_adamw"
    weight_decay: float = 0.0
    no_weight_decay_1d: bool = True
    no_weight_decay_patterns: tuple[str, ...] = ()
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    lr_scale: float = 2.0


def test_plugin_optimizer_and_augmentation_registries():
    register_config("augmentation")(IdentityAugmentationConfig)

    @register_validator(IdentityAugmentationConfig)
    def _validate_identity(config, **context):
        del context
        if config.scale <= 0:
            raise ValueError("scale must be > 0")

    AUGMENTATIONS.register("test_identity", lambda config: IdentityAugmentation(config))
    pipeline = AugmentationPipeline(
        AudioAugmentorConfig(augmentations=(IdentityAugmentationConfig(scale=2.0),))
    )
    assert isinstance(pipeline.augmentations[0], IdentityAugmentation)

    register_config("optimizer")(ScaleAdamWConfig)

    @register_validator(ScaleAdamWConfig)
    def _validate_scale_adamw(config, **context):
        del context
        if config.lr_scale <= 0:
            raise ValueError("lr_scale must be > 0")

    def _scale_adamw(groups, config):
        return torch.optim.AdamW(groups, lr=config.lr * config.lr_scale)

    OPTIMIZERS.register("test_scale_adamw", _scale_adamw)
    module = nn.Linear(2, 2)
    opt = build_optimizer(module, ScaleAdamWConfig(lr=0.1, lr_scale=3.0))
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.3)

    assert validate_config(
        AudioAugmentorConfig(
            augmentations=(
                LoudnessAugmentationConfig(p=1.0, min_gain_db=-1.0, max_gain_db=1.0),
            )
        )
    )
    built_in = build_optimizer(module, AdamWOptimizerConfig(lr=1e-3))
    assert isinstance(built_in, torch.optim.AdamW)

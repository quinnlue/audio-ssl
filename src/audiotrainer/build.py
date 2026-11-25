"""Config-driven construction of a fully wired :class:`Trainer`."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from audiossl.config import AudioSSLTrainingConfig, WandBConfig, validate_config
from audiossl.config.run import RunConfig
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives import build_objective
from audiossl.rng import new_manual_seed_generator
from audiossl.runtime import derive_seed

from .augment import AugmentationPipeline
from .callbacks import Callback, ValidationCallback, WandBLogger
from .data import DataModule
from .evaluation import ClassificationEvaluator, ObjectiveLossEvaluator
from .trainer import Trainer

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**31
    torch.manual_seed(worker_seed + worker_id)


def build_trainer(
    train_config: AudioSSLTrainingConfig,
    data: DataModule,
    *,
    run_config: RunConfig | None = None,
    callbacks: Sequence[Callback] = (),
    model: AudioModel | None = None,
) -> Trainer:
    """Build model, objective, loaders, evaluator, and trainer from config + data."""

    validate_config(train_config)
    run_config = run_config or RunConfig()
    validate_config(run_config)

    if not isinstance(data, DataModule):
        raise TypeError(
            f"build_trainer expected DataModule, got {type(data).__name__}"
        )

    if model is None:
        model = AudioModel(train_config.model)
    else:
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"build_trainer expected AudioModel or None, got {type(model).__name__}"
            )
        if model.config != train_config.model:
            raise ValueError(
                "build_trainer: injected model.config does not match train_config.model"
            )
    seed = model.applied_runtime.seed
    rng_device = model.applied_runtime.device

    augmentor = None
    if train_config.trainer.augmentor is not None:
        aug_gen = new_manual_seed_generator(derive_seed(seed, "augment"), rng_device)
        augmentor = AugmentationPipeline(train_config.trainer.augmentor, generator=aug_gen)

    objective_cls_uses_generator = False
    from audiossl.modelling.registry import get_objective_registry

    try:
        objective_cls = get_objective_registry()[train_config.objective.type]
        objective_cls_uses_generator = bool(getattr(objective_cls, "uses_generator", False))
    except Exception:
        objective_cls_uses_generator = getattr(train_config.objective, "type", None) in {
            "jepa", "mae", "ufo", "wav2vec2", "hubert",
        }

    generator = None
    if objective_cls_uses_generator:
        generator = new_manual_seed_generator(derive_seed(seed, "objective/masking"), rng_device)

    objective = build_objective(
        model,
        train_config.objective,
        augmentor=augmentor,
        generator=generator,
    )

    train_gen = torch.Generator().manual_seed(derive_seed(seed, "data/train"))
    train_loader = data.make_train_loader(
        train_config.trainer.batch,
        run_config.loader,
        generator=train_gen,
    )
    if getattr(train_loader, "num_workers", 0) and train_loader.worker_init_fn is None:
        train_loader.worker_init_fn = _worker_init_fn

    user_callbacks = list(callbacks)
    if run_config.evaluation is not None:
        val_gen = torch.Generator().manual_seed(derive_seed(seed, "data/val"))
        val_loader = data.make_val_loader(
            train_config.trainer.batch,
            run_config.loader,
            generator=val_gen,
        )
        if val_loader is None:
            raise ValueError(
                "RunConfig.evaluation is set but DataModule.make_val_loader returned None"
            )
        mp = train_config.trainer.mixed_precision
        eval_kwargs = {
            "use_autocast": mp is not None,
            "autocast_dtype": mp.dtype if mp is not None else None,
            "eval_seed": derive_seed(seed, "validation"),
        }
        if train_config.objective.type == "sft":
            evaluator = ClassificationEvaluator(objective, val_loader, **eval_kwargs)
        else:
            evaluator = ObjectiveLossEvaluator(objective, val_loader, **eval_kwargs)
        if not any(isinstance(cb, ValidationCallback) for cb in user_callbacks):
            user_callbacks.insert(0, ValidationCallback(evaluator))

    for logger_cfg in run_config.loggers:
        if isinstance(logger_cfg, WandBConfig):
            if not any(isinstance(cb, WandBLogger) for cb in user_callbacks):
                user_callbacks.append(
                    WandBLogger(config=logger_cfg, train_config=train_config)
                )
        else:
            raise TypeError(
                f"Unsupported logger config {type(logger_cfg).__name__}; "
                "register a factory or pass an explicit Callback"
            )

    return Trainer(
        model,
        objective,
        train_loader,
        train_config,
        run_config=run_config,
        data_fingerprint=data.fingerprint(),
        callbacks=user_callbacks,
    )


__all__ = ["build_trainer"]

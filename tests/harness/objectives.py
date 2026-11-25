"""Registry of tiny, trainable objective setups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from audiossl.config import (
    AudioModelConfig,
    AudioSSLTrainingConfig,
    AugmentationConfig,
    BYOLObjectiveConfig,
    ChannelMaskingConfig,
    ContrastiveObjectiveConfig,
    DINOHeadConfig,
    DINOObjectiveConfig,
    ExponentialDecayScalarSchedulerConfig,
    JEPAObjectiveConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    LoudnessAugmentationConfig,
    MAEMaskingConfig,
    MAEObjectiveConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    ProjectionHeadConfig,
    SpanMaskingConfig,
    TemporalMaskingConfig,
    TransformerEncoderConfig,
    Wav2Vec2ObjectiveConfig,
    Wav2Vec2QuantizerConfig,
)
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.config.run import RunConfig
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives._base import Objective
from audiossl.modelling.objectives.objective import build_objective
from audiossl.runtime import derive_seed
from torch.utils.data import DataLoader

from audiotrainer import AugmentationPipeline, Callback, Trainer, collate_fn

from .configs import (
    loudness_augmentation,
    run_config as make_run_config,
    tiny_conv_model_config,
    tiny_mel_model_config,
    tiny_wav2vec2_model_config,
    training_config,
)
from .recording import TrainingTrace

DATA_SEED = 1234
LOUDNESS_AUGMENTATION: tuple[AugmentationConfig, ...] = (loudness_augmentation(),)


def collate_waveforms(items: list[torch.Tensor]) -> AudioBatch:
    return collate_fn({"x": list(items)})


def waveform_loader(*, num_batches: int, batch_size: int, length: int) -> DataLoader:
    """Fixed-order loader over waveforms drawn from a private generator."""

    generator = torch.Generator().manual_seed(DATA_SEED)
    samples = [
        torch.randn(length, generator=generator, dtype=torch.float32)
        for _ in range(num_batches * batch_size)
    ]
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_waveforms,
    )


@dataclass(frozen=True)
class ObjectiveCase:
    name: str
    model_config: Callable[[], AudioModelConfig]
    objective_config: Callable[[], object]
    waveform_length: int = 64
    augmentations: tuple[AugmentationConfig, ...] = field(default=())
    uses_generator: bool = False

    def training_config(self, **trainer_overrides) -> AudioSSLTrainingConfig:
        # Operational keys belong on RunConfig; ignore them here for call-site compat.
        for key in (
            "checkpoint",
            "evaluation",
            "stdout_logging",
            "log_every_n_steps",
            "run_dir",
        ):
            trainer_overrides.pop(key, None)
        trainer_overrides.setdefault("augmentations", self.augmentations)
        return training_config(
            self.objective_config(), model=self.model_config(), **trainer_overrides
        )

    def build(self, config: AudioSSLTrainingConfig) -> tuple[AudioModel, Objective]:
        from audiossl.modelling.registry import get_objective_registry

        model = AudioModel(config.model)
        seed = model.applied_runtime.seed
        augmentor = None
        from audiossl.rng import new_manual_seed_generator

        rng_device = model.applied_runtime.device
        if config.trainer.augmentor is not None:
            aug_gen = new_manual_seed_generator(derive_seed(seed, "augment"), rng_device)
            augmentor = AugmentationPipeline(config.trainer.augmentor, generator=aug_gen)
        cls = get_objective_registry()[config.objective.type]
        generator = None
        if getattr(cls, "uses_generator", False):
            generator = new_manual_seed_generator(
                derive_seed(seed, "objective/masking"), rng_device
            )
        return model, build_objective(
            model, config.objective, augmentor=augmentor, generator=generator
        )

    def run(
        self,
        *,
        num_batches: int = 4,
        config: AudioSSLTrainingConfig | None = None,
        run: RunConfig | None = None,
        **trainer_overrides,
    ) -> Run:
        """Fresh model, objective, and loader. Pass ``config`` to reuse one exactly."""

        checkpoint = trainer_overrides.pop("checkpoint", None)
        evaluation = trainer_overrides.pop("evaluation", None)
        stdout_logging = trainer_overrides.pop("stdout_logging", False)
        log_every_n_steps = trainer_overrides.pop("log_every_n_steps", 1)
        if config is None:
            config = self.training_config(**trainer_overrides)
        elif trainer_overrides:
            raise TypeError("pass either config or trainer overrides, not both")
        if run is None:
            run = make_run_config(
                checkpoint=checkpoint,
                evaluation=evaluation,
                stdout_logging=stdout_logging,
                log_every_n_steps=log_every_n_steps,
            )
        model, objective = self.build(config)
        loader = waveform_loader(
            num_batches=num_batches,
            batch_size=config.trainer.batch.per_device_batch_size,
            length=self.waveform_length,
        )
        return Run(
            case=self,
            config=config,
            run_config=run,
            model=model,
            objective=objective,
            loader=loader,
        )

    def __str__(self) -> str:
        return self.name


@dataclass
class Run:
    """One fully-constructed training setup; build a new one for each fit."""

    case: ObjectiveCase
    config: AudioSSLTrainingConfig
    model: AudioModel
    objective: Objective
    loader: DataLoader
    run_config: RunConfig = field(default_factory=RunConfig)
    data_fingerprint: str = "harness/waveforms"

    def trainer(self, *callbacks: Callback, **trainer_kwargs) -> Trainer:
        return Trainer(
            self.model,
            self.objective,
            self.loader,
            self.config,
            run_config=self.run_config,
            data_fingerprint=self.data_fingerprint,
            callbacks=list(callbacks),
            **trainer_kwargs,
        )

    def fit(self, *callbacks: Callback, **trainer_kwargs) -> TrainingTrace:
        trace = TrainingTrace()
        self.trainer(trace, *callbacks, **trainer_kwargs).fit()
        return trace

    def resume(self, checkpoint: str | Path, *callbacks: Callback) -> Trainer:
        return Trainer.from_checkpoint(
            checkpoint,
            model=self.model,
            objective=self.objective,
            train_loader=self.loader,
            train_config=self.config,
            run_config=self.run_config,
            data_fingerprint=self.data_fingerprint,
            callbacks=list(callbacks),
        )


def _head(**overrides) -> ProjectionHeadConfig:
    values = {
        "depth": 2,
        "hidden_dim": 16,
        "output_dim": 8,
        "activation": "gelu",
        "final_l2_normalize": False,
    }
    values.update(overrides)
    return ProjectionHeadConfig(**values)


CONTRASTIVE = ObjectiveCase(
    name="contrastive_dropout",
    model_config=lambda: tiny_conv_model_config(mlp_dropout_p=0.1),
    objective_config=lambda: ContrastiveObjectiveConfig(
        view_strategy="dropout",
        projection=_head(output_dim=12, final_l2_normalize=True),
        temperature=0.2,
    ),
)

CONTRASTIVE_AUGMENTED = ObjectiveCase(
    name="contrastive_augmented",
    model_config=tiny_conv_model_config,
    objective_config=lambda: ContrastiveObjectiveConfig(
        view_strategy="augmentation",
        projection=_head(output_dim=12, final_l2_normalize=True),
        temperature=0.2,
    ),
    augmentations=LOUDNESS_AUGMENTATION,
)

BYOL = ObjectiveCase(
    name="byol",
    model_config=tiny_conv_model_config,
    objective_config=lambda: BYOLObjectiveConfig(
        projector=_head(),
        predictor=_head(),
        teacher_mode="ema",
        teacher_ema_decay=0.9,
    ),
    augmentations=LOUDNESS_AUGMENTATION,
)

DINO = ObjectiveCase(
    name="dino",
    model_config=tiny_conv_model_config,
    objective_config=lambda: DINOObjectiveConfig(
        head=DINOHeadConfig(
            depth=2,
            hidden_dim=16,
            bottleneck_dim=8,
            num_prototypes=11,
            activation="gelu",
        ),
        teacher_mode="ema",
        teacher_ema_decay=0.9,
        teacher_temperature=0.07,
        student_temperature=0.2,
        center_momentum=0.8,
    ),
    augmentations=LOUDNESS_AUGMENTATION,
)

JEPA = ObjectiveCase(
    name="jepa",
    model_config=tiny_conv_model_config,
    objective_config=lambda: JEPAObjectiveConfig(
        loss_style="mse",
        mask_token_init_scale=0.02,
        teacher_mode="ema",
        teacher_ema_decay=0.9,
        predictor_config=MLPConfig(d_ff=16),
        masking=TemporalMaskingConfig(mask_ratio=0.5, min_span_len=1, max_span_len=3),
    ),
    uses_generator=True,
)

MAE = ObjectiveCase(
    name="mae",
    model_config=tiny_mel_model_config,
    objective_config=lambda: MAEObjectiveConfig(
        decoder_d_model=16,
        decoder=TransformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=4),
            mlp=MLPConfig(d_ff=32, activation="gelu"),
            num_layers=1,
            final_norm=True,
        ),
        decoder_positional_encoding=LearnableTimeFrequencyPositionalEncodingConfig(
            max_t=32, max_f=1
        ),
        masking=MAEMaskingConfig(mask_ratio=0.5),
    ),
    waveform_length=400,
    uses_generator=True,
)

WAV2VEC2 = ObjectiveCase(
    name="wav2vec2",
    model_config=tiny_wav2vec2_model_config,
    objective_config=lambda: Wav2Vec2ObjectiveConfig(
        quantizer=Wav2Vec2QuantizerConfig(
            temperature_scheduler=ExponentialDecayScalarSchedulerConfig(
                start=2.0, end=0.5, decay=0.9
            ),
            num_groups=2,
            num_vars=8,
            vq_dim=16,
        ),
        time_masking=SpanMaskingConfig(mask_prob=0.2, mask_length=2),
        feature_masking=ChannelMaskingConfig(mask_prob=0.0, mask_length=2),
        final_dim=16,
        num_negatives=3,
    ),
    waveform_length=4000,
    uses_generator=True,
)

OBJECTIVE_CASES: tuple[ObjectiveCase, ...] = (
    CONTRASTIVE,
    CONTRASTIVE_AUGMENTED,
    BYOL,
    DINO,
    JEPA,
    MAE,
    WAV2VEC2,
)

DEFAULT_CASE = JEPA

over_objectives = pytest.mark.parametrize("case", OBJECTIVE_CASES, ids=str)


__all__ = [
    "BYOL",
    "CONTRASTIVE",
    "CONTRASTIVE_AUGMENTED",
    "DEFAULT_CASE",
    "DINO",
    "JEPA",
    "LOUDNESS_AUGMENTATION",
    "MAE",
    "OBJECTIVE_CASES",
    "ObjectiveCase",
    "Run",
    "WAV2VEC2",
    "collate_waveforms",
    "over_objectives",
    "waveform_loader",
]

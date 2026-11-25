"""Tiny, fully explicit configs for trainer tests."""

from __future__ import annotations

from dataclasses import replace

from audiossl.config import (
    LoggerConfig,
    AdamWOptimizerConfig,
    AudioAugmentorConfig,
    AudioModelConfig,
    AudioSSLTrainingConfig,
    AugmentationConfig,
    BatchConfig,
    ConstantSchedulerConfig,
    ConvolutionalProcessorConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    LinearEmbeddingConfig,
    LoudnessAugmentationConfig,
    LRSchedulerConfig,
    MelSpectrogramProcessorConfig,
    MixedPrecisionConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoPositionalEncodingConfig,
    OptimizerConfig,
    PatchifyTokenizerConfig,
    RopePositionalEncodingConfig,
    RuntimeConfig,
    TrainerConfig,
    TrainingLoopConfig,
    TransformerEncoderConfig,
    validate_config,
)
from audiossl.config.run import CheckpointConfig, EvaluationConfig, RunConfig
from audiossl.integrations.wav2vec2 import make_wav2vec2_base_config


def tiny_conv_model_config(*, seed: int = 0, mlp_dropout_p: float = 0.0) -> AudioModelConfig:
    return AudioModelConfig(
        d_model=8,
        processor=ConvolutionalProcessorConfig(
            in_channels=1,
            out_channels=4,
            kernel_sizes=(3, 3),
            strides=(2, 2),
        ),
        tokenizer=PatchifyTokenizerConfig(patch_shape=(4, 1), padding_strategy="truncate"),
        embedding=LinearEmbeddingConfig(positional_encoding=NoPositionalEncodingConfig()),
        encoder=TransformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2),
            mlp=MLPConfig(d_ff=16, dropout_p=mlp_dropout_p),
            num_layers=1,
            positional_encoding=RopePositionalEncodingConfig(max_len=128),
        ),
        runtime=RuntimeConfig(seed=seed),
    )


def tiny_mel_model_config(*, seed: int = 0) -> AudioModelConfig:
    return AudioModelConfig(
        d_model=16,
        processor=MelSpectrogramProcessorConfig(
            n_mels=4,
            n_fft=64,
            f_max=8000.0,
            win_length=64,
            hop_length=16,
            log_mel=True,
            batch_norm=True,
            frame_validity="overlap_any",
        ),
        tokenizer=PatchifyTokenizerConfig(patch_shape=(4, 4), padding_strategy="truncate"),
        embedding=LinearEmbeddingConfig(
            positional_encoding=LearnableTimeFrequencyPositionalEncodingConfig(max_t=32, max_f=1)
        ),
        encoder=TransformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=4),
            mlp=MLPConfig(d_ff=32),
            num_layers=1,
            final_norm=True,
        ),
        runtime=RuntimeConfig(seed=seed),
    )


def tiny_wav2vec2_model_config(*, seed: int = 0) -> AudioModelConfig:
    return make_wav2vec2_base_config(
        runtime=RuntimeConfig(seed=seed),
        d_model=32,
        depth=1,
        num_heads=4,
        ffn_dim=64,
        conv_dim=(8, 8, 8, 8, 8, 8, 8),
        conv_pos_kernel_size=16,
        conv_pos_groups=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        feature_projection_dropout=0.0,
    )


def default_optimizer() -> AdamWOptimizerConfig:
    return AdamWOptimizerConfig(lr=1e-3, no_weight_decay_patterns=("bias", "norm"))


def trainer_config(
    *,
    optimizer: OptimizerConfig | None = None,
    scheduler: LRSchedulerConfig | None = None,
    batch_size: int = 2,
    grad_accum: int = 1,
    max_epochs: int = 1,
    grad_clip: float | None = None,
    mixed_precision: MixedPrecisionConfig | None = None,
    augmentations: tuple[AugmentationConfig, ...] = (),
) -> TrainerConfig:
    return TrainerConfig(
        augmentor=AudioAugmentorConfig(augmentations=augmentations) if augmentations else None,
        optimizer=optimizer or default_optimizer(),
        scheduler=scheduler or ConstantSchedulerConfig(),
        batch=BatchConfig(per_device_batch_size=batch_size, gradient_accumulation_steps=grad_accum),
        loop=TrainingLoopConfig(
            max_epochs=max_epochs,
            gradient_clip_norm=grad_clip,
        ),
        mixed_precision=mixed_precision,
    )


def run_config(
    *,
    checkpoint: CheckpointConfig | None = None,
    evaluation: EvaluationConfig | None = None,
    stdout_logging: bool = False,
    log_every_n_steps: int = 1,
    run_dir: str | None = None,
    loggers: tuple[LoggerConfig, ...] = (),
) -> RunConfig:
    return RunConfig(
        run_dir=run_dir,
        checkpoint=checkpoint,
        evaluation=evaluation,
        stdout_logging=stdout_logging,
        log_every_n_steps=log_every_n_steps,
        loggers=loggers,
    )


def training_config(
    objective,
    *,
    model: AudioModelConfig,
    **trainer_overrides,
) -> AudioSSLTrainingConfig:
    config = AudioSSLTrainingConfig(
        model=model,
        objective=objective,
        trainer=trainer_config(**trainer_overrides),
    )
    validate_config(config)
    return config


def with_seed(config: AudioSSLTrainingConfig, seed: int) -> AudioSSLTrainingConfig:
    runtime = replace(config.model.runtime, seed=seed)
    return replace(config, model=replace(config.model, runtime=runtime))


def with_device(config: AudioSSLTrainingConfig, device: str) -> AudioSSLTrainingConfig:
    runtime = replace(config.model.runtime, device=device)
    return replace(config, model=replace(config.model, runtime=runtime))


def loudness_augmentation(
    *,
    p: float = 0.5,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
) -> LoudnessAugmentationConfig:
    return LoudnessAugmentationConfig(p=p, min_gain_db=min_gain_db, max_gain_db=max_gain_db)


__all__ = [
    "default_optimizer",
    "loudness_augmentation",
    "run_config",
    "tiny_conv_model_config",
    "tiny_mel_model_config",
    "tiny_wav2vec2_model_config",
    "trainer_config",
    "training_config",
    "with_device",
    "with_seed",
]

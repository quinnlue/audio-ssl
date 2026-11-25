"""Tiny config builders for every registered modelling component.

Every builder returns a validated-by-construction config small enough to run
a forward/backward pass on CPU in milliseconds. Dropout defaults to zero so
outputs are a deterministic function of inputs, parameters, and injected
generators; cases that need stochastic dropout opt in explicitly.

Builders take keyword overrides so tests can vary one field without
re-specifying the rest.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from audiossl.config import (
    AdamWOptimizerConfig,
    AltTransformerEncoderConfig,
    AudioAugmentorConfig,
    AudioModelConfig,
    AudioSSLTrainingConfig,
    BatchConfig,
    BranchformerEncoderConfig,
    BranchMergeConfig,
    BYOLObjectiveConfig,
    CGMLPConfig,
    ChannelMaskingConfig,
    ConformerEncoderConfig,
    ConstantSchedulerConfig,
    ContrastiveObjectiveConfig,
    ConvEmbeddingConfig,
    ConvModuleConfig,
    ConvolutionalPositionalEncodingConfig,
    ConvolutionalProcessorConfig,
    DINOHeadConfig,
    DINOObjectiveConfig,
    EATEmbeddingConfig,
    EBranchformerEncoderConfig,
    EBranchMergeConfig,
    EmbeddingSubsamplingConfig,
    ExponentialDecayScalarSchedulerConfig,
    HuBERTObjectiveConfig,
    HuBERTTeacherConfig,
    JEPAObjectiveConfig,
    KaldiFbankProcessorConfig,
    LearnablePositionalEncodingConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    LinearEmbeddingConfig,
    MAEMaskingConfig,
    MAEObjectiveConfig,
    MelSpectrogramProcessorConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoOpTokenizerConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    ProjectionHeadConfig,
    RopePositionalEncodingConfig,
    RuntimeConfig,
    Sinusoidal2DPositionalEncodingConfig,
    SinusoidalPositionalEncodingConfig,
    SpanMaskingConfig,
    TemporalMaskingConfig,
    TrainerConfig,
    TrainingLoopConfig,
    TransformerEncoderConfig,
    UFODecoderConfig,
    UFOMaskingConfig,
    UFOObjectiveConfig,
    Wav2Vec2FeatureExtractorProcessorConfig,
    Wav2Vec2ObjectiveConfig,
    Wav2Vec2QuantizerConfig,
)
from audiossl.integrations.eat import make_eat_scaled_config
from audiossl.integrations.wav2vec2 import make_wav2vec2_base_config

D_MODEL = 16
NUM_HEADS = 2
D_FF = 32
NUM_LAYERS = 2
N_MELS = 8
PATCH_SHAPE = (2, 4)
FREQ_PATCHES = N_MELS // PATCH_SHAPE[1]
MAX_TIME_PATCHES = 128
MAX_TOKENS = MAX_TIME_PATCHES * FREQ_PATCHES

# Waveform lengths for the default melspec front-end. Deliberately ragged so
# every batch exercises padding.
DEFAULT_LENGTHS = (1600, 1184, 896)


def _with(config: Any, overrides: dict[str, Any]) -> Any:
    return replace(config, **overrides) if overrides else config


# ---------------------------------------------------------------- runtime ---


def runtime(*, seed: int = 0, dtype: str = "float32", **overrides: Any) -> RuntimeConfig:
    return RuntimeConfig(seed=seed, device="cpu", dtype=dtype, **overrides)  # type: ignore[arg-type]


# -------------------------------------------------------------- processors ---


def melspec(**overrides: Any) -> MelSpectrogramProcessorConfig:
    return _with(
        MelSpectrogramProcessorConfig(
            n_mels=N_MELS,
            n_fft=64,
            win_length=64,
            hop_length=16,
            f_max=8000.0,
        ),
        overrides,
    )


def conv_processor(**overrides: Any) -> ConvolutionalProcessorConfig:
    return _with(
        ConvolutionalProcessorConfig(
            in_channels=1,
            out_channels=N_MELS,
            kernel_sizes=(5, 3),
            strides=(4, 2),
            activation="relu",
        ),
        overrides,
    )


def wav2vec2_processor(**overrides: Any) -> Wav2Vec2FeatureExtractorProcessorConfig:
    return _with(
        Wav2Vec2FeatureExtractorProcessorConfig(
            conv_dim=(N_MELS, N_MELS, N_MELS),
            conv_kernel=(10, 3, 3),
            conv_stride=(5, 2, 2),
            group_norm_num_groups=N_MELS,
            extractor_mode="layer_norm",
        ),
        overrides,
    )


def kaldi_fbank(**overrides: Any) -> KaldiFbankProcessorConfig:
    return _with(
        KaldiFbankProcessorConfig(
            num_mel_bins=N_MELS,
            target_length=32,
            mask_all_frames=False,
        ),
        overrides,
    )


# -------------------------------------------------------------- tokenizers ---


def patchify(**overrides: Any) -> PatchifyTokenizerConfig:
    return _with(
        PatchifyTokenizerConfig(patch_shape=PATCH_SHAPE, padding_strategy="truncate"),
        overrides,
    )


def noop_tokenizer(output_dim: int = N_MELS) -> NoOpTokenizerConfig:
    return NoOpTokenizerConfig(_output_dim=output_dim)


# ----------------------------------------------------- positional encoding ---


def pe_none() -> NoPositionalEncodingConfig:
    return NoPositionalEncodingConfig()


def pe_learnable() -> LearnablePositionalEncodingConfig:
    return LearnablePositionalEncodingConfig(max_len=MAX_TOKENS)


def pe_sinusoidal() -> SinusoidalPositionalEncodingConfig:
    return SinusoidalPositionalEncodingConfig(max_len=MAX_TOKENS)


def pe_learnable_2d(**overrides: Any) -> LearnableTimeFrequencyPositionalEncodingConfig:
    return _with(
        LearnableTimeFrequencyPositionalEncodingConfig(
            max_t=MAX_TIME_PATCHES, max_f=FREQ_PATCHES
        ),
        overrides,
    )


def pe_sinusoidal_2d() -> Sinusoidal2DPositionalEncodingConfig:
    return Sinusoidal2DPositionalEncodingConfig(
        max_time_patches=MAX_TIME_PATCHES, max_freq_patches=FREQ_PATCHES
    )


def pe_conv1d() -> ConvolutionalPositionalEncodingConfig:
    return ConvolutionalPositionalEncodingConfig(kernel_size=4, groups=2)


def rope() -> RopePositionalEncodingConfig:
    return RopePositionalEncodingConfig(max_len=MAX_TOKENS + 1)


# -------------------------------------------------------------- embeddings ---


def linear_embedding(positional_encoding: Any = None, **overrides: Any) -> LinearEmbeddingConfig:
    return _with(
        LinearEmbeddingConfig(positional_encoding=positional_encoding or pe_none()),
        overrides,
    )


def conv_embedding(positional_encoding: Any = None, **overrides: Any) -> ConvEmbeddingConfig:
    return _with(
        ConvEmbeddingConfig(
            positional_encoding=positional_encoding or pe_none(),
            kernel_size=3,
            layer_norm=True,
        ),
        overrides,
    )


def eat_embedding(**overrides: Any) -> EATEmbeddingConfig:
    return _with(EATEmbeddingConfig(positional_encoding=pe_sinusoidal_2d()), overrides)


def subsampling(factor: int = 2) -> EmbeddingSubsamplingConfig:
    return EmbeddingSubsamplingConfig(factor=factor)


# ---------------------------------------------------------------- encoders ---


def mha(**overrides: Any) -> MultiHeadAttentionConfig:
    return _with(MultiHeadAttentionConfig(num_heads=NUM_HEADS, dropout_p=0.0), overrides)


def mlp(**overrides: Any) -> MLPConfig:
    return _with(MLPConfig(d_ff=D_FF, dropout_p=0.0), overrides)


def transformer(**overrides: Any) -> TransformerEncoderConfig:
    return _with(
        TransformerEncoderConfig(
            mha=mha(),
            mlp=mlp(),
            num_layers=NUM_LAYERS,
            positional_encoding=rope(),
        ),
        overrides,
    )


def conformer(**overrides: Any) -> ConformerEncoderConfig:
    return _with(
        ConformerEncoderConfig(
            mha=mha(),
            mlp=mlp(),
            conv=ConvModuleConfig(kernel_size=3),
            num_layers=NUM_LAYERS,
            positional_encoding=rope(),
        ),
        overrides,
    )


def branchformer(**overrides: Any) -> BranchformerEncoderConfig:
    return _with(
        BranchformerEncoderConfig(
            mha=mha(),
            cgmlp=CGMLPConfig(linear_units=D_FF, conv_kernel_size=3),
            merge=BranchMergeConfig(method="concat"),
            num_layers=NUM_LAYERS,
            positional_encoding=rope(),
        ),
        overrides,
    )


def ebranchformer(**overrides: Any) -> EBranchformerEncoderConfig:
    return _with(
        EBranchformerEncoderConfig(
            mha=mha(),
            cgmlp=CGMLPConfig(linear_units=D_FF, conv_kernel_size=3),
            merge=EBranchMergeConfig(conv_kernel_size=3),
            mlp=mlp(),
            use_ffn=True,
            macaron_ffn=True,
            num_layers=NUM_LAYERS,
            positional_encoding=rope(),
        ),
        overrides,
    )


def alt_transformer(**overrides: Any) -> AltTransformerEncoderConfig:
    return _with(
        AltTransformerEncoderConfig(mha=mha(), mlp=mlp(), num_layers=NUM_LAYERS),
        overrides,
    )


# ------------------------------------------------------------------ models ---


def model(**overrides: Any) -> AudioModelConfig:
    """Base model: melspec -> patchify -> linear(no PE) -> transformer(RoPE)."""
    values: dict[str, Any] = {
        "d_model": D_MODEL,
        "processor": melspec(),
        "tokenizer": patchify(),
        "embedding": linear_embedding(),
        "encoder": transformer(),
        "runtime": runtime(),
    }
    values.update(overrides)
    return AudioModelConfig(**values)


def wav2vec2_model(**overrides: Any) -> AudioModelConfig:
    """Tiny facebook/wav2vec2-base recipe (GroupNorm extractor, conv1d PE)."""
    values: dict[str, Any] = {
        "runtime": runtime(),
        "d_model": D_MODEL,
        "depth": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "ffn_dim": D_FF,
        "conv_dim": (N_MELS, N_MELS, N_MELS),
        "conv_kernel": (10, 3, 3),
        "conv_stride": (5, 2, 2),
        "conv_pos_kernel_size": 4,
        "conv_pos_groups": 2,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "feature_projection_dropout": 0.0,
    }
    values.update(overrides)
    return make_wav2vec2_base_config(**values)


EAT_TARGET_LENGTH = 64
EAT_LENGTHS = (10400, 9600, 8000)


def eat_model(**overrides: Any) -> AudioModelConfig:
    """Tiny EAT recipe (Kaldi fbank, EAT embedding with CLS, AltBlocks)."""
    values: dict[str, Any] = {
        "d_model": D_MODEL,
        "depth": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "target_length": EAT_TARGET_LENGTH,
        "max_time_patches": EAT_TARGET_LENGTH // 16,
        "runtime": runtime(),
    }
    values.update(overrides)
    return make_eat_scaled_config(**values)


def dasheng_like_model(**overrides: Any) -> AudioModelConfig:
    """Tiny Dasheng-style recipe (lenient mel framing, log-mel BN, F_patches=1)."""
    values: dict[str, Any] = {
        "processor": melspec(
            log_mel=True, batch_norm=True, frame_validity="overlap_any"
        ),
        "tokenizer": patchify(patch_shape=(4, N_MELS)),
        "embedding": linear_embedding(pe_learnable_2d(max_t=MAX_TIME_PATCHES, max_f=1)),
        "encoder": transformer(positional_encoding=None, final_norm=True),
    }
    values.update(overrides)
    return model(**values)


def mae_model(**overrides: Any) -> AudioModelConfig:
    values: dict[str, Any] = {
        "processor": melspec(log_mel=True, batch_norm=True),
        "embedding": linear_embedding(pe_learnable_2d()),
        "encoder": transformer(positional_encoding=None, final_norm=True),
    }
    values.update(overrides)
    return model(**values)


def dropout_model(p: float = 0.2) -> AudioModelConfig:
    return model(encoder=transformer(mha=mha(dropout_p=p), mlp=mlp(dropout_p=p)))


# -------------------------------------------------------------- objectives ---


def projection_head(**overrides: Any) -> ProjectionHeadConfig:
    return _with(
        ProjectionHeadConfig(
            depth=2,
            hidden_dim=D_FF,
            output_dim=D_MODEL,
            activation="gelu",
            final_l2_normalize=False,
        ),
        overrides,
    )


def contrastive_objective(**overrides: Any) -> ContrastiveObjectiveConfig:
    return _with(
        ContrastiveObjectiveConfig(
            view_strategy="augmentation",
            projection=projection_head(final_l2_normalize=True),
            temperature=0.5,
        ),
        overrides,
    )


def byol_objective(**overrides: Any) -> BYOLObjectiveConfig:
    return _with(
        BYOLObjectiveConfig(
            projector=projection_head(),
            predictor=projection_head(),
            teacher_ema_decay=0.9,
        ),
        overrides,
    )


def dino_objective(**overrides: Any) -> DINOObjectiveConfig:
    return _with(
        DINOObjectiveConfig(
            head=DINOHeadConfig(
                depth=2,
                hidden_dim=D_FF,
                bottleneck_dim=8,
                num_prototypes=11,
                activation="gelu",
            ),
            teacher_ema_decay=0.9,
            teacher_temperature=0.07,
            student_temperature=0.2,
            # With 2 views x 3 clips the center tracks a 6-sample mean; faster
            # momentum makes fixed-batch targets chase the student (loss rises
            # toward ln K), which is genuine DINO dynamics, not a defect.
            center_momentum=0.99,
        ),
        overrides,
    )


def jepa_objective(**overrides: Any) -> JEPAObjectiveConfig:
    return _with(
        JEPAObjectiveConfig(
            loss_style="mse",
            mask_token_init_scale=0.1,
            teacher_ema_decay=0.9,
            predictor_config=mlp(),
            masking=TemporalMaskingConfig(mask_ratio=0.35, min_span_len=1, max_span_len=3),
        ),
        overrides,
    )


def mae_objective(**overrides: Any) -> MAEObjectiveConfig:
    return _with(
        MAEObjectiveConfig(
            decoder_d_model=D_MODEL,
            decoder=transformer(num_layers=1, positional_encoding=None, final_norm=True),
            decoder_positional_encoding=pe_learnable_2d(),
            masking=MAEMaskingConfig(mask_ratio=0.5),
        ),
        overrides,
    )


def ufo_objective(**overrides: Any) -> UFOObjectiveConfig:
    return _with(
        UFOObjectiveConfig(
            decoder=UFODecoderConfig(num_layers=2, kernel_size=3),
            masking=UFOMaskingConfig(mask_ratio=0.5, block_size=(2, 2), num_masks=2),
            teacher_ema_decay=0.9,
        ),
        overrides,
    )


def wav2vec2_objective(**overrides: Any) -> Wav2Vec2ObjectiveConfig:
    return _with(
        Wav2Vec2ObjectiveConfig(
            quantizer=Wav2Vec2QuantizerConfig(
                # Decays visibly within a few steps so resume tests cover scheduler state.
                temperature_scheduler=ExponentialDecayScalarSchedulerConfig(
                    start=2.0, end=0.5, decay=0.8
                ),
                num_groups=2,
                num_vars=8,
                vq_dim=D_MODEL,
            ),
            time_masking=SpanMaskingConfig(mask_prob=0.3, mask_length=2),
            feature_masking=ChannelMaskingConfig(mask_prob=0.1, mask_length=2),
            final_dim=D_MODEL,
            num_negatives=3,
        ),
        overrides,
    )


HUBERT_NUM_CODES = 8


def hubert_objective(**overrides: Any) -> HuBERTObjectiveConfig:
    return _with(
        HuBERTObjectiveConfig(
            teacher=HuBERTTeacherConfig(num_codes=HUBERT_NUM_CODES, encoder_revision="0" * 40),
            time_masking=SpanMaskingConfig(mask_prob=0.3, mask_length=2),
            final_dim=D_MODEL,
            unmasked_loss_weight=0.5,
        ),
        overrides,
    )


# ---------------------------------------------------------------- training ---


def training(
    objective: Any, model_config: AudioModelConfig | None = None
) -> AudioSSLTrainingConfig:
    """Minimal one-epoch training config around ``objective`` and ``model_config``."""
    return AudioSSLTrainingConfig(
        model=model_config if model_config is not None else model(),
        objective=objective,
        trainer=TrainerConfig(
            augmentor=AudioAugmentorConfig(augmentations=[]),
            optimizer=AdamWOptimizerConfig(lr=1e-3),
            scheduler=ConstantSchedulerConfig(),
            batch=BatchConfig(per_device_batch_size=2),
            loop=TrainingLoopConfig(max_epochs=1),
        ),
    )

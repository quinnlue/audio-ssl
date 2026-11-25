"""Config validation for frozen dataclass configs.

Checks field types, allowed literals, and cross-field relationships before
runtime objects are built. Validation policy and exception messages are defined
in ``docs/contracts``; this module implements those checks.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from audiossl.modelling.registry import ACTIVATION_REGISTRY, INITIALIZATION_REGISTRY

from .capabilities import PREFIX_TOKEN, check_objective_compatibility, component_capabilities
from .config import (
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
    ConstantScalarSchedulerConfig,
    ConstantSchedulerConfig,
    ContrastiveObjectiveConfig,
    ConvEmbeddingConfig,
    ConvModuleConfig,
    ConvolutionalPositionalEncodingConfig,
    ConvolutionalProcessorConfig,
    CosineAnnealingSchedulerConfig,
    DINOHeadConfig,
    DINOObjectiveConfig,
    EATEmbeddingConfig,
    EBranchformerEncoderConfig,
    EBranchMergeConfig,
    EmbeddingPositionalEncodingConfig,
    EmbeddingSubsamplingConfig,
    ExponentialDecayScalarSchedulerConfig,
    HuBERTObjectiveConfig,
    HuBERTTeacherConfig,
    JEPAObjectiveConfig,
    KaldiFbankProcessorConfig,
    LARSOptimizerConfig,
    LearnablePositionalEncodingConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    LinearEmbeddingConfig,
    LoudnessAugmentationConfig,
    MAEMaskingConfig,
    MAEObjectiveConfig,
    MaskingStrategyConfig,
    MelSpectrogramProcessorConfig,
    MixedPrecisionConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoOpTokenizerConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    PerBlockPositionalEncodingConfig,
    PositionalEncodingConfig,
    ProjectionHeadConfig,
    RopePositionalEncodingConfig,
    RuntimeConfig,
    ScalarSchedulerConfig,
    SGDOptimizerConfig,
    Sinusoidal2DPositionalEncodingConfig,
    SinusoidalPositionalEncodingConfig,
    SpanMaskingConfig,
    StepLRSchedulerConfig,
    TemporalMaskingConfig,
    TrainerConfig,
    TrainingLoopConfig,
    TransformerEncoderConfig,
    UFODecoderConfig,
    UFOMaskingConfig,
    UFOObjectiveConfig,
    WarmupCosineSchedulerConfig,
    Wav2Vec2FeatureExtractorProcessorConfig,
    Wav2Vec2ObjectiveConfig,
    Wav2Vec2QuantizerConfig,
)
from .logging import WandBConfig
from .registry import in_category, validator_for
from .run import CheckpointConfig, EvaluationConfig, LoaderConfig, RunConfig
from .logging import LoggerConfig

_MELSPEC_PAD_MODES = {"constant", "reflect", "replicate", "circular"}
_MELSPEC_WINDOW_FNS = {"hann"}
_SPECTROGRAM_NORMALIZED = {"window", "frame_length"}
_MEL_SCALE_NORMS = {"slaney"}
_MEL_SCALES = {"htk", "slaney"}
_CONV_PROCESSOR_ACTIVATIONS = {"gelu", "relu", "silu", "gelu_fast"}
_CONTRASTIVE_VIEW_STRATEGIES = {"augmentation", "dropout"}
_JEPA_TEACHER_MODES = {"ema", "stop_grad"}
_JEPA_OUTPUT_NORM_TYPES = {"l2", "layernorm"}
_MAE_NORM_PIX_LOSS = {"per_patch", "global", "none"}
_W2V2_MASK_SELECTIONS = {"static", "uniform", "normal", "poisson"}
_BRANCH_MERGE_METHODS = {"concat", "learned_ave", "fixed_ave"}
_CGMLP_GATE_ACTIVATIONS = {"identity", "gelu", "relu", "silu", "gelu_fast", "sigmoid", "tanh"}
_OPTIMIZER_TYPES = {"adamw", "sgd", "lars"}
_SCHEDULER_TYPES = {"constant", "warmup_cosine", "cosine_annealing", "step_lr"}
_WANDB_MODES = {"online", "offline", "disabled"}


def _validate_registered(config: Any, *, ctx: str, expected: str, **context: Any) -> None:
    """Validate a plug-in config through its registered validator or raise ``TypeError``."""
    validator = validator_for(config)
    if validator is None:
        raise TypeError(
            f"{ctx} must be {expected}, or a config with a validator registered through "
            f"audiossl.config.registry.register_validator; got {type(config).__name__}"
        )
    validator(config, **context)


def _require_type(name: str, value: Any, expected_type: type) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value).__name__}")


def _require_str(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_int(name: str, value: Any, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")


def _require_int_sequence(name: str, value: Any, *, minimum: int) -> None:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be list[int] or tuple[int, ...], got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must contain at least one value")
    for index, item in enumerate(value):
        _require_int(f"{name}[{index}]", item, minimum=minimum)


def _validate_no_weight_decay_patterns(name: str, patterns: Any) -> None:
    if not isinstance(patterns, tuple):
        raise TypeError(
            f"{name} must be tuple[str, ...], got {type(patterns).__name__}"
        )
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, str):
            raise TypeError(
                f"{name}[{index}] must be str, got {type(pattern).__name__}"
            )
        if not pattern:
            raise ValueError(f"{name}[{index}] must not be empty")
    if len(set(patterns)) != len(patterns):
        raise ValueError(f"{name} must not contain duplicate patterns, got {patterns!r}")


def _require_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        if exclusive_minimum and numeric_value <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {value!r}")
        if not exclusive_minimum and numeric_value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and numeric_value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")


def _validate_runtime(config: RuntimeConfig) -> None:
    _require_str("RuntimeConfig.device", config.device)
    _require_int("RuntimeConfig.seed", config.seed, minimum=0)
    if config.dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError(f"Unsupported RuntimeConfig.dtype {config.dtype!r}")
    _require_bool("RuntimeConfig.deterministic", config.deterministic)
    _require_bool("RuntimeConfig.allow_tf32", config.allow_tf32)


def _validate_mha(config: MultiHeadAttentionConfig, *, d_model: int | None = None) -> None:
    _require_int("MultiHeadAttentionConfig.num_heads", config.num_heads, minimum=1)
    if d_model is not None and d_model % config.num_heads != 0:
        raise ValueError(
            "AudioModelConfig.d_model must be divisible by "
            f"MultiHeadAttentionConfig.num_heads, got {d_model} and {config.num_heads}"
        )
    _require_number(
        "MultiHeadAttentionConfig.dropout_p",
        config.dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str("MultiHeadAttentionConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported MultiHeadAttentionConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_mlp(config: MLPConfig) -> None:
    _require_int("MLPConfig.d_ff", config.d_ff, minimum=1)
    _require_number("MLPConfig.dropout_p", config.dropout_p, minimum=0.0, maximum=1.0)
    _require_str("MLPConfig.activation", config.activation)
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported MLPConfig.activation {config.activation!r}; "
            f"expected one of {sorted(ACTIVATION_REGISTRY)}"
        )
    _require_str("MLPConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported MLPConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_int_pair(name: str, value: object) -> None:
    if not isinstance(value, tuple | list):
        raise TypeError(
            f"{name} must be tuple[int, int], "
            f"got {type(value).__name__}"
        )
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    for index, item in enumerate(value):
        _require_int(
            f"{name}[{index}]",
            item,
            minimum=1,
        )


def _validate_patchify_tokenizer(config: PatchifyTokenizerConfig) -> None:
    if config.type != "patchify":
        raise ValueError(f"Unsupported PatchifyTokenizerConfig.type {config.type!r}")
    _validate_int_pair("PatchifyTokenizerConfig.patch_shape", config.patch_shape)
    if config.stride is not None:
        _validate_int_pair("PatchifyTokenizerConfig.stride", config.stride)
    if config.padding_strategy not in {"pad", "truncate"}:
        raise ValueError(
            "PatchifyTokenizerConfig.padding_strategy must be one of "
            "{'pad', 'truncate'}"
        )


def _validate_noop_tokenizer(config: NoOpTokenizerConfig) -> None:
    if config.type != "none":
        raise ValueError(f"Unsupported NoOpTokenizerConfig.type {config.type!r}")
    _require_int("NoOpTokenizerConfig._output_dim", config._output_dim, minimum=1)


def _validate_tokenizer(
    config: PatchifyTokenizerConfig | NoOpTokenizerConfig,
) -> None:
    if isinstance(config, PatchifyTokenizerConfig):
        _validate_patchify_tokenizer(config)
    elif isinstance(config, NoOpTokenizerConfig):
        _validate_noop_tokenizer(config)
    else:
        _validate_registered(
            config,
            ctx="AudioModelConfig.tokenizer",
            expected="PatchifyTokenizerConfig or NoOpTokenizerConfig",
        )


def _validate_embedding_subsampling(
    config: EmbeddingSubsamplingConfig | None,
    *,
    ctx: str,
) -> None:
    if config is None:
        return
    _require_type(ctx, config, EmbeddingSubsamplingConfig)
    if config.type != "strided_select":
        raise ValueError(f"Unsupported {ctx}.type {config.type!r}")
    _require_int(f"{ctx}.factor", config.factor, minimum=1)


def _validate_linear_embedding(config: LinearEmbeddingConfig) -> None:
    if config.type != "linear":
        raise ValueError(f"Unsupported LinearEmbeddingConfig.type {config.type!r}")
    _require_number(
        "LinearEmbeddingConfig.dropout_p",
        config.dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str("LinearEmbeddingConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported LinearEmbeddingConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )
    _validate_embedding_positional_encoding(config.positional_encoding)
    _validate_embedding_subsampling(
        config.subsampling,
        ctx="LinearEmbeddingConfig.subsampling",
    )
    _require_bool("LinearEmbeddingConfig.input_layer_norm", config.input_layer_norm)
    _require_number(
        "LinearEmbeddingConfig.input_layer_norm_eps",
        config.input_layer_norm_eps,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_bool("LinearEmbeddingConfig.post_pos_layer_norm", config.post_pos_layer_norm)
    _require_number(
        "LinearEmbeddingConfig.post_pos_layer_norm_eps",
        config.post_pos_layer_norm_eps,
        minimum=0.0,
        exclusive_minimum=True,
    )


def _validate_conv_embedding(config: ConvEmbeddingConfig) -> None:
    if config.type != "conv":
        raise ValueError(f"Unsupported ConvEmbeddingConfig.type {config.type!r}")
    _require_int("ConvEmbeddingConfig.kernel_size", config.kernel_size, minimum=1)
    if config.kernel_size % 2 == 0:
        raise ValueError(
            "ConvEmbeddingConfig.kernel_size must be odd so same-padding preserves "
            f"the token count, got {config.kernel_size!r}"
        )
    _require_number(
        "ConvEmbeddingConfig.dropout_p",
        config.dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str("ConvEmbeddingConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported ConvEmbeddingConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )
    _require_bool("ConvEmbeddingConfig.layer_norm", config.layer_norm)
    _validate_embedding_positional_encoding(config.positional_encoding)
    _validate_embedding_subsampling(
        config.subsampling,
        ctx="ConvEmbeddingConfig.subsampling",
    )


def _validate_eat_embedding(config: EATEmbeddingConfig) -> None:
    if config.type != "eat":
        raise ValueError(f"Unsupported EATEmbeddingConfig.type {config.type!r}")
    _require_number(
        "EATEmbeddingConfig.dropout_p",
        config.dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str("EATEmbeddingConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported EATEmbeddingConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )
    _require_type(
        "EATEmbeddingConfig.positional_encoding",
        config.positional_encoding,
        Sinusoidal2DPositionalEncodingConfig,
    )
    _validate_sinusoidal_2d_positional_encoding(config.positional_encoding)
    _require_number(
        "EATEmbeddingConfig.entry_layer_norm_eps",
        config.entry_layer_norm_eps,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_number(
        "EATEmbeddingConfig.cls_init_std",
        config.cls_init_std,
        minimum=0.0,
        exclusive_minimum=True,
    )


def _validate_embedding(
    config: (
        LinearEmbeddingConfig
        | ConvEmbeddingConfig
        | EATEmbeddingConfig
    ),
) -> None:
    if isinstance(config, LinearEmbeddingConfig):
        _validate_linear_embedding(config)
    elif isinstance(config, ConvEmbeddingConfig):
        _validate_conv_embedding(config)
    elif isinstance(config, EATEmbeddingConfig):
        _validate_eat_embedding(config)
    else:
        _validate_registered(
            config,
            ctx="AudioModelConfig.embedding",
            expected="LinearEmbeddingConfig, ConvEmbeddingConfig, or EATEmbeddingConfig",
        )


def _validate_conv_module(config: ConvModuleConfig) -> None:
    _require_int("ConvModuleConfig.kernel_size", config.kernel_size, minimum=1)
    if config.kernel_size % 2 == 0:
        raise ValueError(
            "ConvModuleConfig.kernel_size must be odd, "
            f"got {config.kernel_size!r}"
        )
    _require_int("ConvModuleConfig.expansion_factor", config.expansion_factor, minimum=1)
    if config.expansion_factor % 2 != 0:
        raise ValueError(
            "ConvModuleConfig.expansion_factor must be even (GLU halves channels), "
            f"got {config.expansion_factor!r}"
        )
    _require_number(
        "ConvModuleConfig.dropout_p",
        config.dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str("ConvModuleConfig.activation", config.activation)
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported ConvModuleConfig.activation {config.activation!r}; "
            f"expected one of {sorted(ACTIVATION_REGISTRY)}"
        )


def _validate_cgmlp(config: CGMLPConfig) -> None:
    _require_int("CGMLPConfig.linear_units", config.linear_units, minimum=1)
    if config.linear_units % 2 != 0:
        raise ValueError(
            "CGMLPConfig.linear_units must be even so the up-projection can split "
            f"into equal gate/value halves, got {config.linear_units!r}"
        )
    _require_int("CGMLPConfig.conv_kernel_size", config.conv_kernel_size, minimum=1)
    if config.conv_kernel_size % 2 == 0:
        raise ValueError(
            "CGMLPConfig.conv_kernel_size must be odd, "
            f"got {config.conv_kernel_size!r}"
        )
    _require_str("CGMLPConfig.activation", config.activation)
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported CGMLPConfig.activation {config.activation!r}; "
            f"expected one of {sorted(ACTIVATION_REGISTRY)}"
        )
    _require_str("CGMLPConfig.gate_activation", config.gate_activation)
    if config.gate_activation not in _CGMLP_GATE_ACTIVATIONS:
        raise ValueError(
            f"Unsupported CGMLPConfig.gate_activation {config.gate_activation!r}; "
            f"expected one of {sorted(_CGMLP_GATE_ACTIVATIONS)}"
        )
    _require_number("CGMLPConfig.dropout_p", config.dropout_p, minimum=0.0, maximum=1.0)


def _validate_branch_merge(config: BranchMergeConfig) -> None:
    _require_str("BranchMergeConfig.method", config.method)
    if config.method not in _BRANCH_MERGE_METHODS:
        raise ValueError(
            f"Unsupported BranchMergeConfig.method {config.method!r}; "
            f"expected one of {sorted(_BRANCH_MERGE_METHODS)}"
        )
    _require_number(
        "BranchMergeConfig.fixed_weight",
        config.fixed_weight,
        minimum=0.0,
        maximum=1.0,
    )


def _validate_ebranch_merge(config: EBranchMergeConfig) -> None:
    _require_int(
        "EBranchMergeConfig.conv_kernel_size",
        config.conv_kernel_size,
        minimum=1,
    )
    if config.conv_kernel_size % 2 == 0:
        raise ValueError(
            "EBranchMergeConfig.conv_kernel_size must be a positive odd integer, "
            f"got {config.conv_kernel_size!r}"
        )


def _validate_rope_positional_encoding(config: RopePositionalEncodingConfig) -> None:
    _require_int("RopePositionalEncodingConfig.max_len", config.max_len, minimum=1)
    if config.type != "rope":
        raise ValueError(f"Unsupported RopePositionalEncodingConfig.type {config.type!r}")
    _require_number(
        "RopePositionalEncodingConfig.base",
        config.base,
        minimum=0.0,
        exclusive_minimum=True,
    )


def _validate_no_positional_encoding(config: NoPositionalEncodingConfig) -> None:
    if config.type != "none":
        raise ValueError(f"Unsupported NoPositionalEncodingConfig.type {config.type!r}")


def _validate_learnable_positional_encoding(
    config: LearnablePositionalEncodingConfig,
) -> None:
    _require_int("LearnablePositionalEncodingConfig.max_len", config.max_len, minimum=1)
    if config.type != "learnable":
        raise ValueError(
            f"Unsupported LearnablePositionalEncodingConfig.type {config.type!r}"
        )


def _validate_sinusoidal_positional_encoding(
    config: SinusoidalPositionalEncodingConfig,
) -> None:
    _require_int("SinusoidalPositionalEncodingConfig.max_len", config.max_len, minimum=1)
    if config.type != "sinusoidal":
        raise ValueError(
            f"Unsupported SinusoidalPositionalEncodingConfig.type {config.type!r}"
        )
    _require_number(
        "SinusoidalPositionalEncodingConfig.base",
        config.base,
        minimum=0.0,
        exclusive_minimum=True,
    )


def _validate_sinusoidal_2d_positional_encoding(
    config: Sinusoidal2DPositionalEncodingConfig,
) -> None:
    if config.type != "sinusoidal_2d":
        raise ValueError(
            f"Unsupported Sinusoidal2DPositionalEncodingConfig.type {config.type!r}"
        )
    _require_int(
        "Sinusoidal2DPositionalEncodingConfig.max_time_patches",
        config.max_time_patches,
        minimum=1,
    )
    _require_int(
        "Sinusoidal2DPositionalEncodingConfig.max_freq_patches",
        config.max_freq_patches,
        minimum=1,
    )


def _validate_kaldi_fbank_processor(config: KaldiFbankProcessorConfig) -> None:
    if config.type != "kaldi_fbank":
        raise ValueError(f"Unsupported KaldiFbankProcessorConfig.type {config.type!r}")
    _require_int("KaldiFbankProcessorConfig.sr", config.sr, minimum=1)
    _require_int("KaldiFbankProcessorConfig.num_mel_bins", config.num_mel_bins, minimum=1)
    _require_int("KaldiFbankProcessorConfig.frame_shift_ms", config.frame_shift_ms, minimum=1)
    _require_int("KaldiFbankProcessorConfig.target_length", config.target_length, minimum=1)
    if config.target_length % 16 != 0:
        raise ValueError("KaldiFbankProcessorConfig.target_length must be a multiple of 16")
    _require_bool(
        "KaldiFbankProcessorConfig.remove_waveform_mean",
        config.remove_waveform_mean,
    )
    _require_bool("KaldiFbankProcessorConfig.htk_compat", config.htk_compat)
    _require_bool("KaldiFbankProcessorConfig.use_energy", config.use_energy)
    _require_number("KaldiFbankProcessorConfig.dither", config.dither, minimum=0.0)
    _require_str("KaldiFbankProcessorConfig.window_type", config.window_type)
    if config.window_type != "hanning":
        raise ValueError(
            f"Unsupported KaldiFbankProcessorConfig.window_type {config.window_type!r}"
        )


def _validate_learnable_tf_positional_encoding(
    config: LearnableTimeFrequencyPositionalEncodingConfig,
) -> None:
    if config.type != "learnable_2d":
        raise ValueError(
            "Unsupported LearnableTimeFrequencyPositionalEncodingConfig.type "
            f"{config.type!r}"
        )
    _require_int(
        "LearnableTimeFrequencyPositionalEncodingConfig.max_t",
        config.max_t,
        minimum=1,
    )
    _require_int(
        "LearnableTimeFrequencyPositionalEncodingConfig.max_f",
        config.max_f,
        minimum=1,
    )


def _validate_convolutional_positional_encoding(
    config: ConvolutionalPositionalEncodingConfig,
) -> None:
    if config.type != "conv1d":
        raise ValueError(
            f"Unsupported ConvolutionalPositionalEncodingConfig.type {config.type!r}"
        )
    _require_int("ConvolutionalPositionalEncodingConfig.kernel_size", config.kernel_size, minimum=1)
    _require_int("ConvolutionalPositionalEncodingConfig.groups", config.groups, minimum=1)
    _require_str("ConvolutionalPositionalEncodingConfig.activation", config.activation)
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            "Unsupported ConvolutionalPositionalEncodingConfig.activation "
            f"{config.activation!r}; expected one of {sorted(ACTIVATION_REGISTRY)}"
        )
    _require_str("ConvolutionalPositionalEncodingConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            "Unsupported ConvolutionalPositionalEncodingConfig.initialization "
            f"{config.initialization!r}; expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_embedding_positional_encoding(
    config: EmbeddingPositionalEncodingConfig,
) -> None:
    if isinstance(config, NoPositionalEncodingConfig):
        _validate_no_positional_encoding(config)
    elif isinstance(config, LearnablePositionalEncodingConfig):
        _validate_learnable_positional_encoding(config)
    elif isinstance(config, SinusoidalPositionalEncodingConfig):
        _validate_sinusoidal_positional_encoding(config)
    elif isinstance(config, LearnableTimeFrequencyPositionalEncodingConfig):
        _validate_learnable_tf_positional_encoding(config)
    elif isinstance(config, Sinusoidal2DPositionalEncodingConfig):
        _validate_sinusoidal_2d_positional_encoding(config)
    elif isinstance(config, ConvolutionalPositionalEncodingConfig):
        _validate_convolutional_positional_encoding(config)
    else:
        _validate_registered(
            config,
            ctx="Embedding positional encoding config",
            expected=(
                "NoPositionalEncodingConfig, LearnablePositionalEncodingConfig, "
                "SinusoidalPositionalEncodingConfig, "
                "LearnableTimeFrequencyPositionalEncodingConfig, "
                "Sinusoidal2DPositionalEncodingConfig, "
                "or ConvolutionalPositionalEncodingConfig"
            ),
        )


def _validate_per_block_positional_encoding(
    config: PerBlockPositionalEncodingConfig,
) -> None:
    if isinstance(config, RopePositionalEncodingConfig):
        _validate_rope_positional_encoding(config)
    else:
        _validate_registered(
            config,
            ctx="Encoder per-block positional encoding config",
            expected="RopePositionalEncodingConfig",
        )


def _validate_positional_encoding(config: PositionalEncodingConfig) -> None:
    if isinstance(config, RopePositionalEncodingConfig):
        _validate_per_block_positional_encoding(config)
    else:
        _validate_embedding_positional_encoding(config)


def _validate_melspectrogram_processor(config: MelSpectrogramProcessorConfig) -> None:
    if config.type != "melspec":
        raise ValueError(f"Unsupported MelSpectrogramProcessorConfig.type {config.type!r}")
    _require_int("MelSpectrogramProcessorConfig.n_mels", config.n_mels, minimum=1)
    _require_int("MelSpectrogramProcessorConfig.n_fft", config.n_fft, minimum=1)
    _require_number("MelSpectrogramProcessorConfig.f_min", config.f_min, minimum=0.0)
    _require_int("MelSpectrogramProcessorConfig.sr", config.sr, minimum=1)
    nyquist_hz = float(config.sr) / 2.0
    if float(config.f_min) >= nyquist_hz:
        raise ValueError("MelSpectrogramProcessorConfig.f_min must be less than sr / 2")
    if config.f_max is not None:
        _require_number(
            "MelSpectrogramProcessorConfig.f_max",
            config.f_max,
            exclusive_minimum=True,
            minimum=0.0,
        )
        if float(config.f_max) <= float(config.f_min):
            raise ValueError("MelSpectrogramProcessorConfig.f_max must be greater than f_min")
        if float(config.f_max) > nyquist_hz:
            raise ValueError("MelSpectrogramProcessorConfig.f_max must be <= sr / 2")
    _require_int("MelSpectrogramProcessorConfig.win_length", config.win_length, minimum=1)
    if config.win_length > config.n_fft:
        raise ValueError("MelSpectrogramProcessorConfig.win_length must be <= n_fft")
    _require_int("MelSpectrogramProcessorConfig.hop_length", config.hop_length, minimum=1)
    _require_int("MelSpectrogramProcessorConfig.pad", config.pad, minimum=0)
    _require_str("MelSpectrogramProcessorConfig.window_fn", config.window_fn)
    if config.window_fn not in _MELSPEC_WINDOW_FNS:
        raise ValueError(
            f"Unsupported MelSpectrogramProcessorConfig.window_fn {config.window_fn!r}"
        )
    _require_number(
        "MelSpectrogramProcessorConfig.power",
        config.power,
        minimum=0.0,
        exclusive_minimum=True,
    )
    if isinstance(config.normalized, bool):
        pass
    elif isinstance(config.normalized, str):
        if config.normalized not in _SPECTROGRAM_NORMALIZED:
            raise ValueError(
                f"Unsupported MelSpectrogramProcessorConfig.normalized {config.normalized!r}"
            )
    else:
        raise TypeError(
            "MelSpectrogramProcessorConfig.normalized must be bool or one of "
            f"{sorted(_SPECTROGRAM_NORMALIZED)!r}, got {type(config.normalized).__name__}"
        )
    _require_bool("MelSpectrogramProcessorConfig.center", config.center)
    _require_str("MelSpectrogramProcessorConfig.pad_mode", config.pad_mode)
    if config.pad_mode not in _MELSPEC_PAD_MODES:
        raise ValueError(f"Unsupported MelSpectrogramProcessorConfig.pad_mode {config.pad_mode!r}")
    if config.norm is not None:
        _require_str("MelSpectrogramProcessorConfig.norm", config.norm)
        if config.norm not in _MEL_SCALE_NORMS:
            raise ValueError(f"Unsupported MelSpectrogramProcessorConfig.norm {config.norm!r}")
    _require_str("MelSpectrogramProcessorConfig.mel_scale", config.mel_scale)
    if config.mel_scale not in _MEL_SCALES:
        raise ValueError(
            f"Unsupported MelSpectrogramProcessorConfig.mel_scale {config.mel_scale!r}"
        )
    _require_bool("MelSpectrogramProcessorConfig.log_mel", config.log_mel)
    if config.log_mel_top_db is not None:
        _require_number(
            "MelSpectrogramProcessorConfig.log_mel_top_db",
            config.log_mel_top_db,
            minimum=0.0,
            exclusive_minimum=True,
        )
    _require_bool("MelSpectrogramProcessorConfig.batch_norm", config.batch_norm)
    _require_number(
        "MelSpectrogramProcessorConfig.batch_norm_momentum",
        config.batch_norm_momentum,
        minimum=0.0,
        maximum=1.0,
    )
    _require_number(
        "MelSpectrogramProcessorConfig.batch_norm_eps",
        config.batch_norm_eps,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_str("MelSpectrogramProcessorConfig.frame_validity", config.frame_validity)
    if config.frame_validity not in {"strict_window", "overlap_any"}:
        raise ValueError(
            "MelSpectrogramProcessorConfig.frame_validity must be one of "
            "{'strict_window', 'overlap_any'}"
        )


def _validate_convolutional_processor(config: ConvolutionalProcessorConfig) -> None:
    if config.type != "conv":
        raise ValueError(f"Unsupported ConvolutionalProcessorConfig.type {config.type!r}")
    _require_int("ConvolutionalProcessorConfig.in_channels", config.in_channels, minimum=1)
    if config.in_channels != 1:
        raise ValueError("ConvolutionalProcessorConfig.in_channels must be 1 for raw mono audio")
    _require_int("ConvolutionalProcessorConfig.out_channels", config.out_channels, minimum=1)
    _require_int_sequence(
        "ConvolutionalProcessorConfig.kernel_sizes",
        config.kernel_sizes,
        minimum=1,
    )
    _require_int_sequence("ConvolutionalProcessorConfig.strides", config.strides, minimum=1)
    if len(config.kernel_sizes) != len(config.strides):
        raise ValueError(
            "ConvolutionalProcessorConfig.kernel_sizes and strides must have the same length, "
            f"got {len(config.kernel_sizes)} and {len(config.strides)}"
        )
    _require_int("ConvolutionalProcessorConfig.groups", config.groups, minimum=1)
    _require_int("ConvolutionalProcessorConfig.padding", config.padding, minimum=0)
    _require_int("ConvolutionalProcessorConfig.dilation", config.dilation, minimum=1)
    _require_bool("ConvolutionalProcessorConfig.bias", config.bias)
    _require_str("ConvolutionalProcessorConfig.activation", config.activation)
    if config.activation not in _CONV_PROCESSOR_ACTIVATIONS:
        raise ValueError(
            f"Unsupported ConvolutionalProcessorConfig.activation {config.activation!r}"
        )
    _require_str("ConvolutionalProcessorConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported ConvolutionalProcessorConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )
    if config.in_channels % config.groups != 0:
        raise ValueError("ConvolutionalProcessorConfig.groups must divide in_channels")
    if config.out_channels % config.groups != 0:
        raise ValueError("ConvolutionalProcessorConfig.groups must divide out_channels")


def _validate_wav2vec2_feature_extractor_processor(
    config: Wav2Vec2FeatureExtractorProcessorConfig,
) -> None:
    if config.type != "wav2vec2_feature_extractor":
        raise ValueError(
            "Unsupported Wav2Vec2FeatureExtractorProcessorConfig.type "
            f"{config.type!r}"
        )
    _require_int("Wav2Vec2FeatureExtractorProcessorConfig.sr", config.sr, minimum=1)
    _require_int_sequence(
        "Wav2Vec2FeatureExtractorProcessorConfig.conv_dim",
        config.conv_dim,
        minimum=1,
    )
    _require_int_sequence(
        "Wav2Vec2FeatureExtractorProcessorConfig.conv_kernel",
        config.conv_kernel,
        minimum=1,
    )
    _require_int_sequence(
        "Wav2Vec2FeatureExtractorProcessorConfig.conv_stride",
        config.conv_stride,
        minimum=1,
    )
    if not (
        len(config.conv_dim)
        == len(config.conv_kernel)
        == len(config.conv_stride)
    ):
        raise ValueError(
            "Wav2Vec2FeatureExtractorProcessorConfig conv_dim, conv_kernel, "
            "and conv_stride must have the same length, got "
            f"{len(config.conv_dim)}, {len(config.conv_kernel)}, "
            f"and {len(config.conv_stride)}"
        )
    _require_bool(
        "Wav2Vec2FeatureExtractorProcessorConfig.conv_bias",
        config.conv_bias,
    )
    _require_str(
        "Wav2Vec2FeatureExtractorProcessorConfig.activation",
        config.activation,
    )
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            "Unsupported Wav2Vec2FeatureExtractorProcessorConfig.activation "
            f"{config.activation!r}; expected one of {sorted(ACTIVATION_REGISTRY)}"
        )
    if config.extractor_mode not in {"default", "layer_norm"}:
        raise ValueError(
            "Unsupported Wav2Vec2FeatureExtractorProcessorConfig.extractor_mode "
            f"{config.extractor_mode!r}; expected one of "
            "{'default', 'layer_norm'}"
        )
    _require_int(
        "Wav2Vec2FeatureExtractorProcessorConfig.group_norm_num_groups",
        config.group_norm_num_groups,
        minimum=1,
    )
    if (
        config.extractor_mode == "default"
        and config.conv_dim[0] % config.group_norm_num_groups != 0
    ):
        raise ValueError(
            "Wav2Vec2FeatureExtractorProcessorConfig.group_norm_num_groups "
            "must divide conv_dim[0]"
        )
    _require_str(
        "Wav2Vec2FeatureExtractorProcessorConfig.initialization",
        config.initialization,
    )
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            "Unsupported Wav2Vec2FeatureExtractorProcessorConfig.initialization "
            f"{config.initialization!r}; expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_processor(
    config: (
        MelSpectrogramProcessorConfig
        | ConvolutionalProcessorConfig
        | Wav2Vec2FeatureExtractorProcessorConfig
        | KaldiFbankProcessorConfig
    ),
) -> None:
    if isinstance(config, MelSpectrogramProcessorConfig):
        _validate_melspectrogram_processor(config)
    elif isinstance(config, ConvolutionalProcessorConfig):
        _validate_convolutional_processor(config)
    elif isinstance(config, Wav2Vec2FeatureExtractorProcessorConfig):
        _validate_wav2vec2_feature_extractor_processor(config)
    elif isinstance(config, KaldiFbankProcessorConfig):
        _validate_kaldi_fbank_processor(config)
    else:
        _validate_registered(
            config,
            ctx="AudioModelConfig.processor",
            expected=(
                "MelSpectrogramProcessorConfig, ConvolutionalProcessorConfig, "
                "Wav2Vec2FeatureExtractorProcessorConfig, or KaldiFbankProcessorConfig"
            ),
        )


def _validate_encoder(
    config: (
        TransformerEncoderConfig
        | ConformerEncoderConfig
        | BranchformerEncoderConfig
        | EBranchformerEncoderConfig
        | AltTransformerEncoderConfig
    ),
    *,
    d_model: int,
) -> None:
    if isinstance(config, AltTransformerEncoderConfig):
        if config.type != "alt_transformer":
            raise ValueError(f"Unsupported AltTransformerEncoderConfig.type {config.type!r}")
        _require_type(f"{type(config).__name__}.mha", config.mha, MultiHeadAttentionConfig)
        _require_type(f"{type(config).__name__}.mlp", config.mlp, MLPConfig)
        _validate_mha(config.mha, d_model=d_model)
        _validate_mlp(config.mlp)
        _require_int(f"{type(config).__name__}.num_layers", config.num_layers, minimum=1)
        _require_bool(f"{type(config).__name__}.layer_norm_first", config.layer_norm_first)
        _require_bool(f"{type(config).__name__}.qkv_bias", config.qkv_bias)
        _require_number(
            f"{type(config).__name__}.post_mlp_dropout_p",
            config.post_mlp_dropout_p,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            f"{type(config).__name__}.residual_dropout_p",
            config.residual_dropout_p,
            minimum=0.0,
            maximum=1.0,
        )
        _require_str(f"{type(config).__name__}.pooling", config.pooling)
        if config.pooling not in {"masked_mean", "cls"}:
            raise ValueError(
                f"Unsupported {type(config).__name__}.pooling {config.pooling!r}; "
                "expected one of ['masked_mean', 'cls']"
            )
        _require_number(
            f"{type(config).__name__}.layer_norm_eps",
            config.layer_norm_eps,
            minimum=0.0,
            exclusive_minimum=True,
        )
        _require_bool(f"{type(config).__name__}.final_norm", config.final_norm)
        return
    if isinstance(config, TransformerEncoderConfig):
        if config.type != "transformer":
            raise ValueError(f"Unsupported TransformerEncoderConfig.type {config.type!r}")
        _require_type(f"{type(config).__name__}.mha", config.mha, MultiHeadAttentionConfig)
        _require_type(f"{type(config).__name__}.mlp", config.mlp, MLPConfig)
        _validate_mha(config.mha, d_model=d_model)
        _validate_mlp(config.mlp)
    elif isinstance(config, ConformerEncoderConfig):
        if config.type != "conformer":
            raise ValueError(f"Unsupported ConformerEncoderConfig.type {config.type!r}")
        _require_type(f"{type(config).__name__}.mha", config.mha, MultiHeadAttentionConfig)
        _require_type(f"{type(config).__name__}.mlp", config.mlp, MLPConfig)
        _require_type(f"{type(config).__name__}.conv", config.conv, ConvModuleConfig)
        _validate_mha(config.mha, d_model=d_model)
        _validate_mlp(config.mlp)
        _validate_conv_module(config.conv)
    elif isinstance(config, BranchformerEncoderConfig):
        if config.type != "branchformer":
            raise ValueError(
                f"Unsupported BranchformerEncoderConfig.type {config.type!r}"
            )
        _require_type(f"{type(config).__name__}.mha", config.mha, MultiHeadAttentionConfig)
        _require_type(f"{type(config).__name__}.cgmlp", config.cgmlp, CGMLPConfig)
        _require_type(f"{type(config).__name__}.merge", config.merge, BranchMergeConfig)
        _validate_mha(config.mha, d_model=d_model)
        _validate_cgmlp(config.cgmlp)
        _validate_branch_merge(config.merge)
        _require_number(
            "BranchformerEncoderConfig.attention_branch_drop_p",
            config.attention_branch_drop_p,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            "BranchformerEncoderConfig.cgmlp_branch_drop_p",
            config.cgmlp_branch_drop_p,
            minimum=0.0,
            maximum=1.0,
        )
    elif isinstance(config, EBranchformerEncoderConfig):
        if config.type != "ebranchformer":
            raise ValueError(
                f"Unsupported EBranchformerEncoderConfig.type {config.type!r}"
            )
        _require_type(f"{type(config).__name__}.mha", config.mha, MultiHeadAttentionConfig)
        _require_type(f"{type(config).__name__}.cgmlp", config.cgmlp, CGMLPConfig)
        _require_type(f"{type(config).__name__}.merge", config.merge, EBranchMergeConfig)
        _validate_mha(config.mha, d_model=d_model)
        _validate_cgmlp(config.cgmlp)
        _validate_ebranch_merge(config.merge)
        _require_bool(f"{type(config).__name__}.use_ffn", config.use_ffn)
        _require_bool(f"{type(config).__name__}.macaron_ffn", config.macaron_ffn)
        if config.use_ffn or config.macaron_ffn:
            if config.mlp is None:
                raise ValueError(
                    "EBranchformerEncoderConfig.mlp must be set when "
                    "use_ffn or macaron_ffn is True"
                )
            _require_type(f"{type(config).__name__}.mlp", config.mlp, MLPConfig)
            _validate_mlp(config.mlp)
        elif config.mlp is not None:
            _require_type(f"{type(config).__name__}.mlp", config.mlp, MLPConfig)
            _validate_mlp(config.mlp)
    else:
        _validate_registered(
            config,
            ctx="Encoder config",
            expected=(
                "TransformerEncoderConfig, ConformerEncoderConfig, BranchformerEncoderConfig, "
                "EBranchformerEncoderConfig, or AltTransformerEncoderConfig"
            ),
            d_model=d_model,
        )
        return

    _require_int(f"{type(config).__name__}.num_layers", config.num_layers, minimum=1)
    _require_bool(f"{type(config).__name__}.pre_norm", config.pre_norm)
    _require_number(
        f"{type(config).__name__}.residual_dropout_p",
        config.residual_dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str(f"{type(config).__name__}.pooling", config.pooling)
    if config.pooling != "masked_mean":
        raise ValueError(
            f"Unsupported {type(config).__name__}.pooling {config.pooling!r}; "
            "expected one of ['masked_mean']"
        )
    if config.positional_encoding is not None:
        _validate_per_block_positional_encoding(config.positional_encoding)
    _require_number(
        f"{type(config).__name__}.layer_norm_eps",
        config.layer_norm_eps,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_bool(f"{type(config).__name__}.final_norm", config.final_norm)


def _validate_model(config: AudioModelConfig) -> None:
    _require_int("AudioModelConfig.d_model", config.d_model, minimum=1)
    _validate_processor(config.processor)
    _validate_tokenizer(config.tokenizer)
    if (
        isinstance(config.tokenizer, NoOpTokenizerConfig)
        and config.tokenizer.output_dim != config.processor.output_dim
    ):
        raise ValueError(
            "NoOpTokenizerConfig.output_dim must match processor.output_dim, "
            f"got {config.tokenizer.output_dim} and {config.processor.output_dim}"
        )
    _validate_embedding(config.embedding)
    positional_encoding = getattr(config.embedding, "positional_encoding", None)
    if (
        isinstance(positional_encoding, ConvolutionalPositionalEncodingConfig)
        and config.d_model % positional_encoding.groups != 0
    ):
        raise ValueError(
            "AudioModelConfig.d_model must be divisible by "
            "ConvolutionalPositionalEncodingConfig.groups, "
            f"got {config.d_model} and {positional_encoding.groups}"
        )
    _require_type("AudioModelConfig.runtime", config.runtime, RuntimeConfig)
    _validate_runtime(config.runtime)
    _validate_encoder(config.encoder, d_model=config.d_model)
    if (
        getattr(config.encoder, "pooling", None) == "cls"
        and PREFIX_TOKEN not in component_capabilities(config.embedding)
    ):
        raise ValueError(
            f"{type(config.encoder).__name__}.pooling='cls' requires an embedding that adds "
            f"a prefix token; {type(config.embedding).__name__} does not"
        )


def _validate_projection_head_config(
    config: ProjectionHeadConfig,
    *,
    ctx: str = "ProjectionHeadConfig",
) -> None:
    _require_type(ctx, config, ProjectionHeadConfig)
    _require_int(f"{ctx}.depth", config.depth, minimum=1)
    _require_int(f"{ctx}.hidden_dim", config.hidden_dim, minimum=1)
    _require_int(f"{ctx}.output_dim", config.output_dim, minimum=1)
    _require_str(f"{ctx}.activation", config.activation)
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported {ctx}.activation {config.activation!r}; "
            f"expected one of {sorted(ACTIVATION_REGISTRY)}"
        )
    _require_bool(f"{ctx}.final_l2_normalize", config.final_l2_normalize)
    _require_number(
        f"{ctx}.dropout_p",
        config.dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str(f"{ctx}.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported {ctx}.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_dino_head(config: DINOHeadConfig, *, ctx: str = "DINOHeadConfig") -> None:
    _require_type(ctx, config, DINOHeadConfig)
    _require_int(f"{ctx}.depth", config.depth, minimum=1)
    _require_int(f"{ctx}.hidden_dim", config.hidden_dim, minimum=1)
    _require_int(f"{ctx}.bottleneck_dim", config.bottleneck_dim, minimum=1)
    _require_int(f"{ctx}.num_prototypes", config.num_prototypes, minimum=1)
    _require_str(f"{ctx}.activation", config.activation)
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported {ctx}.activation {config.activation!r}; "
            f"expected one of {sorted(ACTIVATION_REGISTRY)}"
        )
    _require_number(
        f"{ctx}.dropout_p",
        config.dropout_p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_str(f"{ctx}.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported {ctx}.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_momentum_teacher_mode(ctx: str, value: str) -> None:
    _require_str(ctx, value)
    if value not in _JEPA_TEACHER_MODES:
        raise ValueError(
            f"Unsupported {ctx} {value!r}; expected one of {sorted(_JEPA_TEACHER_MODES)}"
        )


def _validate_open_unit_interval(ctx: str, value: float) -> None:
    _require_number(
        ctx,
        value,
        minimum=0.0,
        exclusive_minimum=True,
        maximum=1.0,
    )
    if float(value) >= 1.0:
        raise ValueError(f"{ctx} must be < 1.0, got {value!r}")


def _validate_contrastive_objective(config: ContrastiveObjectiveConfig) -> None:
    _require_str("ContrastiveObjectiveConfig.type", config.type)
    if config.type != "contrastive":
        raise ValueError(f"Unsupported ContrastiveObjectiveConfig.type {config.type!r}")
    _require_number(
        "ContrastiveObjectiveConfig.temperature",
        config.temperature,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_str("ContrastiveObjectiveConfig.view_strategy", config.view_strategy)
    if config.view_strategy not in _CONTRASTIVE_VIEW_STRATEGIES:
        raise ValueError(
            f"Unsupported ContrastiveObjectiveConfig.view_strategy {config.view_strategy!r}; "
            f"expected one of {sorted(_CONTRASTIVE_VIEW_STRATEGIES)}"
        )
    _require_bool("ContrastiveObjectiveConfig.symmetric", config.symmetric)
    _require_bool("ContrastiveObjectiveConfig.compute_stats", config.compute_stats)
    _validate_projection_head_config(
        config.projection,
        ctx="ContrastiveObjectiveConfig.projection",
    )


def _validate_byol_objective(config: BYOLObjectiveConfig) -> None:
    _require_str("BYOLObjectiveConfig.type", config.type)
    if config.type != "byol":
        raise ValueError(f"Unsupported BYOLObjectiveConfig.type {config.type!r}")
    _validate_momentum_teacher_mode(
        "BYOLObjectiveConfig.teacher_mode",
        config.teacher_mode,
    )
    _validate_open_unit_interval(
        "BYOLObjectiveConfig.teacher_ema_decay",
        config.teacher_ema_decay,
    )
    _validate_projection_head_config(
        config.projector,
        ctx="BYOLObjectiveConfig.projector",
    )
    _validate_projection_head_config(
        config.predictor,
        ctx="BYOLObjectiveConfig.predictor",
    )
    if config.projector.final_l2_normalize:
        raise ValueError("BYOLObjectiveConfig.projector.final_l2_normalize must be False")
    if config.predictor.final_l2_normalize:
        raise ValueError("BYOLObjectiveConfig.predictor.final_l2_normalize must be False")
    if config.predictor.output_dim != config.projector.output_dim:
        raise ValueError(
            "BYOLObjectiveConfig.predictor.output_dim must match "
            "projector.output_dim, got "
            f"{config.predictor.output_dim} and {config.projector.output_dim}"
        )


def _validate_dino_objective(config: DINOObjectiveConfig) -> None:
    _require_str("DINOObjectiveConfig.type", config.type)
    if config.type != "dino":
        raise ValueError(f"Unsupported DINOObjectiveConfig.type {config.type!r}")
    _validate_momentum_teacher_mode(
        "DINOObjectiveConfig.teacher_mode",
        config.teacher_mode,
    )
    _validate_open_unit_interval(
        "DINOObjectiveConfig.teacher_ema_decay",
        config.teacher_ema_decay,
    )
    _validate_open_unit_interval(
        "DINOObjectiveConfig.center_momentum",
        config.center_momentum,
    )
    _require_number(
        "DINOObjectiveConfig.teacher_temperature",
        config.teacher_temperature,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_number(
        "DINOObjectiveConfig.student_temperature",
        config.student_temperature,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _validate_dino_head(config.head, ctx="DINOObjectiveConfig.head")


def _validate_temporal_masking(config: TemporalMaskingConfig) -> None:
    _require_number(
        "TemporalMaskingConfig.mask_ratio",
        config.mask_ratio,
        minimum=0.0,
        maximum=1.0,
        exclusive_minimum=True,
    )
    if float(config.mask_ratio) >= 1.0:
        raise ValueError(
            "TemporalMaskingConfig.mask_ratio must be < 1.0 so every sample can "
            f"retain visible context, got {config.mask_ratio!r}"
        )
    _require_int("TemporalMaskingConfig.min_span_len", config.min_span_len, minimum=1)
    _require_int("TemporalMaskingConfig.max_span_len", config.max_span_len, minimum=1)
    if config.max_span_len < config.min_span_len:
        raise ValueError(
            "TemporalMaskingConfig.max_span_len must be >= min_span_len, "
            f"got {config.max_span_len} and {config.min_span_len}"
        )
    _require_bool("TemporalMaskingConfig.allow_overlap", config.allow_overlap)


def _validate_span_masking(config: SpanMaskingConfig) -> None:
    _require_str("SpanMaskingConfig.type", config.type)
    if config.type != "span":
        raise ValueError(f"Unsupported SpanMaskingConfig.type {config.type!r}")
    _require_number("SpanMaskingConfig.mask_prob", config.mask_prob, minimum=0.0, maximum=1.0)
    _require_int("SpanMaskingConfig.mask_length", config.mask_length, minimum=1)
    _require_str("SpanMaskingConfig.mask_selection", config.mask_selection)
    if config.mask_selection not in _W2V2_MASK_SELECTIONS:
        raise ValueError(
            f"Unsupported SpanMaskingConfig.mask_selection {config.mask_selection!r}; "
            f"expected one of {sorted(_W2V2_MASK_SELECTIONS)}"
        )
    _require_number("SpanMaskingConfig.mask_other", config.mask_other, minimum=0.0)
    _require_int("SpanMaskingConfig.min_space", config.min_space, minimum=0)
    _require_bool("SpanMaskingConfig.no_overlap", config.no_overlap)
    _require_bool("SpanMaskingConfig.require_same_masks", config.require_same_masks)


def _validate_channel_masking(config: ChannelMaskingConfig) -> None:
    _require_str("ChannelMaskingConfig.type", config.type)
    if config.type != "channel":
        raise ValueError(f"Unsupported ChannelMaskingConfig.type {config.type!r}")
    _require_number("ChannelMaskingConfig.mask_prob", config.mask_prob, minimum=0.0, maximum=1.0)
    _require_int("ChannelMaskingConfig.mask_length", config.mask_length, minimum=1)
    _require_str("ChannelMaskingConfig.mask_selection", config.mask_selection)
    if config.mask_selection not in _W2V2_MASK_SELECTIONS:
        raise ValueError(
            f"Unsupported ChannelMaskingConfig.mask_selection {config.mask_selection!r}; "
            f"expected one of {sorted(_W2V2_MASK_SELECTIONS)}"
        )
    _require_number("ChannelMaskingConfig.mask_other", config.mask_other, minimum=0.0)
    _require_int("ChannelMaskingConfig.min_space", config.min_space, minimum=0)
    _require_bool("ChannelMaskingConfig.no_overlap", config.no_overlap)


def _validate_masking_strategy(config: MaskingStrategyConfig) -> None:
    if isinstance(config, SpanMaskingConfig):
        _validate_span_masking(config)
    elif isinstance(config, ChannelMaskingConfig):
        _validate_channel_masking(config)
    else:
        _validate_registered(
            config,
            ctx="Masking strategy config",
            expected="SpanMaskingConfig or ChannelMaskingConfig",
        )


def _validate_constant_scalar_scheduler(config: ConstantScalarSchedulerConfig) -> None:
    _require_str("ConstantScalarSchedulerConfig.type", config.type)
    if config.type != "constant":
        raise ValueError(f"Unsupported ConstantScalarSchedulerConfig.type {config.type!r}")
    _require_number("ConstantScalarSchedulerConfig.value", config.value)


def _validate_exponential_decay_scalar_scheduler(
    config: ExponentialDecayScalarSchedulerConfig,
) -> None:
    _require_str("ExponentialDecayScalarSchedulerConfig.type", config.type)
    if config.type != "exponential_decay":
        raise ValueError(
            f"Unsupported ExponentialDecayScalarSchedulerConfig.type {config.type!r}"
        )
    _require_number("ExponentialDecayScalarSchedulerConfig.start", config.start)
    _require_number("ExponentialDecayScalarSchedulerConfig.end", config.end)
    _require_number(
        "ExponentialDecayScalarSchedulerConfig.decay",
        config.decay,
        minimum=0.0,
        maximum=1.0,
        exclusive_minimum=True,
    )
    if config.end > config.start:
        raise ValueError(
            "ExponentialDecayScalarSchedulerConfig.end must be <= start, "
            f"got {config.end} and {config.start}"
        )


def _validate_scalar_scheduler(config: ScalarSchedulerConfig) -> None:
    if isinstance(config, ConstantScalarSchedulerConfig):
        _validate_constant_scalar_scheduler(config)
    elif isinstance(config, ExponentialDecayScalarSchedulerConfig):
        _validate_exponential_decay_scalar_scheduler(config)
    else:
        raise TypeError(f"Unsupported scalar scheduler config {type(config).__name__}")


def _validate_wav2vec2_quantizer(config: Wav2Vec2QuantizerConfig) -> None:
    _require_str("Wav2Vec2QuantizerConfig.type", config.type)
    if config.type != "gumbel_product":
        raise ValueError(f"Unsupported Wav2Vec2QuantizerConfig.type {config.type!r}")
    _require_int("Wav2Vec2QuantizerConfig.num_groups", config.num_groups, minimum=1)
    _require_int("Wav2Vec2QuantizerConfig.num_vars", config.num_vars, minimum=1)
    _require_int("Wav2Vec2QuantizerConfig.vq_dim", config.vq_dim, minimum=1)
    if config.vq_dim % config.num_groups != 0:
        raise ValueError(
            "Wav2Vec2QuantizerConfig.vq_dim must be divisible by num_groups, "
            f"got {config.vq_dim} and {config.num_groups}"
        )
    if not isinstance(config.temperature_scheduler, ScalarSchedulerConfig):
        raise TypeError(
            "Wav2Vec2QuantizerConfig.temperature_scheduler must be a scalar scheduler config, "
            f"got {type(config.temperature_scheduler).__name__}"
        )
    _validate_scalar_scheduler(config.temperature_scheduler)
    _require_bool("Wav2Vec2QuantizerConfig.hard", config.hard)
    _require_str("Wav2Vec2QuantizerConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported Wav2Vec2QuantizerConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_jepa_objective(config: JEPAObjectiveConfig) -> None:
    _require_str("JEPAObjectiveConfig.loss_style", config.loss_style)
    if config.loss_style not in {"mse", "mae"}:
        raise ValueError(f"Unsupported JEPAObjectiveConfig.loss_style {config.loss_style!r}")
    _require_str("JEPAObjectiveConfig.teacher_mode", config.teacher_mode)
    if config.teacher_mode not in _JEPA_TEACHER_MODES:
        raise ValueError(
            f"Unsupported JEPAObjectiveConfig.teacher_mode {config.teacher_mode!r}; "
            f"expected one of {sorted(_JEPA_TEACHER_MODES)}"
        )
    _require_str("JEPAObjectiveConfig.output_norm_type", config.output_norm_type)
    if config.output_norm_type not in _JEPA_OUTPUT_NORM_TYPES:
        raise ValueError(
            f"Unsupported JEPAObjectiveConfig.output_norm_type {config.output_norm_type!r}; "
            f"expected one of {sorted(_JEPA_OUTPUT_NORM_TYPES)}"
        )
    _require_bool("JEPAObjectiveConfig.teacher_output_norm", config.teacher_output_norm)
    _require_bool("JEPAObjectiveConfig.student_output_norm", config.student_output_norm)
    _require_number(
        "JEPAObjectiveConfig.teacher_ema_decay",
        config.teacher_ema_decay,
        minimum=0.0,
        maximum=1.0,
    )
    if float(config.teacher_ema_decay) >= 1.0:
        raise ValueError(
            "JEPAObjectiveConfig.teacher_ema_decay must be < 1.0, "
            f"got {config.teacher_ema_decay!r}"
        )
    _require_number(
        "JEPAObjectiveConfig.mask_token_init_scale",
        config.mask_token_init_scale,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_type("JEPAObjectiveConfig.predictor_config", config.predictor_config, MLPConfig)
    _validate_mlp(config.predictor_config)
    _require_type("JEPAObjectiveConfig.masking", config.masking, TemporalMaskingConfig)
    _validate_temporal_masking(config.masking)


def _validate_mae_masking(config: MAEMaskingConfig) -> None:
    _require_number(
        "MAEMaskingConfig.mask_ratio",
        config.mask_ratio,
        minimum=0.0,
        maximum=1.0,
        exclusive_minimum=True,
    )
    if float(config.mask_ratio) >= 1.0:
        raise ValueError(
            "MAEMaskingConfig.mask_ratio must be < 1.0 so each sample retains "
            f"visible context, got {config.mask_ratio!r}"
        )
    if config.group_factor is not None:
        _require_int("MAEMaskingConfig.group_factor", config.group_factor, minimum=1)


def _validate_mae_objective(config: MAEObjectiveConfig) -> None:
    _require_str("MAEObjectiveConfig.type", config.type)
    if config.type != "mae":
        raise ValueError(f"Unsupported MAEObjectiveConfig.type {config.type!r}")
    _require_int("MAEObjectiveConfig.decoder_d_model", config.decoder_d_model, minimum=1)
    _validate_encoder(config.decoder, d_model=config.decoder_d_model)
    if getattr(config.decoder, "pooling", None) == "cls":
        raise ValueError("MAEObjectiveConfig.decoder pooling must not be 'cls'")
    if not isinstance(config.decoder_positional_encoding, EmbeddingPositionalEncodingConfig):
        raise TypeError(
            "MAEObjectiveConfig.decoder_positional_encoding must be an embedding "
            "positional encoding config, "
            f"got {type(config.decoder_positional_encoding).__name__}"
        )
    _validate_embedding_positional_encoding(config.decoder_positional_encoding)
    _require_number(
        "MAEObjectiveConfig.mask_token_init_scale",
        config.mask_token_init_scale,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_str("MAEObjectiveConfig.norm_pix_loss", config.norm_pix_loss)
    if config.norm_pix_loss not in _MAE_NORM_PIX_LOSS:
        raise ValueError(
            f"Unsupported MAEObjectiveConfig.norm_pix_loss {config.norm_pix_loss!r}; "
            f"expected one of {sorted(_MAE_NORM_PIX_LOSS)}"
        )
    _require_number(
        "MAEObjectiveConfig.norm_pix_eps",
        config.norm_pix_eps,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_str("MAEObjectiveConfig.target_patch_order", config.target_patch_order)
    if config.target_patch_order not in {"time_frequency", "frequency_time"}:
        raise ValueError(
            "MAEObjectiveConfig.target_patch_order must be one of "
            "{'time_frequency', 'frequency_time'}"
        )
    _require_type("MAEObjectiveConfig.masking", config.masking, MAEMaskingConfig)
    _validate_mae_masking(config.masking)


def _validate_wav2vec2_objective(config: Wav2Vec2ObjectiveConfig) -> None:
    _require_str("Wav2Vec2ObjectiveConfig.type", config.type)
    if config.type != "wav2vec2":
        raise ValueError(f"Unsupported Wav2Vec2ObjectiveConfig.type {config.type!r}")
    _require_type(
        "Wav2Vec2ObjectiveConfig.quantizer",
        config.quantizer,
        Wav2Vec2QuantizerConfig,
    )
    _validate_wav2vec2_quantizer(config.quantizer)
    _require_type("Wav2Vec2ObjectiveConfig.time_masking", config.time_masking, SpanMaskingConfig)
    _validate_span_masking(config.time_masking)
    if config.feature_masking is not None:
        _require_type(
            "Wav2Vec2ObjectiveConfig.feature_masking",
            config.feature_masking,
            ChannelMaskingConfig,
        )
        _validate_channel_masking(config.feature_masking)
    _require_number(
        "Wav2Vec2ObjectiveConfig.mask_emb_init_scale",
        config.mask_emb_init_scale,
        minimum=0.0,
    )
    _require_int("Wav2Vec2ObjectiveConfig.final_dim", config.final_dim, minimum=1)
    _require_int("Wav2Vec2ObjectiveConfig.num_negatives", config.num_negatives, minimum=0)
    _require_int(
        "Wav2Vec2ObjectiveConfig.cross_sample_negatives",
        config.cross_sample_negatives,
        minimum=0,
    )
    if config.num_negatives == 0 and config.cross_sample_negatives == 0:
        raise ValueError("Wav2Vec2ObjectiveConfig requires at least one negative sample")
    _require_number(
        "Wav2Vec2ObjectiveConfig.logit_temp",
        config.logit_temp,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_number(
        "Wav2Vec2ObjectiveConfig.diversity_loss_weight",
        config.diversity_loss_weight,
        minimum=0.0,
    )
    _require_number(
        "Wav2Vec2ObjectiveConfig.features_pen_weight",
        config.features_pen_weight,
        minimum=0.0,
    )
    _require_number(
        "Wav2Vec2ObjectiveConfig.feature_grad_mult",
        config.feature_grad_mult,
        minimum=0.0,
    )
    _require_str("Wav2Vec2ObjectiveConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported Wav2Vec2ObjectiveConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_hubert_teacher(config: HuBERTTeacherConfig) -> None:
    _require_str("HuBERTTeacherConfig.type", config.type)
    if config.type != "pretrained_hubert_kmeans":
        raise ValueError(f"Unsupported HuBERTTeacherConfig.type {config.type!r}")
    _require_str("HuBERTTeacherConfig.encoder_id", config.encoder_id)
    if config.encoder_revision is None:
        raise ValueError(
            "HuBERTTeacherConfig.encoder_revision is required; pin a commit hash "
            "for reproducible pseudo-labels"
        )
    _require_str("HuBERTTeacherConfig.encoder_revision", config.encoder_revision)
    _require_int("HuBERTTeacherConfig.feature_layer", config.feature_layer, minimum=1)
    _require_int("HuBERTTeacherConfig.num_codes", config.num_codes, minimum=1)
    _require_int("HuBERTTeacherConfig.target_sample_rate", config.target_sample_rate, minimum=1)
    if config.km_url is not None:
        _require_str("HuBERTTeacherConfig.km_url", config.km_url)
    if config.km_sha256 is not None:
        _require_str("HuBERTTeacherConfig.km_sha256", config.km_sha256)
    if (config.km_url is None) != (config.km_sha256 is None):
        raise ValueError("HuBERTTeacherConfig.km_url and km_sha256 must be set together")


def _validate_hubert_objective(config: HuBERTObjectiveConfig) -> None:
    _require_str("HuBERTObjectiveConfig.type", config.type)
    if config.type != "hubert":
        raise ValueError(f"Unsupported HuBERTObjectiveConfig.type {config.type!r}")
    _require_type("HuBERTObjectiveConfig.teacher", config.teacher, HuBERTTeacherConfig)
    _validate_hubert_teacher(config.teacher)
    _require_type("HuBERTObjectiveConfig.time_masking", config.time_masking, SpanMaskingConfig)
    _validate_span_masking(config.time_masking)
    if config.feature_masking is not None:
        _require_type(
            "HuBERTObjectiveConfig.feature_masking",
            config.feature_masking,
            ChannelMaskingConfig,
        )
        _validate_channel_masking(config.feature_masking)
    _require_number("HuBERTObjectiveConfig.mask_emb_init_scale", config.mask_emb_init_scale, minimum=0.0)
    _require_int("HuBERTObjectiveConfig.final_dim", config.final_dim, minimum=1)
    _require_number("HuBERTObjectiveConfig.logit_temp", config.logit_temp, minimum=0.0, exclusive_minimum=True)
    _require_number("HuBERTObjectiveConfig.label_emb_init_scale", config.label_emb_init_scale, minimum=0.0)
    _require_number("HuBERTObjectiveConfig.masked_loss_weight", config.masked_loss_weight, minimum=0.0)
    _require_number("HuBERTObjectiveConfig.unmasked_loss_weight", config.unmasked_loss_weight, minimum=0.0)
    if config.masked_loss_weight == 0.0 and config.unmasked_loss_weight == 0.0:
        raise ValueError("HuBERTObjectiveConfig requires at least one non-zero loss weight")
    _require_number("HuBERTObjectiveConfig.feature_grad_mult", config.feature_grad_mult, minimum=0.0)
    _require_str("HuBERTObjectiveConfig.initialization", config.initialization)
    if config.initialization not in INITIALIZATION_REGISTRY:
        raise ValueError(
            f"Unsupported HuBERTObjectiveConfig.initialization {config.initialization!r}; "
            f"expected one of {sorted(INITIALIZATION_REGISTRY)}"
        )


def _validate_ufo_decoder(config: UFODecoderConfig) -> None:
    _require_int("UFODecoderConfig.num_layers", config.num_layers, minimum=1)
    _require_int("UFODecoderConfig.kernel_size", config.kernel_size, minimum=1)
    _require_str("UFODecoderConfig.activation", config.activation)
    if config.activation not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported UFODecoderConfig.activation {config.activation!r}; "
            f"expected one of {sorted(ACTIVATION_REGISTRY)}"
        )
    _require_number(
        "UFODecoderConfig.layer_norm_eps",
        config.layer_norm_eps,
        minimum=0.0,
        exclusive_minimum=True,
    )


def _validate_ufo_masking(config: UFOMaskingConfig) -> None:
    _require_number("UFOMaskingConfig.mask_ratio", config.mask_ratio, minimum=0.0, maximum=1.0)
    if not (0.0 < float(config.mask_ratio) < 1.0):
        raise ValueError("UFOMaskingConfig.mask_ratio must be in (0, 1)")
    if len(config.block_size) != 2:
        raise ValueError("UFOMaskingConfig.block_size must have length 2")
    block_t, block_f = config.block_size
    _require_int("UFOMaskingConfig.block_size[0]", block_t, minimum=1)
    _require_int("UFOMaskingConfig.block_size[1]", block_f, minimum=1)
    _require_int("UFOMaskingConfig.num_masks", config.num_masks, minimum=1)


def _validate_ufo_objective(config: UFOObjectiveConfig) -> None:
    _require_str("UFOObjectiveConfig.type", config.type)
    if config.type != "ufo":
        raise ValueError(f"Unsupported UFOObjectiveConfig.type {config.type!r}")
    _require_type("UFOObjectiveConfig.decoder", config.decoder, UFODecoderConfig)
    _validate_ufo_decoder(config.decoder)
    _require_type("UFOObjectiveConfig.masking", config.masking, UFOMaskingConfig)
    _validate_ufo_masking(config.masking)
    _require_number(
        "UFOObjectiveConfig.utterance_loss_weight",
        config.utterance_loss_weight,
        minimum=0.0,
    )
    _require_str("UFOObjectiveConfig.teacher_mode", config.teacher_mode)
    if config.teacher_mode not in _JEPA_TEACHER_MODES:
        raise ValueError(
            f"Unsupported UFOObjectiveConfig.teacher_mode {config.teacher_mode!r}; "
            f"expected one of {sorted(_JEPA_TEACHER_MODES)}"
        )
    _require_number(
        "UFOObjectiveConfig.teacher_ema_decay",
        config.teacher_ema_decay,
        minimum=0.0,
        maximum=1.0,
        exclusive_minimum=True,
    )
    if float(config.teacher_ema_decay) >= 1.0:
        raise ValueError(
            f"UFOObjectiveConfig.teacher_ema_decay must be < 1.0, got {config.teacher_ema_decay!r}"
        )
    _require_number(
        "UFOObjectiveConfig.mask_token_init_scale",
        config.mask_token_init_scale,
        minimum=0.0,
        exclusive_minimum=True,
    )


def _validate_objective(config) -> None:
    if isinstance(config, ContrastiveObjectiveConfig):
        _validate_contrastive_objective(config)
    elif isinstance(config, BYOLObjectiveConfig):
        _validate_byol_objective(config)
    elif isinstance(config, DINOObjectiveConfig):
        _validate_dino_objective(config)
    elif isinstance(config, JEPAObjectiveConfig):
        _validate_jepa_objective(config)
    elif isinstance(config, MAEObjectiveConfig):
        _validate_mae_objective(config)
    elif isinstance(config, UFOObjectiveConfig):
        _validate_ufo_objective(config)
    elif isinstance(config, Wav2Vec2ObjectiveConfig):
        _validate_wav2vec2_objective(config)
    elif isinstance(config, HuBERTObjectiveConfig):
        _validate_hubert_objective(config)
    else:
        _validate_registered(
            config,
            ctx="Objective config",
            expected=(
                "ContrastiveObjectiveConfig, BYOLObjectiveConfig, DINOObjectiveConfig, "
                "JEPAObjectiveConfig, MAEObjectiveConfig, UFOObjectiveConfig, "
                "Wav2Vec2ObjectiveConfig, or HuBERTObjectiveConfig"
            ),
        )


def _validate_loudness_augmentation(config: LoudnessAugmentationConfig) -> None:
    if config.type != "loudness":
        raise ValueError(
            f"Unsupported LoudnessAugmentationConfig.type {config.type!r}"
        )
    _require_number(
        "LoudnessAugmentationConfig.p",
        config.p,
        minimum=0.0,
        maximum=1.0,
    )
    _require_number("LoudnessAugmentationConfig.min_gain_db", config.min_gain_db)
    _require_number("LoudnessAugmentationConfig.max_gain_db", config.max_gain_db)
    if float(config.min_gain_db) > float(config.max_gain_db):
        raise ValueError(
            "LoudnessAugmentationConfig.min_gain_db must be <= max_gain_db, "
            f"got {config.min_gain_db!r} and {config.max_gain_db!r}"
        )


def _validate_augmentation(config: LoudnessAugmentationConfig) -> None:
    if isinstance(config, LoudnessAugmentationConfig):
        _validate_loudness_augmentation(config)
    elif in_category(config, "augmentation"):
        _validate_registered(
            config,
            ctx="Augmentation config",
            expected="LoudnessAugmentationConfig or a registered augmentation config",
        )
    else:
        raise TypeError(
            "Augmentation config must be LoudnessAugmentationConfig or a config "
            f"registered under the augmentation category; got {type(config).__name__}"
        )


def _validate_augmentor(config: AudioAugmentorConfig) -> None:
    if not isinstance(config.augmentations, tuple):
        raise TypeError(
            "AudioAugmentorConfig.augmentations must be tuple[AugmentationConfig, ...], "
            f"got {type(config.augmentations).__name__}"
        )
    for index, augmentation in enumerate(config.augmentations):
        try:
            _validate_augmentation(augmentation)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"AudioAugmentorConfig.augmentations[{index}]: {exc}"
            ) from exc


def _validate_adamw_optimizer(config: AdamWOptimizerConfig) -> None:
    if config.type != "adamw":
        raise ValueError(f"Unsupported AdamWOptimizerConfig.type {config.type!r}")
    _require_number(
        "AdamWOptimizerConfig.lr",
        config.lr,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_number("AdamWOptimizerConfig.weight_decay", config.weight_decay, minimum=0.0)
    if len(config.betas) != 2:
        raise ValueError(
            "AdamWOptimizerConfig.betas must contain exactly two values, "
            f"got {len(config.betas)}"
        )
    for index, beta in enumerate(config.betas):
        _require_number(
            f"AdamWOptimizerConfig.betas[{index}]",
            beta,
            minimum=0.0,
            exclusive_minimum=True,
        )
        if float(beta) >= 1.0:
            raise ValueError(
                f"AdamWOptimizerConfig.betas[{index}] must be < 1, got {beta!r}"
            )
    _require_number(
        "AdamWOptimizerConfig.eps",
        config.eps,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _validate_no_weight_decay_patterns(
        "AdamWOptimizerConfig.no_weight_decay_patterns",
        config.no_weight_decay_patterns,
    )


def _validate_sgd_optimizer(config: SGDOptimizerConfig) -> None:
    if config.type != "sgd":
        raise ValueError(f"Unsupported SGDOptimizerConfig.type {config.type!r}")
    _require_number(
        "SGDOptimizerConfig.lr",
        config.lr,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_number(
        "SGDOptimizerConfig.momentum",
        config.momentum,
        minimum=0.0,
        maximum=1.0,
    )
    _require_number("SGDOptimizerConfig.weight_decay", config.weight_decay, minimum=0.0)
    _require_bool("SGDOptimizerConfig.nesterov", config.nesterov)
    _validate_no_weight_decay_patterns(
        "SGDOptimizerConfig.no_weight_decay_patterns",
        config.no_weight_decay_patterns,
    )


def _validate_lars_optimizer(config: LARSOptimizerConfig) -> None:
    if config.type != "lars":
        raise ValueError(f"Unsupported LARSOptimizerConfig.type {config.type!r}")
    _require_number(
        "LARSOptimizerConfig.lr",
        config.lr,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _require_number(
        "LARSOptimizerConfig.momentum",
        config.momentum,
        minimum=0.0,
        maximum=1.0,
    )
    _require_number("LARSOptimizerConfig.weight_decay", config.weight_decay, minimum=0.0)
    _require_number(
        "LARSOptimizerConfig.eta",
        config.eta,
        minimum=0.0,
        exclusive_minimum=True,
    )
    _validate_no_weight_decay_patterns(
        "LARSOptimizerConfig.no_weight_decay_patterns",
        config.no_weight_decay_patterns,
    )


def _validate_optimizer(
    config: AdamWOptimizerConfig | SGDOptimizerConfig | LARSOptimizerConfig,
) -> None:
    if isinstance(config, AdamWOptimizerConfig):
        _validate_adamw_optimizer(config)
    elif isinstance(config, SGDOptimizerConfig):
        _validate_sgd_optimizer(config)
    elif isinstance(config, LARSOptimizerConfig):
        _validate_lars_optimizer(config)
    else:
        raise TypeError(
            "Optimizer config must be AdamWOptimizerConfig, SGDOptimizerConfig, "
            f"or LARSOptimizerConfig, got {type(config).__name__}"
        )


def _validate_constant_scheduler(config: ConstantSchedulerConfig) -> None:
    if config.type != "constant":
        raise ValueError(f"Unsupported ConstantSchedulerConfig.type {config.type!r}")


def _validate_warmup_cosine_scheduler(config: WarmupCosineSchedulerConfig) -> None:
    if config.type != "warmup_cosine":
        raise ValueError(
            f"Unsupported WarmupCosineSchedulerConfig.type {config.type!r}"
        )
    _require_int("WarmupCosineSchedulerConfig.warmup_steps", config.warmup_steps, minimum=0)
    _require_number(
        "WarmupCosineSchedulerConfig.min_lr_scale",
        config.min_lr_scale,
        minimum=0.0,
        maximum=1.0,
    )


def _validate_cosine_annealing_scheduler(config: CosineAnnealingSchedulerConfig) -> None:
    if config.type != "cosine_annealing":
        raise ValueError(
            f"Unsupported CosineAnnealingSchedulerConfig.type {config.type!r}"
        )
    _require_number(
        "CosineAnnealingSchedulerConfig.min_lr_scale",
        config.min_lr_scale,
        minimum=0.0,
        maximum=1.0,
    )


def _validate_step_lr_scheduler(config: StepLRSchedulerConfig) -> None:
    if config.type != "step_lr":
        raise ValueError(f"Unsupported StepLRSchedulerConfig.type {config.type!r}")
    _require_int("StepLRSchedulerConfig.step_size", config.step_size, minimum=1)
    _require_number(
        "StepLRSchedulerConfig.gamma",
        config.gamma,
        minimum=0.0,
        exclusive_minimum=True,
        maximum=1.0,
    )


def _validate_scheduler(
    config: (
        ConstantSchedulerConfig
        | WarmupCosineSchedulerConfig
        | CosineAnnealingSchedulerConfig
        | StepLRSchedulerConfig
    ),
) -> None:
    if isinstance(config, ConstantSchedulerConfig):
        _validate_constant_scheduler(config)
    elif isinstance(config, WarmupCosineSchedulerConfig):
        _validate_warmup_cosine_scheduler(config)
    elif isinstance(config, CosineAnnealingSchedulerConfig):
        _validate_cosine_annealing_scheduler(config)
    elif isinstance(config, StepLRSchedulerConfig):
        _validate_step_lr_scheduler(config)
    else:
        raise TypeError(
            "Scheduler config must be ConstantSchedulerConfig, "
            "WarmupCosineSchedulerConfig, CosineAnnealingSchedulerConfig, "
            f"or StepLRSchedulerConfig, got {type(config).__name__}"
        )


def _validate_batch(config: BatchConfig) -> None:
    _require_int("BatchConfig.per_device_batch_size", config.per_device_batch_size, minimum=1)
    _require_int(
        "BatchConfig.gradient_accumulation_steps",
        config.gradient_accumulation_steps,
        minimum=1,
    )
    _require_bool("BatchConfig.drop_last", config.drop_last)
    _require_bool("BatchConfig.shuffle", config.shuffle)


def _validate_training_loop(config: TrainingLoopConfig) -> None:
    if config.max_epochs is None and config.max_steps is None:
        raise ValueError(
            "TrainingLoopConfig requires at least one of max_epochs or max_steps"
        )
    if config.max_epochs is not None:
        _require_int("TrainingLoopConfig.max_epochs", config.max_epochs, minimum=1)
    if config.max_steps is not None:
        _require_int("TrainingLoopConfig.max_steps", config.max_steps, minimum=1)
    if config.gradient_clip_norm is not None:
        _require_number(
            "TrainingLoopConfig.gradient_clip_norm",
            config.gradient_clip_norm,
            minimum=0.0,
            exclusive_minimum=True,
        )
    if config.nonfinite_policy not in {"raise", "skip"}:
        raise ValueError(
            "TrainingLoopConfig.nonfinite_policy must be one of {'raise', 'skip'}, "
            f"got {config.nonfinite_policy!r}"
        )
    _require_int(
        "TrainingLoopConfig.max_consecutive_skipped_steps",
        config.max_consecutive_skipped_steps,
        minimum=1,
    )


def _validate_evaluation(config: EvaluationConfig) -> None:
    if config.every_n_epochs is not None:
        _require_int(
            "EvaluationConfig.every_n_epochs",
            config.every_n_epochs,
            minimum=1,
        )
    if config.every_n_steps is not None:
        _require_int(
            "EvaluationConfig.every_n_steps",
            config.every_n_steps,
            minimum=1,
        )
    if config.every_n_epochs is not None and config.every_n_steps is not None:
        raise ValueError(
            "EvaluationConfig.every_n_epochs and EvaluationConfig.every_n_steps "
            "cannot both be set"
        )
    _require_bool("EvaluationConfig.before_train", config.before_train)
    _require_bool("EvaluationConfig.at_train_end", config.at_train_end)
    if (
        config.every_n_epochs is None
        and config.every_n_steps is None
        and not config.before_train
        and not config.at_train_end
    ):
        raise ValueError(
            "EvaluationConfig must enable at least one validation trigger: "
            "every_n_epochs, every_n_steps, before_train, or at_train_end"
        )


def _validate_checkpoint(config: CheckpointConfig) -> None:
    _require_str("CheckpointConfig.save_dir", config.save_dir)
    if config.save_every_n_epochs is not None:
        _require_int(
            "CheckpointConfig.save_every_n_epochs",
            config.save_every_n_epochs,
            minimum=1,
        )
    if config.save_every_n_steps is not None:
        _require_int(
            "CheckpointConfig.save_every_n_steps",
            config.save_every_n_steps,
            minimum=1,
        )
    if config.save_every_n_epochs is None and config.save_every_n_steps is None:
        raise ValueError(
            "CheckpointConfig must set save_every_n_epochs, save_every_n_steps, or both"
        )
    _require_int("CheckpointConfig.keep_last_n", config.keep_last_n, minimum=1)
    _require_bool("CheckpointConfig.save_best", config.save_best)
    _require_str("CheckpointConfig.monitor", config.monitor)
    if not (
        config.monitor.startswith("train/")
        or config.monitor.startswith("val/")
    ):
        warnings.warn(
            "CheckpointConfig.monitor is usually a logged metric name such as "
            "'train/loss' or 'val/loss'; "
            f"got {config.monitor!r}",
            stacklevel=2,
        )
    _require_str("CheckpointConfig.mode", config.mode)
    if config.mode not in {"min", "max"}:
        raise ValueError(
            f"CheckpointConfig.mode must be one of {{'min', 'max'}}, got {config.mode!r}"
        )


def _validate_mixed_precision(config: MixedPrecisionConfig) -> None:
    _require_str("MixedPrecisionConfig.dtype", config.dtype)
    if config.dtype not in {"bfloat16", "float16"}:
        raise ValueError(
            "MixedPrecisionConfig.dtype must be one of {'bfloat16', 'float16'}, "
            f"got {config.dtype!r}"
        )
    _require_bool("MixedPrecisionConfig.grad_scaler", config.grad_scaler)
    if config.grad_scaler and config.dtype != "float16":
        raise ValueError(
            "MixedPrecisionConfig.grad_scaler=True requires dtype='float16'; "
            "bfloat16 does not need GradScaler. "
            f"Got dtype={config.dtype!r}, grad_scaler={config.grad_scaler!r}"
        )


def _validate_loader(config: LoaderConfig) -> None:
    _require_int("LoaderConfig.num_workers", config.num_workers, minimum=0)
    _require_bool("LoaderConfig.pin_memory", config.pin_memory)
    if config.prefetch_factor is not None:
        _require_int("LoaderConfig.prefetch_factor", config.prefetch_factor, minimum=1)
        if config.num_workers == 0:
            warnings.warn(
                "LoaderConfig.prefetch_factor is only used when num_workers > 0",
                stacklevel=2,
            )
    _require_bool("LoaderConfig.persistent_workers", config.persistent_workers)
    if config.persistent_workers and config.num_workers == 0:
        raise ValueError(
            "LoaderConfig.persistent_workers requires num_workers > 0, "
            f"got num_workers={config.num_workers!r}"
        )


def _validate_logger(config: LoggerConfig) -> None:
    if isinstance(config, WandBConfig):
        _validate_wandb(config)
    elif in_category(config, "logger"):
        _validate_registered(
            config,
            ctx="Logger config",
            expected="WandBConfig or a registered logger config",
        )
    else:
        raise TypeError(
            "Logger config must be WandBConfig or a config registered under the "
            f"logger category; got {type(config).__name__}"
        )


def _validate_run(config: RunConfig) -> None:
    if config.run_dir is not None:
        _require_str("RunConfig.run_dir", config.run_dir)
    _require_type("RunConfig.loader", config.loader, LoaderConfig)
    _validate_loader(config.loader)
    if config.checkpoint is not None:
        _require_type("RunConfig.checkpoint", config.checkpoint, CheckpointConfig)
        _validate_checkpoint(config.checkpoint)
        if (
            config.checkpoint.save_best
            and config.checkpoint.monitor.startswith("val/")
            and config.evaluation is None
        ):
            raise ValueError(
                "RunConfig.evaluation is required when "
                "CheckpointConfig.save_best=True with a val/* monitor"
            )
    if config.evaluation is not None:
        _require_type("RunConfig.evaluation", config.evaluation, EvaluationConfig)
        _validate_evaluation(config.evaluation)
        if config.checkpoint is None:
            warnings.warn(
                "RunConfig.evaluation is set but checkpoint is None; "
                "computing val metrics but not saving any checkpoint",
                stacklevel=2,
            )
    _require_bool("RunConfig.stdout_logging", config.stdout_logging)
    _require_int("RunConfig.log_every_n_steps", config.log_every_n_steps, minimum=1)
    if not isinstance(config.loggers, tuple):
        raise TypeError(
            "RunConfig.loggers must be tuple[LoggerConfig, ...], "
            f"got {type(config.loggers).__name__}"
        )
    for index, logger in enumerate(config.loggers):
        try:
            _validate_logger(logger)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"RunConfig.loggers[{index}]: {exc}") from exc
    _require_bool("RunConfig.require_clean_git", config.require_clean_git)
    if config.provenance_mismatch not in {"warn", "raise"}:
        raise ValueError(
            "RunConfig.provenance_mismatch must be one of {'warn', 'raise'}, "
            f"got {config.provenance_mismatch!r}"
        )


def _validate_trainer(config: TrainerConfig) -> None:
    if config.augmentor is not None:
        _require_type("TrainerConfig.augmentor", config.augmentor, AudioAugmentorConfig)
        _validate_augmentor(config.augmentor)
    _validate_optimizer(config.optimizer)
    _validate_scheduler(config.scheduler)
    _validate_batch(config.batch)
    _validate_training_loop(config.loop)
    if config.mixed_precision is not None:
        _require_type(
            "TrainerConfig.mixed_precision",
            config.mixed_precision,
            MixedPrecisionConfig,
        )
        _validate_mixed_precision(config.mixed_precision)


def _validate_wandb(config: WandBConfig) -> None:
    _require_str("WandBConfig.project", config.project)
    if config.name is not None:
        _require_str("WandBConfig.name", config.name)
    if config.entity is not None:
        _require_str("WandBConfig.entity", config.entity)
    if not isinstance(config.tags, tuple):
        raise TypeError(
            "WandBConfig.tags must be tuple[str, ...], "
            f"got {type(config.tags).__name__}"
        )
    for index, tag in enumerate(config.tags):
        if not isinstance(tag, str):
            raise TypeError(
                f"WandBConfig.tags[{index}] must be str, got {type(tag).__name__}"
            )
        if not tag:
            raise ValueError(f"WandBConfig.tags[{index}] must not be empty")
    _require_str("WandBConfig.mode", config.mode)
    if config.mode not in _WANDB_MODES:
        raise ValueError(
            f"Unsupported WandBConfig.mode {config.mode!r}; "
            f"expected one of {sorted(_WANDB_MODES)}"
        )
    if not isinstance(config.notes, str):
        raise TypeError(
            f"WandBConfig.notes must be str, got {type(config.notes).__name__}"
        )
    _require_int("WandBConfig.log_every_n_steps", config.log_every_n_steps, minimum=1)
    _require_bool("WandBConfig.aggregate_over_steps", config.aggregate_over_steps)
    _require_bool("WandBConfig.upload_checkpoints", config.upload_checkpoints)
    _require_bool("WandBConfig.resume", config.resume)


def _validate_mae_training_cross_config(training: AudioSSLTrainingConfig) -> None:
    """Cross-config checks specific to the MAE objective."""

    assert isinstance(training.objective, MAEObjectiveConfig)
    objective = training.objective
    model = training.model

    decoder_pe = objective.decoder_positional_encoding
    embedding_pe = model.embedding.positional_encoding
    if (
        isinstance(decoder_pe, LearnableTimeFrequencyPositionalEncodingConfig)
        and isinstance(embedding_pe, LearnableTimeFrequencyPositionalEncodingConfig)
    ):
        if decoder_pe.max_t != embedding_pe.max_t:
            raise ValueError(
                "MAEObjectiveConfig.decoder_positional_encoding.max_t must match "
                "model.embedding.positional_encoding.max_t, "
                f"got {decoder_pe.max_t} and {embedding_pe.max_t}"
            )
        if decoder_pe.max_f != embedding_pe.max_f:
            raise ValueError(
                "MAEObjectiveConfig.decoder_positional_encoding.max_f must match "
                "model.embedding.positional_encoding.max_f, "
                f"got {decoder_pe.max_f} and {embedding_pe.max_f}"
            )


def _validate_training(config: AudioSSLTrainingConfig) -> None:
    _require_type("AudioSSLTrainingConfig.model", config.model, AudioModelConfig)
    _require_type("AudioSSLTrainingConfig.trainer", config.trainer, TrainerConfig)
    _validate_model(config.model)
    _validate_objective(config.objective)
    _validate_trainer(config.trainer)
    check_objective_compatibility(config.model, config.objective)
    if isinstance(config.objective, MAEObjectiveConfig):
        _validate_mae_training_cross_config(config)


def validate_config(config):
    """Validate a config dataclass and return ``True`` when all checks pass.

    Parameters
    ----------
    config
        Any supported frozen config dataclass, including nested model,
        objective, and trainer configs.

    Returns
    -------
    bool
        Always ``True`` on success. Raises ``TypeError`` or ``ValueError`` when
        a check fails.

    Raises
    ------
    TypeError
        When ``config`` is unsupported or a nested field has the wrong type.
    ValueError
        When a field value or cross-field relationship is invalid.
    """
    if isinstance(config, AudioModelConfig):
        _validate_model(config)
    elif in_category(config, "objective"):
        _validate_objective(config)
    elif isinstance(config, HuBERTTeacherConfig):
        _validate_hubert_teacher(config)
    elif isinstance(config, UFODecoderConfig):
        _validate_ufo_decoder(config)
    elif isinstance(config, UFOMaskingConfig):
        _validate_ufo_masking(config)
    elif isinstance(config, DINOHeadConfig):
        _validate_dino_head(config)
    elif isinstance(config, TemporalMaskingConfig):
        _validate_temporal_masking(config)
    elif isinstance(config, MAEMaskingConfig):
        _validate_mae_masking(config)
    elif in_category(config, "masking_strategy"):
        _validate_masking_strategy(config)
    elif isinstance(config, ConstantScalarSchedulerConfig | ExponentialDecayScalarSchedulerConfig):
        _validate_scalar_scheduler(config)
    elif isinstance(config, Wav2Vec2QuantizerConfig):
        _validate_wav2vec2_quantizer(config)
    elif isinstance(config, LoudnessAugmentationConfig):
        _validate_loudness_augmentation(config)
    elif in_category(config, "augmentation"):
        _validate_augmentation(config)
    elif isinstance(config, AudioAugmentorConfig):
        _validate_augmentor(config)
    elif isinstance(config, AudioSSLTrainingConfig):
        _validate_training(config)
    elif isinstance(config, AdamWOptimizerConfig | SGDOptimizerConfig | LARSOptimizerConfig):
        _validate_optimizer(config)
    elif isinstance(
        config,
        ConstantSchedulerConfig
        | WarmupCosineSchedulerConfig
        | CosineAnnealingSchedulerConfig
        | StepLRSchedulerConfig,
    ):
        _validate_scheduler(config)
    elif isinstance(config, BatchConfig):
        _validate_batch(config)
    elif isinstance(config, TrainingLoopConfig):
        _validate_training_loop(config)
    elif isinstance(config, EvaluationConfig):
        _validate_evaluation(config)
    elif isinstance(config, LoaderConfig):
        _validate_loader(config)
    elif isinstance(config, RunConfig):
        _validate_run(config)
    elif isinstance(config, TrainerConfig):
        _validate_trainer(config)
    elif isinstance(config, CheckpointConfig):
        _validate_checkpoint(config)
    elif isinstance(config, MixedPrecisionConfig):
        _validate_mixed_precision(config)
    elif isinstance(config, WandBConfig):
        _validate_wandb(config)
    elif in_category(config, "logger"):
        _validate_logger(config)
    elif isinstance(config, RuntimeConfig):
        _validate_runtime(config)
    elif in_category(config, "processor"):
        _validate_processor(config)
    elif isinstance(config, MultiHeadAttentionConfig):
        _validate_mha(config)
    elif isinstance(config, MLPConfig):
        _validate_mlp(config)
    elif isinstance(config, ConvModuleConfig):
        _validate_conv_module(config)
    elif isinstance(config, CGMLPConfig):
        _validate_cgmlp(config)
    elif isinstance(config, BranchMergeConfig):
        _validate_branch_merge(config)
    elif isinstance(config, EBranchMergeConfig):
        _validate_ebranch_merge(config)
    elif isinstance(config, ProjectionHeadConfig):
        _validate_projection_head_config(config)
    elif in_category(config, "tokenizer"):
        _validate_tokenizer(config)
    elif isinstance(config, EmbeddingSubsamplingConfig):
        _validate_embedding_subsampling(config, ctx="EmbeddingSubsamplingConfig")
    elif in_category(config, "embedding"):
        _validate_embedding(config)
    elif in_category(config, "per_block_positional_encoding"):
        _validate_per_block_positional_encoding(config)
    elif in_category(config, "positional_encoding"):
        _validate_embedding_positional_encoding(config)
    elif in_category(config, "encoder"):
        raise ValueError("Encoder config validation requires the parent model d_model")
    else:
        validator = validator_for(config)
        if validator is None:
            raise TypeError(f"Unsupported config type {type(config).__name__}")
        validator(config)
    return True


__all__ = ["validate_config"]





"""Frozen dataclass configs for models, objectives, and training.

These types are the JSON-serializable configuration surface for ``audiossl``.
Use :func:`~audiossl.config.load_config` to construct an
:class:`AudioSSLTrainingConfig` and :func:`~audiossl.config.validate_config`
to check field relationships before building runtime objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from .capabilities import (
    PATCH_GRID,
    PREFIX_TOKEN,
    RECONSTRUCTION_TARGET,
    SEQUENCE_CONVOLUTION,
    TOKEN_SUBSAMPLING,
)


@dataclass(kw_only=True, frozen=True)
class RuntimeConfig:
    """Device, RNG, and tensor dtype settings applied when building a model.

    ``allow_tf32`` controls TF32 matmul/convolution kernels on CUDA. It is off by
    default because TF32 changes float32 numerics; enabling it is a research
    decision and therefore part of the fingerprinted config.
    """

    device: str = "cpu"
    seed: int = 0
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    deterministic: bool = False
    allow_tf32: bool = False


@dataclass(kw_only=True, frozen=True)
class MultiHeadAttentionConfig:
    """Multi-head self-attention block parameters shared by encoder variants."""

    num_heads: int
    dropout_p: float = 0.0
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class MLPConfig:
    """Feed-forward sublayer parameters shared by transformer-style blocks."""

    d_ff: int
    dropout_p: float = 0.0
    activation: str = "silu"
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class ProjectionConfig:
    """Linear projection between feature dimensions."""

    input_dim: int
    output_dim: int
    dropout_p: float = 0.0
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class ConvModuleConfig:
    """Conformer-style depthwise-separable convolution module.

    ``kernel_size`` is measured over the time axis. ``expansion_factor`` controls
    the pointwise expansion before GLU, so even values preserve a clean channel
    split.
    """
    kernel_size: int
    expansion_factor: int = 2
    activation: str = "silu"
    dropout_p: float = 0.0


@dataclass(kw_only=True, frozen=True)
class CGMLPConfig:
    """Convolutional Gating MLP (CSGU) used as the local branch of Branchformer.

    `linear_units` is the gated-MLP inner dim before the depthwise conv; it is
    split in half over the up-projection so the depthwise conv runs on
    `linear_units` channels. `gate_activation` is applied to the gating branch
    after the depthwise conv (``identity`` is the paper default).
    """
    linear_units: int
    conv_kernel_size: int
    activation: str = "gelu"
    gate_activation: str = "identity"
    dropout_p: float = 0.0


@dataclass(kw_only=True, frozen=True)
class BranchMergeConfig:
    """Merge policy for the parallel attention and cgMLP branches."""
    method: Literal["concat", "learned_ave", "fixed_ave"]
    fixed_weight: float = 0.5


@dataclass(kw_only=True, frozen=True)
class EBranchMergeConfig:
    """Enhanced merge for E-Branchformer: concat branches, DwConv fusion, linear projection."""
    conv_kernel_size: int = 31


@dataclass(kw_only=True, frozen=True)
class TransformerEncoderConfig:
    """Standard transformer encoder stack with optional per-block RoPE."""

    mha: MultiHeadAttentionConfig
    mlp: MLPConfig
    type: Literal["transformer"] = "transformer"
    num_layers: int = 1
    pre_norm: bool = True
    residual_dropout_p: float = 0.0
    pooling: Literal["masked_mean"] = "masked_mean"
    positional_encoding: PerBlockPositionalEncodingConfig | None = None
    layer_norm_eps: float = 1e-5
    final_norm: bool = False


@dataclass(kw_only=True, frozen=True)
class ConformerEncoderConfig:
    """Macaron-style Conformer encoder with parallel MHA, conv, and FFN branches."""
    mha: MultiHeadAttentionConfig
    mlp: MLPConfig
    conv: ConvModuleConfig
    type: Literal["conformer"] = "conformer"
    num_layers: int = 1
    pre_norm: bool = True
    residual_dropout_p: float = 0.0
    pooling: Literal["masked_mean"] = "masked_mean"
    positional_encoding: PerBlockPositionalEncodingConfig | None = None
    layer_norm_eps: float = 1e-5
    final_norm: bool = False

    capabilities: ClassVar[frozenset[str]] = frozenset({SEQUENCE_CONVOLUTION})


@dataclass(kw_only=True, frozen=True)
class BranchformerEncoderConfig:
    """Vanilla Branchformer (Peng et al., 2022): parallel MHA + cgMLP, merged once."""
    mha: MultiHeadAttentionConfig
    cgmlp: CGMLPConfig
    merge: BranchMergeConfig
    type: Literal["branchformer"] = "branchformer"
    num_layers: int = 1
    pre_norm: bool = True
    residual_dropout_p: float = 0.0
    pooling: Literal["masked_mean"] = "masked_mean"
    positional_encoding: PerBlockPositionalEncodingConfig | None = None
    attention_branch_drop_p: float = 0.0
    cgmlp_branch_drop_p: float = 0.0
    layer_norm_eps: float = 1e-5
    final_norm: bool = False

    capabilities: ClassVar[frozenset[str]] = frozenset({SEQUENCE_CONVOLUTION})


@dataclass(kw_only=True, frozen=True)
class EBranchformerEncoderConfig:
    """E-Branchformer (Kim et al., 2022): parallel MHA + cgMLP with enhanced DwConv merge."""
    mha: MultiHeadAttentionConfig
    cgmlp: CGMLPConfig
    merge: EBranchMergeConfig
    mlp: MLPConfig | None = None
    use_ffn: bool = False
    macaron_ffn: bool = False
    type: Literal["ebranchformer"] = "ebranchformer"
    num_layers: int = 1
    pre_norm: bool = True
    residual_dropout_p: float = 0.0
    pooling: Literal["masked_mean"] = "masked_mean"
    positional_encoding: PerBlockPositionalEncodingConfig | None = None
    layer_norm_eps: float = 1e-5
    final_norm: bool = False

    capabilities: ClassVar[frozenset[str]] = frozenset({SEQUENCE_CONVOLUTION})


@dataclass(kw_only=True, frozen=True)
class AltTransformerEncoderConfig:
    """EAT / data2vec-style AltBlock encoder stack."""

    mha: MultiHeadAttentionConfig
    mlp: MLPConfig
    type: Literal["alt_transformer"] = "alt_transformer"
    num_layers: int = 1
    layer_norm_first: bool = False
    qkv_bias: bool = True
    post_mlp_dropout_p: float = 0.0
    residual_dropout_p: float = 0.0
    pooling: Literal["masked_mean", "cls"] = "cls"
    layer_norm_eps: float = 1e-6
    final_norm: bool = False


@dataclass(kw_only=True, frozen=True)
class MelSpectrogramProcessorConfig:
    """Mel-spectrogram front-end for raw waveform batches.

    Frequency limits and window sizes are in Hz and samples at ``sr``. Frame
    validity controls how partial overlap at sequence edges affects the output
    mask.
    """

    n_mels: int
    type: Literal["melspec"] = "melspec"
    n_fft: int = 2048
    f_min: float = 0.0
    f_max: float | None = None
    sr: int = 16000
    win_length: int = 2048
    hop_length: int = 512
    pad: int = 0
    window_fn: Literal["hann"] = "hann"
    power: float = 2.0
    normalized: bool | Literal["window", "frame_length"] = False
    center: bool = True
    pad_mode: Literal["constant", "reflect", "replicate", "circular"] = "reflect"
    norm: Literal["slaney"] | None = None
    mel_scale: Literal["htk", "slaney"] = "htk"
    log_mel: bool = False
    log_mel_top_db: float | None = 120.0
    batch_norm: bool = False
    batch_norm_momentum: float = 0.01
    batch_norm_eps: float = 1e-5
    frame_validity: Literal["strict_window", "overlap_any"] = "strict_window"

    capabilities: ClassVar[frozenset[str]] = frozenset({RECONSTRUCTION_TARGET})

    @property
    def output_dim(self) -> int:
        return self.n_mels


@dataclass(kw_only=True, frozen=True)
class ConvolutionalProcessorConfig:
    """Stacked 1-D convolutions on mono raw waveform ``(B, S)``."""

    type: Literal["conv"] = "conv"
    in_channels: int
    out_channels: int
    kernel_sizes: tuple[int, ...]
    strides: tuple[int, ...]
    groups: int = 1
    padding: int = 0
    dilation: int = 1
    bias: bool = True
    activation: str = "relu"
    initialization: str = "kaiming_uniform"

    @property
    def output_dim(self) -> int:
        return self.out_channels

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_sizes", tuple(self.kernel_sizes))
        object.__setattr__(self, "strides", tuple(self.strides))


@dataclass(kw_only=True, frozen=True)
class Wav2Vec2FeatureExtractorProcessorConfig:
    """Fairseq-style multi-layer conv feature extractor used by wav2vec2/HuBERT."""

    type: Literal["wav2vec2_feature_extractor"] = "wav2vec2_feature_extractor"
    sr: int = 16000
    conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = False
    activation: str = "gelu"
    extractor_mode: Literal["default", "layer_norm"] = "default"
    group_norm_num_groups: int = 512
    initialization: str = "kaiming_normal"

    @property
    def output_dim(self) -> int:
        return self.conv_dim[-1]

    def __post_init__(self) -> None:
        object.__setattr__(self, "conv_dim", tuple(self.conv_dim))
        object.__setattr__(self, "conv_kernel", tuple(self.conv_kernel))
        object.__setattr__(self, "conv_stride", tuple(self.conv_stride))


@dataclass(kw_only=True, frozen=True)
class KaldiFbankProcessorConfig:
    """Kaldi fbank front-end used by EAT with dataset mean/std normalization."""

    type: Literal["kaldi_fbank"] = "kaldi_fbank"
    sr: int = 16000
    num_mel_bins: int = 128
    frame_shift_ms: int = 10
    target_length: int = 1024
    norm_mean: float = -4.268
    norm_std: float = 4.569
    remove_waveform_mean: bool = True
    htk_compat: bool = True
    use_energy: bool = False
    dither: float = 0.0
    window_type: Literal["hanning"] = "hanning"
    mask_all_frames: bool = False

    @property
    def output_dim(self) -> int:
        return self.num_mel_bins


ProcessorConfig = (
    MelSpectrogramProcessorConfig
    | ConvolutionalProcessorConfig
    | Wav2Vec2FeatureExtractorProcessorConfig
    | KaldiFbankProcessorConfig
)


@dataclass(kw_only=True, frozen=True)
class PatchifyTokenizerConfig:
    """Patch grid tokenizer over processor features ``(B, T, F)``.

    ``patch_shape`` and ``stride`` are measured in time and frequency bins.
    ``padding_strategy`` applies when the time axis does not fit a complete
    patch grid.
    """

    type: Literal["patchify"] = "patchify"
    patch_shape: tuple[int, int]
    padding_strategy: Literal["pad", "truncate"]
    stride: tuple[int, int] | None = None

    capabilities: ClassVar[frozenset[str]] = frozenset({PATCH_GRID})

    @property
    def output_dim(self) -> int:
        patch_t, patch_f = self.patch_shape
        return patch_t * patch_f

    @property
    def effective_stride(self) -> tuple[int, int]:
        if self.stride is None:
            return self.patch_shape
        return self.stride

    def __post_init__(self) -> None:
        if isinstance(self.patch_shape, list):
            object.__setattr__(self, "patch_shape", tuple(self.patch_shape))
        if isinstance(self.stride, list):
            object.__setattr__(self, "stride", tuple(self.stride))


@dataclass(kw_only=True, frozen=True)
class NoOpTokenizerConfig:
    """Identity tokenizer when processor output dim already matches token dim."""

    type: Literal["none"] = "none"
    _output_dim: int

    @property
    def output_dim(self) -> int:
        return self._output_dim


TokenizerConfig = PatchifyTokenizerConfig | NoOpTokenizerConfig


@dataclass(kw_only=True, frozen=True)
class EmbeddingSubsamplingConfig:
    """Optional strided token subsampling before the embedding projection."""

    type: Literal["strided_select"] = "strided_select"
    factor: int = 1


def _subsampling_capabilities(config: EmbeddingSubsamplingConfig | None) -> frozenset[str]:
    if config is not None and config.factor != 1:
        return frozenset({TOKEN_SUBSAMPLING})
    return frozenset()


@dataclass(kw_only=True, frozen=True)
class LinearEmbeddingConfig:
    """Linear token projection with optional entry normalization and positional encoding."""

    positional_encoding: EmbeddingPositionalEncodingConfig
    type: Literal["linear"] = "linear"
    dropout_p: float = 0.0
    initialization: str = "xavier_uniform"
    subsampling: EmbeddingSubsamplingConfig | None = None
    input_layer_norm: bool = False
    input_layer_norm_eps: float = 1e-5
    post_pos_layer_norm: bool = False
    post_pos_layer_norm_eps: float = 1e-5

    @property
    def capabilities(self) -> frozenset[str]:
        return _subsampling_capabilities(self.subsampling)


@dataclass(kw_only=True, frozen=True)
class ConvEmbeddingConfig:
    """1-D conv token projection with optional layer norm and positional encoding."""

    positional_encoding: EmbeddingPositionalEncodingConfig
    type: Literal["conv"] = "conv"
    kernel_size: int = 1
    dropout_p: float = 0.0
    initialization: str = "kaiming_uniform"
    layer_norm: bool = False
    subsampling: EmbeddingSubsamplingConfig | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        return _subsampling_capabilities(self.subsampling)


@dataclass(kw_only=True, frozen=True)
class EATEmbeddingConfig:
    """EAT patch projection with 2-D sin-cos PE, CLS token, and entry layer norm."""

    positional_encoding: "Sinusoidal2DPositionalEncodingConfig"
    type: Literal["eat"] = "eat"
    dropout_p: float = 0.0
    initialization: str = "xavier_uniform"
    entry_layer_norm_eps: float = 1e-6
    cls_init_std: float = 0.02

    capabilities: ClassVar[frozenset[str]] = frozenset({PREFIX_TOKEN})


EmbeddingConfig = (
    LinearEmbeddingConfig
    | ConvEmbeddingConfig
    | EATEmbeddingConfig
)


@dataclass(kw_only=True, frozen=True)
class ProjectionHeadConfig:
    """SimCLR-style MLP applied to pooled encoder output (``cls``) before InfoNCE."""
    depth: int
    hidden_dim: int
    output_dim: int
    activation: str
    final_l2_normalize: bool
    dropout_p: float = 0.0
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class DINOHeadConfig:
    """DiNO-style projection head with prototype bottleneck."""

    depth: int
    hidden_dim: int
    bottleneck_dim: int
    num_prototypes: int
    activation: str
    dropout_p: float = 0.0
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class ContrastiveObjectiveConfig:
    """SimCLR-style contrastive objective with two augmented views."""

    type: Literal["contrastive"] = "contrastive"
    view_strategy: Literal["augmentation", "dropout"]
    projection: ProjectionHeadConfig
    temperature: float = 0.1
    symmetric: bool = True
    compute_stats: bool = True


@dataclass(kw_only=True, frozen=True)
class BYOLObjectiveConfig:
    """Bootstrap-your-own-latent objective with EMA or stop-gradient teacher."""

    type: Literal["byol"] = "byol"
    projector: ProjectionHeadConfig
    predictor: ProjectionHeadConfig
    teacher_mode: Literal["ema", "stop_grad"] = "ema"
    teacher_ema_decay: float = 0.996


@dataclass(kw_only=True, frozen=True)
class DINOObjectiveConfig:
    """Self-distillation objective with prototype head and centering."""

    type: Literal["dino"] = "dino"
    head: DINOHeadConfig
    teacher_mode: Literal["ema", "stop_grad"] = "ema"
    teacher_ema_decay: float = 0.996
    teacher_temperature: float = 0.04
    student_temperature: float = 0.1
    center_momentum: float = 0.9


@dataclass(kw_only=True, frozen=True)
class TemporalMaskingConfig:
    """Span masking schedule for JEPA-style encoder targets."""

    mask_ratio: float = 0.5
    min_span_len: int = 1
    max_span_len: int = 10
    allow_overlap: bool = True


@dataclass(kw_only=True, frozen=True)
class SpanMaskingConfig:
    """Time-axis span masking as used in wav2vec2/HuBERT."""

    mask_prob: float
    mask_length: int
    type: Literal["span"] = "span"
    mask_selection: Literal["static", "uniform", "normal", "poisson"] = "static"
    mask_other: float = 0.0
    min_space: int = 0
    no_overlap: bool = False
    require_same_masks: bool = False


@dataclass(kw_only=True, frozen=True)
class ChannelMaskingConfig:
    """Feature-channel span masking as used in wav2vec2/HuBERT."""

    mask_prob: float
    mask_length: int
    type: Literal["channel"] = "channel"
    mask_selection: Literal["static", "uniform", "normal", "poisson"] = "static"
    mask_other: float = 0.0
    min_space: int = 0
    no_overlap: bool = False


MaskingStrategyConfig = SpanMaskingConfig | ChannelMaskingConfig


@dataclass(kw_only=True, frozen=True)
class ConstantScalarSchedulerConfig:
    """Scalar schedule that holds a fixed value."""

    value: float
    type: Literal["constant"] = "constant"


@dataclass(kw_only=True, frozen=True)
class ExponentialDecayScalarSchedulerConfig:
    """Scalar schedule that exponentially decays from ``start`` to ``end``."""

    start: float
    end: float
    decay: float
    type: Literal["exponential_decay"] = "exponential_decay"


ScalarSchedulerConfig = (
    ConstantScalarSchedulerConfig | ExponentialDecayScalarSchedulerConfig
)


@dataclass(kw_only=True, frozen=True)
class Wav2Vec2QuantizerConfig:
    """Gumbel-product vector quantizer for wav2vec2 contrastive targets."""

    temperature_scheduler: ScalarSchedulerConfig
    type: Literal["gumbel_product"] = "gumbel_product"
    num_groups: int = 2
    num_vars: int = 320
    vq_dim: int = 256
    hard: bool = True
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class JEPAObjectiveConfig:
    """Joint embedding predictive architecture with EMA teacher and predictor."""

    type: Literal["jepa"] = "jepa"
    loss_style: Literal["mse", "mae"]
    mask_token_init_scale: float
    teacher_mode: Literal["ema", "stop_grad"] = "ema"
    output_norm_type: Literal["l2", "layernorm"] = "l2"
    teacher_output_norm: bool = True
    student_output_norm: bool = True

    teacher_ema_decay: float = 0.999
    predictor_config: MLPConfig
    masking: TemporalMaskingConfig = field(default_factory=TemporalMaskingConfig)


@dataclass(kw_only=True, frozen=True)
class MAEMaskingConfig:
    """Masked autoencoder token removal ratio and optional grouping factor."""

    mask_ratio: float = 0.75
    group_factor: int | None = None


@dataclass(kw_only=True, frozen=True)
class MAEObjectiveConfig:
    """Masked autoencoder with lightweight decoder and patch reconstruction targets."""

    required_capabilities: ClassVar[frozenset[str]] = frozenset(
        {RECONSTRUCTION_TARGET, PATCH_GRID}
    )
    incompatible_capabilities: ClassVar[frozenset[str]] = frozenset(
        {TOKEN_SUBSAMPLING, SEQUENCE_CONVOLUTION}
    )

    type: Literal["mae"] = "mae"
    decoder_d_model: int
    decoder: EncoderConfig
    decoder_positional_encoding: EmbeddingPositionalEncodingConfig
    mask_token_init_scale: float = 0.02
    norm_pix_loss: Literal["per_patch", "global", "none"] = "per_patch"
    norm_pix_eps: float = 1e-6
    target_patch_order: Literal["time_frequency", "frequency_time"] = "time_frequency"
    masking: MAEMaskingConfig = field(default_factory=MAEMaskingConfig)


@dataclass(kw_only=True, frozen=True)
class UFODecoderConfig:
    """Lightweight 2-D CNN decoder for EAT frame-level UFO targets."""

    num_layers: int = 6
    kernel_size: int = 3
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6


@dataclass(kw_only=True, frozen=True)
class UFOMaskingConfig:
    """Inverse block masking for EAT UFO pre-training."""

    mask_ratio: float = 0.8
    block_size: tuple[int, int] = (2, 2)
    num_masks: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_size", tuple(self.block_size))


@dataclass(kw_only=True, frozen=True)
class UFOObjectiveConfig:
    """EAT Utterance-Frame Objective with EMA teacher and CNN decoder."""

    required_capabilities: ClassVar[frozenset[str]] = frozenset({PATCH_GRID, PREFIX_TOKEN})
    incompatible_capabilities: ClassVar[frozenset[str]] = frozenset(
        {TOKEN_SUBSAMPLING, SEQUENCE_CONVOLUTION}
    )

    type: Literal["ufo"] = "ufo"
    decoder: UFODecoderConfig
    masking: UFOMaskingConfig = field(default_factory=UFOMaskingConfig)
    utterance_loss_weight: float = 1.0
    teacher_mode: Literal["ema", "stop_grad"] = "ema"
    teacher_ema_decay: float = 0.999
    mask_token_init_scale: float = 0.02


@dataclass(kw_only=True, frozen=True)
class Wav2Vec2ObjectiveConfig:
    """wav2vec2 contrastive plus diversity and feature penalty losses."""

    incompatible_capabilities: ClassVar[frozenset[str]] = frozenset({TOKEN_SUBSAMPLING})

    quantizer: Wav2Vec2QuantizerConfig
    time_masking: SpanMaskingConfig
    type: Literal["wav2vec2"] = "wav2vec2"
    feature_masking: ChannelMaskingConfig | None = None
    mask_emb_init_scale: float = 0.5
    final_dim: int = 256
    num_negatives: int = 100
    cross_sample_negatives: int = 0
    logit_temp: float = 0.1
    diversity_loss_weight: float = 0.1
    features_pen_weight: float = 10.0
    feature_grad_mult: float = 0.1
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class HuBERTTeacherConfig:
    """Offline HuBERT k-means teacher used to generate pseudo-labels."""

    type: Literal["pretrained_hubert_kmeans"] = "pretrained_hubert_kmeans"
    encoder_id: str = "facebook/hubert-base-ls960"
    encoder_revision: str | None = None
    feature_layer: int = 9
    num_codes: int = 500
    target_sample_rate: int = 16000
    km_url: str | None = None
    km_sha256: str | None = None


@dataclass(kw_only=True, frozen=True)
class HuBERTObjectiveConfig:
    """HuBERT masked prediction with optional feature masking."""

    incompatible_capabilities: ClassVar[frozenset[str]] = frozenset({TOKEN_SUBSAMPLING})

    teacher: HuBERTTeacherConfig
    time_masking: SpanMaskingConfig
    type: Literal["hubert"] = "hubert"
    feature_masking: ChannelMaskingConfig | None = None
    mask_emb_init_scale: float = 0.5
    final_dim: int = 256
    logit_temp: float = 0.1
    label_emb_init_scale: float = 0.02
    masked_loss_weight: float = 1.0
    unmasked_loss_weight: float = 0.0
    feature_grad_mult: float = 0.1
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class SFTHeadConfig:
    """MLP classification head on pooled encoder output."""

    depth: int
    hidden_dim: int
    activation: str = "gelu"
    dropout_p: float = 0.0
    initialization: str = "xavier_uniform"


@dataclass(kw_only=True, frozen=True)
class SFTObjectiveConfig:
    """Supervised fine-tuning for single-label audio classification."""

    type: Literal["sft"] = "sft"
    num_classes: int
    head: SFTHeadConfig
    freeze_backbone: bool = False
    backbone_lr_scale: float = 1.0
    label_smoothing: float = 0.0


ObjectiveConfig = (
    ContrastiveObjectiveConfig
    | BYOLObjectiveConfig
    | DINOObjectiveConfig
    | JEPAObjectiveConfig
    | MAEObjectiveConfig
    | UFOObjectiveConfig
    | Wav2Vec2ObjectiveConfig
    | HuBERTObjectiveConfig
    | SFTObjectiveConfig
)


@dataclass(kw_only=True, frozen=True)
class RopePositionalEncodingConfig:
    """Rotary positional encoding applied inside attention blocks."""

    type: Literal["rope"] = "rope"
    max_len: int = 1024
    base: float = 10000.0


@dataclass(kw_only=True, frozen=True)
class NoPositionalEncodingConfig:
    """Skip positional encoding."""

    type: Literal["none"] = "none"


@dataclass(kw_only=True, frozen=True)
class LearnablePositionalEncodingConfig:
    """Learnable absolute positional embeddings up to ``max_len`` tokens."""

    type: Literal["learnable"] = "learnable"
    max_len: int = 1024


@dataclass(kw_only=True, frozen=True)
class SinusoidalPositionalEncodingConfig:
    """Fixed sinusoidal absolute positional embeddings."""

    type: Literal["sinusoidal"] = "sinusoidal"
    max_len: int = 1024
    base: float = 10000.0


@dataclass(kw_only=True, frozen=True)
class LearnableTimeFrequencyPositionalEncodingConfig:
    """Learnable 2-D positional embeddings over time and frequency patch grids."""

    type: Literal["learnable_2d"] = "learnable_2d"
    max_t: int
    max_f: int


@dataclass(kw_only=True, frozen=True)
class Sinusoidal2DPositionalEncodingConfig:
    """Fixed 2-D sinusoidal positional embeddings over patch grids."""

    type: Literal["sinusoidal_2d"] = "sinusoidal_2d"
    max_time_patches: int
    max_freq_patches: int


@dataclass(kw_only=True, frozen=True)
class ConvolutionalPositionalEncodingConfig:
    """Depthwise-separable conv positional encoding over the token sequence."""

    type: Literal["conv1d"] = "conv1d"
    kernel_size: int
    groups: int = 1
    activation: str = "gelu"
    initialization: str = "xavier_uniform"


PerBlockPositionalEncodingConfig = RopePositionalEncodingConfig

EmbeddingPositionalEncodingConfig = (
    NoPositionalEncodingConfig
    | LearnablePositionalEncodingConfig
    | SinusoidalPositionalEncodingConfig
    | LearnableTimeFrequencyPositionalEncodingConfig
    | Sinusoidal2DPositionalEncodingConfig
    | ConvolutionalPositionalEncodingConfig
)

PositionalEncodingConfig = (
    EmbeddingPositionalEncodingConfig | PerBlockPositionalEncodingConfig
)


@dataclass(kw_only=True, frozen=True)
class AudioModelConfig:
    """Full audio representation model: processor through encoder."""

    d_model: int
    processor: ProcessorConfig
    tokenizer: TokenizerConfig
    embedding: EmbeddingConfig
    encoder: (
        TransformerEncoderConfig
        | ConformerEncoderConfig
        | BranchformerEncoderConfig
        | EBranchformerEncoderConfig
        | AltTransformerEncoderConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass(kw_only=True, frozen=True)
class AudioSSLTrainingConfig:
    """Top-level self-supervised training configuration."""

    model: AudioModelConfig
    objective: ObjectiveConfig
    trainer: TrainerConfig


EncoderConfig = (
    TransformerEncoderConfig
    | ConformerEncoderConfig
    | BranchformerEncoderConfig
    | EBranchformerEncoderConfig
    | AltTransformerEncoderConfig
)


@dataclass(kw_only=True, frozen=True)
class LoudnessAugmentationConfig:
    """Random per-example gain in dB, applied with probability ``p``."""

    p: float
    min_gain_db: float
    max_gain_db: float
    type: Literal["loudness"] = "loudness"


AugmentationConfig = LoudnessAugmentationConfig


@dataclass(kw_only=True, frozen=True)
class AudioAugmentorConfig:
    """Ordered waveform augmentations applied to every view an objective requests."""

    augmentations: tuple[AugmentationConfig, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "augmentations", tuple(self.augmentations))


@dataclass(kw_only=True, frozen=True)
class AdamWOptimizerConfig:
    """AdamW optimizer with configurable weight-decay exclusions.

    ``no_weight_decay_1d`` exempts every rank <= 1 parameter (biases, norm
    scales, mask tokens). ``no_weight_decay_patterns`` exempts additional
    parameters by qualified-name substring; every pattern must match at least
    one trainable parameter.
    """

    lr: float
    type: Literal["adamw"] = "adamw"
    weight_decay: float = 0.01
    no_weight_decay_1d: bool = True
    no_weight_decay_patterns: tuple[str, ...] = ()
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        object.__setattr__(
            self, "no_weight_decay_patterns", tuple(self.no_weight_decay_patterns)
        )


@dataclass(kw_only=True, frozen=True)
class SGDOptimizerConfig:
    """SGD optimizer with optional Nesterov momentum."""

    lr: float
    type: Literal["sgd"] = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    no_weight_decay_1d: bool = True
    no_weight_decay_patterns: tuple[str, ...] = ()
    nesterov: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "no_weight_decay_patterns", tuple(self.no_weight_decay_patterns)
        )


@dataclass(kw_only=True, frozen=True)
class LARSOptimizerConfig:
    """Layer-wise adaptive rate scaling optimizer."""

    lr: float
    type: Literal["lars"] = "lars"
    momentum: float = 0.9
    weight_decay: float = 0.0
    no_weight_decay_1d: bool = True
    no_weight_decay_patterns: tuple[str, ...] = ()
    eta: float = 0.001

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "no_weight_decay_patterns", tuple(self.no_weight_decay_patterns)
        )


OptimizerConfig = AdamWOptimizerConfig | SGDOptimizerConfig | LARSOptimizerConfig


@dataclass(kw_only=True, frozen=True)
class ConstantSchedulerConfig:
    """Learning-rate schedule that holds the base optimizer rate."""

    horizon_dependent: ClassVar[bool] = False

    type: Literal["constant"] = "constant"


@dataclass(kw_only=True, frozen=True)
class WarmupCosineSchedulerConfig:
    """Linear warmup over ``warmup_steps`` optimizer steps, then cosine decay to the run horizon."""

    horizon_dependent: ClassVar[bool] = True

    type: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_steps: int = 0
    min_lr_scale: float = 0.0


@dataclass(kw_only=True, frozen=True)
class CosineAnnealingSchedulerConfig:
    """Cosine decay from the base rate to ``min_lr_scale`` at the run horizon."""

    horizon_dependent: ClassVar[bool] = True

    type: Literal["cosine_annealing"] = "cosine_annealing"
    min_lr_scale: float = 0.0


@dataclass(kw_only=True, frozen=True)
class StepLRSchedulerConfig:
    """Multiply the rate by ``gamma`` every ``step_size`` optimizer steps."""

    horizon_dependent: ClassVar[bool] = False

    step_size: int
    type: Literal["step_lr"] = "step_lr"
    gamma: float = 0.1


LRSchedulerConfig = (
    ConstantSchedulerConfig
    | WarmupCosineSchedulerConfig
    | CosineAnnealingSchedulerConfig
    | StepLRSchedulerConfig
)


@dataclass(kw_only=True, frozen=True)
class BatchConfig:
    """Batch composition: everything about batching that changes the trained model.

    Worker counts and pinning are operational and live in ``RunConfig.loader``.
    """

    per_device_batch_size: int
    gradient_accumulation_steps: int = 1
    drop_last: bool = True
    shuffle: bool = True


@dataclass(kw_only=True, frozen=True)
class TrainingLoopConfig:
    """Training horizon, gradient clipping, and non-finite handling.

    At least one of ``max_epochs`` / ``max_steps`` must be set; ``max_steps``
    is required when the train loader has no length. ``nonfinite_policy``
    decides what a non-finite loss or gradient norm does: ``"raise"`` aborts,
    ``"skip"`` drops that optimizer step and counts it. More than
    ``max_consecutive_skipped_steps`` skipped steps in a row (from either cause
    or from GradScaler overflow) always aborts.
    """

    max_epochs: int | None = None
    max_steps: int | None = None
    gradient_clip_norm: float | None = None
    nonfinite_policy: Literal["raise", "skip"] = "raise"
    max_consecutive_skipped_steps: int = 20


@dataclass(kw_only=True, frozen=True)
class MixedPrecisionConfig:
    """Autocast dtype and optional GradScaler for float16 training."""

    dtype: Literal["bfloat16", "float16"]
    grad_scaler: bool = False


@dataclass(kw_only=True, frozen=True)
class TrainerConfig:
    """Optimization hyperparameters: everything the trainer does that changes the model.

    Operational settings (checkpointing, validation cadence, logging, worker
    counts) live in :class:`~audiossl.config.RunConfig` and are not part of
    the run fingerprint.
    """

    optimizer: OptimizerConfig
    scheduler: LRSchedulerConfig
    batch: BatchConfig
    loop: TrainingLoopConfig
    augmentor: AudioAugmentorConfig | None = None
    mixed_precision: MixedPrecisionConfig | None = None


__all__ = [
    "AdamWOptimizerConfig",
    "AltTransformerEncoderConfig",
    "AudioAugmentorConfig",
    "AudioModelConfig",
    "AudioSSLTrainingConfig",
    "AugmentationConfig",
    "BatchConfig",
    "BYOLObjectiveConfig",
    "ChannelMaskingConfig",
    "ConstantScalarSchedulerConfig",
    "ConstantSchedulerConfig",
    "CosineAnnealingSchedulerConfig",
    "BranchformerEncoderConfig",
    "BranchMergeConfig",
    "EBranchMergeConfig",
    "EBranchformerEncoderConfig",
    "CGMLPConfig",
    "ConformerEncoderConfig",
    "ConvolutionalProcessorConfig",
    "ConvModuleConfig",
    "ContrastiveObjectiveConfig",
    "ConvolutionalPositionalEncodingConfig",
    "DINOHeadConfig",
    "DINOObjectiveConfig",
    "HuBERTObjectiveConfig",
    "HuBERTTeacherConfig",
    "JEPAObjectiveConfig",
    "ProcessorConfig",
    "EncoderConfig",
    "EmbeddingSubsamplingConfig",
    "EmbeddingConfig",
    "EmbeddingPositionalEncodingConfig",
    "LoudnessAugmentationConfig",
    "MAEMaskingConfig",
    "MAEObjectiveConfig",
    "ConvEmbeddingConfig",
    "EATEmbeddingConfig",
    "LinearEmbeddingConfig",
    "MLPConfig",
    "MelSpectrogramProcessorConfig",
    "MixedPrecisionConfig",
    "MultiHeadAttentionConfig",
    "NoOpTokenizerConfig",
    "PatchifyTokenizerConfig",
    "PerBlockPositionalEncodingConfig",
    "PositionalEncodingConfig",
    "ProjectionConfig",
    "ProjectionHeadConfig",
    "RopePositionalEncodingConfig",
    "NoPositionalEncodingConfig",
    "LARSOptimizerConfig",
    "LearnablePositionalEncodingConfig",
    "LearnableTimeFrequencyPositionalEncodingConfig",
    "KaldiFbankProcessorConfig",
    "LRSchedulerConfig",
    "MaskingStrategyConfig",
    "OptimizerConfig",
    "SFTHeadConfig",
    "SFTObjectiveConfig",
    "SGDOptimizerConfig",
    "Sinusoidal2DPositionalEncodingConfig",
    "SinusoidalPositionalEncodingConfig",
    "RuntimeConfig",
    "ScalarSchedulerConfig",
    "StepLRSchedulerConfig",
    "SpanMaskingConfig",
    "TrainingLoopConfig",
    "TrainerConfig",
    "TemporalMaskingConfig",
    "TokenizerConfig",
    "TransformerEncoderConfig",
    "UFODecoderConfig",
    "UFOMaskingConfig",
    "UFOObjectiveConfig",
    "WarmupCosineSchedulerConfig",
    "Wav2Vec2FeatureExtractorProcessorConfig",
    "ExponentialDecayScalarSchedulerConfig",
    "Wav2Vec2ObjectiveConfig",
    "Wav2Vec2QuantizerConfig",
]

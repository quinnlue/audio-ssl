"""AudioModelConfig recipe for EAT-base inference."""

from __future__ import annotations

from audiossl.config import (
    AltTransformerEncoderConfig,
    AudioModelConfig,
    EATEmbeddingConfig,
    KaldiFbankProcessorConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    PatchifyTokenizerConfig,
    RuntimeConfig,
    Sinusoidal2DPositionalEncodingConfig,
)

_EAT_BASE_D_MODEL = 768
_EAT_BASE_DEPTH = 12
_EAT_BASE_NUM_HEADS = 12
_EAT_BASE_MLP_RATIO = 4
_EAT_PATCH_SIZE = 16
_EAT_N_MELS = 128
_EAT_SAMPLE_RATE = 16000
_EAT_TARGET_LENGTH = 1024
_EAT_MAX_TIME_PATCHES = 768
_EAT_LAYER_NORM_EPS = 1e-6


def make_eat_scaled_config(
    *,
    d_model: int,
    depth: int,
    num_heads: int,
    mlp_ratio: int = _EAT_BASE_MLP_RATIO,
    target_length: int = _EAT_TARGET_LENGTH,
    max_time_patches: int = _EAT_MAX_TIME_PATCHES,
    runtime: RuntimeConfig | None = None,
) -> AudioModelConfig:
    """Build an EAT AudioModelConfig with arbitrary transformer width and depth."""
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model!r}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth!r}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads!r}")
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
    if target_length <= 0 or target_length % _EAT_PATCH_SIZE != 0:
        raise ValueError(
            "target_length must be a positive multiple of "
            f"{_EAT_PATCH_SIZE}, got {target_length!r}"
        )

    freq_patches = _EAT_N_MELS // _EAT_PATCH_SIZE
    processor = KaldiFbankProcessorConfig(
        sr=_EAT_SAMPLE_RATE,
        num_mel_bins=_EAT_N_MELS,
        target_length=target_length,
        mask_all_frames=True,
    )
    tokenizer = PatchifyTokenizerConfig(
        patch_shape=(_EAT_PATCH_SIZE, _EAT_PATCH_SIZE),
        padding_strategy="truncate",
    )
    embedding = EATEmbeddingConfig(
        positional_encoding=Sinusoidal2DPositionalEncodingConfig(
            max_time_patches=max_time_patches,
            max_freq_patches=freq_patches,
        ),
        entry_layer_norm_eps=_EAT_LAYER_NORM_EPS,
    )
    encoder = AltTransformerEncoderConfig(
        mha=MultiHeadAttentionConfig(num_heads=num_heads),
        mlp=MLPConfig(
            d_ff=d_model * mlp_ratio,
            activation="gelu",
        ),
        num_layers=depth,
        layer_norm_first=False,
        qkv_bias=True,
        pooling="cls",
        layer_norm_eps=_EAT_LAYER_NORM_EPS,
        final_norm=False,
    )

    return AudioModelConfig(
        d_model=d_model,
        processor=processor,
        tokenizer=tokenizer,
        embedding=embedding,
        encoder=encoder,
        runtime=runtime if runtime is not None else RuntimeConfig(),
    )


def make_eat_base_config(
    *,
    target_length: int = _EAT_TARGET_LENGTH,
    runtime: RuntimeConfig | None = None,
) -> AudioModelConfig:
    """Build the AudioModelConfig that reproduces EAT-base inference."""
    return make_eat_scaled_config(
        d_model=_EAT_BASE_D_MODEL,
        depth=_EAT_BASE_DEPTH,
        num_heads=_EAT_BASE_NUM_HEADS,
        mlp_ratio=_EAT_BASE_MLP_RATIO,
        target_length=target_length,
        runtime=runtime,
    )


_EAT_500M_D_MODEL = 1024
_EAT_500M_DEPTH = 36
_EAT_500M_NUM_HEADS = 16


def make_eat_500m_config(
    *,
    target_length: int = _EAT_TARGET_LENGTH,
    runtime: RuntimeConfig | None = None,
) -> AudioModelConfig:
    """Build an EAT config sized for roughly 500M parameters."""
    return make_eat_scaled_config(
        d_model=_EAT_500M_D_MODEL,
        depth=_EAT_500M_DEPTH,
        num_heads=_EAT_500M_NUM_HEADS,
        target_length=target_length,
        runtime=runtime,
    )


__all__ = [
    "make_eat_500m_config",
    "make_eat_base_config",
    "make_eat_scaled_config",
]

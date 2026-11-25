"""AudioModelConfig recipes for Hugging Face HuBERT backbones."""

from __future__ import annotations

from typing import Any, Literal

from audiossl.config import (
    AudioModelConfig,
    ConvolutionalPositionalEncodingConfig,
    LinearEmbeddingConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoOpTokenizerConfig,
    RuntimeConfig,
    TransformerEncoderConfig,
    Wav2Vec2FeatureExtractorProcessorConfig,
)

_HUBERT_BASE_EMBED_DIM = 768
_HUBERT_BASE_DEPTH = 12
_HUBERT_BASE_NUM_HEADS = 12
_HUBERT_BASE_FFN_DIM = 3072
_HUBERT_CONV_DIM = (512, 512, 512, 512, 512, 512, 512)
_HUBERT_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
_HUBERT_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
_HUBERT_CONV_POS_KERNEL = 128
_HUBERT_CONV_POS_GROUPS = 16
_HUBERT_LAYER_NORM_EPS = 1e-5
_HUBERT_SAMPLE_RATE = 16000
_HUBERT_FEAT_PROJ_DROPOUT = 0.0
_HUBERT_HIDDEN_DROPOUT = 0.1
_HUBERT_ATTENTION_DROPOUT = 0.1


def make_hubert_base_ls960_config(
    *,
    runtime: RuntimeConfig | None = None,
    d_model: int = _HUBERT_BASE_EMBED_DIM,
    depth: int = _HUBERT_BASE_DEPTH,
    num_heads: int = _HUBERT_BASE_NUM_HEADS,
    ffn_dim: int = _HUBERT_BASE_FFN_DIM,
    conv_dim: tuple[int, ...] = _HUBERT_CONV_DIM,
    conv_kernel: tuple[int, ...] = _HUBERT_CONV_KERNEL,
    conv_stride: tuple[int, ...] = _HUBERT_CONV_STRIDE,
    conv_pos_kernel_size: int = _HUBERT_CONV_POS_KERNEL,
    conv_pos_groups: int = _HUBERT_CONV_POS_GROUPS,
    layer_norm_eps: float = _HUBERT_LAYER_NORM_EPS,
    sample_rate: int = _HUBERT_SAMPLE_RATE,
    feature_projection_dropout: float = _HUBERT_FEAT_PROJ_DROPOUT,
    hidden_dropout: float = _HUBERT_HIDDEN_DROPOUT,
    attention_dropout: float = _HUBERT_ATTENTION_DROPOUT,
    extractor_mode: Literal["default", "layer_norm"] = "default",
    feature_projection_layer_norm: bool = True,
) -> AudioModelConfig:
    """Build an ``AudioModelConfig`` compatible with ``facebook/hubert-base-ls960``.

    Parameters
    ----------
    runtime
        Optional runtime override. Defaults to a fresh :class:`RuntimeConfig`.
    d_model, depth, num_heads, ffn_dim
        Transformer width and depth. Defaults match the released checkpoint.
    conv_dim, conv_kernel, conv_stride
        Feature-extractor convolution stack. ``sample_rate`` is in Hz.
    conv_pos_kernel_size, conv_pos_groups
        Convolutional positional encoding on the projected feature sequence.
    layer_norm_eps, feature_projection_dropout, hidden_dropout, attention_dropout
        Normalization and dropout hyperparameters from the reference config.
    extractor_mode
        ``"default"`` uses group norm in the conv stack; ``"layer_norm"`` uses
        per-layer norm after each conv, matching ``feat_extract_norm`` in HF.
    feature_projection_layer_norm
        Whether the feature projection applies layer norm before the linear map.

    Notes
    -----
    The encoder uses post-norm blocks (``pre_norm=False``, ``final_norm=False``),
    matching non-stable HuBERT layouts.
    """

    if extractor_mode not in {"default", "layer_norm"}:
        raise ValueError(
            "HuBERT extractor_mode must be 'default' or 'layer_norm', "
            f"got {extractor_mode!r}"
        )

    processor = Wav2Vec2FeatureExtractorProcessorConfig(
        sr=sample_rate,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        conv_bias=False,
        activation="gelu",
        extractor_mode=extractor_mode,
        group_norm_num_groups=conv_dim[0],
        initialization="kaiming_normal",
    )
    tokenizer = NoOpTokenizerConfig(_output_dim=processor.output_dim)
    embedding = LinearEmbeddingConfig(
        positional_encoding=ConvolutionalPositionalEncodingConfig(
            kernel_size=conv_pos_kernel_size,
            groups=conv_pos_groups,
            activation="gelu",
            initialization="xavier_uniform",
        ),
        dropout_p=feature_projection_dropout,
        initialization="xavier_uniform",
        input_layer_norm=feature_projection_layer_norm,
        input_layer_norm_eps=layer_norm_eps,
        post_pos_layer_norm=True,
        post_pos_layer_norm_eps=layer_norm_eps,
    )
    encoder = TransformerEncoderConfig(
        mha=MultiHeadAttentionConfig(
            num_heads=num_heads,
            dropout_p=attention_dropout,
            initialization="xavier_uniform",
        ),
        mlp=MLPConfig(
            d_ff=ffn_dim,
            dropout_p=hidden_dropout,
            activation="gelu",
            initialization="xavier_uniform",
        ),
        num_layers=depth,
        pre_norm=False,
        residual_dropout_p=hidden_dropout,
        positional_encoding=None,
        layer_norm_eps=layer_norm_eps,
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


def make_hubert_config_from_hf(
    hf_config: Any,
    *,
    runtime: RuntimeConfig | None = None,
) -> AudioModelConfig:
    """Convert a compatible Hugging Face ``HubertConfig`` to ``AudioModelConfig``.

    Parameters
    ----------
    hf_config
        Hugging Face ``HubertConfig`` instance or a duck-typed equivalent.
    runtime
        Optional runtime override. Defaults to a fresh :class:`RuntimeConfig`.

    Notes
    -----
    Stable layer norm and ``conv_pos_batch_norm`` layouts are rejected.
    """

    if getattr(hf_config, "do_stable_layer_norm", False):
        raise ValueError("audiossl HuBERT integration does not support stable layer norm")
    if getattr(hf_config, "conv_pos_batch_norm", False):
        raise ValueError("audiossl HuBERT integration does not support conv_pos_batch_norm")

    feat_extract_norm = getattr(hf_config, "feat_extract_norm", "group")
    if feat_extract_norm == "group":
        extractor_mode = "default"
    elif feat_extract_norm == "layer":
        extractor_mode = "layer_norm"
    else:
        raise ValueError(f"Unsupported HuBERT feat_extract_norm {feat_extract_norm!r}")

    return make_hubert_base_ls960_config(
        runtime=runtime,
        d_model=hf_config.hidden_size,
        depth=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        ffn_dim=hf_config.intermediate_size,
        conv_dim=tuple(hf_config.conv_dim),
        conv_kernel=tuple(hf_config.conv_kernel),
        conv_stride=tuple(hf_config.conv_stride),
        conv_pos_kernel_size=hf_config.num_conv_pos_embeddings,
        conv_pos_groups=hf_config.num_conv_pos_embedding_groups,
        layer_norm_eps=hf_config.layer_norm_eps,
        feature_projection_dropout=hf_config.feat_proj_dropout,
        hidden_dropout=hf_config.hidden_dropout,
        attention_dropout=hf_config.attention_dropout,
        extractor_mode=extractor_mode,
        feature_projection_layer_norm=getattr(hf_config, "feat_proj_layer_norm", True),
    )


__all__ = [
    "make_hubert_base_ls960_config",
    "make_hubert_config_from_hf",
]

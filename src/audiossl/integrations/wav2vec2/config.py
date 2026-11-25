"""AudioModelConfig recipe for facebook/wav2vec2-base inference."""

from __future__ import annotations

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

_W2V2_BASE_EMBED_DIM = 768
_W2V2_BASE_DEPTH = 12
_W2V2_BASE_NUM_HEADS = 12
_W2V2_BASE_FFN_DIM = 3072
_W2V2_CONV_DIM = (512, 512, 512, 512, 512, 512, 512)
_W2V2_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
_W2V2_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
_W2V2_CONV_POS_KERNEL = 128
_W2V2_CONV_POS_GROUPS = 16
_W2V2_LAYER_NORM_EPS = 1e-5
_W2V2_SAMPLE_RATE = 16000
_W2V2_FEAT_PROJ_DROPOUT = 0.0
_W2V2_HIDDEN_DROPOUT = 0.1
_W2V2_ATTENTION_DROPOUT = 0.1


def make_wav2vec2_base_config(
    *,
    runtime: RuntimeConfig | None = None,
    d_model: int = _W2V2_BASE_EMBED_DIM,
    depth: int = _W2V2_BASE_DEPTH,
    num_heads: int = _W2V2_BASE_NUM_HEADS,
    ffn_dim: int = _W2V2_BASE_FFN_DIM,
    conv_dim: tuple[int, ...] = _W2V2_CONV_DIM,
    conv_kernel: tuple[int, ...] = _W2V2_CONV_KERNEL,
    conv_stride: tuple[int, ...] = _W2V2_CONV_STRIDE,
    conv_pos_kernel_size: int = _W2V2_CONV_POS_KERNEL,
    conv_pos_groups: int = _W2V2_CONV_POS_GROUPS,
    layer_norm_eps: float = _W2V2_LAYER_NORM_EPS,
    sample_rate: int = _W2V2_SAMPLE_RATE,
    feature_projection_dropout: float = _W2V2_FEAT_PROJ_DROPOUT,
    hidden_dropout: float = _W2V2_HIDDEN_DROPOUT,
    attention_dropout: float = _W2V2_ATTENTION_DROPOUT,
) -> AudioModelConfig:
    """Build the ``AudioModelConfig`` that reproduces ``facebook/wav2vec2-base`` inference.

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

    Notes
    -----
    The encoder uses post-norm blocks (``pre_norm=False``, ``final_norm=False``),
    matching the inference-only wav2vec2-base layout.
    """

    processor = Wav2Vec2FeatureExtractorProcessorConfig(
        sr=sample_rate,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        conv_bias=False,
        activation="gelu",
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
        input_layer_norm=True,
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


__all__ = ["make_wav2vec2_base_config"]

"""AudioModelConfig recipe for Dasheng ``dasheng_base`` inference."""

from __future__ import annotations

from audiossl.config import (
    AudioModelConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    LinearEmbeddingConfig,
    MLPConfig,
    MelSpectrogramProcessorConfig,
    MultiHeadAttentionConfig,
    PatchifyTokenizerConfig,
    RuntimeConfig,
    TransformerEncoderConfig,
)

# These are the dasheng_base hyperparameters from
# eval/models/dasheng-inference/dasheng/dasheng/train/models.py (dasheng_base
# and AudioTransformerMAE_Encoder defaults). Patch is (freq=64, time=4); since
# n_mels == patch_freq == 64, every released variant has F_patches=1.
_DASHENG_BASE_EMBED_DIM = 768
_DASHENG_BASE_DEPTH = 12
_DASHENG_BASE_NUM_HEADS = 12
_DASHENG_BASE_MLP_RATIO = 4
_DASHENG_PATCH_TIME = 4
_DASHENG_PATCH_FREQ = 64
_DASHENG_N_MELS = 64
_DASHENG_N_FFT = 512
_DASHENG_HOP = 160
_DASHENG_WIN = 512
_DASHENG_SAMPLE_RATE = 16000
_DASHENG_F_MIN = 0.0
_DASHENG_F_MAX = 8000.0
_DASHENG_TOP_DB = 120.0
_DASHENG_BN_MOMENTUM = 0.01
_DASHENG_LAYER_NORM_EPS = 1e-6


def make_dasheng_base_config(
    *,
    target_length: int = 1008,
    runtime: RuntimeConfig | None = None,
) -> AudioModelConfig:
    """Build the AudioModelConfig that reproduces dasheng_base inference.

    `target_length` is the number of mel frames the model is designed for
    (1008 in the released checkpoint). The encoder's max sequence length is
    ``target_length // patch_time = 252`` (``target_length_in_patches`` in the
    reference). Inputs that produce more than this many time patches will
    diverge from the reference, which chunks them.

    Parameters
    ----------
    target_length
        Expected mel-frame count for a single forward pass. Must be a positive
        multiple of the patch time size (4).
    runtime
        Optional runtime override. Defaults to a fresh :class:`RuntimeConfig`.
    """
    if target_length <= 0 or target_length % _DASHENG_PATCH_TIME != 0:
        raise ValueError(
            "target_length must be a positive multiple of "
            f"{_DASHENG_PATCH_TIME}, got {target_length!r}"
        )

    max_patches = target_length // _DASHENG_PATCH_TIME

    processor = MelSpectrogramProcessorConfig(
        n_mels=_DASHENG_N_MELS,
        n_fft=_DASHENG_N_FFT,
        f_min=_DASHENG_F_MIN,
        f_max=_DASHENG_F_MAX,
        sr=_DASHENG_SAMPLE_RATE,
        win_length=_DASHENG_WIN,
        hop_length=_DASHENG_HOP,
        center=True,
        log_mel=True,
        log_mel_top_db=_DASHENG_TOP_DB,
        batch_norm=True,
        batch_norm_momentum=_DASHENG_BN_MOMENTUM,
        mel_scale="htk",
        power=2.0,
        norm=None,
        # Dasheng does not propagate input masks through its mel front-end;
        # every STFT frame contributes to the patch sequence.
        frame_validity="overlap_any",
    )
    # Our PatchifyTokenizer's patch_shape convention is (pt, pf). Dasheng's
    # Conv2d kernel is (F=64, T=4) on (B, 1, F, T); truncate matches Dasheng's
    # Conv2d which has no implicit padding so trailing frames are dropped.
    tokenizer = PatchifyTokenizerConfig(
        patch_shape=(_DASHENG_PATCH_TIME, _DASHENG_PATCH_FREQ),
        padding_strategy="truncate",
    )
    embedding = LinearEmbeddingConfig(
        positional_encoding=LearnableTimeFrequencyPositionalEncodingConfig(
            max_t=max_patches,
            max_f=1,
        )
    )
    encoder = TransformerEncoderConfig(
        mha=MultiHeadAttentionConfig(num_heads=_DASHENG_BASE_NUM_HEADS),
        mlp=MLPConfig(
            d_ff=_DASHENG_BASE_EMBED_DIM * _DASHENG_BASE_MLP_RATIO,
            activation="gelu",
        ),
        num_layers=_DASHENG_BASE_DEPTH,
        pre_norm=True,
        positional_encoding=None,
        layer_norm_eps=_DASHENG_LAYER_NORM_EPS,
        final_norm=True,
    )

    return AudioModelConfig(
        d_model=_DASHENG_BASE_EMBED_DIM,
        processor=processor,
        tokenizer=tokenizer,
        embedding=embedding,
        encoder=encoder,
        runtime=runtime if runtime is not None else RuntimeConfig(),
    )


__all__ = ["make_dasheng_base_config"]

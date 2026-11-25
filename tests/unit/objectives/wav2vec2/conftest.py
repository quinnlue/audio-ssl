import pytest
import torch

from audiossl.config import (
    ChannelMaskingConfig,
    ConstantScalarSchedulerConfig,
    RuntimeConfig,
    SpanMaskingConfig,
    Wav2Vec2ObjectiveConfig,
    Wav2Vec2QuantizerConfig,
)
from audiossl.integrations.wav2vec2 import make_wav2vec2_base_config
from audiossl.modelling.model import AudioModel


@pytest.fixture()
def tiny_wav2vec2_model():
    config = make_wav2vec2_base_config(
        runtime=RuntimeConfig(seed=123, deterministic=True),
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
    return AudioModel(config)


@pytest.fixture()
def tiny_wav2vec2_objective_config():
    return Wav2Vec2ObjectiveConfig(
        quantizer=Wav2Vec2QuantizerConfig(
            temperature_scheduler=ConstantScalarSchedulerConfig(value=2.0),
            num_groups=2,
            num_vars=8,
            vq_dim=16,
        ),
        time_masking=SpanMaskingConfig(mask_prob=0.2, mask_length=2),
        feature_masking=ChannelMaskingConfig(mask_prob=0.0, mask_length=2),
        final_dim=16,
        num_negatives=3,
        diversity_loss_weight=0.1,
        features_pen_weight=10.0,
    )


@pytest.fixture()
def waveform_batch():
    from audiossl.config.batch_dataclasses import AudioBatch

    generator = torch.Generator().manual_seed(7)
    return AudioBatch(
        x=torch.randn(2, 4000, generator=generator),
        mask=torch.ones(2, 4000, dtype=torch.bool),
    )


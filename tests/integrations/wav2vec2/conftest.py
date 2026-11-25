from __future__ import annotations

import types

import pytest


@pytest.fixture(scope="session")
def hf_wav2vec2():
    """Reference modules supplied by the locked transformers dependency."""
    from transformers.models.wav2vec2 import configuration_wav2vec2, modeling_wav2vec2

    return types.SimpleNamespace(config=configuration_wav2vec2, modeling=modeling_wav2vec2)


@pytest.fixture(scope="session")
def hf_w2v2_base_config(hf_wav2vec2):
    """Tiny HF config matching the base architecture for fast synthetic tests."""

    return hf_wav2vec2.config.Wav2Vec2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4,
        layer_norm_eps=1e-5,
        feat_extract_activation="gelu",
        hidden_act="gelu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=0.0,
    )

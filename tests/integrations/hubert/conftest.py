from __future__ import annotations

import types

import pytest


@pytest.fixture(scope="session")
def hf_hubert():
    """Reference modules supplied by the locked transformers dependency."""
    from transformers.models.hubert import configuration_hubert, modeling_hubert

    return types.SimpleNamespace(config=configuration_hubert, modeling=modeling_hubert)


@pytest.fixture(scope="session")
def hf_tiny_hubert_config(hf_hubert):
    return hf_hubert.config.HubertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        conv_dim=(8, 8, 8),
        conv_kernel=(5, 3, 3),
        conv_stride=(2, 2, 2),
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        num_conv_pos_embeddings=8,
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
        vocab_size=13,
    )

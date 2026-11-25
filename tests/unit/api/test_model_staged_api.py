from __future__ import annotations

import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.integrations.wav2vec2 import make_wav2vec2_base_config
from audiossl.modelling.model import AudioModel


def test_staged_audio_model_api_composes_to_pipeline() -> None:
    torch.manual_seed(0)
    model = AudioModel(
        make_wav2vec2_base_config(
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
    ).eval()
    batch = AudioBatch(
        x=torch.randn(2, 4000),
        mask=torch.ones(2, 4000, dtype=torch.bool),
    )

    with torch.inference_mode():
        pipeline = model.pipeline(batch)
        processed = model.process(batch)
        layout = model.token_layout(processed)
        tokenized = model.tokenize(processed)
        embedded = model.embed(tokenized, layout=layout)
        staged = model.encode_embedded(embedded)

    torch.testing.assert_close(staged.z, pipeline.z)
    torch.testing.assert_close(staged.mask, pipeline.mask)
    torch.testing.assert_close(staged.lengths, pipeline.lengths)

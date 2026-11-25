from __future__ import annotations

import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.processors.wav2vec2 import Wav2Vec2FeatureExtractorProcessor


def test_student_processor_output_length_matches_stub_teacher_labels():
    from audiossl.config import Wav2Vec2FeatureExtractorProcessorConfig

    processor = Wav2Vec2FeatureExtractorProcessor(
        config=Wav2Vec2FeatureExtractorProcessorConfig(
            conv_dim=(8, 8, 8, 8, 8, 8, 8),
            group_norm_num_groups=8,
        ),
    )
    batch = AudioBatch(
        x=torch.randn(2, 4000),
        mask=torch.ones(2, 4000, dtype=torch.bool),
    )
    features = processor(batch)
    expected = processor.output_lengths(torch.tensor([4000, 4000]))
    assert features.mask.shape == (2, int(expected[0].item()))

from __future__ import annotations

import pytest
import torch

from audiossl.modelling.masking.span import SpanMaskingStrategy
from audiossl.modelling.objectives.wav2vec2.loss import sample_negatives
from audiossl.rng import generator_device, new_manual_seed_generator, training_generator_device
from audiotrainer.augment.waveform import LoudnessAugmentation
from audiossl.config import SpanMaskingConfig

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_training_generator_device_policy():
    assert training_generator_device(torch.device("cpu")) == torch.device("cpu")
    assert training_generator_device(torch.device("cuda:1")) == torch.device("cuda:1")


@requires_cuda
def test_sample_negatives_accepts_cuda_generator_and_features():
    generator = new_manual_seed_generator(0, torch.device("cuda"))
    quantized = torch.randn(2, 6, 4, device="cuda", generator=generator)
    mask = torch.tensor(
        [[True, True, False, False, False, False], [False, True, True, False, False, False]],
        device="cuda",
    )
    sample_negatives(
        quantized,
        mask,
        num_negatives=2,
        cross_sample_negatives=2,
        generator=generator,
    )


@requires_cuda
def test_span_masking_uses_cuda_generator_with_gpu_valid_mask():
    strategy = SpanMaskingStrategy(config=SpanMaskingConfig(mask_prob=0.5, mask_length=2))
    generator = new_manual_seed_generator(4, torch.device("cuda"))
    valid = torch.ones(2, 12, dtype=torch.bool, device="cuda")
    mask = strategy.compute_mask(shape=(2, 12), valid_mask=valid, generator=generator)
    assert mask.device.type == "cuda"
    assert mask.any()


@requires_cuda
def test_loudness_augmentation_on_cuda_waveform():
    config = {"p": 1.0, "min_gain_db": -3.0, "max_gain_db": 3.0}
    generator = new_manual_seed_generator(9, torch.device("cuda"))
    aug = LoudnessAugmentation(config, generator=generator)
    x = torch.randn(8000, device="cuda")
    y = aug._process_single(x)
    assert y.device.type == "cuda"
    assert generator_device(generator).type == "cuda"
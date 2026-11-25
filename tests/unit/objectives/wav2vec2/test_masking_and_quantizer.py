import torch

from audiossl.config import (
    ConstantScalarSchedulerConfig,
    SpanMaskingConfig,
    Wav2Vec2QuantizerConfig,
)
from audiossl.modelling.masking.span import SpanMaskingStrategy
from audiossl.modelling.objectives.wav2vec2.quantizer import GumbelProductQuantizer
from audiossl.modelling.objectives.wav2vec2.scalar_scheduler import build_scalar_scheduler


def test_span_masking_is_deterministic_for_seed():
    strategy = SpanMaskingStrategy(config=SpanMaskingConfig(mask_prob=0.5, mask_length=2))
    valid = torch.ones(2, 12, dtype=torch.bool)
    m1 = strategy.compute_mask(
        shape=(2, 12),
        valid_mask=valid,
        generator=torch.Generator().manual_seed(4),
    )
    m2 = strategy.compute_mask(
        shape=(2, 12),
        valid_mask=valid,
        generator=torch.Generator().manual_seed(4),
    )
    torch.testing.assert_close(m1, m2)
    assert m1.dtype is torch.bool
    assert m1.any()


def test_quantizer_eval_is_deterministic_and_shapes():
    config = Wav2Vec2QuantizerConfig(
        temperature_scheduler=ConstantScalarSchedulerConfig(value=2.0),
        num_groups=2,
        num_vars=4,
        vq_dim=8,
    )
    quantizer = GumbelProductQuantizer(
        input_dim=6,
        config=config,
        temperature_scheduler=build_scalar_scheduler(config.temperature_scheduler),
    )
    quantizer.eval()
    x = torch.randn(2, 5, 6)
    with torch.inference_mode():
        out1 = quantizer(x)
        out2 = quantizer(x)
    torch.testing.assert_close(out1.quantized, out2.quantized)
    assert out1.quantized.shape == (2, 5, 8)
    assert out1.codebook_indices.shape == (2, 5, 2)


import torch

from audiossl.modelling.objectives.components import GradMultiply as _GradMultiply
from audiossl.modelling.objectives.wav2vec2.loss import (
    diversity_loss,
    feature_penalty,
    wav2vec2_contrastive_loss,
)


def test_diversity_loss_matches_formula():
    loss = diversity_loss(torch.tensor(3.0), torch.tensor(6.0), 10)
    torch.testing.assert_close(loss, torch.tensor(0.4))


def test_feature_penalty_is_l2_mean():
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    torch.testing.assert_close(feature_penalty(x), x.pow(2).mean())


def test_grad_multiply_scales_backward_only():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = _GradMultiply.apply(x, 0.25)
    torch.testing.assert_close(y, x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.25))


def test_contrastive_loss_shapes_and_zero_targets():
    generator = torch.Generator().manual_seed(0)
    encoder = torch.randn(2, 4, 8, generator=generator)
    quantized = torch.randn(2, 4, 8, generator=generator)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    loss, logits, targets = wav2vec2_contrastive_loss(
        encoder,
        quantized,
        mask,
        num_negatives=2,
        logit_temp=0.1,
        generator=torch.Generator().manual_seed(1),
    )
    assert loss.ndim == 0
    assert logits.shape == (4, 3)
    torch.testing.assert_close(targets, torch.zeros(4, dtype=torch.long))


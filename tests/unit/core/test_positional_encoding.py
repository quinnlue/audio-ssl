from __future__ import annotations

import unittest

import torch

from audiossl.config import (
    LearnablePositionalEncodingConfig,
    MultiHeadAttentionConfig,
    RopePositionalEncodingConfig,
    SinusoidalPositionalEncodingConfig,
)
from audiossl.modelling.core.mha import MultiHeadAttention
from audiossl.modelling.core.positional_encoding import (
    LearnablePositionalEncoding,
    RoPE,
    SinusoidalPositionalEncoding,
)


class PositionalEncodingTests(unittest.TestCase):
    def test_rope_uses_configured_base(self) -> None:
        config = RopePositionalEncodingConfig(max_len=8, base=2.0)
        rope = RoPE(config, d_model=4)
        expected = torch.cos(torch.tensor(1.0))
        self.assertAlmostEqual(float(rope.cos_cached[1, 0]), float(expected), places=6)

    def test_rope_raises_when_sequence_exceeds_max_len(self) -> None:
        rope = RoPE(RopePositionalEncodingConfig(max_len=2, base=1000.0), d_model=4)
        x = torch.randn(1, 3, 4)
        with self.assertRaisesRegex(ValueError, "exceeds configured max_len"):
            rope(x)

    def test_rope_requires_even_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "even embedding dimension"):
            RoPE(RopePositionalEncodingConfig(max_len=4, base=1000.0), d_model=3)

    def test_learnable_allocates_expected_shape_and_validates_last_dim(self) -> None:
        pe = LearnablePositionalEncoding(LearnablePositionalEncodingConfig(max_len=5), d_model=6)
        self.assertEqual(tuple(pe.pe.shape), (1, 5, 6))

        with self.assertRaisesRegex(ValueError, "last dimension mismatch"):
            pe(torch.randn(2, 4, 7))

    def test_sinusoidal_allocates_expected_shape_and_validates_guards(self) -> None:
        pe = SinusoidalPositionalEncoding(
            SinusoidalPositionalEncodingConfig(max_len=5, base=2.0),
            d_model=5,
        )
        self.assertEqual(tuple(pe.pe.shape), (1, 5, 5))

        with self.assertRaisesRegex(ValueError, "last dimension mismatch"):
            pe(torch.randn(2, 4, 6))

        with self.assertRaisesRegex(ValueError, "exceeds configured max_len"):
            pe(torch.randn(1, 6, 5))

    def test_sinusoidal_uses_configured_base(self) -> None:
        pe = SinusoidalPositionalEncoding(
            SinusoidalPositionalEncodingConfig(max_len=8, base=2.0),
            d_model=4,
        )
        self.assertAlmostEqual(float(pe.pe[0, 1, 0]), float(torch.sin(torch.tensor(1.0))), places=6)
        self.assertAlmostEqual(float(pe.pe[0, 1, 1]), float(torch.cos(torch.tensor(1.0))), places=6)

    def test_rope_mha_does_not_rotate_values(self) -> None:
        mha = MultiHeadAttention(
            d_model=4,
            config=MultiHeadAttentionConfig(num_heads=1, dropout_p=0.0),
            positional_encoding=RopePositionalEncodingConfig(max_len=8, base=2.0),
        )
        mha.eval()

        with torch.no_grad():
            mha.mha.in_proj_weight.zero_()
            mha.mha.in_proj_weight[8:12] = torch.eye(4)
            mha.mha.in_proj_bias.zero_()
            mha.mha.out_proj.weight.copy_(torch.eye(4))
            mha.mha.out_proj.bias.zero_()

        x = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, -1.0, 0.5, -3.0],
                    [100.0, -100.0, 50.0, -50.0],
                ]
            ]
        )
        attention_mask = torch.tensor([[True, True, False]])

        out = mha(x, attention_mask)
        expected = x[:, :2].mean(dim=1, keepdim=True).expand_as(x)

        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

if __name__ == "__main__":
    unittest.main()

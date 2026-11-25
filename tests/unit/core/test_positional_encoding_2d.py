from __future__ import annotations

import unittest

import torch

from audiossl.config import LearnableTimeFrequencyPositionalEncodingConfig
from audiossl.config.batch_dataclasses import TokenLayout
from audiossl.modelling.core.positional_encoding import (
    LearnableTimeFrequencyPositionalEncoding,
)


def _grid(time_patches: int, frequency_patches: int) -> TokenLayout:
    return TokenLayout(
        num_tokens=time_patches * frequency_patches,
        time_patches=time_patches,
        frequency_patches=frequency_patches,
        patch_shape=(1, 1),
        stride=(1, 1),
        flatten_order="time_major",
        output_dim=1,
        source_shape=(time_patches, frequency_patches),
    )


class LearnableTimeFrequencyPositionalEncodingTests(unittest.TestCase):
    def test_requires_layout(self) -> None:
        pe = LearnableTimeFrequencyPositionalEncoding(
            LearnableTimeFrequencyPositionalEncodingConfig(max_t=3, max_f=2),
            d_model=4,
        )

        with self.assertRaisesRegex(ValueError, "requires a TokenLayout"):
            pe(torch.randn(1, 6, 4))

    def test_forward_matches_hand_broadcast(self) -> None:
        torch.manual_seed(0)
        pe = LearnableTimeFrequencyPositionalEncoding(
            LearnableTimeFrequencyPositionalEncodingConfig(max_t=3, max_f=2),
            d_model=4,
        )
        x = torch.randn(2, 6, 4)

        with torch.no_grad():
            pe.time_pe.copy_(torch.arange(12, dtype=torch.float32).reshape(1, 3, 4))
            pe.freq_pe.copy_(100 + torch.arange(8, dtype=torch.float32).reshape(1, 2, 4))

        expected = (
            x
            + pe.time_pe.repeat_interleave(2, dim=1)
            + pe.freq_pe.repeat(1, 3, 1)
        )
        out = pe(x, layout=_grid(3, 2))

        self.assertEqual(tuple(out.shape), (2, 6, 4))
        torch.testing.assert_close(out, expected)

    def test_rejects_token_count_not_divisible_by_frequency_grid(self) -> None:
        pe = LearnableTimeFrequencyPositionalEncoding(
            LearnableTimeFrequencyPositionalEncodingConfig(max_t=3, max_f=2),
            d_model=4,
        )

        with self.assertRaisesRegex(ValueError, "divisible by frequency_patches"):
            pe(torch.randn(1, 5, 4), layout=_grid(3, 2))

    def test_rejects_time_grid_overflow(self) -> None:
        pe = LearnableTimeFrequencyPositionalEncoding(
            LearnableTimeFrequencyPositionalEncodingConfig(max_t=2, max_f=2),
            d_model=4,
        )

        with self.assertRaisesRegex(ValueError, "exceeds configured max_t"):
            pe(torch.randn(1, 6, 4), layout=_grid(3, 2))

    def test_uses_token_layout_frequency_patches(self) -> None:
        pe = LearnableTimeFrequencyPositionalEncoding(
            LearnableTimeFrequencyPositionalEncodingConfig(max_t=3, max_f=4),
            d_model=2,
        )
        layout = TokenLayout(
            num_tokens=6,
            time_patches=3,
            frequency_patches=2,
            patch_shape=(1, 1),
            stride=(1, 1),
            flatten_order="time_major",
            output_dim=1,
            source_shape=(3, 2),
        )

        out = pe(torch.zeros(1, 6, 2), layout=layout)

        self.assertEqual(tuple(out.shape), (1, 6, 2))

    def test_rejects_last_dim_mismatch(self) -> None:
        pe = LearnableTimeFrequencyPositionalEncoding(
            LearnableTimeFrequencyPositionalEncodingConfig(max_t=2, max_f=2),
            d_model=4,
        )

        with self.assertRaisesRegex(ValueError, "last dimension mismatch"):
            pe(torch.randn(1, 4, 5))


if __name__ == "__main__":
    unittest.main()

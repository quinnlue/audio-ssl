from __future__ import annotations

import unittest

import torch

from audiossl.config import ConvModuleConfig
from audiossl.modelling.core.conv_module import ConvModule


def _config(**overrides) -> ConvModuleConfig:
    values: dict = {
        "kernel_size": 3,
        "expansion_factor": 2,
        "activation": "silu",
        "dropout_p": 0.0,
    }
    values.update(overrides)
    return ConvModuleConfig(**values)


def _mask(batch: int, time_steps: int, *, valid_lengths: list[int] | None = None) -> torch.Tensor:
    if valid_lengths is None:
        return torch.ones(batch, time_steps, dtype=torch.bool)
    if len(valid_lengths) != batch:
        raise AssertionError("valid_lengths length must equal batch")
    mask = torch.zeros(batch, time_steps, dtype=torch.bool)
    for i, valid in enumerate(valid_lengths):
        mask[i, :valid] = True
    return mask


class ConvModuleTests(unittest.TestCase):
    def test_canonical_channel_geometry_after_glu(self) -> None:
        d_model = 8
        module = ConvModule(d_model=d_model, config=_config(expansion_factor=2))
        self.assertEqual(module.pw1.out_channels, d_model * 2)
        self.assertEqual(module.depthwise.in_channels, d_model)
        self.assertEqual(module.depthwise.out_channels, d_model)
        self.assertEqual(module.pw2.in_channels, d_model)

    def test_forward_preserves_shape(self) -> None:
        module = ConvModule(d_model=8, config=_config())
        x = torch.randn(2, 5, 8)
        mask = _mask(2, 5)

        output = module(x, mask)

        self.assertEqual(tuple(output.shape), (2, 5, 8))

    def test_padded_frames_are_zero_after_forward(self) -> None:
        module = ConvModule(d_model=8, config=_config())
        module.eval()
        x = torch.randn(2, 6, 8)
        mask = _mask(2, 6, valid_lengths=[4, 3])

        output = module(x, mask)

        for batch_idx, valid in enumerate([4, 3]):
            zeroed = output[batch_idx, valid:]
            self.assertTrue(
                torch.equal(zeroed, torch.zeros_like(zeroed)),
                f"padded frames not zeroed for batch {batch_idx}",
            )

    def test_padding_does_not_leak_into_valid_frames(self) -> None:
        torch.manual_seed(2026)
        module = ConvModule(d_model=8, config=_config(kernel_size=5))
        module.eval()

        valid_len = 4
        padded_len = 9
        sample = torch.randn(1, valid_len, 8)
        sample_mask = _mask(1, valid_len)

        padded = torch.randn(2, padded_len, 8)
        padded[0, :valid_len] = sample[0]
        padded[0, valid_len:] = 100.0 * torch.randn(padded_len - valid_len, 8)
        padded[1] = -100.0 * torch.randn(padded_len, 8)
        padded_mask = _mask(2, padded_len, valid_lengths=[valid_len, padded_len - 1])

        with torch.no_grad():
            expected = module(sample, sample_mask)[0]
            actual = module(padded, padded_mask)[0, :valid_len]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_invalid_kernel_size_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            ConvModule(d_model=8, config=_config(kernel_size=4))
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            ConvModule(d_model=8, config=_config(kernel_size=0))

    def test_invalid_expansion_factor_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "expansion_factor"):
            ConvModule(d_model=8, config=_config(expansion_factor=3))
        with self.assertRaisesRegex(ValueError, "expansion_factor"):
            ConvModule(d_model=8, config=_config(expansion_factor=0))

    def test_invalid_activation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation"):
            ConvModule(d_model=8, config=_config(activation="not_an_activation"))


if __name__ == "__main__":
    unittest.main()

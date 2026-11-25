from __future__ import annotations

import unittest

import torch

from audiossl.config import CGMLPConfig
from audiossl.modelling.core.cgmlp import CGMLP


def _config(**overrides) -> CGMLPConfig:
    values: dict = {
        "linear_units": 16,
        "conv_kernel_size": 3,
        "activation": "gelu",
        "gate_activation": "identity",
        "dropout_p": 0.0,
    }
    values.update(overrides)
    return CGMLPConfig(**values)


def _mask(batch: int, time_steps: int, *, valid_lengths: list[int] | None = None) -> torch.Tensor:
    if valid_lengths is None:
        return torch.ones(batch, time_steps, dtype=torch.bool)
    if len(valid_lengths) != batch:
        raise AssertionError("valid_lengths length must equal batch")
    mask = torch.zeros(batch, time_steps, dtype=torch.bool)
    for i, valid in enumerate(valid_lengths):
        mask[i, :valid] = True
    return mask


class CGMLPTests(unittest.TestCase):
    def test_forward_preserves_shape(self) -> None:
        module = CGMLP(d_model=8, config=_config())
        x = torch.randn(2, 5, 8)
        mask = _mask(2, 5)

        output = module(x, mask)

        self.assertEqual(tuple(output.shape), (2, 5, 8))

    def test_padded_frames_are_zero_after_forward(self) -> None:
        module = CGMLP(d_model=8, config=_config())
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
        module = CGMLP(d_model=8, config=_config(conv_kernel_size=5))
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

    def test_gating_is_multiplicative(self) -> None:
        """If the gate path (v) is forced to zero, the output should be zero (only bias term remains)."""
        torch.manual_seed(0)
        module = CGMLP(d_model=8, config=_config(linear_units=16, gate_activation="identity"))
        module.eval()
        with torch.no_grad():
            module.gate_norm.weight.zero_()
            module.gate_norm.bias.zero_()
            module.depthwise.weight.zero_()
            if module.depthwise.bias is not None:
                module.depthwise.bias.zero_()
            module.fc2.bias.zero_()

        x = torch.randn(2, 4, 8)
        mask = _mask(2, 4)
        output = module(x, mask)

        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_invalid_conv_kernel_size_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            CGMLP(d_model=8, config=_config(conv_kernel_size=4))
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            CGMLP(d_model=8, config=_config(conv_kernel_size=0))

    def test_invalid_linear_units_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "linear_units"):
            CGMLP(d_model=8, config=_config(linear_units=15))
        with self.assertRaisesRegex(ValueError, "linear_units"):
            CGMLP(d_model=8, config=_config(linear_units=0))

    def test_invalid_activation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation"):
            CGMLP(d_model=8, config=_config(activation="not_an_activation"))

    def test_invalid_gate_activation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate_activation"):
            CGMLP(d_model=8, config=_config(gate_activation="not_an_activation"))

    def test_gate_activation_aliases_resolve(self) -> None:
        for alias in ("identity", "sigmoid", "tanh", "gelu", "relu", "silu", "gelu_fast"):
            with self.subTest(gate_activation=alias):
                module = CGMLP(d_model=8, config=_config(gate_activation=alias))
                output = module(torch.randn(1, 3, 8), _mask(1, 3))
                self.assertEqual(tuple(output.shape), (1, 3, 8))


if __name__ == "__main__":
    unittest.main()

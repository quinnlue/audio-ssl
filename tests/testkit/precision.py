"""Numerical comparison settings for training parity."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Precision:
    rtol: float
    atol: float
    eps: float

    def close(self, actual: torch.Tensor, expected: torch.Tensor, *, label: str) -> None:
        for tensor in (actual, expected):
            assert torch.isfinite(tensor).all(), f"{label}: non-finite value"
        torch.testing.assert_close(
            actual, expected, rtol=self.rtol, atol=self.atol,
            msg=lambda message: f"{label}: {message}",
        )


FP32 = Precision(rtol=1e-4, atol=1e-6, eps=1e-5)
FP64 = Precision(rtol=1e-8, atol=1e-10, eps=1e-6)

"""Convolutional gating MLP branch for Branchformer-style encoders."""

import torch
import torch.nn as nn

from audiossl.config import CGMLPConfig

from ..registry import ACTIVATION_REGISTRY, build_activation


_EXTRA_GATE_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


def _resolve_gate_activation(name: str) -> nn.Module:
    """Resolve cgMLP gate activation names from local and shared registries."""
    if name in _EXTRA_GATE_ACTIVATIONS:
        return _EXTRA_GATE_ACTIVATIONS[name]()
    if name in ACTIVATION_REGISTRY:
        return build_activation(name)
    raise ValueError(
        f"Unsupported CGMLPConfig.gate_activation {name!r}; "
        f"expected one of {sorted(set(ACTIVATION_REGISTRY) | set(_EXTRA_GATE_ACTIVATIONS))}"
    )


class CGMLP(nn.Module):
    """Convolutional Gating MLP (CSGU) used as Branchformer's local branch.

    The block preserves `(B, N, D)` encoder states and uses a boolean
    attention mask `(B, N)`, where `True` means valid.

    `linear_units` follows ESPnet convention: it is the *pre-split* dim, so
    each gating half has `linear_units // 2` channels. This is enforced to be
    even by config validation.
    """

    def __init__(self, *, d_model: int, config: CGMLPConfig, layer_norm_eps: float = 1e-5):
        super().__init__()
        kernel_size = config.conv_kernel_size
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "CGMLP conv_kernel_size must be a positive odd integer, "
                f"got {kernel_size}"
            )
        if config.linear_units <= 0 or config.linear_units % 2 != 0:
            raise ValueError(
                "CGMLP linear_units must be a positive even integer (it is split "
                f"into equal gate/value halves), got {config.linear_units}"
            )
        if config.activation not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unsupported CGMLPConfig.activation {config.activation!r}; "
                f"expected one of {sorted(ACTIVATION_REGISTRY)}"
            )

        inner = config.linear_units // 2

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.fc1 = nn.Linear(d_model, config.linear_units)
        self.activation = build_activation(config.activation)

        self.gate_norm = nn.LayerNorm(inner, eps=layer_norm_eps)
        self.depthwise = nn.Conv1d(
            inner,
            inner,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner,
        )
        self.gate_activation = _resolve_gate_activation(config.gate_activation)

        self.dropout = nn.Dropout(config.dropout_p)
        self.fc2 = nn.Linear(inner, d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply the mask-aware local cgMLP branch to `(B, N, D)` states."""
        mask_btD = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        mask_bDt = attention_mask.unsqueeze(1).to(dtype=x.dtype)

        y = self.norm(x)
        y = self.fc1(y)
        y = self.activation(y)
        u, v = y.chunk(2, dim=-1)

        v = self.gate_norm(v).transpose(1, 2)
        v = v * mask_bDt
        v = self.depthwise(v)
        v = v.transpose(1, 2)
        v = self.gate_activation(v)

        y = u * v
        y = self.dropout(y)
        y = self.fc2(y)
        y = y * mask_btD
        y = self.dropout(y)
        return y

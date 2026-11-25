"""Feed-forward blocks used inside encoder layers."""

import torch
import torch.nn as nn

from audiossl.config import MLPConfig
from ..registry import build_activation
from .initialization import initialize_linear


class MLP(nn.Module):
    """Transformer-style positionwise feed-forward network.

    Parameters
    ----------
    d_model
        Input and output feature dimension `D`.
    config
        Hidden size, activation registry key, dropout, and initialization route.
    """

    def __init__(self, *, d_model: int, config: MLPConfig):
        super().__init__()
        self.fc1 = nn.Linear(d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, d_model)
        self.activation = build_activation(config.activation)
        self.dropout = nn.Dropout(config.dropout_p)
        initialize_linear(self.fc1, config.initialization)
        initialize_linear(self.fc2, config.initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to states with shape `(B, N, D)`."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

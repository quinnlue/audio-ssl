"""Projection helpers that bridge component feature dimensions."""

from dataclasses import replace

import torch.nn as nn

from audiossl.config import ProjectionConfig
from audiossl.config.batch_dataclasses import AudioBatch
from .initialization import initialize_linear


class Projector(nn.Module):
    """Apply a configured linear projection to an `AudioBatch`.

    The projector is used at component boundaries such as
    `tokenizer.output_dim -> d_model`. It preserves the batch mask unchanged.

    Parameters
    ----------
    config
        Projection dimensions, dropout probability, and initialization route.
    """

    def __init__(self, config: ProjectionConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.input_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout_p)
        initialize_linear(self.projection, config.initialization)

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Project `batch.x` from `(B, N, input_dim)` to `(B, N, output_dim)`."""
        return replace(batch, x=self.dropout(self.projection(batch.x)))

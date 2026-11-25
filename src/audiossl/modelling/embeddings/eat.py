"""EAT-specific embedding: projection, 2-D PE, CLS token, and entry layer norm."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from audiossl.config import EATEmbeddingConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout

from ..core.initialization import initialize_linear
from ..registry import get_embedding_positional_encoding_registry
from .base import BaseEmbedding


class EATEmbedding(BaseEmbedding):
    """Patch projection and EAT front-end ordering through entry layer norm.

    Stages: ``project`` is the patch projection; ``position`` adds the 2-D
    sin-cos PE (requires a ``TokenLayout``); ``finalize`` prepends the CLS
    token, applies the entry layer norm and dropout, and marks one prefix
    token. ``finalize`` works on any token subset, which is how UFO encodes
    only the kept patches.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        config: EATEmbeddingConfig,
    ) -> None:
        super().__init__()
        if not isinstance(config, EATEmbeddingConfig):
            raise TypeError(
                "EATEmbedding.__init__ expected EATEmbeddingConfig, "
                f"got {type(config).__name__}"
            )
        self.config = config
        self.input_dim = input_dim
        self.d_model = d_model
        self.projection = nn.Linear(input_dim, d_model, bias=True)
        initialize_linear(self.projection, config.initialization)
        pe_cls = get_embedding_positional_encoding_registry()[config.positional_encoding.type]
        self.positional_encoding = pe_cls(config.positional_encoding, d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=config.cls_init_std)
        self.entry_layer_norm = nn.LayerNorm(d_model, eps=config.entry_layer_norm_eps)
        self.pos_drop = nn.Dropout(config.dropout_p)

    @property
    def num_prefix_tokens(self) -> int:
        return 1

    def project(self, batch: AudioBatch) -> AudioBatch:
        self._validate_batch(batch)
        return replace(batch, x=self.projection(batch.x))

    def position(self, batch: AudioBatch, *, layout: TokenLayout | None = None) -> AudioBatch:
        if layout is None:
            raise ValueError("EATEmbedding.position requires TokenLayout")
        return super().position(batch, layout=layout)

    def finalize(self, batch: AudioBatch) -> AudioBatch:
        self._validate_projected(batch, stage="finalize")
        if batch.num_prefix_tokens:
            raise ValueError(
                "EATEmbedding.finalize received a batch that already has prefix tokens"
            )
        batch_size = batch.x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(dtype=batch.x.dtype)
        x = torch.cat([cls_tokens, batch.x], dim=1)
        mask = torch.cat([batch.mask.new_ones((batch_size, 1)), batch.mask], dim=1)
        positions = None
        if batch.positions is not None:
            positions = torch.cat(
                [batch.positions.new_zeros((batch_size, 1)), batch.positions + 1], dim=1
            )
        x = self.pos_drop(self.entry_layer_norm(x))
        return AudioBatch(x=x, mask=mask, positions=positions, num_prefix_tokens=1)


__all__ = ["EATEmbedding"]

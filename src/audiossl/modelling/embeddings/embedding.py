"""Embedding wrapper for audio model configs."""

from dataclasses import replace

import torch
import torch.nn as nn

from audiossl.config import AudioModelConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout

from ..registry import get_embedding_registry
from .base import BaseEmbedding


class Embedding(nn.Module):
    """Dispatch embedding configs to concrete embedding modules.

    The wrapper constructs the registered embedding with
    ``input_dim=config.tokenizer.output_dim`` and ``d_model=config.d_model`` and
    re-exports the :class:`BaseEmbedding` stage API (``prepare``, ``project``,
    ``position``, ``finalize``). ``prepare`` additionally casts token features
    to the embedding parameter dtype.
    """

    def __init__(self, config: AudioModelConfig) -> None:
        super().__init__()
        self.config = config
        embedding_cls = get_embedding_registry()[config.embedding.type]
        self.embedding = embedding_cls(
            input_dim=config.tokenizer.output_dim,
            d_model=config.d_model,
            config=config.embedding,
        )
        if not isinstance(self.embedding, BaseEmbedding):
            raise TypeError(
                f"Registered embedding {config.embedding.type!r} must subclass BaseEmbedding, "
                f"got {type(self.embedding).__name__}"
            )

    @property
    def num_prefix_tokens(self) -> int:
        return self.embedding.num_prefix_tokens

    @property
    def subsampling_factor(self) -> int:
        return self.embedding.subsampling_factor

    def forward(
        self,
        batch: AudioBatch,
        *,
        layout: TokenLayout | None = None,
    ) -> AudioBatch:
        """Embed tokenizer output ``(B, N, tokenizer.output_dim)`` for the encoder."""
        batch = self.prepare(batch)
        batch = self.embedding.project(batch)
        batch = self.embedding.position(batch, layout=layout)
        return self.embedding.finalize(batch)

    def prepare(self, batch: AudioBatch) -> AudioBatch:
        if not isinstance(batch, AudioBatch):
            raise TypeError(f"Embedding.prepare expected AudioBatch, got {type(batch).__name__}")
        dtype = self._parameter_dtype()
        if dtype is not None and batch.x.dtype != dtype:
            batch = replace(batch, x=batch.x.to(dtype=dtype))
        return self.embedding.prepare(batch)

    def project(self, batch: AudioBatch) -> AudioBatch:
        return self.embedding.project(batch)

    def position(self, batch: AudioBatch, *, layout: TokenLayout | None = None) -> AudioBatch:
        return self.embedding.position(batch, layout=layout)

    def finalize(self, batch: AudioBatch) -> AudioBatch:
        return self.embedding.finalize(batch)

    def output_mask(self, token_mask: torch.Tensor) -> torch.Tensor:
        return self.embedding.output_mask(token_mask)

    def _parameter_dtype(self) -> torch.dtype | None:
        for param in self.embedding.parameters():
            if param.is_floating_point():
                return param.dtype
        return None

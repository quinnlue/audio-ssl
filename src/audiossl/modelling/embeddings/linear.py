"""Linear token embedding implementation."""

from dataclasses import replace

import torch.nn as nn

from audiossl.config import LinearEmbeddingConfig, ProjectionConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout

from ..core.projector import Projector
from ..registry import get_embedding_positional_encoding_registry
from .base import BaseEmbedding


class LinearEmbedding(BaseEmbedding):
    """Linear tokenizer-output to ``d_model`` embedding.

    Stages: ``prepare`` applies optional input layer norm; ``project`` is the
    linear map; ``position`` applies embedding PE and optional post-PE layer
    norm; ``finalize`` applies dropout and optional strided subsampling.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        config: LinearEmbeddingConfig,
    ) -> None:
        super().__init__()
        if not isinstance(config, LinearEmbeddingConfig):
            raise TypeError(
                "LinearEmbedding.__init__ expected LinearEmbeddingConfig, "
                f"got {type(config).__name__}"
            )
        self.config = config
        self.input_dim = input_dim
        self.d_model = d_model
        self._factor = self._subsampling_factor(config.subsampling)
        self.projector = Projector(
            ProjectionConfig(
                input_dim=input_dim,
                output_dim=d_model,
                dropout_p=config.dropout_p,
                initialization=config.initialization,
            )
        )
        self.pre_layer_norm = (
            nn.LayerNorm(input_dim, eps=config.input_layer_norm_eps)
            if config.input_layer_norm
            else None
        )
        self.post_layer_norm = (
            nn.LayerNorm(d_model, eps=config.post_pos_layer_norm_eps)
            if config.post_pos_layer_norm
            else None
        )
        pe_cls = get_embedding_positional_encoding_registry()[config.positional_encoding.type]
        self.positional_encoding = pe_cls(config.positional_encoding, d_model=d_model)

    @property
    def projection(self) -> nn.Linear:
        """Underlying linear projection from tokenizer ``output_dim`` to ``d_model``."""
        return self.projector.projection

    @property
    def subsampling_factor(self) -> int:
        return self._factor

    def prepare(self, batch: AudioBatch) -> AudioBatch:
        self._validate_batch(batch)
        if self.pre_layer_norm is None:
            return batch
        return replace(batch, x=self.pre_layer_norm(batch.x))

    def project(self, batch: AudioBatch) -> AudioBatch:
        self._validate_batch(batch)
        return replace(batch, x=self.projection(batch.x))

    def position(self, batch: AudioBatch, *, layout: TokenLayout | None = None) -> AudioBatch:
        batch = super().position(batch, layout=layout)
        if self.post_layer_norm is None:
            return batch
        return replace(batch, x=self.post_layer_norm(batch.x))

    def finalize(self, batch: AudioBatch) -> AudioBatch:
        batch = replace(batch, x=self.projector.dropout(batch.x))
        return self.subsample_batch(batch, self.config.subsampling)

"""Convolutional token embedding implementation."""

from dataclasses import replace

import torch.nn as nn

from audiossl.config import ConvEmbeddingConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout

from ..core.initialization import initialize_conv1d
from ..registry import get_embedding_positional_encoding_registry
from .base import BaseEmbedding


class ConvEmbedding(BaseEmbedding):
    """Conv1d-across-tokens embedding inspired by Dasheng's AudioPatchEmbed.

    Dasheng (XiaoMi/dasheng ``train/models.py``) projects mel frames into
    ``embed_dim`` with a single ``Conv2d(1, embed_dim, kernel=patch, stride=patch)``
    that jointly patchifies and projects. In this codebase the patchify step
    is owned by the tokenizer, so this module runs a Conv1d **across the
    token axis** of the already-patchified sequence with stride=1 and
    same-padding, preserving ``N_tokens``.

    ``kernel_size=1`` is equivalent to ``LinearEmbedding``; ``kernel_size>1``
    lets each output token mix a local window of neighboring input tokens.

    Stages: ``project`` is the masked conv; ``position`` applies embedding PE
    and the optional layer norm; ``finalize`` applies dropout and optional
    strided subsampling.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        config: ConvEmbeddingConfig,
    ) -> None:
        super().__init__()
        if not isinstance(config, ConvEmbeddingConfig):
            raise TypeError(
                "ConvEmbedding.__init__ expected ConvEmbeddingConfig, "
                f"got {type(config).__name__}"
            )
        if config.kernel_size % 2 == 0:
            raise ValueError(
                "ConvEmbedding requires odd kernel_size for same-padding, "
                f"got {config.kernel_size!r}"
            )
        self.config = config
        self.input_dim = input_dim
        self.d_model = d_model
        self._factor = self._subsampling_factor(config.subsampling)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=config.kernel_size,
            stride=1,
            padding=(config.kernel_size - 1) // 2,
        )
        self.dropout = nn.Dropout(config.dropout_p)
        self.norm = nn.LayerNorm(d_model) if config.layer_norm else None
        initialize_conv1d(self.conv, config.initialization)
        pe_cls = get_embedding_positional_encoding_registry()[config.positional_encoding.type]
        self.positional_encoding = pe_cls(config.positional_encoding, d_model=d_model)

    @property
    def subsampling_factor(self) -> int:
        return self._factor

    def project(self, batch: AudioBatch) -> AudioBatch:
        self._validate_batch(batch)
        # Zero invalid positions so kernel windows wider than one token cannot
        # leak padded content into valid outputs.
        x = batch.x.masked_fill(~batch.mask.unsqueeze(-1), 0.0)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return replace(batch, x=x)

    def position(self, batch: AudioBatch, *, layout: TokenLayout | None = None) -> AudioBatch:
        batch = super().position(batch, layout=layout)
        if self.norm is None:
            return batch
        return replace(batch, x=self.norm(batch.x))

    def finalize(self, batch: AudioBatch) -> AudioBatch:
        batch = replace(batch, x=self.dropout(batch.x))
        return self.subsample_batch(batch, self.config.subsampling)

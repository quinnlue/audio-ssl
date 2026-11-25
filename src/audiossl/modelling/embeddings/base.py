"""Shared stage contract and helpers for embedding modules."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from audiossl.config import EmbeddingSubsamplingConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout


class BaseEmbedding(nn.Module):
    """Base class for tokenizer-to-encoder embedding implementations.

    Embeddings bridge tokenizer ``output_dim`` to ``d_model`` through four
    public stages, so objectives can intervene between them without reaching
    into private submodules:

    ``prepare``
        Validate token input and apply optional input normalization.
        Shape-preserving on ``(B, N, input_dim)``.
    ``project``
        Map ``(B, N, input_dim)`` to ``(B, N, d_model)`` per token.
    ``position``
        Apply embedding-side positional encoding (and any post-PE norm).
        Token count and order must be the full tokenizer grid, so masking
        objectives substitute mask tokens *before* this stage and drop tokens
        *after* it.
    ``finalize``
        Everything that may change the token axis or must run on the final
        sequence: dropout, subsampling, prefix tokens. Safe to call on a
        subset of tokens (for example after MAE-style token dropping); it
        updates ``positions`` and ``num_prefix_tokens`` metadata.

    ``forward`` is exactly ``finalize(position(project(prepare(batch))))``.
    Embeddings do not own encoder blocks, pooling, or objective masks.
    """

    input_dim: int
    d_model: int

    @property
    def num_prefix_tokens(self) -> int:
        """Tokens prepended by :meth:`finalize` (for example a CLS token)."""
        return 0

    @property
    def subsampling_factor(self) -> int:
        """Stride applied to the token axis by :meth:`finalize`."""
        return 1

    def prepare(self, batch: AudioBatch) -> AudioBatch:
        self._validate_batch(batch)
        return batch

    def project(self, batch: AudioBatch) -> AudioBatch:
        raise NotImplementedError(f"{type(self).__name__} must implement project")

    def position(self, batch: AudioBatch, *, layout: TokenLayout | None = None) -> AudioBatch:
        self._validate_projected(batch, stage="position")
        x = self.positional_encoding(batch.x, layout, mask=batch.mask)
        return replace(batch, x=x)

    def finalize(self, batch: AudioBatch) -> AudioBatch:
        raise NotImplementedError(f"{type(self).__name__} must implement finalize")

    def forward(
        self,
        batch: AudioBatch,
        *,
        layout: TokenLayout | None = None,
    ) -> AudioBatch:
        """Embed tokenizer output ``(B, N, input_dim)`` for the encoder."""
        batch = self.prepare(batch)
        batch = self.project(batch)
        batch = self.position(batch, layout=layout)
        return self.finalize(batch)

    def output_mask(self, token_mask: torch.Tensor) -> torch.Tensor:
        """Map a tokenizer-axis boolean mask ``(B, N)`` onto the output token axis.

        Subsampling keeps every ``subsampling_factor``-th entry; prefix tokens
        are reported as ``False`` so objective masks never select them.
        """
        mask = token_mask[:, :: self.subsampling_factor]
        if self.num_prefix_tokens:
            prefix = mask.new_zeros((mask.shape[0], self.num_prefix_tokens))
            mask = torch.cat([prefix, mask], dim=1)
        return mask

    def _validate_batch(self, batch: AudioBatch) -> None:
        class_name = self.__class__.__name__
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                f"{class_name}.forward expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                f"{class_name}.forward expected batch.x to have shape "
                f"(B, T_tokens, input_dim), got {tuple(batch.x.shape)}"
            )
        if batch.x.shape[-1] != self.input_dim:
            raise ValueError(
                f"{class_name}.forward batch.x last dimension must match input_dim, "
                f"got {batch.x.shape[-1]} and {self.input_dim}"
            )
        self._validate_mask(batch)

    def _validate_projected(self, batch: AudioBatch, *, stage: str) -> None:
        class_name = self.__class__.__name__
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                f"{class_name}.{stage} expected AudioBatch, got {type(batch).__name__}"
            )
        if batch.x.ndim != 3 or batch.x.shape[-1] != self.d_model:
            raise ValueError(
                f"{class_name}.{stage} expected batch.x shape (B, N, {self.d_model}), "
                f"got {tuple(batch.x.shape)}"
            )
        self._validate_mask(batch)

    def _validate_mask(self, batch: AudioBatch) -> None:
        class_name = self.__class__.__name__
        if batch.mask.ndim != 2:
            raise ValueError(
                f"{class_name}.forward expected batch.mask to have shape "
                f"(B, T_tokens), got {tuple(batch.mask.shape)}"
            )
        if batch.mask.shape != batch.x.shape[:2]:
            raise ValueError(
                f"{class_name}.forward batch.mask shape must match batch.x first two dimensions, "
                f"got {tuple(batch.mask.shape)} and {tuple(batch.x.shape[:2])}"
            )
        if batch.mask.dtype != torch.bool:
            raise ValueError(
                f"{class_name}.forward expected batch.mask dtype torch.bool, "
                f"got {batch.mask.dtype}"
            )

    @classmethod
    def _subsampling_factor(cls, config: EmbeddingSubsamplingConfig | None) -> int:
        if config is None:
            return 1
        if not isinstance(config, EmbeddingSubsamplingConfig):
            raise TypeError(
                "Embedding subsampling config must be EmbeddingSubsamplingConfig, "
                f"got {type(config).__name__}"
            )
        if config.type != "strided_select":
            raise ValueError(f"Unsupported EmbeddingSubsamplingConfig.type {config.type!r}")
        if isinstance(config.factor, bool) or not isinstance(config.factor, int):
            raise TypeError(
                "EmbeddingSubsamplingConfig.factor must be int, "
                f"got {type(config.factor).__name__}"
            )
        if config.factor < 1:
            raise ValueError(
                "EmbeddingSubsamplingConfig.factor must be >= 1, "
                f"got {config.factor!r}"
            )
        return config.factor

    @classmethod
    def subsample_batch(
        cls,
        batch: AudioBatch,
        config: EmbeddingSubsamplingConfig | None,
    ) -> AudioBatch:
        """Select every ``factor``-th token (and mask/positions entry) from a batch."""
        factor = cls._subsampling_factor(config)
        if factor == 1:
            return batch
        positions = None if batch.positions is None else batch.positions[:, ::factor]
        return replace(
            batch,
            x=batch.x[:, ::factor, :],
            mask=batch.mask[:, ::factor],
            positions=positions,
        )

    @classmethod
    def subsample_mask(
        cls,
        mask: torch.Tensor,
        config: EmbeddingSubsamplingConfig | None,
    ) -> torch.Tensor:
        """Select every ``factor``-th entry of a ``(B, N)`` boolean mask."""
        factor = cls._subsampling_factor(config)
        if factor == 1:
            return mask
        return mask[:, ::factor]

    @classmethod
    def scatter_subsampled_mask(
        cls,
        mask: torch.Tensor,
        *,
        original_length: int,
        config: EmbeddingSubsamplingConfig | None,
    ) -> torch.Tensor:
        """Scatter a subsampled ``(B, N_sub)`` mask back to ``(B, original_length)``.

        Positions not selected by the strided subsampling are ``False``.
        """
        factor = cls._subsampling_factor(config)
        if factor == 1:
            return mask

        scattered = mask.new_zeros((mask.shape[0], original_length))
        target = scattered[:, ::factor]
        if target.shape != mask.shape:
            raise ValueError(
                "Subsampled mask shape must match strided positions in original mask, "
                f"got {tuple(mask.shape)} and {tuple(target.shape)}"
            )
        target.copy_(mask)
        return scattered


__all__ = ["BaseEmbedding"]

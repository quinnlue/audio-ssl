"""Tokenizer dispatcher for audio model configs."""

import torch.nn as nn

from audiossl.config import AudioModelConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout
from ..registry import get_tokenizer_registry

class Tokenizer(nn.Module):
    """Dispatch feature batches to the configured tokenizer implementation.

    The wrapper is the modelling pipeline boundary that converts processor
    features to tokenizer outputs. Concrete tokenizers own feature-to-token
    reshaping, output lengths, `TokenLayout`, patch target order, and mask
    collapse. Projection to `d_model` and positional encoding remain embedding
    responsibilities.

    Parameters
    ----------
    config
        Audio model config whose `tokenizer.type` selects a registered
        tokenizer implementation.
    """

    def __init__(self, config: AudioModelConfig) -> None:
        """Construct the registered tokenizer from `config.tokenizer`."""
        super().__init__()
        self.config = config
        tokenizer_cls = get_tokenizer_registry()[config.tokenizer.type]
        self.tokenizer = tokenizer_cls(
            config=config.tokenizer,
            input_dim=int(config.processor.output_dim),
        )

    def output_lengths(self, input_lengths):
        return self.tokenizer.output_lengths(input_lengths)

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Convert processor features to tokens.

        Parameters
        ----------
        batch
            Feature batch with `x` shape `(B, T, F)` and boolean `mask` shape
            `(B, T)`, where `True` means valid.

        Returns
        -------
        AudioBatch
            Token batch with `x` shape `(B, N, output_dim)` and boolean `mask`
            shape `(B, N)`, where `True` means valid.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "Tokenizer.forward expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        return self.tokenizer(batch)

    def layout(self, batch: AudioBatch) -> TokenLayout:
        """Return tokenizer layout metadata for a feature batch.

        Parameters
        ----------
        batch
            Feature batch with `x` shape `(B, T, F)`.

        Returns
        -------
        TokenLayout
            Layout metadata from the configured tokenizer for the batch's
            feature time and feature dimensions.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "Tokenizer.layout expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                "Tokenizer.layout expected batch.x with rank 3, "
                f"got rank {batch.x.ndim}"
            )
        return self.tokenizer.layout(T_feat=batch.x.shape[1], F_in=batch.x.shape[2])

    def reorder_patch_targets(
        self,
        target,
        *,
        layout: TokenLayout,
        order: str,
    ):
        """Reorder patch targets according to the configured tokenizer.

        Parameters
        ----------
        target
            Patch targets, usually with shape `(B, N, output_dim)`.
        layout
            Token layout associated with `target`.
        order
            Target order requested by an objective. `"time_frequency"` is the
            identity order; patch-aware tokenizers may also support
            `"frequency_time"`.

        Returns
        -------
        Any
            The original or reordered target object.
        """
        if not hasattr(self.tokenizer, "reorder_patch_targets"):
            if order != "time_frequency":
                raise ValueError(
                    f"Tokenizer {type(self.tokenizer).__name__} does not support "
                    f"patch target order {order!r}"
                )
            return target
        return self.tokenizer.reorder_patch_targets(target, layout=layout, order=order)

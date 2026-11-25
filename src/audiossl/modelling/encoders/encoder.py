"""Encoder wrapper for d_model token sequences."""

import torch
import torch.nn as nn

from audiossl.config import (
    EncoderConfig,
)
from ..registry import get_encoder_block_registry
from audiossl.config.batch_dataclasses import EncoderOutput, AudioBatch



class Encoder(nn.Module):
    """Stack encoder blocks and return sequence-level encoder states.

    The encoder consumes embedded tokens with shape `(B, N, d_model)` and a
    boolean validity mask with shape `(B, N)`, where `True` means valid. It owns
    the block stack, optional per-block positional encoding through the block
    configuration, and optional final normalization. Entry positional encoding,
    pooling, losses, and class-token extraction are owned by other modelling
    layers.

    Parameters
    ----------
    d_model
        Token embedding dimension consumed and produced by every block.
    config
        Encoder variant configuration used to select and construct the block
        stack.
    """

    def __init__(self, *, d_model: int, config: EncoderConfig) -> None:
        """Create an encoder block stack for `d_model` tokens."""
        super().__init__()
        self.d_model = d_model
        self.config = config

        if d_model % config.mha.num_heads != 0:
            raise ValueError(
                "Encoder d_model must be divisible by MultiHeadAttentionConfig.num_heads, "
                f"got {d_model} and {config.mha.num_heads}"
            )

        block_type = config.type
        num_layers = config.num_layers

        layer_cls = get_encoder_block_registry()[block_type]
        self.blocks = nn.ModuleList(
            [layer_cls(d_model=d_model, config=config) for _ in range(num_layers)]
        )
        self.final_norm = (
            nn.LayerNorm(d_model, eps=config.layer_norm_eps)
            if config.final_norm
            else None
        )

    @property
    def layers(self) -> nn.ModuleList:
        """Block stack alias for compatibility with layer-oriented callers."""
        return self.blocks

    def forward(
        self,
        batch: AudioBatch,
        *,
        return_hidden_states: bool = False,
    ) -> EncoderOutput:
        """Encode embedded tokens and preserve their validity mask.

        Parameters
        ----------
        batch
            Embedded token batch. `batch.x` has shape `(B, N, d_model)`.
            `batch.mask` has shape `(B, N)`, dtype `torch.bool`, where `True`
            means valid.
        return_hidden_states
            When ``True``, include per-block outputs in ``EncoderOutput.hidden_states``.

        Returns
        -------
        EncoderOutput
            Encoded token states `z` with shape `(B, N, d_model)`, the input
            mask, `lengths=batch.mask.sum(dim=1)`, and `cls=None`. Pooling and
            class-token extraction are handled by `AudioModel.get_cls`.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "Encoder.forward expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                "Encoder.forward expected batch.x to have shape (B, T_tokens, d_model), "
                f"got {tuple(batch.x.shape)}"
            )
        if batch.x.shape[-1] != self.d_model:
            raise ValueError(
                "Encoder.forward batch.x last dimension must match d_model, "
                f"got {batch.x.shape[-1]} and {self.d_model}"
            )
        if batch.mask.ndim != 2:
            raise ValueError(
                "Encoder.forward expected batch.mask to have shape (B, T_tokens), "
                f"got {tuple(batch.mask.shape)}"
            )
        if batch.mask.shape != batch.x.shape[:2]:
            raise ValueError(
                "Encoder.forward batch.mask shape must match batch.x first two dimensions, "
                f"got {tuple(batch.mask.shape)} and {tuple(batch.x.shape[:2])}"
            )
        if batch.mask.dtype != torch.bool:
            raise ValueError(
                "Encoder.forward expected batch.mask dtype torch.bool, "
                f"got {batch.mask.dtype}"
            )
        lengths = batch.mask.sum(dim=1)
        if bool((lengths == 0).any()):
            empty = torch.nonzero(lengths == 0).flatten().tolist()
            raise ValueError(
                "Encoder.forward received rows with no valid tokens (attention would be "
                f"undefined): rows {empty}"
            )

        x = batch.x
        block_kwargs = {} if batch.positions is None else {"positions": batch.positions}
        hidden_states: list[torch.Tensor] | None = [] if return_hidden_states else None
        for block in self.blocks:
            x = block(x, attention_mask=batch.mask, **block_kwargs)
            if hidden_states is not None:
                hidden_states.append(x)
        if self.final_norm is not None:
            x = self.final_norm(x)

        return EncoderOutput(
            z=x,
            mask=batch.mask,
            lengths=lengths,
            cls=None,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            num_prefix_tokens=batch.num_prefix_tokens,
        )

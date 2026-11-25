"""Identity tokenizer for already-token-shaped feature sequences."""

import torch
import torch.nn as nn

from audiossl.config import NoOpTokenizerConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout


class NoOpTokenizer(nn.Module):
    """Pass processor features through as token sequences unchanged.

    This tokenizer preserves `x` and `mask` identities, exposes a
    `TokenLayout`, and leaves projection to `d_model` and positional encoding
    to the embedding layer.

    Parameters
    ----------
    config
        No-op tokenizer configuration. `config.output_dim` is expected to
        match the upstream processor feature dimension.
    """

    def __init__(self, *, config: NoOpTokenizerConfig, input_dim: int | None = None) -> None:
        """Initialize the identity tokenizer from a frozen config."""
        super().__init__()
        if not isinstance(config, NoOpTokenizerConfig):
            raise TypeError(
                "NoOpTokenizer.__init__ expected NoOpTokenizerConfig, "
                f"got {type(config).__name__}"
            )
        if input_dim is not None and input_dim != config.output_dim:
            raise ValueError(
                "NoOpTokenizer output_dim must equal the processor output_dim, "
                f"got {config.output_dim} and {input_dim}"
            )
        self.config = config

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Validate and return a feature batch unchanged.

        Parameters
        ----------
        batch
            Feature batch with `x` shape `(B, T, F)` and boolean `mask` shape
            `(B, T)`, where `True` means valid.

        Returns
        -------
        AudioBatch
            The original batch, preserving `x`, `mask`, and object identity.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "NoOpTokenizer.forward expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                "NoOpTokenizer.forward expected batch.x with rank 3, "
                f"got rank {batch.x.ndim}"
            )
        if batch.mask.dtype != torch.bool:
            raise ValueError(
                "NoOpTokenizer.forward expected batch.mask dtype torch.bool, "
                f"got {batch.mask.dtype}"
            )
        if batch.mask.shape != batch.x.shape[:2]:
            raise ValueError(
                "NoOpTokenizer.forward expected batch.mask shape to match "
                f"batch.x.shape[:2], got {tuple(batch.mask.shape)} and "
                f"{tuple(batch.x.shape[:2])}"
            )
        return batch

    def layout(self, *, T_feat: int, F_in: int) -> TokenLayout:
        """Describe the identity token layout for processor features.

        Parameters
        ----------
        T_feat
            Number of input feature frames and output tokens.
        F_in
            Number of input feature bins. Must equal `config.output_dim`.

        Returns
        -------
        TokenLayout
            Layout metadata with `num_tokens == T_feat`, one frequency patch,
            patch shape `(1, F_in)`, and time-major flatten order.
        """
        if isinstance(T_feat, bool) or not isinstance(T_feat, int):
            raise TypeError(f"NoOpTokenizer.layout T_feat must be int, got {type(T_feat).__name__}")
        if isinstance(F_in, bool) or not isinstance(F_in, int):
            raise TypeError(f"NoOpTokenizer.layout F_in must be int, got {type(F_in).__name__}")
        if T_feat < 0:
            raise ValueError(f"NoOpTokenizer.layout T_feat must be >= 0, got {T_feat!r}")
        if F_in != self.config.output_dim:
            raise ValueError(
                "NoOpTokenizer.layout F_in must match config.output_dim, "
                f"got {F_in} and {self.config.output_dim}"
            )
        return TokenLayout(
            num_tokens=T_feat,
            time_patches=T_feat,
            frequency_patches=1,
            patch_shape=(1, F_in),
            stride=(1, F_in),
            flatten_order="time_major",
            output_dim=F_in,
            source_shape=(T_feat, F_in),
        )

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Return valid feature-frame counts as valid token counts.

        Parameters
        ----------
        input_lengths
            Rank-1 tensor `(B,)` containing per-row valid frame counts.

        Returns
        -------
        torch.Tensor
            The original `input_lengths` tensor unchanged.
        """
        if not isinstance(input_lengths, torch.Tensor):
            raise TypeError(
                "NoOpTokenizer.output_lengths expected torch.Tensor, "
                f"got {type(input_lengths).__name__}"
            )
        if input_lengths.ndim != 1:
            raise ValueError(
                "NoOpTokenizer.output_lengths expected rank-1 input_lengths, "
                f"got rank {input_lengths.ndim}"
            )
        return input_lengths

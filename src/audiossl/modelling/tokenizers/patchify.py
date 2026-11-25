"""Patch-based feature tokenizer and layout helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import PatchifyTokenizerConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout


class PatchifyTokenizer(nn.Module):
    """Convert feature frames to time-major patch tokens.

    Tokenizers consume processor features and do not project to `d_model` or
    apply positional encoding. This variant owns the patch target order, mask
    collapse, output-length mapping, and `TokenLayout` metadata for patch-aware
    downstream components.

    Parameters
    ----------
    config
        Patch tokenizer configuration. `patch_shape` is `(patch_t, patch_f)`;
        `stride` defaults to `patch_shape` through `effective_stride`; and
        `padding_strategy` controls incomplete trailing time windows.

    Notes
    -----
    Input batches carry features with shape `(B, T, F)` and boolean masks with
    shape `(B, T)`, where `True` means valid. Output tokens have shape
    `(B, N, patch_t * patch_f)` and masks have shape `(B, N)`.
    Tokens are flattened in time-major order: all frequency patches for time
    patch 0, then all frequency patches for time patch 1, and so on. A patch
    token is valid iff every input frame contributing to its time patch is
    valid; padded frames count as invalid.
    """

    def __init__(self, *, config: PatchifyTokenizerConfig, input_dim: int | None = None) -> None:
        """Initialize the tokenizer from a frozen patchify config.

        ``input_dim`` is the processor feature width. When given, the frequency
        grid is fixed at construction and every input is checked against it, so
        ``output_lengths`` is a pure function of the config.
        """
        super().__init__()
        if not isinstance(config, PatchifyTokenizerConfig):
            raise TypeError(
                "PatchifyTokenizer.__init__ expected PatchifyTokenizerConfig, "
                f"got {type(config).__name__}"
            )
        self.config = config
        self.input_dim = input_dim
        self._num_feat_patches: int | None = None
        if input_dim is not None:
            patch_f = config.patch_shape[1]
            stride_f = config.effective_stride[1]
            self._num_feat_patches = self._feature_grid(input_dim, patch_f, stride_f)

    def forward(self, batch: AudioBatch) -> AudioBatch:
        """Patchify feature batches into token batches.

        Parameters
        ----------
        batch
            Feature batch with `x` shape `(B, T, F)` and boolean `mask` shape
            `(B, T)`, where `True` means valid.

        Returns
        -------
        AudioBatch
            Token batch with `x` shape `(B, N, output_dim)` and collapsed
            boolean `mask` shape `(B, N)`, where `True` means valid.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "PatchifyTokenizer.forward expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                "PatchifyTokenizer.forward expected batch.x with rank 3, "
                f"got rank {batch.x.ndim}"
            )
        if batch.mask.dtype != torch.bool:
            raise ValueError(
                "PatchifyTokenizer.forward expected batch.mask dtype torch.bool, "
                f"got {batch.mask.dtype}"
            )
        if batch.mask.shape != batch.x.shape[:2]:
            raise ValueError(
                "PatchifyTokenizer.forward expected batch.mask shape to match "
                f"batch.x.shape[:2], got {tuple(batch.mask.shape)} and "
                f"{tuple(batch.x.shape[:2])}"
            )

        x = batch.x
        mask = batch.mask

        B, T, F_dim = x.shape
        pt, pf = self.config.patch_shape
        st, sf = self.config.effective_stride

        layout = self.layout(T_feat=T, F_in=F_dim)
        num_feat_patches = layout.frequency_patches

        # Handle time dimension
        num_time_patches, adjusted_t = self._time_grid(T, pt, st)
        if adjusted_t != T:
            if self.config.padding_strategy == "pad" and adjusted_t > T:
                pad_len = adjusted_t - T
                x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
                mask = F.pad(mask, (0, pad_len), value=False)
                T = adjusted_t
            elif self.config.padding_strategy == "truncate" and adjusted_t < T:
                T = adjusted_t
                x = x[:, :T, :]
                mask = mask[:, :T]
            else:
                raise ValueError(
                    f"Invalid padding strategy: {self.config.padding_strategy}. "
                    "Must be one of 'pad' or 'truncate'"
                )

        if num_time_patches == 0:
            x = x.new_empty((B, 0, self.config.output_dim))
            patch_mask = mask.new_empty((B, 0))
            return AudioBatch(x=x, mask=patch_mask)

        # Extract patches: (B, N, pt*pf)
        x = x.unfold(1, pt, st).unfold(2, pf, sf).contiguous()
        x = x.reshape(B, num_time_patches * num_feat_patches, pt * pf)
        if x.shape[-1] != self.config.output_dim:
            raise ValueError(
                "PatchifyTokenizer.forward produced output dimension "
                f"{x.shape[-1]}, expected {self.config.output_dim}"
            )

        # Build patch mask (B, N)
        time_patch_mask = (
            mask
            .unfold(1, pt, st)
            .all(dim=-1)
        )
        patch_mask = (
            time_patch_mask
            .unsqueeze(-1)
            .expand(-1, -1, num_feat_patches)
            .reshape(B, num_time_patches * num_feat_patches)
        )

        return AudioBatch(x=x, mask=patch_mask)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map valid feature-frame counts to valid token counts.

        Parameters
        ----------
        input_lengths
            Rank-1 tensor `(B,)` containing per-row valid frame counts.

        Returns
        -------
        torch.Tensor
            Rank-1 tensor `(B,)` on the same device as `input_lengths`
            containing per-row valid token counts. Values must not exceed the
            corresponding input lengths.

        Notes
        -----
        The feature-grid multiplier comes from the ``input_dim`` passed at
        construction, so the mapping does not depend on previous calls.
        """
        if not isinstance(input_lengths, torch.Tensor):
            raise TypeError(
                "PatchifyTokenizer.output_lengths expected torch.Tensor, "
                f"got {type(input_lengths).__name__}"
            )
        if input_lengths.ndim != 1:
            raise ValueError(
                "PatchifyTokenizer.output_lengths expected rank-1 input_lengths, "
                f"got rank {input_lengths.ndim}"
            )
        if self._num_feat_patches is None:
            raise ValueError(
                "PatchifyTokenizer.output_lengths requires the tokenizer to be constructed "
                "with input_dim (the processor feature width)"
            )

        patch_t, _ = self.config.patch_shape
        stride_t, _ = self.config.effective_stride
        full_time_patches = input_lengths.new_zeros(input_lengths.shape)
        has_full_patch = input_lengths >= patch_t
        full_time_patches[has_full_patch] = (
            torch.div(
                input_lengths[has_full_patch] - patch_t,
                stride_t,
                rounding_mode="floor",
            )
            + 1
        )
        output_lengths = full_time_patches * self._num_feat_patches
        if torch.any(output_lengths > input_lengths):
            raise ValueError("PatchifyTokenizer.output_lengths would upsample input lengths")
        return output_lengths

    def patch_grid(self, *, T_feat: int, F_in: int) -> tuple[int, int]:
        """Return `(T_patches, F_patches)` for a fixed-shape feature tensor.

        The calculation is a pure function of this tokenizer's config and
        mirrors the time and feature grid math used by `forward`. Tokens are
        ordered time-major over this grid.
        """
        layout = self.layout(T_feat=T_feat, F_in=F_in)
        return layout.time_patches, layout.frequency_patches

    def layout(self, *, T_feat: int, F_in: int) -> TokenLayout:
        """Describe the token layout for processor features.

        Parameters
        ----------
        T_feat
            Number of input feature frames.
        F_in
            Number of input feature bins.

        Returns
        -------
        TokenLayout
            Frozen layout metadata containing token count, time and frequency
            patch counts, patch shape, stride, time-major flatten order,
            tokenizer output dimension, and source feature shape.
        """
        if isinstance(T_feat, bool) or not isinstance(T_feat, int):
            raise TypeError(f"PatchifyTokenizer.layout T_feat must be int, got {type(T_feat).__name__}")
        if isinstance(F_in, bool) or not isinstance(F_in, int):
            raise TypeError(f"PatchifyTokenizer.layout F_in must be int, got {type(F_in).__name__}")
        if T_feat < 0:
            raise ValueError(f"PatchifyTokenizer.layout T_feat must be >= 0, got {T_feat!r}")
        if F_in < 0:
            raise ValueError(f"PatchifyTokenizer.layout F_in must be >= 0, got {F_in!r}")
        if self.input_dim is not None and F_in != self.input_dim:
            raise ValueError(
                "PatchifyTokenizer feature dimension must match the processor output_dim "
                f"given at construction, got {F_in} and {self.input_dim}"
            )

        patch_t, patch_f = self.config.patch_shape
        stride_t, stride_f = self.config.effective_stride
        num_time_patches, _ = self._time_grid(T_feat, patch_t, stride_t)
        num_feat_patches = self._feature_grid(F_in, patch_f, stride_f)
        return TokenLayout(
            num_tokens=num_time_patches * num_feat_patches,
            time_patches=num_time_patches,
            frequency_patches=num_feat_patches,
            patch_shape=(patch_t, patch_f),
            stride=(stride_t, stride_f),
            flatten_order="time_major",
            output_dim=self.config.output_dim,
            source_shape=(T_feat, F_in),
        )

    @staticmethod
    def reorder_patch_targets(
        target: torch.Tensor,
        *,
        layout: TokenLayout,
        order: str,
    ) -> torch.Tensor:
        """Reorder per-patch targets within each flattened patch vector.

        Parameters
        ----------
        target
            Patch targets with shape `(B, N, patch_t * patch_f)`.
        layout
            Token layout that supplies `patch_shape` and the expected
            time-major token grid.
        order
            Target order. `"time_frequency"` leaves targets unchanged;
            `"frequency_time"` transposes each patch target to Dasheng
            reconstruction order.

        Returns
        -------
        torch.Tensor
            Reordered patch targets with shape `(B, N, patch_t * patch_f)`.
        """
        if order == "time_frequency":
            return target
        if order != "frequency_time":
            raise ValueError(
                "PatchifyTokenizer.reorder_patch_targets order must be "
                f"'time_frequency' or 'frequency_time', got {order!r}"
            )
        patch_t, patch_f = layout.patch_shape
        if target.ndim != 3:
            raise ValueError(
                "PatchifyTokenizer.reorder_patch_targets expected rank-3 target, "
                f"got {tuple(target.shape)}"
            )
        if target.shape[-1] != patch_t * patch_f:
            raise ValueError(
                "PatchifyTokenizer.reorder_patch_targets expected final dim "
                f"{patch_t * patch_f}, got {target.shape[-1]}"
            )
        return (
            target.view(target.shape[0], target.shape[1], patch_t, patch_f)
            .transpose(-2, -1)
            .reshape(target.shape[0], target.shape[1], patch_t * patch_f)
        )

    def _feature_grid(self, feature_dim: int, patch_f: int, stride_f: int) -> int:
        if feature_dim < patch_f:
            raise ValueError(
                f"Feature dimension {feature_dim} is smaller than "
                f"patch_feature size {patch_f}."
            )
        if self.config.stride is None and feature_dim % patch_f != 0:
            raise ValueError(
                f"Feature dimension {feature_dim} is not divisible by "
                f"patch_feature size {patch_f}."
            )
        return ((feature_dim - patch_f) // stride_f) + 1

    def _time_grid(self, time_length: int, patch_t: int, stride_t: int) -> tuple[int, int]:
        if self.config.padding_strategy == "pad":
            if time_length <= patch_t:
                return 1, patch_t
            steps = (time_length - patch_t + stride_t - 1) // stride_t
            return steps + 1, patch_t + steps * stride_t
        if self.config.padding_strategy == "truncate":
            if time_length < patch_t:
                return 0, 0
            steps = (time_length - patch_t) // stride_t
            return steps + 1, patch_t + steps * stride_t
        raise ValueError(
            f"Invalid padding strategy: {self.config.padding_strategy}. "
            "Must be one of 'pad' or 'truncate'"
        )

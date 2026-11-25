"""Positional encoding modules for embedding entry PE and per-block RoPE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from audiossl.config import (
    ConvolutionalPositionalEncodingConfig,
    LearnablePositionalEncodingConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    NoPositionalEncodingConfig,
    RopePositionalEncodingConfig,
    Sinusoidal2DPositionalEncodingConfig,
    SinusoidalPositionalEncodingConfig,
)
from audiossl.config.batch_dataclasses import TokenLayout
from audiossl.modelling.core.initialization import initialize_conv1d
from audiossl.modelling.registry import build_activation


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate paired feature channels for rotary positional encoding."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


class LearnablePositionalEncoding(nn.Module):
    """Learned 1D absolute positional encoding for token embeddings.

    Parameters
    ----------
    config
        Maximum supported token length and initialization fields.
    d_model
        Embedding dimension `D`.
    """

    def __init__(self, config: LearnablePositionalEncodingConfig, *, d_model: int):
        super().__init__()
        self.config = config
        self.max_len = int(config.max_len)
        self.d_model = int(d_model)
        if self.max_len <= 0:
            raise ValueError(f"LearnablePositionalEncoding.max_len must be > 0, got {self.max_len}")
        if self.d_model <= 0:
            raise ValueError(f"LearnablePositionalEncoding.d_model must be > 0, got {self.d_model}")
        self.pe = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))

    def forward(
        self,
        x: torch.Tensor,
        layout: TokenLayout | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add learned position vectors to token states with shape `(B, N, D)`."""
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                "LearnablePositionalEncoding input last dimension mismatch: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured max_len {self.max_len}"
            )
        return x + self.pe[:, :seq_len]


class NoPositionalEncoding(nn.Module):
    """Identity positional encoding used when entry PE is disabled explicitly."""

    def __init__(self, config: NoPositionalEncodingConfig, *, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError(f"NoPositionalEncoding.d_model must be > 0, got {self.d_model}")

    def forward(
        self,
        x: torch.Tensor,
        layout: TokenLayout | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return token states unchanged."""
        return x


class LearnableTimeFrequencyPositionalEncoding(nn.Module):
    """Learned separable time/frequency PE for time-major patch token grids.

    The module adds a time table and a frequency table after embedding
    projection. A `TokenLayout` is required: `layout.frequency_patches` and
    `layout.time_patches` define the `(T_patches, F_patches)` grid.

    Parameters
    ----------
    config
        Maximum time/frequency patch counts for the learnable tables.
    d_model
        Embedding dimension `D`.
    """

    def __init__(
        self,
        config: LearnableTimeFrequencyPositionalEncodingConfig,
        *,
        d_model: int,
    ):
        super().__init__()
        self.config = config
        self.max_t = int(config.max_t)
        self.max_f = int(config.max_f)
        self.d_model = int(d_model)
        if self.max_t <= 0:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding.max_t must be > 0, "
                f"got {self.max_t}"
            )
        if self.max_f <= 0:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding.max_f must be > 0, "
                f"got {self.max_f}"
            )
        if self.d_model <= 0:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding.d_model must be > 0, "
                f"got {self.d_model}"
            )
        self.time_pe = nn.Parameter(torch.empty(1, self.max_t, self.d_model))
        self.freq_pe = nn.Parameter(torch.empty(1, self.max_f, self.d_model))
        nn.init.normal_(self.time_pe, mean=0.0, std=0.02)
        nn.init.normal_(self.freq_pe, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        layout: TokenLayout | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add time and frequency PE to token states with shape `(B, N, D)`."""
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding input last dimension mismatch: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )
        if layout is None:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding requires a TokenLayout describing "
                "the (time, frequency) patch grid"
            )
        num_tokens = x.shape[1]
        frequency_patches = layout.frequency_patches
        if frequency_patches <= 0:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding layout.frequency_patches "
                f"must be > 0, got {frequency_patches}"
            )
        if frequency_patches > self.max_f:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding frequency patch count "
                f"{frequency_patches} exceeds configured max_f {self.max_f}"
            )
        if num_tokens % frequency_patches != 0:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding token count must be divisible "
                f"by frequency_patches={frequency_patches}, got {num_tokens}"
            )
        t_patches = num_tokens // frequency_patches
        if layout.time_patches != t_patches:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding layout time_patches does not "
                f"match token count: got {layout.time_patches} and {t_patches}"
            )
        if t_patches > self.max_t:
            raise ValueError(
                "LearnableTimeFrequencyPositionalEncoding time patch count "
                f"{t_patches} exceeds configured max_t {self.max_t}"
            )

        time_pe = self.time_pe[:, :t_patches].repeat_interleave(frequency_patches, dim=1)
        freq_pe = self.freq_pe[:, :frequency_patches].repeat(1, t_patches, 1)
        return x + time_pe + freq_pe


class ConvolutionalPositionalEncoding(nn.Module):
    """Grouped Conv1d positional residual used by wav2vec-style embeddings.

    The module operates on embedding token states `(B, N, D)`, applies grouped
    temporal convolution over `N`, then adds the activated result back to `x`.

    Parameters
    ----------
    config
        Kernel size, group count, activation, and initialization route.
    d_model
        Embedding dimension `D`.
    """

    def __init__(
        self,
        config: ConvolutionalPositionalEncodingConfig,
        *,
        d_model: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = int(d_model)
        self.kernel_size = int(config.kernel_size)
        if self.d_model <= 0:
            raise ValueError(
                f"ConvolutionalPositionalEncoding.d_model must be > 0, got {self.d_model}"
            )
        if self.kernel_size <= 0:
            raise ValueError(
                "ConvolutionalPositionalEncoding.kernel_size must be > 0, "
                f"got {self.kernel_size}"
            )
        if self.d_model % config.groups != 0:
            raise ValueError(
                "ConvolutionalPositionalEncoding groups must divide d_model, "
                f"got {config.groups} and {self.d_model}"
            )
        self.conv = nn.Conv1d(
            self.d_model,
            self.d_model,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=config.groups,
        )
        self.num_pad_remove = 1 if self.kernel_size % 2 == 0 else 0
        self.activation = build_activation(config.activation)
        initialize_conv1d(self.conv, config.initialization)

    def forward(
        self,
        x: torch.Tensor,
        layout: TokenLayout | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add a convolutional positional residual to `(B, N, D)` token states."""
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                "ConvolutionalPositionalEncoding input last dimension mismatch: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )
        conv_input = x if mask is None else x.masked_fill(~mask.unsqueeze(-1), 0.0)
        pos = self.conv(conv_input.transpose(1, 2))
        if self.num_pad_remove > 0:
            pos = pos[:, :, : -self.num_pad_remove]
        return x + self.activation(pos).transpose(1, 2)


class RoPE(nn.Module):
    """Rotary positional encoding for per-block attention heads.

    RoPE is used inside `MultiHeadAttention` when the encoder config requests
    per-block positional encoding. Inputs are per-head states with shape
    `(B * H, N, head_dim)`.

    Parameters
    ----------
    config
        Maximum sequence length and rotary base.
    d_model
        Per-head feature dimension. It must be even.
    """

    def __init__(self, config: RopePositionalEncodingConfig, *, d_model: int):
        super().__init__()
        self.config = config
        self.max_len = int(config.max_len)
        self.d_model = int(d_model)
        if self.max_len <= 0:
            raise ValueError(f"RoPE.max_len must be > 0, got {self.max_len}")
        if self.d_model <= 0:
            raise ValueError(f"RoPE.d_model must be > 0, got {self.d_model}")
        self.base = float(config.base)
        if not torch.isfinite(torch.tensor(self.base)) or self.base <= 0.0:
            raise ValueError(f"RoPE.base must be a positive finite float, got {self.base!r}")

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(dim=self.d_model, device=torch.device("cpu"))

    def _build_cache(self, dim: int, device: torch.device) -> None:
        """Build cosine/sine tables on the requested device."""
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even embedding dimension, got {dim}")
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        positions = torch.arange(self.max_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Apply rotary position mixing to states with shape `(B * H, N, D)`.

        ``positions`` optionally gives integer token positions with shape
        ``(B * H, N)``; ``None`` means ``arange(N)``. Objectives that drop or
        reorder tokens pass true positions so rotations stay consistent.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        _, seq_len, dim = x.shape
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even embedding dimension, got {dim}")
        if dim != self.d_model:
            raise ValueError(f"RoPE input dimension mismatch: expected {self.d_model}, got {dim}")

        if self.cos_cached.device != x.device:
            self._build_cache(dim=dim, device=x.device)

        if positions is None:
            if seq_len > self.max_len:
                raise ValueError(
                    f"Input sequence length {seq_len} exceeds configured max_len {self.max_len}"
                )
            cos = self.cos_cached[:seq_len].to(dtype=x.dtype).unsqueeze(0)
            sin = self.sin_cached[:seq_len].to(dtype=x.dtype).unsqueeze(0)
        else:
            if positions.shape != x.shape[:2]:
                raise ValueError(
                    f"RoPE positions must have shape {tuple(x.shape[:2])}, "
                    f"got {tuple(positions.shape)}"
                )
            if positions.numel() > 0 and int(positions.max().item()) >= self.max_len:
                raise ValueError(
                    f"RoPE position {int(positions.max().item())} exceeds configured "
                    f"max_len {self.max_len}"
                )
            cos = self.cos_cached[positions].to(dtype=x.dtype)
            sin = self.sin_cached[positions].to(dtype=x.dtype)
        return (x * cos) + (_rotate_half(x) * sin)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed 1D sinusoidal absolute positional encoding for embeddings.

    Parameters
    ----------
    config
        Maximum supported token length and sinusoidal base.
    d_model
        Embedding dimension `D`.
    """

    def __init__(self, config: SinusoidalPositionalEncodingConfig, *, d_model: int):
        super().__init__()
        self.config = config
        self.max_len = int(config.max_len)
        self.d_model = int(d_model)
        if self.max_len <= 0:
            raise ValueError(f"SinusoidalPositionalEncoding.max_len must be > 0, got {self.max_len}")
        if self.d_model <= 0:
            raise ValueError(f"SinusoidalPositionalEncoding.d_model must be > 0, got {self.d_model}")
        self.base = float(config.base)
        if not math.isfinite(self.base) or self.base <= 0.0:
            raise ValueError(
                f"SinusoidalPositionalEncoding.base must be a positive finite float, got {self.base!r}"
            )
        self.register_buffer("pe", torch.empty(0), persistent=False)
        self._build_table(device=torch.device("cpu"))

    def _build_table(self, device: torch.device) -> None:
        """Build the fixed sinusoidal lookup table on the requested device."""
        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float32, device=device)
        position = torch.arange(self.max_len, dtype=torch.float32, device=device).unsqueeze(1)
        even_idx = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
        div_term = self.base ** (even_idx / self.d_model)
        angles = position / div_term.unsqueeze(0)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
        self.pe = pe.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        layout: TokenLayout | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add sinusoidal position vectors to token states with shape `(B, N, D)`."""
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                "SinusoidalPositionalEncoding input last dimension mismatch: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured max_len {self.max_len}"
            )
        if self.pe.device != x.device:
            self._build_table(device=x.device)
        return x + self.pe[:, :seq_len].to(dtype=x.dtype)


def build_2d_sincos_pos_embed(
    *,
    embed_dim: int,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Build fixed 2-D sin-cos table matching EAT ``model_core.py``."""
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    mesh_w, mesh_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack([mesh_w, mesh_h], dim=0).reshape(2, 1, grid_size[0], grid_size[1])

    emb_h = _build_1d_sincos_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _build_1d_sincos_from_grid(embed_dim // 2, grid[1].reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1)


def _build_1d_sincos_from_grid(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


class Sinusoidal2DPositionalEncoding(nn.Module):
    """Fixed 2-D sinusoidal positional encoding over patch token grids."""

    def __init__(self, config: Sinusoidal2DPositionalEncodingConfig, *, d_model: int) -> None:
        super().__init__()
        self.config = config
        self.d_model = int(d_model)
        self.max_time_patches = int(config.max_time_patches)
        self.max_freq_patches = int(config.max_freq_patches)
        max_tokens = self.max_time_patches * self.max_freq_patches
        table = build_2d_sincos_pos_embed(
            embed_dim=self.d_model,
            grid_size=(self.max_time_patches, self.max_freq_patches),
        )
        self.register_buffer("pe", table.unsqueeze(0), persistent=True)

    def forward(
        self,
        x: torch.Tensor,
        layout: TokenLayout | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, N, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                "Sinusoidal2DPositionalEncoding input last dimension mismatch: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )
        if layout is None:
            raise ValueError(
                "Sinusoidal2DPositionalEncoding requires a TokenLayout describing the "
                "(time, frequency) patch grid"
            )
        time_patches, frequency_patches = layout.time_patches, layout.frequency_patches
        if time_patches * frequency_patches != x.shape[1]:
            raise ValueError(
                "Sinusoidal2DPositionalEncoding layout grid does not match token count: "
                f"{time_patches} x {frequency_patches} != {x.shape[1]}"
            )
        if time_patches > self.max_time_patches:
            raise ValueError(
                f"Sinusoidal2DPositionalEncoding time patch count {time_patches} exceeds "
                f"configured max_time_patches {self.max_time_patches}"
            )
        if frequency_patches > self.max_freq_patches:
            raise ValueError(
                f"Sinusoidal2DPositionalEncoding frequency patch count {frequency_patches} "
                f"exceeds configured max_freq_patches {self.max_freq_patches}"
            )
        if frequency_patches == self.max_freq_patches:
            table = self.pe[:, : time_patches * frequency_patches]
        else:
            t_idx = torch.arange(time_patches, device=self.pe.device)
            f_idx = torch.arange(frequency_patches, device=self.pe.device)
            index = (t_idx[:, None] * self.max_freq_patches + f_idx[None, :]).reshape(-1)
            table = self.pe[:, index]
        return x + table.to(dtype=x.dtype)


"""MAE decoder stack for reconstructing removed patch targets."""

from __future__ import annotations

import torch
import torch.nn as nn

from audiossl.config import MAEObjectiveConfig
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout
from audiossl.modelling.encoders.encoder import Encoder
from audiossl.modelling.registry import get_embedding_positional_encoding_registry


class MAEDecoder(nn.Module):
    """Transformer decoder reconstructing patch pixels from kept-token latents.

    Parameters
    ----------
    encoder_d_model
        Feature dimension of kept encoder latents.
    config
        MAE objective config that owns decoder width, decoder encoder stack,
        positional encoding, and mask-token scale.
    patch_t, patch_f
        Patch shape used to set the reconstruction dimension
        ``patch_t * patch_f``.

    This mirrors XiaoMi/dasheng `AudioTransformerMAE_Decoder` except that the
    positional encoding site is audiossl's embedding-side PE module, commonly
    the 2D separable PE used by dasheng encoders, rather than dasheng's 1D
    learnable `pos_embed`.
    """

    def __init__(
        self,
        *,
        encoder_d_model: int,
        config: MAEObjectiveConfig,
        patch_t: int,
        patch_f: int,
    ) -> None:
        super().__init__()
        if not isinstance(encoder_d_model, int) or encoder_d_model < 1:
            raise ValueError(
                "MAEDecoder.__init__ expected encoder_d_model >= 1, "
                f"got {encoder_d_model!r}"
            )
        if not isinstance(config, MAEObjectiveConfig):
            raise TypeError(
                "MAEDecoder.__init__ expected MAEObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        if not isinstance(patch_t, int) or patch_t < 1:
            raise ValueError(f"MAEDecoder.__init__ expected patch_t >= 1, got {patch_t!r}")
        if not isinstance(patch_f, int) or patch_f < 1:
            raise ValueError(f"MAEDecoder.__init__ expected patch_f >= 1, got {patch_f!r}")

        self.config = config
        self.encoder_d_model = encoder_d_model
        self.decoder_d_model = config.decoder_d_model
        self.patch_t = patch_t
        self.patch_f = patch_f

        self.input_proj = nn.Linear(encoder_d_model, config.decoder_d_model)
        self.mask_token = nn.Parameter(torch.empty(1, 1, config.decoder_d_model))
        nn.init.normal_(self.mask_token, std=float(config.mask_token_init_scale))

        pe_registry = get_embedding_positional_encoding_registry()
        pe_cls = pe_registry[config.decoder_positional_encoding.type]
        self.positional_encoding = pe_cls(
            config.decoder_positional_encoding,
            d_model=config.decoder_d_model,
        )

        self.blocks = Encoder(
            d_model=config.decoder_d_model,
            config=config.decoder,
        )
        self.outputlayer = nn.Linear(config.decoder_d_model, patch_t * patch_f)

    def forward(
        self,
        x_kept: torch.Tensor,
        ids_restore: torch.Tensor,
        *,
        kept_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        layout: TokenLayout | None = None,
    ) -> torch.Tensor:
        """Run the decoder and return ``(B, N, patch_t * patch_f)`` predictions.

        Parameters
        ----------
        x_kept
            Encoded kept tokens with shape ``(B, len_keep, encoder_d_model)``.
        ids_restore
            Inverse shuffle indices with shape ``(B, N)`` used to restore the
            full token order before decoding.
        kept_mask
            Optional boolean ``(B, len_keep)``; ``False`` slots were padding for
            the encoder and are replaced by the mask token. Defaults to all kept.
        token_mask
            Optional boolean ``(B, N)`` validity of the full token axis, used as
            the decoder attention mask. Defaults to all valid.
        layout
            Token layout of the full token axis, required by layout-aware PEs.
        """

        if not isinstance(x_kept, torch.Tensor):
            raise TypeError(
                f"MAEDecoder.forward expected x_kept torch.Tensor, got {type(x_kept).__name__}"
            )
        if x_kept.ndim != 3:
            raise ValueError(
                f"MAEDecoder.forward expected x_kept rank-3, got shape {tuple(x_kept.shape)}"
            )
        if x_kept.shape[-1] != self.encoder_d_model:
            raise ValueError(
                "MAEDecoder.forward expected x_kept last dim == encoder_d_model, "
                f"got {x_kept.shape[-1]} and {self.encoder_d_model}"
            )
        if not isinstance(ids_restore, torch.Tensor):
            raise TypeError(
                "MAEDecoder.forward expected ids_restore torch.Tensor, "
                f"got {type(ids_restore).__name__}"
            )
        if ids_restore.ndim != 2:
            raise ValueError(
                "MAEDecoder.forward expected ids_restore rank-2 [B, N], "
                f"got shape {tuple(ids_restore.shape)}"
            )
        if ids_restore.shape[0] != x_kept.shape[0]:
            raise ValueError(
                "MAEDecoder.forward expected ids_restore.shape[0] == x_kept.shape[0], "
                f"got {ids_restore.shape[0]} and {x_kept.shape[0]}"
            )
        if ids_restore.shape[1] < x_kept.shape[1]:
            raise ValueError(
                "MAEDecoder.forward expected ids_restore.shape[1] >= x_kept.shape[1], "
                f"got {ids_restore.shape[1]} and {x_kept.shape[1]}"
            )

        x = self.input_proj(x_kept)
        batch_size, len_keep, dim = x.shape
        num_tokens = ids_restore.shape[1]
        self._check_mask("kept_mask", kept_mask, (batch_size, len_keep))
        self._check_mask("token_mask", token_mask, (batch_size, num_tokens))

        num_masked = num_tokens - len_keep
        mask_tokens = self.mask_token.to(x.dtype).expand(batch_size, num_masked, dim)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, dim),
        )
        if kept_mask is not None:
            encoded = torch.cat([kept_mask, kept_mask.new_zeros(batch_size, num_masked)], dim=1)
            encoded = torch.gather(encoded, dim=1, index=ids_restore)
            x_full = torch.where(encoded.unsqueeze(-1), x_full, mask_tokens[:, :1])

        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=x_full.device)
        x_full = self.positional_encoding(x_full, layout, mask=token_mask)
        decoder_out = self.blocks(AudioBatch(x=x_full, mask=token_mask))
        return self.outputlayer(decoder_out.z)

    @staticmethod
    def _check_mask(name: str, mask: torch.Tensor | None, shape: tuple[int, int]) -> None:
        if mask is None:
            return
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError(f"MAEDecoder.forward expected boolean {name} tensor")
        if tuple(mask.shape) != shape:
            raise ValueError(
                f"MAEDecoder.forward expected {name} shape {shape}, got {tuple(mask.shape)}"
            )


__all__ = ["MAEDecoder"]

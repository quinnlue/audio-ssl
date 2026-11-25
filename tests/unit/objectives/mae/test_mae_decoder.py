from __future__ import annotations

import pytest
import torch

from audiossl.config import (
    LearnableTimeFrequencyPositionalEncodingConfig,
    MAEObjectiveConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    TransformerEncoderConfig,
)
from audiossl.config.batch_dataclasses import TokenLayout
from audiossl.modelling.objectives.mae.decoder import MAEDecoder


def _grid(time_patches: int, frequency_patches: int) -> TokenLayout:
    return TokenLayout(
        num_tokens=time_patches * frequency_patches,
        time_patches=time_patches,
        frequency_patches=frequency_patches,
        patch_shape=(1, 1),
        stride=(1, 1),
        flatten_order="time_major",
        output_dim=1,
        source_shape=(time_patches, frequency_patches),
    )


def _mae_config(
    *,
    decoder_d_model: int = 32,
    max_t: int = 10,
    max_f: int = 2,
) -> MAEObjectiveConfig:
    return MAEObjectiveConfig(
        decoder_d_model=decoder_d_model,
        decoder=TransformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=4),
            mlp=MLPConfig(d_ff=64),
            num_layers=2,
            pre_norm=True,
            residual_dropout_p=0.0,
            positional_encoding=None,
            final_norm=True,
        ),
        decoder_positional_encoding=LearnableTimeFrequencyPositionalEncodingConfig(
            max_t=max_t,
            max_f=max_f,
        ),
        mask_token_init_scale=0.02,
    )


class TestMAEDecoder:
    def test_output_shape(self) -> None:
        decoder = MAEDecoder(
            encoder_d_model=16,
            config=_mae_config(),
            patch_t=4,
            patch_f=8,
        )
        x_kept = torch.randn(2, 5, 16)
        ids_restore = torch.stack([torch.randperm(20) for _ in range(2)])

        out = decoder(x_kept, ids_restore, layout=_grid(10, 2))

        assert out.shape == (2, 20, 32)

    def test_unkept_slots_and_padding_do_not_influence_valid_predictions(self) -> None:
        torch.manual_seed(0)
        decoder = MAEDecoder(
            encoder_d_model=16,
            config=_mae_config(),
            patch_t=4,
            patch_f=8,
        ).eval()
        ids_restore = torch.stack([torch.randperm(20) for _ in range(2)])
        kept_mask = torch.ones(2, 5, dtype=torch.bool)
        kept_mask[0, 3:] = False
        token_mask = torch.ones(2, 20, dtype=torch.bool)
        token_mask[0, 16:] = False
        x_kept = torch.randn(2, 5, 16)
        perturbed = x_kept.clone()
        perturbed[0, 3:] += 100.0

        kwargs = {"kept_mask": kept_mask, "token_mask": token_mask, "layout": _grid(10, 2)}
        with torch.no_grad():
            a = decoder(x_kept, ids_restore, **kwargs)
            b = decoder(perturbed, ids_restore, **kwargs)

        torch.testing.assert_close(a[token_mask], b[token_mask], rtol=0, atol=0)

    def test_mask_token_is_parameter(self) -> None:
        decoder = MAEDecoder(
            encoder_d_model=16,
            config=_mae_config(),
            patch_t=4,
            patch_f=8,
        )

        assert isinstance(decoder.mask_token, torch.nn.Parameter)
        assert decoder.mask_token.shape == (1, 1, 32)
        assert decoder.mask_token.abs().sum() > 0

    def test_backward_reaches_decoder_parameters(self) -> None:
        decoder = MAEDecoder(
            encoder_d_model=16,
            config=_mae_config(),
            patch_t=2,
            patch_f=4,
        )
        x_kept = torch.randn(2, 6, 16, requires_grad=True)
        ids_restore = torch.stack([torch.randperm(12) for _ in range(2)])

        out = decoder(x_kept, ids_restore, layout=_grid(6, 2))
        out.square().mean().backward()

        assert x_kept.grad is not None
        assert decoder.mask_token.grad is not None
        assert decoder.input_proj.weight.grad is not None
        assert decoder.outputlayer.weight.grad is not None

    def test_init_validates_args(self) -> None:
        config = _mae_config()
        with pytest.raises(ValueError):
            MAEDecoder(encoder_d_model=0, config=config, patch_t=4, patch_f=8)
        with pytest.raises(ValueError):
            MAEDecoder(encoder_d_model=16, config=config, patch_t=0, patch_f=8)
        with pytest.raises(TypeError):
            MAEDecoder(
                encoder_d_model=16,
                config="not config",  # type: ignore[arg-type]
                patch_t=4,
                patch_f=8,
            )

    def test_forward_validates_shapes(self) -> None:
        decoder = MAEDecoder(
            encoder_d_model=16,
            config=_mae_config(),
            patch_t=4,
            patch_f=8,
        )

        with pytest.raises(ValueError, match="rank-3"):
            decoder(torch.randn(2, 16), torch.arange(4).unsqueeze(0))
        with pytest.raises(ValueError, match="last dim"):
            decoder(torch.randn(2, 4, 15), torch.arange(4).repeat(2, 1))
        with pytest.raises(ValueError, match="ids_restore"):
            decoder(torch.randn(2, 4, 16), torch.arange(4))

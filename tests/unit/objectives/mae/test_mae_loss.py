from __future__ import annotations

import pytest
import torch

from audiossl.modelling.objectives.mae.loss import mae_reconstruction_loss


class TestMAEReconstructionLoss:
    def test_norm_none_matches_hand_rolled(self) -> None:
        torch.manual_seed(0)
        prediction = torch.randn(2, 8, 16)
        target = torch.randn(2, 8, 16)
        target_mask = torch.zeros(2, 8, dtype=torch.bool)
        target_mask[:, ::2] = True

        out = mae_reconstruction_loss(
            prediction,
            target,
            target_mask,
            norm_pix_loss="none",
        )

        expected = ((prediction - target) ** 2).mean(dim=-1)[target_mask].mean()
        torch.testing.assert_close(out.loss, expected, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(out.per_patch_loss, ((prediction - target) ** 2).mean(dim=-1))

    def test_per_patch_z_scores_target(self) -> None:
        torch.manual_seed(0)
        prediction = torch.randn(2, 8, 16)
        target = torch.randn(2, 8, 16) * 5.0 + 3.0
        mask = torch.ones(2, 8, dtype=torch.bool)

        out_none = mae_reconstruction_loss(
            prediction,
            target,
            mask,
            norm_pix_loss="none",
        )
        out_per_patch = mae_reconstruction_loss(
            prediction,
            target,
            mask,
            norm_pix_loss="per_patch",
        )

        assert not torch.allclose(out_none.loss, out_per_patch.loss)

    def test_global_z_scores_target(self) -> None:
        torch.manual_seed(0)
        prediction = torch.randn(2, 8, 16)
        target = torch.randn(2, 8, 16) * 5.0 + 3.0
        mask = torch.ones(2, 8, dtype=torch.bool)

        out = mae_reconstruction_loss(
            prediction,
            target,
            mask,
            norm_pix_loss="global",
        )

        assert out.loss.ndim == 0
        assert out.loss.dtype == torch.float32

    def test_fp32_under_bfloat16_autocast(self) -> None:
        prediction = torch.randn(2, 4, 8)
        target = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = mae_reconstruction_loss(
                prediction,
                target,
                mask,
                norm_pix_loss="per_patch",
            )

        assert out.loss.dtype == torch.float32

    def test_empty_mask_raises(self) -> None:
        prediction = torch.randn(2, 4, 8)
        target = torch.randn(2, 4, 8)
        mask = torch.zeros(2, 4, dtype=torch.bool)

        with pytest.raises(ValueError, match="select"):
            mae_reconstruction_loss(
                prediction,
                target,
                mask,
                norm_pix_loss="per_patch",
            )

    @pytest.mark.parametrize("norm", ["random", "", "Per_Patch"])
    def test_invalid_norm_raises(self, norm: str) -> None:
        prediction = torch.randn(2, 4, 8)
        target = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)

        with pytest.raises(ValueError):
            mae_reconstruction_loss(
                prediction,
                target,
                mask,
                norm_pix_loss=norm,  # type: ignore[arg-type]
            )

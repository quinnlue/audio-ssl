"""Forward-equivalence parity between MAEObjective and dasheng MAE."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from testkit.external import load_reference

from audiossl.config import (
    LearnableTimeFrequencyPositionalEncodingConfig,
    MAEMaskingConfig,
    MAEObjectiveConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    TransformerEncoderConfig,
)
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.integrations.dasheng import make_dasheng_base_config
from audiossl.integrations.dasheng.porter import (
    port_dasheng_mae_state_dict,
    port_dasheng_state_dict,
)
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives.mae import MAEObjective


def _dasheng_models_module():
    pytest.importorskip("einops")
    pytest.importorskip("torchaudio")
    repo_root = Path(__file__).resolve().parents[3]
    dasheng_pkg_dir = (
        repo_root / "eval" / "models" / "dasheng-inference" / "dasheng" / "dasheng"
    )
    if not dasheng_pkg_dir.is_dir():
        pytest.skip("Dasheng reference source tree is not available")
    if "dasheng" not in sys.modules:
        pkg = types.ModuleType("dasheng")
        pkg.__path__ = [str(dasheng_pkg_dir)]
        pkg.__version__ = "test"
        sys.modules["dasheng"] = pkg
    def load():
        from dasheng.train import models

        return models

    return load_reference(load, "Dasheng reference")


def _objective_config() -> MAEObjectiveConfig:
    return MAEObjectiveConfig(
        decoder_d_model=512,
        decoder=TransformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=16),
            mlp=MLPConfig(d_ff=512 * 4, activation="gelu"),
            num_layers=8,
            pre_norm=True,
            residual_dropout_p=0.0,
            positional_encoding=None,
            layer_norm_eps=1e-6,
            final_norm=True,
        ),
        decoder_positional_encoding=LearnableTimeFrequencyPositionalEncodingConfig(
            max_t=252,
            max_f=1,
        ),
        masking=MAEMaskingConfig(mask_ratio=0.75, group_factor=2),
        norm_pix_loss="per_patch",
        target_patch_order="frequency_time",
    )


def test_mae_dasheng_parity() -> None:
    dasheng_models = _dasheng_models_module()
    torch.manual_seed(0)

    dasheng_mae = dasheng_models.dasheng_base(group_masking=2).eval()
    our_model = AudioModel(make_dasheng_base_config())
    our_obj = MAEObjective(
        our_model, _objective_config(), generator=torch.Generator().manual_seed(42)
    ).eval()

    port_dasheng_state_dict(dasheng_mae.encoder.state_dict(), our_model)
    decoder_state = {
        f"decoder.{key}": value
        for key, value in dasheng_mae.decoder.state_dict().items()
    }
    port_dasheng_mae_state_dict(decoder_state, our_obj)

    with torch.no_grad():
        dasheng_mae.decoder.pos_embed.zero_()
        our_obj.decoder.positional_encoding.time_pe.zero_()
        our_obj.decoder.positional_encoding.freq_pe.zero_()

    # 160000 samples produce 250 patches with Dasheng's frontend, so group
    # masking with group_factor=2 is well-defined on both sides.
    waveform = torch.randn(1, 160000, dtype=torch.float32)

    torch.manual_seed(42)
    with torch.no_grad():
        dasheng_pred, dasheng_target, dasheng_mask = dasheng_mae(
            waveform,
            mask_ratio=0.75,
        )
        dasheng_loss = dasheng_mae.loss_fn(
            dasheng_pred,
            dasheng_target,
            dasheng_mask,
        )

    torch.manual_seed(42)
    our_batch = AudioBatch(
        x=waveform,
        mask=torch.ones_like(waveform, dtype=torch.bool),
    )
    with torch.no_grad():
        our_out = our_obj(our_batch)

    assert torch.equal(our_out.target_mask, dasheng_mask.bool())
    torch.testing.assert_close(our_out.target, dasheng_target, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(our_out.prediction, dasheng_pred, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(our_out.loss, dasheng_loss, atol=1e-5, rtol=1e-5)

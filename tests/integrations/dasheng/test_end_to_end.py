"""End-to-end parity test against the real dasheng_base checkpoint.

Skipped cleanly when the checkpoint isn't available locally. Set
``DASHENG_BASE_CKPT`` to the .pt path to enable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.integrations.dasheng import load_dasheng_base


def _find_checkpoint() -> Path | None:
    env = os.environ.get("DASHENG_BASE_CKPT")
    if env:
        path = Path(env)
        return path if path.is_file() else None
    # Common torch.hub cache locations.
    hub_candidates = [
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dasheng_base.pt",
        Path(os.environ.get("TORCH_HOME", "")) / "hub" / "checkpoints" / "dasheng_base.pt",
    ]
    for candidate in hub_candidates:
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.parametrize("audio_length", [16000, 160000])  # 1 s and 10 s
def test_dasheng_base_end_to_end(dasheng_modules, audio_length: int):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip(
            "dasheng_base checkpoint not found; set DASHENG_BASE_CKPT to the .pt path"
        )

    torch.manual_seed(7)
    waveform = torch.randn(1, audio_length)

    # Reference: dasheng_base() helper - it sets num_heads=12 explicitly.
    # Calling Dasheng.from_pretrained() directly would default to num_heads=8
    # (the AudioTransformerMAE_Encoder default), which silently mis-splits the
    # QKV projection at inference even though the checkpoint loads cleanly.
    ref_model = dasheng_modules.pretrained.dasheng_base(path=str(ckpt)).eval()
    with torch.inference_mode():
        ref_out = ref_model(waveform)  # [B, num_patches, 768]

    # Ours.
    our_model = load_dasheng_base(ckpt).eval()
    mask = torch.ones_like(waveform, dtype=torch.bool)
    with torch.inference_mode():
        ours = our_model(AudioBatch(x=waveform, mask=mask)).z

    # For inputs <= target_length=1008 frames, both should agree to within
    # accumulated fp32 error across 12 blocks.
    assert ours.shape == ref_out.shape, (ours.shape, ref_out.shape)
    torch.testing.assert_close(ours, ref_out, atol=1e-3, rtol=1e-3)

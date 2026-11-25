"""End-to-end parity test against the real EAT-base Hugging Face checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.integrations.eat import load_eat_base, make_eat_base_config
from audiossl.integrations.eat.preprocess import preprocess_eat_mel
from audiossl.integrations.eat.porter import port_eat_state_dict
from audiossl.modelling.model import AudioModel


def _find_checkpoint() -> Path | None:
    env = os.environ.get("EAT_BASE_CKPT")
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    if hub_root.is_dir():
        candidates = sorted(
            hub_root.glob(
                "models--worstchan--EAT-base_epoch30_pretrain/snapshots/*/model.safetensors"
            )
        )
        if candidates:
            return candidates[-1]
    return None


def _preprocess_mel(waveform: torch.Tensor, *, target_length: int = 1024) -> torch.Tensor:
    return preprocess_eat_mel(waveform, target_length=target_length)


@pytest.mark.parametrize("audio_length", [16000, 160000])
def test_eat_base_end_to_end(hf_eat_model, audio_length: int):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip(
            "EAT-base checkpoint not found; set EAT_BASE_CKPT or populate the HF cache"
        )

    torch.manual_seed(11)
    waveform = torch.randn(1, audio_length)
    mel = _preprocess_mel(waveform[0])

    with torch.inference_mode():
        ref_out = hf_eat_model.extract_features(mel)

    our_model = load_eat_base(ckpt).eval()
    mask = torch.ones_like(waveform, dtype=torch.bool)
    with torch.inference_mode():
        ours = our_model.pipeline(AudioBatch(x=waveform, mask=mask)).z

    assert ours.shape == ref_out.shape, (ours.shape, ref_out.shape)
    torch.testing.assert_close(ours, ref_out, atol=1e-3, rtol=1e-3)


def test_port_eat_state_dict_loads(hf_eat_model):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip(
            "EAT-base checkpoint not found; set EAT_BASE_CKPT or populate the HF cache"
        )

    from safetensors.torch import load_file

    state_dict = load_file(str(ckpt), device="cpu")
    model = AudioModel(make_eat_base_config())
    ported = port_eat_state_dict(state_dict, model=model)
    incompatible = model.load_state_dict(ported, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

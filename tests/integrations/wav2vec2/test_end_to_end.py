"""Opt-in end-to-end parity test against the real facebook/wav2vec2-base checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.integrations.wav2vec2 import load_wav2vec2_base
from audiossl.integrations.wav2vec2.porter import (
    _extract_state_dict,
    _normalize_wav2vec2_state_dict,
)


def _find_checkpoint() -> Path | None:
    env = os.environ.get("WAV2VEC2_BASE_CKPT")
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    if hub_root.is_dir():
        candidates = sorted(
            hub_root.glob("models--facebook--wav2vec2-base/snapshots/*/model.safetensors")
        )
        if candidates:
            return candidates[-1]
    return None


def _load_reference_model(hf_wav2vec2, ckpt: Path):
    config = hf_wav2vec2.config.Wav2Vec2Config()
    ref_model = hf_wav2vec2.modeling.Wav2Vec2Model(config)
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt), device="cpu")
    else:
        state_dict = _extract_state_dict(torch.load(ckpt, map_location="cpu"))
    state_dict = _normalize_wav2vec2_state_dict(state_dict)

    old_g = "encoder.pos_conv_embed.conv.weight_g"
    old_v = "encoder.pos_conv_embed.conv.weight_v"
    new_g = "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    new_v = "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    if old_g in state_dict and new_g not in state_dict:
        state_dict[new_g] = state_dict.pop(old_g)
        state_dict[new_v] = state_dict.pop(old_v)

    # masked_spec_embed is only read by SpecAugment during training.
    state_dict.pop("masked_spec_embed", None)
    incompatible = ref_model.load_state_dict(state_dict, strict=False)
    assert set(incompatible.missing_keys) <= {"masked_spec_embed"}, incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys
    return ref_model.eval()


@pytest.mark.parametrize("audio_length", [16000, 160000])
def test_wav2vec2_base_end_to_end(hf_wav2vec2, audio_length: int):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip(
            "wav2vec2-base checkpoint not found; set WAV2VEC2_BASE_CKPT or "
            "populate the Hugging Face cache"
        )

    torch.manual_seed(9)
    waveform = torch.randn(1, audio_length)
    mask = torch.ones_like(waveform, dtype=torch.bool)

    ref_model = _load_reference_model(hf_wav2vec2, ckpt)
    with torch.inference_mode():
        ref_out = ref_model(waveform, attention_mask=mask).last_hidden_state

    our_model = load_wav2vec2_base(ckpt).eval()
    with torch.inference_mode():
        ours = our_model(AudioBatch(x=waveform, mask=mask)).z

    assert ours.shape == ref_out.shape
    torch.testing.assert_close(ours, ref_out, atol=1e-3, rtol=1e-3)

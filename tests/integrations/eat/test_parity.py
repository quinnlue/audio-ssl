"""Component-level parity tests between audiossl EAT pieces and the HF reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from audiossl.integrations.eat import make_eat_base_config
from audiossl.integrations.eat.porter import _convert_patch_embed_weight
from audiossl.modelling.model import AudioModel


def test_patch_weight_conversion_preserves_rectangular_grid_order():
    weight = torch.randn(4, 1, 3, 2)
    features = torch.randn(2, 1, 6, 8)
    patches = F.unfold(features, kernel_size=(3, 2), stride=(3, 2)).transpose(1, 2)
    expected = F.conv2d(features, weight, stride=(3, 2)).flatten(2).transpose(1, 2)
    actual = F.linear(patches, _convert_patch_embed_weight(weight))
    torch.testing.assert_close(actual, expected)


def _require_torchaudio():
    try:
        torchaudio = pytest.importorskip("torchaudio")
    except OSError as exc:
        pytest.skip(f"torchaudio extension unavailable: {exc}")
    try:
        _ = torchaudio.compliance.kaldi
    except OSError as exc:
        pytest.skip(f"torchaudio extension unavailable: {exc}")
    return torchaudio


def test_eat_model_builds():
    _require_torchaudio()
    model = AudioModel(make_eat_base_config())
    assert len(model.encoder.blocks) == 12

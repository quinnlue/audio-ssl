"""The pretrained HuBERT teacher is pinned by config and never checkpointed."""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch
import transformers

from audiossl.config import HuBERTTeacherConfig
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.objectives.hubert.teacher import PretrainedHuBERTTokenizer

NUM_CODES = 4


def _tiny_hubert() -> transformers.HubertModel:
    torch.manual_seed(0)
    return transformers.HubertModel(
        transformers.HubertConfig(
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(8, 8),
            conv_stride=(5, 2),
            conv_kernel=(10, 3),
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
        )
    ).eval()


@pytest.fixture
def offline_teacher(tmp_path, monkeypatch):
    """Teacher factory that never touches the network; records from_pretrained calls."""
    artifact = tmp_path / "km.bin"
    joblib.dump(SimpleNamespace(cluster_centers_=np.eye(NUM_CODES, 8, dtype=np.float32)), artifact)
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setenv("AUDIOSSL_CACHE_DIR", str(tmp_path))

    calls: list[dict] = []

    def from_pretrained(encoder_id, **kwargs):
        calls.append({"encoder_id": encoder_id, **kwargs})
        return _tiny_hubert()

    monkeypatch.setattr(transformers.HubertModel, "from_pretrained", from_pretrained)

    def build(**overrides) -> PretrainedHuBERTTokenizer:
        config = HuBERTTeacherConfig(
            feature_layer=1,
            num_codes=NUM_CODES,
            km_url="https://example.invalid/km.bin",
            km_sha256=sha256,
            **overrides,
        )
        return PretrainedHuBERTTokenizer(config)

    build.calls = calls
    return build


def test_encoder_revision_is_passed_to_from_pretrained(offline_teacher):
    offline_teacher(encoder_revision="abc123")
    assert offline_teacher.calls[-1]["revision"] == "abc123"


def test_unpinned_revision_warns(offline_teacher, caplog):
    with caplog.at_level(logging.WARNING):
        offline_teacher()
    assert "encoder_revision" in caplog.text


def test_frozen_teacher_is_excluded_from_state_dict_and_loads_strictly(offline_teacher):
    teacher = offline_teacher(encoder_revision="abc123")
    assert teacher.state_dict() == {}

    wrapper = torch.nn.ModuleDict({"teacher": teacher, "head": torch.nn.Linear(2, 2)})
    state = wrapper.state_dict()
    assert set(state) == {"head.weight", "head.bias"}
    before = {k: v.clone() for k, v in teacher.encoder.state_dict().items()}
    wrapper.load_state_dict(state, strict=True)
    for key, value in teacher.encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_teacher_labels_are_frame_aligned(offline_teacher):
    teacher = offline_teacher(encoder_revision="abc123")
    x = torch.randn(2, 400)
    mask = torch.ones_like(x, dtype=torch.bool)
    labels = teacher(AudioBatch(x=x, mask=mask))
    expected = int(teacher.output_lengths(torch.tensor([400]))[0])
    assert labels.shape == (2, expected)
    assert labels.dtype == torch.long
    assert labels.min() >= 0 and labels.max() < NUM_CODES

"""Tests for the SFT classification objective."""

from __future__ import annotations

import torch

from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives.sft import SFTObjective
from testkit import batches, configs


def test_sft_forward_computes_loss_and_logits() -> None:
    from dataclasses import replace

    model = AudioModel(configs.model())
    objective = SFTObjective(model, configs.sft_objective(num_classes=4))
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    batch = replace(batches.waveform_batch(configs.DEFAULT_LENGTHS, seed=0), labels=labels)

    output = objective(batch)

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (3, 4)
    assert output.accuracy is not None


def test_sft_missing_labels_raises() -> None:
    model = AudioModel(configs.model())
    objective = SFTObjective(model, configs.sft_objective())
    batch = batches.waveform_batch(configs.DEFAULT_LENGTHS)
    try:
        objective(batch)
    except ValueError as exc:
        assert "labels" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_sft_freeze_backbone_leaves_head_trainable() -> None:
    model = AudioModel(configs.model())
    objective = SFTObjective(model, configs.sft_objective(freeze_backbone=True))
    backbone_trainable = any(
        p.requires_grad for n, p in objective.named_parameters() if n.startswith("model.")
    )
    assert not backbone_trainable
    assert any(p.requires_grad for n, p in objective.named_parameters() if n.startswith("head."))
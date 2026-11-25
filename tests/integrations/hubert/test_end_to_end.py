from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import (
    HuBERTObjectiveConfig,
    HuBERTTeacherConfig,
    RuntimeConfig,
    SpanMaskingConfig,
)
from audiossl.config.batch_dataclasses import AudioBatch, HuBERTOutput
from audiossl.integrations.hubert import make_hubert_config_from_hf, port_hubert_state_dict
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives.hubert.hubert import HuBERTObjective

from testkit.hf_backbone import compare_hf_backbone


class TinyHuBERTKMeansTeacher(nn.Module):
    def __init__(self, encoder: nn.Module, centers: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder.eval()
        self.register_buffer("centers", centers)

    @torch.no_grad()
    def forward(self, batch: AudioBatch) -> torch.Tensor:
        outputs = self.encoder(
            batch.x,
            attention_mask=batch.mask.to(dtype=torch.long),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        centers = self.centers.to(device=hidden.device, dtype=hidden.dtype)
        distances = torch.cdist(hidden, centers.unsqueeze(0).expand(hidden.shape[0], -1, -1))
        return distances.argmin(dim=-1).to(dtype=torch.long)


def _build_ported_model(hf_model, hf_config) -> AudioModel:
    model_config = make_hubert_config_from_hf(
        hf_config,
        runtime=RuntimeConfig(seed=7, deterministic=True),
    )
    model = AudioModel(model_config).to(next(hf_model.parameters()).dtype)
    port_hubert_state_dict(hf_model.state_dict(), model)
    return model


def _manual_loss(
    out: HuBERTOutput,
    *,
    masked_weight: float,
    unmasked_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = out.mask & out.time_mask_indices.to(out.mask.device)
    unmasked = out.mask & ~out.time_mask_indices.to(out.mask.device)
    masked_loss = (
        F.cross_entropy(out.logits[masked], out.targets[masked])
        if masked.any()
        else out.logits.new_zeros(())
    )
    unmasked_loss = (
        F.cross_entropy(out.logits[unmasked], out.targets[unmasked])
        if unmasked.any()
        else out.logits.new_zeros(())
    )
    loss = out.logits.new_zeros(())
    sample_size = out.logits.new_zeros(())
    if masked_weight > 0.0:
        count = masked.sum().to(dtype=out.logits.dtype)
        loss = loss + masked_weight * masked_loss * count
        sample_size = sample_size + count
    if unmasked_weight > 0.0:
        count = unmasked.sum().to(dtype=out.logits.dtype)
        loss = loss + unmasked_weight * unmasked_loss * count
        sample_size = sample_size + count
    return loss / sample_size, masked_loss, unmasked_loss


def test_backbone_training_matches_hf(
    hf_hubert, hf_tiny_hubert_config, parity_dtype, parity_precision,
):
    torch.manual_seed(0)
    hf_tiny_hubert_config = deepcopy(hf_tiny_hubert_config)
    hf_tiny_hubert_config.layer_norm_eps = parity_precision.eps
    ref_model = hf_hubert.modeling.HubertModel(hf_tiny_hubert_config).to(parity_dtype)
    our_model = _build_ported_model(ref_model, hf_tiny_hubert_config).eval()
    compare_hf_backbone(
        our_model, ref_model, hf_tiny_hubert_config, samples=320,
        dtype=parity_dtype, precision=parity_precision,
    )


def test_tiny_hubert_objective_loss_and_latents_end_to_end(
    hf_hubert,
    hf_tiny_hubert_config,
):
    torch.manual_seed(1)
    ref_model = hf_hubert.modeling.HubertModel(hf_tiny_hubert_config).eval()
    student = _build_ported_model(ref_model, hf_tiny_hubert_config)
    teacher = TinyHuBERTKMeansTeacher(
        ref_model,
        centers=torch.randn(7, hf_tiny_hubert_config.hidden_size),
    )
    config = HuBERTObjectiveConfig(
        teacher=HuBERTTeacherConfig(num_codes=7, encoder_revision="0" * 40),
        time_masking=SpanMaskingConfig(mask_prob=0.35, mask_length=2),
        final_dim=16,
        masked_loss_weight=1.0,
        unmasked_loss_weight=0.25,
    )
    objective = HuBERTObjective(
        student,
        config,
        generator=torch.Generator().manual_seed(2),
        teacher=teacher,
    )
    objective.train()

    batch = AudioBatch(
        x=torch.randn(2, 512),
        mask=torch.ones(2, 512, dtype=torch.bool),
    )
    out = objective(batch)
    expected_loss, expected_masked, expected_unmasked = _manual_loss(
        out,
        masked_weight=config.masked_loss_weight,
        unmasked_weight=config.unmasked_loss_weight,
    )

    assert isinstance(out, HuBERTOutput)
    assert out.z.shape[:2] == out.mask.shape == out.targets.shape
    assert out.logits.shape == (*out.targets.shape, 7)
    assert torch.isfinite(out.loss)
    torch.testing.assert_close(out.loss, expected_loss)
    torch.testing.assert_close(out.masked_loss, expected_masked)
    torch.testing.assert_close(out.unmasked_loss, expected_unmasked)

    out.loss.backward()
    assert objective.mask_emb.grad is not None
    assert all(param.grad is None for param in objective.teacher.parameters())

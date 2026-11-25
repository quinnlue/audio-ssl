"""HuBERT-specific behaviour: loss parity with fairseq's HubertCriterion."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from testkit.cases import objective_case


def _fairseq_nce_ce_sum(class_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.numel() == 0:
        return class_logits.new_zeros(())
    pos = class_logits.gather(1, targets.unsqueeze(1))
    negs = class_logits.clone()
    negs[torch.arange(targets.numel()), targets] = float("-inf")
    logits = torch.cat([pos, negs], dim=1)
    return F.cross_entropy(logits, targets.new_zeros(targets.shape), reduction="sum")


def _canonical_fairseq_hubert_loss(*, logits, targets, valid_mask, mask_indices, wm, wu):
    """fairseq ``HubertCriterion`` predictive-loss branch with ``loss_weights=None``."""
    masked = valid_mask & mask_indices
    unmasked = valid_mask & ~mask_indices
    masked_sum = _fairseq_nce_ce_sum(logits[masked], targets[masked])
    unmasked_sum = _fairseq_nce_ce_sum(logits[unmasked], targets[unmasked])
    loss = logits.new_zeros(())
    sample_size = logits.new_zeros(())
    if wm > 0.0:
        loss = loss + wm * masked_sum
        sample_size = sample_size + masked.sum()
    if wu > 0.0:
        loss = loss + wu * unmasked_sum
        sample_size = sample_size + unmasked.sum()
    return loss / sample_size, masked_sum / masked.sum(), unmasked_sum / unmasked.sum()


def test_loss_matches_canonical_fairseq_hubert_criterion():
    base = objective_case("hubert")
    case = replace(base, objective=lambda: replace(base.objective(), unmasked_loss_weight=0.5))
    config = case.objective_config()
    out = case.build().train()(case.batch())

    expected, expected_masked, expected_unmasked = _canonical_fairseq_hubert_loss(
        logits=out.logits,
        targets=out.targets,
        valid_mask=out.mask,
        mask_indices=out.time_mask_indices,
        wm=config.masked_loss_weight,
        wu=config.unmasked_loss_weight,
    )
    torch.testing.assert_close(out.masked_loss, expected_masked)
    torch.testing.assert_close(out.unmasked_loss, expected_unmasked)
    torch.testing.assert_close(out.loss, expected)
    assert out.logits.shape[-1] == config.teacher.num_codes
    assert out.z.shape[:2] == out.mask.shape == out.targets.shape

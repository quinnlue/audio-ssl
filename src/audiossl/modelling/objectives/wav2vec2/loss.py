"""wav2vec 2.0 contrastive and diversity loss helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Wav2Vec2LossOutput:
    """Scalar wav2vec 2.0 loss terms plus contrastive logits and targets."""

    contrastive_loss: torch.Tensor
    diversity_loss: torch.Tensor
    features_pen: torch.Tensor
    loss: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor


def sample_negatives(
    quantized_features: torch.Tensor,
    mask_time_indices: torch.Tensor,
    *,
    num_negatives: int,
    cross_sample_negatives: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample negative quantized targets for masked wav2vec 2.0 positions.

    Parameters
    ----------
    quantized_features
        Quantized target features with shape ``(B, N, D)``.
    mask_time_indices
        Objective-specific boolean mask with shape ``(B, N)``. Here ``True``
        means selected as a masked contrastive prediction position.

    Returns
    -------
    negatives
        Sampled negatives with shape ``(num_masked, K, D)``.
    indices
        Sample indices with shape ``(num_masked, K)``.
    """

    if quantized_features.ndim != 3:
        raise ValueError("quantized_features must be rank-3")
    if mask_time_indices.shape != quantized_features.shape[:2]:
        raise ValueError("mask_time_indices must match quantized batch/time shape")
    positives = quantized_features[mask_time_indices]
    total_masked = positives.shape[0]
    if total_masked == 0:
        raise ValueError("wav2vec2 contrastive loss requires at least one masked position")
    if num_negatives <= 0 and cross_sample_negatives <= 0:
        empty = positives.new_empty((total_masked, 0, positives.shape[-1]))
        return empty, torch.empty((total_masked, 0), dtype=torch.long, device=positives.device)

    per_sample = []
    indices = []
    batch, _, _ = quantized_features.shape
    flat = quantized_features.reshape(-1, quantized_features.shape[-1])
    for batch_index in range(batch):
        row_positions = torch.nonzero(mask_time_indices[batch_index], as_tuple=False).flatten()
        if row_positions.numel() == 0:
            continue
        row_features = quantized_features[batch_index, row_positions]
        for local_index in range(row_positions.numel()):
            choices = []
            choice_indices = []
            if num_negatives > 0:
                local_choices = torch.randint(
                    max(row_features.shape[0] - 1, 1),
                    (num_negatives,),
                    generator=generator,
                    device=quantized_features.device,
                )
                local_choices = local_choices + (local_choices >= local_index).long()
                local_choices = local_choices.clamp_max(row_features.shape[0] - 1)
                choices.append(row_features[local_choices])
                choice_indices.append(local_choices)
            if cross_sample_negatives > 0:
                flat_choices = torch.randint(
                    flat.shape[0],
                    (cross_sample_negatives,),
                    generator=generator,
                    device=quantized_features.device,
                )
                choices.append(flat[flat_choices])
                choice_indices.append(flat_choices)
            per_sample.append(torch.cat(choices, dim=0))
            indices.append(torch.cat(choice_indices, dim=0))
    return torch.stack(per_sample, dim=0), torch.stack(indices, dim=0)


def wav2vec2_contrastive_loss(
    encoder_features: torch.Tensor,
    quantized_features: torch.Tensor,
    mask_time_indices: torch.Tensor,
    *,
    num_negatives: int,
    logit_temp: float,
    cross_sample_negatives: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute wav2vec 2.0 contrastive loss over masked time positions.

    ``mask_time_indices`` is objective-specific: ``True`` means a masked
    prediction position. Returned logits have shape ``(num_masked, 1 + K)``,
    where column 0 is the positive target and the remaining columns are
    negatives.
    """

    if logit_temp <= 0:
        raise ValueError("logit_temp must be > 0")
    if encoder_features.shape != quantized_features.shape:
        raise ValueError(
            "encoder_features and quantized_features must have identical shape, "
            f"got {tuple(encoder_features.shape)} and {tuple(quantized_features.shape)}"
        )
    targets_features = quantized_features[mask_time_indices]
    predicted = encoder_features[mask_time_indices]
    negatives, _ = sample_negatives(
        quantized_features,
        mask_time_indices,
        num_negatives=num_negatives,
        cross_sample_negatives=cross_sample_negatives,
        generator=generator,
    )
    candidates = torch.cat([targets_features.unsqueeze(1), negatives], dim=1)
    logits = F.cosine_similarity(predicted.unsqueeze(1).float(), candidates.float(), dim=-1)
    logits = logits / logit_temp
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, targets)
    return loss, logits, targets


def diversity_loss(
    code_perplexity: torch.Tensor,
    prob_perplexity: torch.Tensor,
    num_vars: int,
) -> torch.Tensor:
    """Return the quantizer diversity penalty as a scalar tensor."""

    del code_perplexity
    return (float(num_vars) - prob_perplexity) / float(num_vars)


def feature_penalty(raw_features: torch.Tensor) -> torch.Tensor:
    """Return the wav2vec 2.0 feature penalty over raw processor features."""

    return raw_features.float().pow(2).mean()


__all__ = [
    "Wav2Vec2LossOutput",
    "diversity_loss",
    "feature_penalty",
    "sample_negatives",
    "wav2vec2_contrastive_loss",
]

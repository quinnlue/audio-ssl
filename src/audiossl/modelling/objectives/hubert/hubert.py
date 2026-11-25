"""HuBERT objective over teacher token targets."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import HuBERTObjectiveConfig
from audiossl.config.batch_dataclasses import AudioBatch, HuBERTOutput, ObjectiveOutput
from audiossl.modelling.core.initialization import initialize_linear
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope

from ...model import AudioModel
from ...registry import get_masking_strategy_registry
from .._base import Objective
from ..components import encode_with_mask_embedding, validate_waveform_batch
from .teacher import PretrainedHuBERTTokenizer

logger = logging.getLogger(__name__)

_ALIGNMENT_PROBE_LENGTHS = (16_000, 16_123, 32_000, 48_517)


class HuBERTObjective(Objective):
    """HuBERT masked prediction objective with a frozen tokenizer teacher.

    The objective owns time and optional feature masking, mask embeddings,
    projection and label heads, teacher label lookup, masked/unmasked losses,
    and metrics. Public ``forward`` accepts raw ``AudioBatch`` inputs with
    waveform and boolean validity mask shape ``(B, S)``.

    Teachers that expose ``output_lengths(sample_lengths)`` are checked at
    construction against the student processor's frame counts, so a frame-rate
    mismatch fails before training instead of on the first batch.
    """

    uses_generator = True

    def __init__(
        self,
        model: AudioModel,
        config: HuBERTObjectiveConfig,
        *,
        generator: torch.Generator | None = None,
        teacher: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(f"HuBERTObjective expected AudioModel, got {type(model).__name__}")
        if not isinstance(config, HuBERTObjectiveConfig):
            raise TypeError(
                "HuBERTObjective expected HuBERTObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        if generator is None:
            raise ValueError("HuBERTObjective requires a non-None generator")
        self.register_generator("generator", generator)
        self.generator = generator
        self.teacher = teacher if teacher is not None else PretrainedHuBERTTokenizer(config.teacher)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        with runtime_seed_scope(model.config.runtime, "objective/hubert"):
            self.mask_emb = nn.Parameter(
                torch.empty(model.config.d_model).uniform_(
                    -config.mask_emb_init_scale,
                    config.mask_emb_init_scale,
                )
            )
            self.project_hid = nn.Linear(model.config.d_model, config.final_dim)
            self.label_emb = nn.Parameter(
                torch.empty(config.teacher.num_codes, config.final_dim).uniform_(
                    -config.label_emb_init_scale,
                    config.label_emb_init_scale,
                )
            )
            initialize_linear(self.project_hid, config.initialization)
            registry = get_masking_strategy_registry()
            self.time_masking = registry[config.time_masking.type](config=config.time_masking)
            self.feature_masking = (
                registry[config.feature_masking.type](config=config.feature_masking)
                if config.feature_masking is not None
                else None
            )

        apply_runtime_to_module(self, model.applied_runtime)
        self._check_frame_rate_alignment()

    def forward(self, batch: AudioBatch) -> HuBERTOutput:
        """Run HuBERT masked classification and return objective output.

        ``batch.mask`` follows the default convention: ``True`` means valid.
        ``time_mask_indices`` in the output is objective-specific: ``True``
        means selected as a masked prediction position.
        """

        validate_waveform_batch(batch, owner="HuBERTObjective")
        with torch.no_grad():
            targets = self.teacher(batch)

        encoded = encode_with_mask_embedding(
            self.model,
            batch,
            mask_emb=self.mask_emb,
            time_masking=self.time_masking,
            feature_masking=self.feature_masking,
            generator=self.generator,
            feature_grad_mult=self.config.feature_grad_mult,
            training=self.training,
            owner="HuBERTObjective",
        )
        if targets.shape != encoded.time_mask.shape:
            raise ValueError(
                "HuBERT teacher/student frame-rate mismatch in forward: "
                f"teacher labels {tuple(targets.shape)}, "
                f"student tokens {tuple(encoded.time_mask.shape)}"
            )
        encoder_out = encoded.encoder_out
        time_mask_indices = encoded.time_mask
        logits = self.project_hid(encoder_out.z) @ self.label_emb.t()
        logits = logits / self.config.logit_temp
        targets = targets.to(device=logits.device)
        valid_mask = encoder_out.mask
        masked_selector = time_mask_indices.to(valid_mask.device) & valid_mask
        unmasked_selector = (~time_mask_indices.to(valid_mask.device)) & valid_mask
        masked_loss = self._ce_loss(logits, targets, masked_selector)
        unmasked_loss = self._ce_loss(logits, targets, unmasked_selector)
        loss = self._weighted_fairseq_loss(
            masked_loss=masked_loss,
            unmasked_loss=unmasked_loss,
            masked_count=masked_selector.sum(),
            unmasked_count=unmasked_selector.sum(),
        )
        entropy, unique_codes = self._target_stats(targets, valid_mask)
        return HuBERTOutput(
            loss=loss,
            z=encoder_out.z,
            mask=encoder_out.mask,
            logits=logits,
            targets=targets,
            time_mask_indices=time_mask_indices,
            masked_loss=masked_loss,
            unmasked_loss=unmasked_loss,
            target_entropy=entropy,
            unique_codes=unique_codes,
        )

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return HuBERT loss and target diagnostics."""

        del ctx
        if not isinstance(output, HuBERTOutput):
            raise TypeError(
                f"HuBERTObjective.metrics expected HuBERTOutput, got {type(output).__name__}"
            )
        return {
            "masked_loss": output.masked_loss.detach(),
            "unmasked_loss": output.unmasked_loss.detach(),
            "mask_coverage": output.time_mask_indices.float().mean(),
            "target_entropy": output.target_entropy.detach(),
            "unique_codes": output.unique_codes.detach(),
        }

    def train(self, mode: bool = True):
        """Set objective mode while keeping the frozen teacher in eval mode."""

        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def _ce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        selector: torch.Tensor,
    ) -> torch.Tensor:
        if not selector.any():
            return logits.new_zeros(())
        return F.cross_entropy(logits[selector], targets[selector], reduction="mean")

    def _weighted_fairseq_loss(
        self,
        *,
        masked_loss: torch.Tensor,
        unmasked_loss: torch.Tensor,
        masked_count: torch.Tensor,
        unmasked_count: torch.Tensor,
    ) -> torch.Tensor:
        loss_sum = masked_loss.new_zeros(())
        sample_size = masked_loss.new_zeros(())
        if self.config.masked_loss_weight > 0.0:
            count = masked_count.to(device=masked_loss.device, dtype=masked_loss.dtype)
            loss_sum = loss_sum + self.config.masked_loss_weight * masked_loss * count
            sample_size = sample_size + count
        if self.config.unmasked_loss_weight > 0.0:
            count = unmasked_count.to(device=unmasked_loss.device, dtype=unmasked_loss.dtype)
            loss_sum = loss_sum + self.config.unmasked_loss_weight * unmasked_loss * count
            sample_size = sample_size + count
        if sample_size.item() == 0:
            raise ValueError("HuBERTObjective requires at least one weighted target frame")
        return loss_sum / sample_size

    @staticmethod
    def _target_stats(
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = targets[valid_mask]
        if selected.numel() == 0:
            return (
                targets.new_zeros((), dtype=torch.float32),
                targets.new_zeros((), dtype=torch.float32),
            )
        counts = torch.bincount(selected, minlength=int(targets.max().item()) + 1).float()
        probs = counts[counts > 0] / counts.sum()
        entropy = -(probs * torch.log(probs)).sum()
        return entropy, torch.tensor(float(probs.numel()), device=targets.device)

    def _check_frame_rate_alignment(self) -> None:
        sample_lengths = torch.tensor(_ALIGNMENT_PROBE_LENGTHS, dtype=torch.long)
        student = self.model.tokenizer.output_lengths(
            self.model.processor.output_lengths(sample_lengths)
        )
        if torch.any(student <= 0):
            raise ValueError(
                "HuBERTObjective student pipeline produced no tokens for probe lengths "
                f"{list(_ALIGNMENT_PROBE_LENGTHS)}: {student.tolist()}"
            )
        teacher_lengths = getattr(self.teacher, "output_lengths", None)
        if teacher_lengths is None:
            logger.warning(
                "HuBERT teacher %s does not expose output_lengths; frame-rate alignment is "
                "only checked per batch in forward",
                type(self.teacher).__name__,
            )
            return
        teacher = teacher_lengths(sample_lengths).to(student.device)
        if not torch.equal(student.to(torch.long), teacher.to(torch.long)):
            raise ValueError(
                "HuBERT teacher and student frame rates disagree for sample lengths "
                f"{list(_ALIGNMENT_PROBE_LENGTHS)}: teacher labels {teacher.tolist()}, "
                f"student tokens {student.tolist()}"
            )


__all__ = ["HuBERTObjective"]

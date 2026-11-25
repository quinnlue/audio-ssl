"""DINO objective with student and teacher global-view paths."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from audiossl.config import DINOObjectiveConfig
from audiossl.config.batch_dataclasses import (
    AudioBatch,
    DINOOutput,
    EncoderOutput,
    ObjectiveOutput,
)
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope
from ..protocols import AugmentationPipeline

from ...model import AudioModel
from .._base import Objective
from ..components import ema_update_, frozen_copy, validate_waveform_batch
from ..contrastive.views import create_contrastive_views
from .head import DINOHead
from .loss import dino_cross_entropy_loss

class DINOObjective(Objective):
    """Two-global-view DINO objective with teacher path and running center.

    The objective owns augmented views, student and teacher heads, DINO
    cross-view loss, center updates, EMA updates when configured, and metrics.
    Public ``forward`` accepts raw ``AudioBatch`` inputs with waveform and
    boolean validity mask shape ``(B, S)``.
    """

    uses_augmentor = True

    def __init__(
        self,
        model: AudioModel,
        config: DINOObjectiveConfig,
        *,
        augmentor: AugmentationPipeline | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"DINOObjective.__init__ expected AudioModel, got {type(model).__name__}"
            )
        if not isinstance(config, DINOObjectiveConfig):
            raise TypeError(
                "DINOObjective.__init__ expected DINOObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        if augmentor is None:
            raise ValueError("DINOObjective requires a non-None augmentor")
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        self.augmentor = augmentor

        with runtime_seed_scope(model.config.runtime, "objective/dino"):
            self.head = DINOHead(input_dim=model.config.d_model, config=config.head)

        self.register_buffer("center", torch.zeros(config.head.num_prototypes))

        if config.teacher_mode == "ema":
            self.teacher_model: AudioModel | None = frozen_copy(model)
            # weight_norm heads cannot be deep-copied; rebuild under a private
            # seed scope so the caller's RNG is untouched, then copy weights.
            with runtime_seed_scope(model.config.runtime, "objective/dino/teacher_head"):
                teacher_head = DINOHead(input_dim=model.config.d_model, config=config.head)
            teacher_head.load_state_dict(self.head.state_dict())
            teacher_head.eval()
            for parameter in teacher_head.parameters():
                parameter.requires_grad_(False)
            self.teacher_head: DINOHead | None = teacher_head
        elif config.teacher_mode == "stop_grad":
            self.teacher_model = None
            self.teacher_head = None
        else:
            raise ValueError(
                "Unsupported DINOObjectiveConfig.teacher_mode "
                f"{config.teacher_mode!r}; expected 'ema' or 'stop_grad'"
            )

        apply_runtime_to_module(self, model.applied_runtime)

    def forward(self, batch: AudioBatch) -> DINOOutput:
        """Run student and teacher views and return DINO loss output.

        ``batch.mask`` uses the default convention: ``True`` means a valid raw
        waveform sample. Student and teacher logits have shape
        ``(B, num_prototypes)``; ``teacher_logits_uncentered`` has shape
        ``(2B, num_prototypes)`` for centering in :meth:`on_after_forward`.
        """

        validate_waveform_batch(batch, owner="DINOObjective")

        views = create_contrastive_views(
            batch,
            strategy="augmentation",
            augmentor=self.augmentor,
        )

        out_i: EncoderOutput = self.model.pipeline(views.batch_i)
        out_j: EncoderOutput = self.model.pipeline(views.batch_j)
        cls_i = self.model.get_cls(out_i)
        cls_j = self.model.get_cls(out_j)
        student_logits_i = self.head(cls_i)
        student_logits_j = self.head(cls_j)

        if self.teacher_model is not None:
            teacher_logits_i = self._ema_teacher_logits(views.batch_i)
            teacher_logits_j = self._ema_teacher_logits(views.batch_j)
        else:
            teacher_logits_i, teacher_logits_j = self._stop_grad_teacher_logits(
                views.batch_i,
                views.batch_j,
            )

        loss_out = dino_cross_entropy_loss(
            student_logits_i,
            student_logits_j,
            teacher_logits_i,
            teacher_logits_j,
            self.center,
            teacher_temperature=self.config.teacher_temperature,
            student_temperature=self.config.student_temperature,
        )
        teacher_logits_uncentered = torch.cat(
            [teacher_logits_i.detach(), teacher_logits_j.detach()],
            dim=0,
        )

        return DINOOutput(
            loss=loss_out.loss,
            student_logits_i=student_logits_i,
            student_logits_j=student_logits_j,
            teacher_logits_i=teacher_logits_i.detach(),
            teacher_logits_j=teacher_logits_j.detach(),
            teacher_logits_uncentered=teacher_logits_uncentered,
            cls_i=cls_i,
            cls_j=cls_j,
            mask_i=out_i.mask,
            mask_j=out_j.mask,
        )

    def on_after_forward(self, output: ObjectiveOutput, ctx: Any = None) -> None:
        """Update the running center from uncentered teacher logits."""
        del ctx

        if not isinstance(output, DINOOutput):
            raise TypeError(
                f"DINOObjective.on_after_forward expected DINOOutput, got {type(output).__name__}"
            )
        if output.teacher_logits_uncentered.ndim != 2:
            raise ValueError(
                "DINOObjective.on_after_forward expected rank-2 teacher_logits_uncentered, "
                f"got shape {tuple(output.teacher_logits_uncentered.shape)}"
            )
        if output.teacher_logits_uncentered.shape[-1] != self.center.shape[0]:
            raise ValueError(
                "DINOObjective.on_after_forward teacher logits prototype dimension "
                f"{output.teacher_logits_uncentered.shape[-1]} does not match center "
                f"{self.center.shape[0]}"
            )
        with torch.no_grad():
            batch_center = output.teacher_logits_uncentered.detach().mean(dim=0)
            batch_center = batch_center.to(device=self.center.device, dtype=self.center.dtype)
            momentum = float(self.config.center_momentum)
            self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return DINO entropy, center, and logit stats."""

        del ctx
        if not isinstance(output, DINOOutput):
            raise TypeError(
                f"DINOObjective.metrics expected DINOOutput, got {type(output).__name__}"
            )
        student_logits = torch.cat(
            [output.student_logits_i.detach(), output.student_logits_j.detach()],
            dim=0,
        )
        teacher_logits = output.teacher_logits_uncentered.detach()
        teacher_probs = F.softmax(
            (teacher_logits - self.center.detach().view(1, -1))
            / float(self.config.teacher_temperature),
            dim=-1,
        )
        student_probs = F.softmax(
            student_logits / float(self.config.student_temperature),
            dim=-1,
        )
        teacher_entropy = -(teacher_probs * teacher_probs.clamp_min(1e-12).log()).sum(
            dim=-1
        ).mean()
        student_entropy = -(student_probs * student_probs.clamp_min(1e-12).log()).sum(
            dim=-1
        ).mean()
        return {
            "center_norm": self.center.detach().norm(),
            "student_entropy": student_entropy,
            "teacher_entropy": teacher_entropy,
            "student_logit_std": student_logits.std(unbiased=False),
            "teacher_logit_std": teacher_logits.std(unbiased=False),
        }

    def on_after_optimizer_step(self, ctx: Any = None) -> None:
        if ctx is not None and getattr(ctx, "step_was_skipped", False):
            return
        self.update_teacher()

    def update_teacher(self) -> None:
        """Update EMA encoder and DINO head after an optimizer step."""

        if self.teacher_model is None or self.teacher_head is None:
            return
        decay = float(self.config.teacher_ema_decay)
        ema_update_(self.teacher_model, self.model, decay=decay)
        ema_update_(self.teacher_head, self.head, decay=decay)

    def _ema_teacher_logits(self, batch: AudioBatch) -> torch.Tensor:
        teacher_model = self.teacher_model
        teacher_head = self.teacher_head
        if teacher_model is None or teacher_head is None:
            raise RuntimeError("DINOObjective._ema_teacher_logits requires EMA teacher")
        with torch.no_grad():
            teacher_model.eval()
            teacher_head.eval()
            out = teacher_model.pipeline(batch)
            return teacher_head(teacher_model.get_cls(out)).detach()

    def _stop_grad_teacher_logits(
        self,
        batch_i: AudioBatch,
        batch_j: AudioBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_model_training = self.model.training
        was_head_training = self.head.training
        try:
            self.model.eval()
            self.head.eval()
            with torch.no_grad():
                out_i = self.model.pipeline(batch_i)
                out_j = self.model.pipeline(batch_j)
                logits_i = self.head(self.model.get_cls(out_i)).detach()
                logits_j = self.head(self.model.get_cls(out_j)).detach()
        finally:
            self.model.train(was_model_training)
            self.head.train(was_head_training)
        return logits_i, logits_j


__all__ = ["DINOObjective"]

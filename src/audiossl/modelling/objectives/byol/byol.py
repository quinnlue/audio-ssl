"""BYOL objective with student prediction and target teacher paths."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from audiossl.config import BYOLObjectiveConfig
from audiossl.config.batch_dataclasses import (
    AudioBatch,
    BYOLOutput,
    EncoderOutput,
    ObjectiveOutput,
)
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope
from ..protocols import AugmentationPipeline

from ...model import AudioModel
from .._base import Objective
from ..components import ema_update_, frozen_copy, validate_waveform_batch
from ..contrastive.projection import ProjectionHead
from ..contrastive.views import create_contrastive_views
from .loss import byol_regression_loss

class BYOLObjective(Objective):
    """Bootstrap Your Own Latent objective over two augmented waveform views.

    The objective owns augmentation views, projector and student-only predictor
    heads, teacher or stop-grad target paths, BYOL regression loss, EMA updates,
    and metrics. Public ``forward`` accepts raw ``AudioBatch`` inputs with
    waveform and boolean validity mask shape ``(B, S)``.
    """

    uses_augmentor = True

    def __init__(
        self,
        model: AudioModel,
        config: BYOLObjectiveConfig,
        *,
        augmentor: AugmentationPipeline | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"BYOLObjective.__init__ expected AudioModel, got {type(model).__name__}"
            )
        if not isinstance(config, BYOLObjectiveConfig):
            raise TypeError(
                "BYOLObjective.__init__ expected BYOLObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        if augmentor is None:
            raise ValueError("BYOLObjective requires a non-None augmentor")
        if config.projector.final_l2_normalize:
            raise ValueError("BYOLObjective projector.final_l2_normalize must be False")
        if config.predictor.final_l2_normalize:
            raise ValueError("BYOLObjective predictor.final_l2_normalize must be False")
        if config.predictor.output_dim != config.projector.output_dim:
            raise ValueError(
                "BYOLObjective predictor.output_dim must match projector.output_dim, "
                f"got {config.predictor.output_dim} and {config.projector.output_dim}"
            )
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        self.augmentor = augmentor

        with runtime_seed_scope(model.config.runtime, "objective/byol"):
            self.projector = ProjectionHead(
                input_dim=model.config.d_model,
                config=config.projector,
            )
            self.predictor = ProjectionHead(
                input_dim=config.projector.output_dim,
                config=config.predictor,
            )

        if config.teacher_mode == "ema":
            self.teacher_model: AudioModel | None = frozen_copy(model)
            self.teacher_projector: ProjectionHead | None = frozen_copy(self.projector)
        elif config.teacher_mode == "stop_grad":
            self.teacher_model = None
            self.teacher_projector = None
        else:
            raise ValueError(
                "Unsupported BYOLObjectiveConfig.teacher_mode "
                f"{config.teacher_mode!r}; expected 'ema' or 'stop_grad'"
            )

        apply_runtime_to_module(self, model.applied_runtime)

    def forward(self, batch: AudioBatch) -> BYOLOutput:
        """Run student and target paths and return BYOL loss output.

        ``batch.mask`` uses the default convention: ``True`` means a valid raw
        waveform sample. Output prediction, projection, and target tensors have
        shape ``(B, projection_dim)``.
        """

        validate_waveform_batch(batch, owner="BYOLObjective")

        views = create_contrastive_views(
            batch,
            strategy="augmentation",
            augmentor=self.augmentor,
        )

        out_i: EncoderOutput = self.model.pipeline(views.batch_i)
        out_j: EncoderOutput = self.model.pipeline(views.batch_j)
        cls_i = self.model.get_cls(out_i)
        cls_j = self.model.get_cls(out_j)

        projection_i = self.projector(cls_i)
        projection_j = self.projector(cls_j)
        prediction_i = self.predictor(projection_i)
        prediction_j = self.predictor(projection_j)

        if self.teacher_model is not None:
            target_i = self._ema_target_projection(views.batch_i)
            target_j = self._ema_target_projection(views.batch_j)
        else:
            target_i, target_j = self._stop_grad_target_projections(
                views.batch_i,
                views.batch_j,
            )

        loss = byol_regression_loss(
            prediction_i,
            target_j,
            prediction_j,
            target_i,
        )

        return BYOLOutput(
            loss=loss,
            prediction_i=prediction_i,
            prediction_j=prediction_j,
            target_i=target_i.detach(),
            target_j=target_j.detach(),
            projection_i=projection_i,
            projection_j=projection_j,
            cls_i=cls_i,
            cls_j=cls_j,
            mask_i=out_i.mask,
            mask_j=out_j.mask,
        )

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return BYOL norm and cosine diagnostics."""

        del ctx
        if not isinstance(output, BYOLOutput):
            raise TypeError(
                f"BYOLObjective.metrics expected BYOLOutput, got {type(output).__name__}"
            )
        predictions = torch.cat(
            [output.prediction_i.detach(), output.prediction_j.detach()],
            dim=0,
        )
        targets = torch.cat([output.target_i.detach(), output.target_j.detach()], dim=0)
        projections = torch.cat(
            [output.projection_i.detach(), output.projection_j.detach()],
            dim=0,
        )
        cosine = 0.5 * (
            F.cosine_similarity(output.prediction_i.detach(), output.target_j.detach(), dim=-1)
            .mean()
            + F.cosine_similarity(
                output.prediction_j.detach(),
                output.target_i.detach(),
                dim=-1,
            ).mean()
        )
        return {
            "prediction_norm": predictions.norm(dim=-1).mean(),
            "target_norm": targets.norm(dim=-1).mean(),
            "projection_norm": projections.norm(dim=-1).mean(),
            "cosine_similarity": cosine,
        }

    def on_after_optimizer_step(self, ctx: Any = None) -> None:
        if ctx is not None and getattr(ctx, "step_was_skipped", False):
            return
        self.update_teacher()

    def update_teacher(self) -> None:
        """Update EMA encoder and projector; BYOL never has a teacher predictor."""

        if self.teacher_model is None or self.teacher_projector is None:
            return
        decay = float(self.config.teacher_ema_decay)
        ema_update_(self.teacher_model, self.model, decay=decay)
        ema_update_(self.teacher_projector, self.projector, decay=decay)

    def _ema_target_projection(self, batch: AudioBatch) -> torch.Tensor:
        teacher_model = self.teacher_model
        teacher_projector = self.teacher_projector
        if teacher_model is None or teacher_projector is None:
            raise RuntimeError("BYOLObjective._ema_target_projection requires EMA teacher")
        with torch.no_grad():
            teacher_model.eval()
            teacher_projector.eval()
            out = teacher_model.pipeline(batch)
            return teacher_projector(teacher_model.get_cls(out)).detach()

    def _stop_grad_target_projections(
        self,
        batch_i: AudioBatch,
        batch_j: AudioBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_model_training = self.model.training
        was_projector_training = self.projector.training
        try:
            self.model.eval()
            self.projector.eval()
            with torch.no_grad():
                out_i = self.model.pipeline(batch_i)
                out_j = self.model.pipeline(batch_j)
                target_i = self.projector(self.model.get_cls(out_i)).detach()
                target_j = self.projector(self.model.get_cls(out_j)).detach()
        finally:
            self.model.train(was_model_training)
            self.projector.train(was_projector_training)
        return target_i, target_j


__all__ = ["BYOLObjective"]

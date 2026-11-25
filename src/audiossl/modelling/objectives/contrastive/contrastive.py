"""Contrastive objective for paired audio representation views."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from audiossl.config import ContrastiveObjectiveConfig
from audiossl.config.batch_dataclasses import (
    AudioBatch,
    ContrastiveOutput,
    EncoderOutput,
    ObjectiveOutput,
)
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope
from ..protocols import AugmentationPipeline

from ...model import AudioModel
from .._base import Objective
from ..components import validate_waveform_batch
from .loss import contrastive_info_nce_loss
from .projection import ProjectionHead
from .views import create_contrastive_views

class ContrastiveObjective(Objective):
    """SimCLR / SimCSE-style contrastive SSL objective.

    This objective owns training-time view creation, the projection head,
    InfoNCE loss, and contrastive metrics. The public ``forward`` boundary
    accepts only raw ``AudioBatch`` instances whose waveform and validity mask
    have shape ``(B, S)``.
    """

    uses_augmentor = True

    def __init__(
        self,
        model: AudioModel,
        config: ContrastiveObjectiveConfig,
        *,
        augmentor: AugmentationPipeline | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"ContrastiveObjective.__init__ expected AudioModel, got {type(model).__name__}"
            )
        if not isinstance(config, ContrastiveObjectiveConfig):
            raise TypeError(
                f"ContrastiveObjective.__init__ expected ContrastiveObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        if config.view_strategy == "augmentation" and augmentor is None:
            raise ValueError(
                "ContrastiveObjective: view_strategy='augmentation' requires a non-None augmentor"
            )
        if config.view_strategy == "dropout" and augmentor is not None:
            raise ValueError(
                "ContrastiveObjective: view_strategy='dropout' forbids passing an augmentor; "
                "the two views must be produced by stochastic model dropout"
            )
        if config.view_strategy == "dropout":
            self.eval_requires_train_mode = True
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        self.augmentor = augmentor
        with runtime_seed_scope(model.config.runtime, "objective/contrastive"):
            self.projection = ProjectionHead(
                input_dim=model.config.d_model,
                config=config.projection,
            )
        apply_runtime_to_module(self.projection, model.applied_runtime)

    def forward(self, batch: AudioBatch) -> ContrastiveOutput:
        """Run two model views and return contrastive loss output.

        Parameters
        ----------
        batch
            Raw waveform ``AudioBatch``. ``batch.mask`` is boolean with shape
            ``(B, S)`` and ``True`` means valid.

        Returns
        -------
        ContrastiveOutput
            Scalar loss, projected view states ``zi`` and ``zj`` with shape
            ``(B, projection_dim)``, pooled CLS states ``(B, d_model)``, and
            token masks from both encoded views.
        """

        validate_waveform_batch(batch, owner="ContrastiveObjective")
        if self.config.view_strategy == "dropout" and not self.training:
            raise RuntimeError(
                "ContrastiveObjective: view_strategy='dropout' requires .train() mode; "
                "in .eval() mode dropout is disabled and both views would be identical"
            )

        views = create_contrastive_views(
            batch,
            strategy=self.config.view_strategy,
            augmentor=self.augmentor,
        )

        out_i: EncoderOutput = self.model.pipeline(views.batch_i)
        out_j: EncoderOutput = self.model.pipeline(views.batch_j)

        cls_i = self.model.get_cls(out_i)
        cls_j = self.model.get_cls(out_j)

        zi = self.projection(cls_i)
        zj = self.projection(cls_j)

        loss_out = contrastive_info_nce_loss(
            zi,
            zj,
            temperature=self.config.temperature,
            symmetric=self.config.symmetric,
        )

        stats = (
            self._compute_stats(zi, zj, loss_out.logits)
            if self.config.compute_stats
            else {}
        )

        return ContrastiveOutput(
            zi=zi,
            zj=zj,
            cls_i=cls_i,
            cls_j=cls_j,
            loss=loss_out.loss,
            positive_loss=loss_out.positive_loss,
            negative_loss=loss_out.negative_loss,
            mask_i=out_i.mask,
            mask_j=out_j.mask,
            **stats,
        )

    @staticmethod
    def _compute_stats(
        zi: torch.Tensor,
        zj: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Diagnostic stats. All scalars. No grad.

        - ``top1``: fraction of rows for which the matching positive logit is
          the row argmax.
        - ``pos_sim``: mean cosine similarity of matched (i, j) pairs.
        - ``neg_sim``: mean cosine similarity of unmatched (i, j) pairs.
        - ``align``: ``E[||zi_n - zj_n||^2]`` over matched pairs (Wang & Isola).
        - ``uniformity``: ``log E[exp(-2 * ||zi_n - zi_n'||^2)]`` over the
          off-diagonal pairs in the i-batch (Wang & Isola).
        """
        with torch.no_grad():
            batch_size = zi.shape[0]
            device = zi.device
            targets = torch.arange(batch_size, device=device)
            top1 = (logits.argmax(dim=-1) == targets).float().mean()

            zi_norm = torch.nn.functional.normalize(zi, dim=-1)
            zj_norm = torch.nn.functional.normalize(zj, dim=-1)
            cosine = zi_norm @ zj_norm.t()

            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            pos_sim = cosine.diagonal().mean()
            if batch_size == 1:
                neg_sim = torch.zeros((), device=device, dtype=cosine.dtype)
            else:
                neg_sim = cosine[~diag_mask].mean()

            align = (zi_norm - zj_norm).pow(2).sum(dim=-1).mean()

            if batch_size == 1:
                uniformity = torch.zeros((), device=device, dtype=cosine.dtype)
            else:
                pairwise_sq = torch.cdist(zi_norm, zi_norm).pow(2)
                off = pairwise_sq[~diag_mask]
                uniformity = torch.log(torch.exp(-2.0 * off).mean())

        return {
            "top1": top1,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "align": align,
            "uniformity": uniformity,
        }

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return contrastive diagnostic scalars."""

        del ctx
        if not isinstance(output, ContrastiveOutput):
            raise TypeError(
                "ContrastiveObjective.metrics expected ContrastiveOutput, "
                f"got {type(output).__name__}"
            )
        result: dict[str, torch.Tensor] = {
            "positive_loss": output.positive_loss.detach(),
            "negative_loss": output.negative_loss.detach(),
        }
        for name in ("top1", "pos_sim", "neg_sim", "align", "uniformity"):
            value = getattr(output, name)
            if value is not None:
                result[name] = value.detach()
        return result


__all__ = ["ContrastiveObjective"]

"""Supervised fine-tuning objective for audio classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import OptimizerConfig, SFTObjectiveConfig
from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput, SFTOutput
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope

from ...model import AudioModel
from .._base import Objective
from ..components import validate_waveform_batch
from ..protocols import AugmentationPipeline
from .head import ClassificationHead


class SFTObjective(Objective):
    """Cross-entropy classification on pooled encoder representations."""

    uses_augmentor = True

    def __init__(
        self,
        model: AudioModel,
        config: SFTObjectiveConfig,
        *,
        augmentor: AugmentationPipeline | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"SFTObjective.__init__ expected AudioModel, got {type(model).__name__}"
            )
        if not isinstance(config, SFTObjectiveConfig):
            raise TypeError(
                f"SFTObjective.__init__ expected SFTObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        self.augmentor = augmentor

        if config.freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

        with runtime_seed_scope(model.config.runtime, "objective/sft"):
            self.head = ClassificationHead(
                input_dim=model.config.d_model,
                num_classes=config.num_classes,
                config=config.head,
            )
        apply_runtime_to_module(self.head, model.applied_runtime)

    def forward(self, batch: AudioBatch) -> SFTOutput:
        validate_waveform_batch(batch, owner="SFTObjective")
        if batch.labels is None:
            raise ValueError("SFTObjective.forward requires batch.labels")

        labels = batch.labels
        if labels.ndim != 1 or labels.shape[0] != batch.x.shape[0]:
            raise ValueError(
                "SFTObjective.forward expected batch.labels with shape (B,), "
                f"got {tuple(labels.shape)} for batch size {batch.x.shape[0]}"
            )

        working = batch
        if self.training and self.augmentor is not None:
            working = self.augmentor(batch)

        encoder_out = self.model.pipeline(working)
        cls = self.model.get_cls(encoder_out)
        logits = self.head(cls)
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=float(self.config.label_smoothing),
        )

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).to(dtype=logits.dtype).mean()

        return SFTOutput(
            loss=loss,
            logits=logits,
            labels=labels,
            accuracy=accuracy,
        )

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        del ctx
        if not isinstance(output, SFTOutput):
            raise TypeError(
                "SFTObjective.metrics expected SFTOutput, "
                f"got {type(output).__name__}"
            )
        metrics: dict[str, torch.Tensor] = {
            "loss": output.loss.detach(),
        }
        if output.accuracy is not None:
            metrics["accuracy"] = output.accuracy.detach()
        return metrics

    def param_groups(self, config: OptimizerConfig) -> list[dict]:
        from audiotrainer.optim import build_param_groups

        scale = float(self.config.backbone_lr_scale)
        if self.config.freeze_backbone or scale == 1.0:
            return build_param_groups(
                self,
                weight_decay=config.weight_decay,
                no_weight_decay_patterns=config.no_weight_decay_patterns,
            )

        base_lr = float(config.lr)
        decay_backbone: list[nn.Parameter] = []
        no_decay_backbone: list[nn.Parameter] = []
        decay_head: list[nn.Parameter] = []
        no_decay_head: list[nn.Parameter] = []
        patterns = config.no_weight_decay_patterns

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith("head.")
            hit = [pattern for pattern in patterns if pattern in name]
            if is_head:
                (no_decay_head if hit else decay_head).append(param)
            else:
                (no_decay_backbone if hit else decay_backbone).append(param)

        groups: list[dict] = []
        if decay_backbone:
            groups.append(
                {
                    "params": decay_backbone,
                    "weight_decay": config.weight_decay,
                    "lr": base_lr * scale,
                }
            )
        if no_decay_backbone:
            groups.append(
                {
                    "params": no_decay_backbone,
                    "weight_decay": 0.0,
                    "lr": base_lr * scale,
                }
            )
        if decay_head:
            groups.append(
                {"params": decay_head, "weight_decay": config.weight_decay, "lr": base_lr}
            )
        if no_decay_head:
            groups.append({"params": no_decay_head, "weight_decay": 0.0, "lr": base_lr})

        if not groups:
            raise ValueError("SFTObjective.param_groups: no trainable parameters")
        return groups


__all__ = ["SFTObjective"]
"""JEPA objective with masked student prediction and teacher targets."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import JEPAObjectiveConfig
from audiossl.config.batch_dataclasses import (
    AudioBatch,
    EncoderOutput,
    JEPAOutput,
    ObjectiveOutput,
    TokenLayout,
)
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope

from ...core.mlp import MLP
from ...model import AudioModel
from .._base import Objective
from ..components import ema_update_, frozen_copy, validate_waveform_batch
from .loss import jepa_prediction_loss
from .masking import create_jepa_target_mask

class JEPAObjective(Objective):
    """Joint-embedding predictive objective over the shared AudioModel.

    The objective owns JEPA-specific masking, prediction, teacher target
    generation, and loss computation. The wrapped AudioModel remains the only
    owner of processor, tokenizer, and encoder definitions.

    Architecturally this is a *data2vec*-style joint-embedding predictor: a
    single shared encoder with learned mask tokens at target positions plus a
    per-token MLP predictor head, rather than I-JEPA's separate context encoder
    with a cross-attention predictor over learned mask queries.

    Target masks live on the embedding's output axis (after subsampling and
    prefix tokens, which are never selected) and are scattered back onto the
    tokenizer axis through ``model.embedding.output_mask`` metadata.
    """

    uses_generator = True

    def __init__(
        self,
        model: AudioModel,
        config: JEPAObjectiveConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"JEPAObjective.__init__ expected AudioModel, got {type(model).__name__}"
            )
        if not isinstance(config, JEPAObjectiveConfig):
            raise TypeError(
                f"JEPAObjective.__init__ expected JEPAObjectiveConfig, got {type(config).__name__}"
            )

        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        if generator is None:
            raise ValueError("JEPAObjective requires a non-None generator")
        self.register_generator("generator", generator)
        self.generator = generator

        encoder_input_dim = int(model.config.tokenizer.output_dim)
        # Mask token lives in pre-embedding tokenizer-output space; `model.embedding`
        # then maps it into d_model alongside real tokens.
        init_scale = float(config.mask_token_init_scale)
        with runtime_seed_scope(model.config.runtime, "objective/jepa"):
            mask_token = torch.empty(
                encoder_input_dim,
                device=model.applied_runtime.device,
                dtype=model.applied_runtime.dtype,
            )
            # Intentionally inline: bounded uniform_(a, b) has no INITIALIZATION_REGISTRY entry.
            nn.init.uniform_(mask_token, -init_scale, init_scale)
            self.mask_token = nn.Parameter(mask_token)
            self.predictor = MLP(d_model=model.config.d_model, config=config.predictor_config)

        if config.teacher_mode == "ema":
            self.teacher_model: AudioModel | None = frozen_copy(model)
        elif config.teacher_mode == "stop_grad":
            self.teacher_model = None
        else:
            raise ValueError(
                "Unsupported JEPAObjectiveConfig.teacher_mode "
                f"{config.teacher_mode!r}; expected 'ema' or 'stop_grad'"
            )

        apply_runtime_to_module(self, model.applied_runtime)

    def forward(self, batch: AudioBatch) -> JEPAOutput:
        """Run JEPA masking, teacher target creation, prediction, and loss.

        Parameters
        ----------
        batch
            Raw waveform ``AudioBatch`` with ``x`` and boolean validity mask
            shape ``(B, S)``. At this public boundary, ``True`` means valid.

        Returns
        -------
        JEPAOutput
            Prediction and target tensors with shape ``(B, N, d_model)``.
            ``target_mask`` is objective-specific: ``True`` means selected for
            the JEPA loss. ``context_mask`` and ``encoder_mask`` are validity
            masks with shape ``(B, N)`` where ``True`` means valid.
        """

        validate_waveform_batch(batch, owner="JEPAObjective")

        token_batch, layout = self._to_token_batch(batch)
        self._validate_token_batch(token_batch)
        embedding = self.model.embedding
        num_prefix = embedding.num_prefix_tokens
        encoder_mask = embedding.output_mask(token_batch.mask)
        target_mask = create_jepa_target_mask(
            encoder_mask,
            config=self.config.masking,
            generator=self.generator,
        )
        context_mask = encoder_mask & ~target_mask
        pre_embedding_target_mask = self._scatter_to_tokens(
            target_mask[:, num_prefix:],
            num_tokens=token_batch.mask.shape[1],
        )
        pre_embedding_context_mask = token_batch.mask & ~pre_embedding_target_mask

        if self.teacher_model is not None:
            teacher_out = self._ema_teacher_output(batch)
        else:
            was_training = self.model.training
            try:
                self.model.eval()
                with torch.no_grad():
                    teacher_out = self.model.encode_tokens(token_batch, layout=layout)
            finally:
                self.model.train(was_training)

        if teacher_out.z.shape[:2] != target_mask.shape:
            raise RuntimeError(
                "Teacher output and JEPA target mask disagree on batch/time shape, "
                f"got {tuple(teacher_out.z.shape[:2])} and {tuple(target_mask.shape)}"
            )

        masked_x = self._apply_mask_token(token_batch.x, pre_embedding_target_mask)
        # Single-encoder data2vec design: `context_mask` is interpreted by the
        # encoder blocks as a *key* mask, so target-position queries still emit
        # outputs but can only attend to context positions.
        student_out = self.model.encode_tokens(
            AudioBatch(x=masked_x, mask=pre_embedding_context_mask),
            layout=layout,
        )
        prediction = self.predictor(student_out.z)
        target = teacher_out.z.detach()

        if self.config.student_output_norm:
            prediction = self._normalize(prediction)
        if self.config.teacher_output_norm:
            target = self._normalize(target)

        loss_out = jepa_prediction_loss(
            prediction,
            target,
            target_mask,
            loss_style=self.config.loss_style,
        )

        return JEPAOutput(
            z=prediction,
            loss=loss_out.loss,
            mask=encoder_mask,
            prediction=prediction,
            target=target,
            target_mask=target_mask,
            context_mask=context_mask,
            encoder_mask=encoder_mask,
        )

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return JEPA norm, variance, and mask stats."""

        del ctx
        if not isinstance(output, JEPAOutput):
            raise TypeError(
                "JEPAObjective.metrics expected JEPAOutput, "
                f"got {type(output).__name__}"
            )
        prediction = output.prediction.detach()
        target = output.target.detach()
        target_mask = output.target_mask
        result: dict[str, torch.Tensor] = {
            "prediction_norm": prediction.norm(dim=-1).mean(),
            "target_norm": target.norm(dim=-1).mean(),
            "prediction_var": prediction.var(),
            "target_var": target.var(),
            "target_mask_coverage": target_mask.float().mean(),
        }
        per_position = ((prediction - target) ** 2).mean(dim=-1)
        if target_mask.any():
            result["per_position_loss_std"] = per_position[target_mask].std()
        return result

    def on_after_optimizer_step(self, ctx: Any = None) -> None:
        if ctx is not None and getattr(ctx, "step_was_skipped", False):
            return
        self.update_teacher()

    def update_teacher(self) -> None:
        """Update the EMA teacher from the student model after an optimizer step."""

        if self.teacher_model is None:
            return
        ema_update_(self.teacher_model, self.model, decay=float(self.config.teacher_ema_decay))

    def _scatter_to_tokens(self, output_mask: torch.Tensor, *, num_tokens: int) -> torch.Tensor:
        """Place an embedding-output-axis mask (prefix removed) onto the token axis."""
        factor = self.model.embedding.subsampling_factor
        scattered = output_mask.new_zeros((output_mask.shape[0], num_tokens))
        target = scattered[:, ::factor]
        if target.shape != output_mask.shape:
            raise ValueError(
                "JEPA target mask does not match the embedding's subsampled token axis, "
                f"got {tuple(output_mask.shape)} for {tuple(target.shape)}"
            )
        target.copy_(output_mask)
        return scattered

    def _to_token_batch(self, batch: AudioBatch) -> tuple[AudioBatch, TokenLayout]:
        features = self.model.process(batch)
        layout = self.model.token_layout(features)
        tokens = self.model.tokenize(features)
        return tokens, layout

    @staticmethod
    def _validate_token_batch(batch: AudioBatch) -> None:
        if not isinstance(batch.x, torch.Tensor):
            raise TypeError(f"token batch x must be torch.Tensor, got {type(batch.x).__name__}")
        if not isinstance(batch.mask, torch.Tensor):
            raise TypeError(
                f"token batch mask must be torch.Tensor, got {type(batch.mask).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                f"token batch x must be rank-3 [B, S, F], got shape {tuple(batch.x.shape)}"
            )
        if batch.mask.shape != batch.x.shape[:2]:
            raise ValueError(
                "token batch mask must match x batch/time dimensions, "
                f"got {tuple(batch.mask.shape)} and {tuple(batch.x.shape[:2])}"
            )
        if batch.mask.dtype != torch.bool:
            raise ValueError(f"token batch mask must be torch.bool, got {batch.mask.dtype}")

    def _ema_teacher_output(self, batch: AudioBatch) -> EncoderOutput:
        teacher_model = self.teacher_model
        if teacher_model is None:
            raise RuntimeError("JEPAObjective._ema_teacher_output requires teacher_model")
        with torch.no_grad():
            teacher_model.eval()
            return teacher_model.pipeline(AudioBatch(x=batch.x, mask=batch.mask))

    def _apply_mask_token(self, x: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.mask_token.shape[0]:
            raise ValueError(
                "JEPA mask token feature dimension does not match encoder input, "
                f"got token dim {self.mask_token.shape[0]} and input dim {x.shape[-1]}"
            )
        mask_token = self.mask_token.to(device=x.device, dtype=x.dtype)
        return torch.where(target_mask.unsqueeze(-1), mask_token.view(1, 1, -1), x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.output_norm_type == "l2":
            return F.normalize(x, dim=-1)
        if self.config.output_norm_type == "layernorm":
            # Intentionally non-parametric: statistic-only normalization for
            # stable predictor/teacher targets without learnable scale/shift.
            return F.layer_norm(x, (x.shape[-1],))
        raise ValueError(
            "Unsupported JEPAObjectiveConfig.output_norm_type "
            f"{self.config.output_norm_type!r}"
        )


__all__ = ["JEPAObjective"]

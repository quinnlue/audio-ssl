"""wav2vec 2.0 objective orchestration and metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from audiossl.config import Wav2Vec2ObjectiveConfig
from audiossl.config.batch_dataclasses import (
    AudioBatch,
    ObjectiveOutput,
    Wav2Vec2Output,
)
from audiossl.modelling.core.initialization import initialize_linear
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope
from .scalar_scheduler import build_scalar_scheduler

from ...model import AudioModel
from ...registry import get_masking_strategy_registry
from .._base import Objective
from ..components import encode_with_mask_embedding, validate_waveform_batch
from .loss import diversity_loss, feature_penalty, wav2vec2_contrastive_loss
from .quantizer import GumbelProductQuantizer

class Wav2Vec2Objective(Objective):
    """wav2vec 2.0 masked contrastive pretraining objective.

    The objective owns time and optional feature masking, mask embeddings,
    quantization, projection heads, contrastive and auxiliary losses, quantizer
    temperature updates, and metrics. Public ``forward`` accepts raw
    ``AudioBatch`` inputs with waveform and boolean validity mask shape
    ``(B, S)``. The quantizer consumes the embedding's ``prepare`` output, so
    any processor/tokenizer/embedding whose ``finalize`` keeps one output per
    token works.
    """

    uses_generator = True

    def __init__(
        self,
        model: AudioModel,
        config: Wav2Vec2ObjectiveConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(f"Wav2Vec2Objective expected AudioModel, got {type(model).__name__}")
        if not isinstance(config, Wav2Vec2ObjectiveConfig):
            raise TypeError(
                "Wav2Vec2Objective expected Wav2Vec2ObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        if generator is None:
            raise ValueError("Wav2Vec2Objective requires a non-None generator")
        self.register_generator("generator", generator)
        self.generator = generator

        with runtime_seed_scope(model.config.runtime, "objective/wav2vec2"):
            self.mask_emb = nn.Parameter(
                torch.empty(model.config.d_model).uniform_(
                    -config.mask_emb_init_scale,
                    config.mask_emb_init_scale,
                )
            )
            self.project_hid = nn.Linear(model.config.d_model, config.final_dim)
            self.project_q = nn.Linear(config.quantizer.vq_dim, config.final_dim)
            initialize_linear(self.project_hid, config.initialization)
            initialize_linear(self.project_q, config.initialization)
            self.quantizer = GumbelProductQuantizer(
                input_dim=int(model.config.tokenizer.output_dim),
                config=config.quantizer,
                temperature_scheduler=build_scalar_scheduler(
                    config.quantizer.temperature_scheduler
                ),
            )
            registry = get_masking_strategy_registry()
            self.time_masking = registry[config.time_masking.type](
                config=config.time_masking
            )
            self.feature_masking = (
                registry[config.feature_masking.type](config=config.feature_masking)
                if config.feature_masking is not None
                else None
            )

        apply_runtime_to_module(self, model.applied_runtime)

    def forward(self, batch: AudioBatch) -> Wav2Vec2Output:
        """Run masked wav2vec 2.0 prediction and return objective output.

        ``batch.mask`` follows the default convention: ``True`` means valid.
        ``time_mask_indices`` in the output is objective-specific: ``True``
        means the token was selected as a masked prediction position. Contrastive
        logits have shape ``(num_masked, 1 + K)``.
        """

        validate_waveform_batch(batch, owner="Wav2Vec2Objective")
        encoded = encode_with_mask_embedding(
            self.model,
            batch,
            mask_emb=self.mask_emb,
            time_masking=self.time_masking,
            feature_masking=self.feature_masking,
            generator=self.generator,
            feature_grad_mult=self.config.feature_grad_mult,
            training=self.training,
            owner="Wav2Vec2Objective",
        )
        encoder_out = encoded.encoder_out
        time_mask_indices = encoded.time_mask
        raw_features = encoded.raw_features
        quantizer_out = self.quantizer(encoded.prepared_tokens)
        proj_hid = self.project_hid(encoder_out.z)
        proj_q = self.project_q(quantizer_out.quantized)
        contrastive_loss, logits, targets = wav2vec2_contrastive_loss(
            encoder_features=proj_hid,
            quantized_features=proj_q,
            mask_time_indices=time_mask_indices,
            num_negatives=self.config.num_negatives,
            logit_temp=self.config.logit_temp,
            cross_sample_negatives=self.config.cross_sample_negatives,
            generator=self.generator,
        )
        div_loss = diversity_loss(
            quantizer_out.code_perplexity,
            quantizer_out.prob_perplexity,
            quantizer_out.num_vars,
        )
        features_pen = feature_penalty(raw_features)
        loss = (
            contrastive_loss
            + self.config.diversity_loss_weight * div_loss
            + self.config.features_pen_weight * features_pen
        )
        return Wav2Vec2Output(
            loss=loss,
            z=encoder_out.z,
            mask=encoder_out.mask,
            contrastive_loss=contrastive_loss,
            diversity_loss=div_loss,
            features_pen=features_pen,
            logits=logits,
            targets=targets,
            time_mask_indices=time_mask_indices,
            prob_perplexity=quantizer_out.prob_perplexity,
            code_perplexity=quantizer_out.code_perplexity,
            num_vars=torch.tensor(quantizer_out.num_vars, device=loss.device),
            temperature=torch.tensor(quantizer_out.temperature, device=loss.device),
        )

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return wav2vec 2.0 loss and quantizer stats."""

        del ctx
        if not isinstance(output, Wav2Vec2Output):
            raise TypeError(
                f"Wav2Vec2Objective.metrics expected Wav2Vec2Output, got {type(output).__name__}"
            )
        return {
            "contrastive_loss": output.contrastive_loss.detach(),
            "diversity_loss": output.diversity_loss.detach(),
            "features_pen": output.features_pen.detach(),
            "prob_perplexity": output.prob_perplexity.detach(),
            "code_perplexity": output.code_perplexity.detach(),
            "temperature": output.temperature.detach(),
            "mask_coverage": output.time_mask_indices.float().mean(),
        }

    def on_after_optimizer_step(self, ctx: Any = None) -> None:
        """Advance the quantizer temperature scheduler after an optimizer step."""
        if ctx is not None and getattr(ctx, "step_was_skipped", False):
            return
        self.quantizer.step()


__all__ = ["Wav2Vec2Objective"]

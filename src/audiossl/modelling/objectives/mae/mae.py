"""Masked autoencoder objective for mel patch reconstruction."""

from __future__ import annotations

from typing import Any

import torch

from audiossl.config import MAEObjectiveConfig
from audiossl.config.batch_dataclasses import AudioBatch, MAEOutput, ObjectiveOutput
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope

from ...model import AudioModel
from .._base import Objective
from ..components import strip_prefix, validate_waveform_batch
from .decoder import MAEDecoder
from .loss import mae_reconstruction_loss
from .masking import kept_token_mask, random_masking, random_masking_group

class MAEObjective(Objective):
    """Masked Autoencoder objective reproducing XiaoMi/dasheng's training path.

    The objective owns training-time token removal, decoder mask tokens,
    reconstruction targets, reconstruction loss, and metrics. Public
    ``forward`` accepts raw ``AudioBatch`` inputs with waveform and boolean
    validity mask shape ``(B, S)``.

    Token removal happens between the embedding's ``position`` and
    ``finalize`` stages, so dropout and any prefix token (for example CLS) are
    applied to kept tokens only, matching dasheng. Padding tokens are never
    kept and never scored; the encoder receives original token indices as
    ``positions``.
    """

    uses_generator = True

    def __init__(
        self,
        model: AudioModel,
        config: MAEObjectiveConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"MAEObjective.__init__ expected AudioModel, got {type(model).__name__}"
            )
        if not isinstance(config, MAEObjectiveConfig):
            raise TypeError(
                "MAEObjective.__init__ expected MAEObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        if generator is None:
            raise ValueError("MAEObjective requires a non-None generator")
        self.register_generator("generator", generator)
        self.generator = generator

        patch_t, patch_f = model.config.tokenizer.patch_shape
        self.patch_t = patch_t
        self.patch_f = patch_f
        with runtime_seed_scope(model.config.runtime, "objective/mae"):
            self.decoder = MAEDecoder(
                encoder_d_model=model.config.d_model,
                config=config,
                patch_t=patch_t,
                patch_f=patch_f,
            )

        apply_runtime_to_module(self, model.applied_runtime)

    def forward(self, batch: AudioBatch) -> MAEOutput:
        """Run MAE masking, encoding, decoding, target creation, and loss.

        ``batch.mask`` follows the default convention: ``True`` means a valid
        raw waveform sample. ``removal_mask`` (B, N) is ``True`` for every
        token the encoder did not see (removed or padding); ``target_mask`` is
        its restriction to valid tokens and selects the reconstruction loss.
        """

        validate_waveform_batch(batch, owner="MAEObjective")

        features_batch = self.model.process(batch)
        layout = self.model.token_layout(features_batch)
        tokens_batch = self.model.tokenize(features_batch)
        token_valid = tokens_batch.mask

        embedding = self.model.embedding
        positioned = embedding.position(
            embedding.project(embedding.prepare(tokens_batch)),
            layout=layout,
        )

        mask_cfg = self.config.masking
        if mask_cfg.group_factor is None:
            x_kept, mask, ids_restore = random_masking(
                positioned.x,
                mask_cfg.mask_ratio,
                generator=self.generator,
                valid_mask=token_valid,
            )
        else:
            x_kept, mask, ids_restore = random_masking_group(
                positioned.x,
                mask_cfg.mask_ratio,
                group_factor=mask_cfg.group_factor,
                generator=self.generator,
                valid_mask=token_valid,
            )

        len_keep = x_kept.shape[1]
        kept_mask = kept_token_mask(mask, ids_restore, len_keep)
        ids_keep = torch.argsort(ids_restore, dim=1)[:, :len_keep]
        kept_batch = embedding.finalize(
            AudioBatch(x=x_kept, mask=kept_mask, positions=ids_keep)
        )
        encoder_out = strip_prefix(self.model.encode_embedded(kept_batch))
        prediction = self.decoder(
            encoder_out.z,
            ids_restore,
            kept_mask=kept_mask,
            token_mask=token_valid,
            layout=layout,
        )

        target_features = self.model.processor.reconstruction_target(batch)
        target_batch = self.model.tokenize(target_features)
        target = self.model.tokenizer.reorder_patch_targets(
            target_batch.x,
            layout=layout,
            order=self.config.target_patch_order,
        ).detach()

        removal_mask = mask.bool()
        target_mask = removal_mask & token_valid
        loss_out = mae_reconstruction_loss(
            prediction,
            target,
            target_mask,
            norm_pix_loss=self.config.norm_pix_loss,
            eps=self.config.norm_pix_eps,
        )

        return MAEOutput(
            loss=loss_out.loss,
            prediction=prediction,
            target=target,
            target_mask=target_mask,
            removal_mask=removal_mask,
            ids_restore=ids_restore,
        )

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return MAE reconstruction diagnostics."""

        del ctx
        if not isinstance(output, MAEOutput):
            raise TypeError(
                f"MAEObjective.metrics expected MAEOutput, got {type(output).__name__}"
            )
        prediction = output.prediction.detach()
        target = output.target.detach()
        target_mask = output.target_mask
        result: dict[str, torch.Tensor] = {
            "mask_coverage": target_mask.float().mean(),
            "prediction_var": prediction.var(),
            "target_var": target.var(),
        }
        per_patch = ((prediction - target) ** 2).mean(dim=-1)
        if target_mask.any():
            result["per_patch_loss_mean"] = per_patch[target_mask].mean()
            result["per_patch_loss_std"] = per_patch[target_mask].std()
        return result

__all__ = ["MAEObjective"]

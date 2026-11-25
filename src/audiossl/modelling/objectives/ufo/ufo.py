"""EAT Utterance-Frame Objective with EMA teacher and CNN decoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from audiossl.config import UFOObjectiveConfig
from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput, UFOOutput
from audiossl.runtime import apply_runtime_to_module, runtime_seed_scope

from ...model import AudioModel
from ...utils.pooling import masked_mean_pool
from .._base import Objective
from ..components import ema_update_, frozen_copy, validate_waveform_batch
from .decoder import UFODecoder
from .loss import combine_ufo_loss, ufo_frame_loss, ufo_utterance_loss
from .masking import inverse_block_masking, restore_patch_tokens

class UFOObjective(Objective):
    """Bootstrap UFO objective for EAT pre-training.

    Requires a patch-grid tokenizer and an embedding with a prefix (CLS)
    token. Inverse block masking runs between the embedding's ``position``
    and ``finalize`` stages, so the prefix token, entry normalization, and
    dropout are applied to the visible tokens only. Padding patches are
    neither preferred for the student view nor scored in either loss.
    """

    uses_generator = True

    def __init__(
        self,
        model: AudioModel,
        config: UFOObjectiveConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, AudioModel):
            raise TypeError(
                f"UFOObjective.__init__ expected AudioModel, got {type(model).__name__}"
            )
        if not isinstance(config, UFOObjectiveConfig):
            raise TypeError(
                "UFOObjective.__init__ expected UFOObjectiveConfig, "
                f"got {type(config).__name__}"
            )
        self._check_model_compatibility(model, config)

        self.model = model
        self.config = config
        if generator is None:
            raise ValueError("UFOObjective requires a non-None generator")
        self.register_generator("generator", generator)
        self.generator = generator

        with runtime_seed_scope(model.config.runtime, "objective/ufo"):
            self.mask_token = nn.Parameter(
                torch.empty(model.config.d_model, device=model.applied_runtime.device)
            )
            nn.init.normal_(self.mask_token, std=float(config.mask_token_init_scale))
            self.decoder = UFODecoder(
                d_model=model.config.d_model,
                config=config.decoder,
            )

        if config.teacher_mode == "ema":
            self.teacher_model: AudioModel | None = frozen_copy(model)
        elif config.teacher_mode == "stop_grad":
            self.teacher_model = None
        else:
            raise ValueError(
                "Unsupported UFOObjectiveConfig.teacher_mode "
                f"{config.teacher_mode!r}; expected 'ema' or 'stop_grad'"
            )

        apply_runtime_to_module(self, model.applied_runtime)

    @staticmethod
    def _average_teacher_targets(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not hidden_states:
            raise ValueError("Teacher hidden states must not be empty")
        stacked = torch.stack(hidden_states, dim=0)
        return stacked.mean(dim=0)

    def _teacher_targets(self, processed: AudioBatch) -> torch.Tensor:
        # ``processed`` (the front-end features) is shared with the student
        # path: the teacher is an EMA copy with an identical, deterministic
        # front-end, so recomputing it would duplicate the same tensor.
        teacher = self.teacher_model if self.teacher_model is not None else self.model
        was_training = teacher.training
        try:
            teacher.eval()
            with torch.no_grad():
                encoded = teacher.encode_tokens(
                    teacher.tokenize(processed),
                    layout=teacher.token_layout(processed),
                    return_hidden_states=True,
                )
            if encoded.hidden_states is None:
                raise RuntimeError("Teacher encoder did not return hidden states")
            targets = self._average_teacher_targets(encoded.hidden_states)
            return targets[:, encoded.num_prefix_tokens:]
        finally:
            teacher.train(was_training)

    def forward(self, batch: AudioBatch) -> UFOOutput:
        validate_waveform_batch(batch, owner="UFOObjective")

        processed = self.model.process(batch)
        layout = self.model.token_layout(processed)
        tokenized = self.model.tokenize(processed)
        embedding = self.model.embedding
        positioned = embedding.position(
            embedding.project(embedding.prepare(tokenized)),
            layout=layout,
        )
        patch_tokens = positioned.x
        patch_mask = positioned.mask

        teacher_patch_targets = self._teacher_targets(processed)
        if teacher_patch_targets.shape[:2] != patch_mask.shape:
            raise RuntimeError(
                "UFOObjective teacher targets do not align with student patch tokens, "
                f"got {tuple(teacher_patch_targets.shape[:2])} and {tuple(patch_mask.shape)}"
            )
        teacher_utterance_target = masked_mean_pool(teacher_patch_targets, patch_mask)

        # Multi-mask amortization: reuse the single (expensive) teacher pass across
        # ``num_masks`` student views by replicating each clip K times. K=1 leaves
        # everything untouched; K>1 trades extra student/decoder compute for more
        # learning signal per teacher pass (faithful to EAT's utterance cloning).
        num_masks = int(self.config.masking.num_masks)
        if num_masks > 1:
            patch_tokens = patch_tokens.repeat_interleave(num_masks, dim=0)
            patch_mask = patch_mask.repeat_interleave(num_masks, dim=0)
            teacher_patch_targets = teacher_patch_targets.repeat_interleave(num_masks, dim=0)
            teacher_utterance_target = teacher_utterance_target.repeat_interleave(
                num_masks, dim=0
            )

        _, removal_mask, ids_restore, ids_keep = inverse_block_masking(
            batch_size=patch_tokens.shape[0],
            time_patches=layout.time_patches,
            frequency_patches=layout.frequency_patches,
            mask_ratio=self.config.masking.mask_ratio,
            block_size=self.config.masking.block_size,
            device=patch_tokens.device,
            generator=self.generator,
            valid_mask=patch_mask,
        )

        dim = patch_tokens.shape[-1]
        kept_patches = torch.gather(patch_tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        kept_mask = torch.gather(patch_mask, 1, ids_keep)
        student_encoded = self.model.encode_embedded(
            embedding.finalize(AudioBatch(x=kept_patches, mask=kept_mask, positions=ids_keep))
        )
        num_prefix = student_encoded.num_prefix_tokens
        student_cls = student_encoded.z[:, 0]
        student_visible_patches = student_encoded.z[:, num_prefix:]

        restored_patches = restore_patch_tokens(
            student_visible_patches,
            mask_token=self.mask_token,
            ids_restore=ids_restore,
        )
        prediction = self.decoder(restored_patches, layout=layout)

        frame_mask = removal_mask & patch_mask
        frame_loss = ufo_frame_loss(
            prediction,
            teacher_patch_targets.detach(),
            frame_mask,
        )
        utterance_loss = ufo_utterance_loss(
            student_cls,
            teacher_utterance_target.detach(),
        )
        loss_out = combine_ufo_loss(
            frame_loss=frame_loss,
            utterance_loss=utterance_loss,
            utterance_loss_weight=self.config.utterance_loss_weight,
        )

        return UFOOutput(
            loss=loss_out.loss,
            frame_loss=loss_out.frame_loss,
            utterance_loss=loss_out.utterance_loss,
            prediction=prediction,
            target=teacher_patch_targets.detach(),
            frame_mask=frame_mask,
            student_cls=student_cls,
            teacher_utterance_target=teacher_utterance_target.detach(),
        )

    def on_after_optimizer_step(self, ctx: Any = None) -> None:
        if ctx is not None and ctx.step_was_skipped:
            return
        self.update_teacher()

    def update_teacher(self) -> None:
        if self.teacher_model is None:
            return
        ema_update_(self.teacher_model, self.model, decay=float(self.config.teacher_ema_decay))

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        del ctx
        if not isinstance(output, UFOOutput):
            raise TypeError(
                f"UFOObjective.metrics expected UFOOutput, got {type(output).__name__}"
            )
        return {
            "frame_loss": output.frame_loss.detach(),
            "utterance_loss": output.utterance_loss.detach(),
            "frame_mask_coverage": output.frame_mask.float().mean(),
        }


__all__ = ["UFOObjective"]

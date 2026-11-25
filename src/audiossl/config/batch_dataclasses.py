"""Batch and output dataclasses exchanged between pipeline stages.

Boolean masks use ``True`` for valid positions unless a field docstring states
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch


@dataclass(frozen=True)
class AudioBatch:
    """Immutable padded batch with features or tokens and a validity mask.

    ``x`` shape depends on the pipeline stage: ``(B, S)`` for waveform,
    ``(B, T, F)`` for features, or ``(B, N, D)`` for embedded tokens.
    ``mask`` has shape ``(B, T)`` or ``(B, N)`` matching the time/token axis of
    ``x``. Stages return new batches (``dataclasses.replace``) instead of
    mutating their input.

    ``positions`` optionally carries integer ``(B, N)`` indices of each token
    in the full, undropped token sequence (prefix tokens included). Objectives
    that drop or reorder tokens set it so position-dependent modules (RoPE)
    see true positions; ``None`` means ``arange(N)``.

    ``num_prefix_tokens`` counts leading non-content tokens (e.g. a CLS token
    prepended by the embedding). Pooling and objectives use it instead of
    hard-coding slice offsets.
    """

    x: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor | None = None
    num_prefix_tokens: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.num_prefix_tokens, bool) or not isinstance(
            self.num_prefix_tokens, int
        ):
            raise TypeError(
                "AudioBatch.num_prefix_tokens must be int, "
                f"got {type(self.num_prefix_tokens).__name__}"
            )
        if self.num_prefix_tokens < 0:
            raise ValueError(
                f"AudioBatch.num_prefix_tokens must be >= 0, got {self.num_prefix_tokens}"
            )
        if self.positions is not None:
            if not isinstance(self.positions, torch.Tensor):
                raise TypeError(
                    "AudioBatch.positions must be torch.Tensor or None, "
                    f"got {type(self.positions).__name__}"
                )
            if self.positions.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"AudioBatch.positions must be an integer tensor, got {self.positions.dtype}"
                )
            if isinstance(self.mask, torch.Tensor) and self.positions.shape != self.mask.shape:
                raise ValueError(
                    "AudioBatch.positions must match mask shape, "
                    f"got {tuple(self.positions.shape)} and {tuple(self.mask.shape)}"
                )

    def to(
        self,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> AudioBatch:
        """Return a copy with every tensor field moved to ``device``.

        Subclasses with extra tensor fields are preserved via ``dataclasses.replace``.
        """
        updates: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                updates[field.name] = value.to(device=device, non_blocking=non_blocking)
        return replace(self, **updates) if updates else self


@dataclass(frozen=True)
class TokenLayout:
    """Public tokenizer layout metadata for patch grids and layout-aware PE."""

    num_tokens: int
    time_patches: int
    frequency_patches: int
    patch_shape: tuple[int, int]
    stride: tuple[int, int]
    flatten_order: str
    output_dim: int
    source_shape: tuple[int, int]


@dataclass
class EncoderOutput:
    """Encoder sequence output with optional pooled ``cls`` representation.

    ``z`` has shape ``(B, N, D)``. ``mask`` and ``lengths`` describe valid tokens.
    ``num_prefix_tokens`` is propagated from the encoder input batch.
    """

    z: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    cls: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    num_prefix_tokens: int = 0


@dataclass
class ContrastiveAudioBatch:
    """Paired augmented views for contrastive objectives."""

    batch_i: AudioBatch
    batch_j: AudioBatch


@dataclass
class ObjectiveOutput:
    """Base training objective return type."""

    loss: torch.Tensor


@dataclass
class ContrastiveOutput(ObjectiveOutput):
    """SimCLR-style contrastive objective outputs and optional alignment metrics."""

    zi: torch.Tensor
    zj: torch.Tensor
    cls_i: torch.Tensor
    cls_j: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    mask_i: torch.Tensor
    mask_j: torch.Tensor
    top1: torch.Tensor | None = None
    pos_sim: torch.Tensor | None = None
    neg_sim: torch.Tensor | None = None
    align: torch.Tensor | None = None
    uniformity: torch.Tensor | None = None


@dataclass
class BYOLOutput(ObjectiveOutput):
    """BYOL objective outputs for online and target branches."""

    loss: torch.Tensor
    prediction_i: torch.Tensor
    prediction_j: torch.Tensor
    target_i: torch.Tensor
    target_j: torch.Tensor
    projection_i: torch.Tensor
    projection_j: torch.Tensor
    cls_i: torch.Tensor
    cls_j: torch.Tensor
    mask_i: torch.Tensor
    mask_j: torch.Tensor


@dataclass
class DINOOutput(ObjectiveOutput):
    """DiNO objective outputs for student and teacher logits."""

    loss: torch.Tensor
    student_logits_i: torch.Tensor
    student_logits_j: torch.Tensor
    teacher_logits_i: torch.Tensor
    teacher_logits_j: torch.Tensor
    teacher_logits_uncentered: torch.Tensor
    cls_i: torch.Tensor
    cls_j: torch.Tensor
    mask_i: torch.Tensor
    mask_j: torch.Tensor


@dataclass
class JEPAOutput(ObjectiveOutput):
    """JEPA predictor outputs and context/target masks."""

    z: torch.Tensor
    mask: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    context_mask: torch.Tensor
    encoder_mask: torch.Tensor


@dataclass
class MAEOutput(ObjectiveOutput):
    """MAE reconstruction outputs and masking metadata."""

    loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    removal_mask: torch.Tensor
    ids_restore: torch.Tensor


@dataclass
class UFOOutput(ObjectiveOutput):
    """EAT UFO frame and utterance regression outputs."""

    loss: torch.Tensor
    frame_loss: torch.Tensor
    utterance_loss: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    frame_mask: torch.Tensor
    student_cls: torch.Tensor
    teacher_utterance_target: torch.Tensor


@dataclass
class Wav2Vec2Output(ObjectiveOutput):
    """wav2vec2 contrastive, diversity, and quantizer diagnostics."""

    z: torch.Tensor
    mask: torch.Tensor
    contrastive_loss: torch.Tensor
    diversity_loss: torch.Tensor
    features_pen: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    time_mask_indices: torch.Tensor
    prob_perplexity: torch.Tensor
    code_perplexity: torch.Tensor
    num_vars: torch.Tensor
    temperature: torch.Tensor


@dataclass
class HuBERTOutput(ObjectiveOutput):
    """HuBERT masked and unmasked prediction outputs."""

    z: torch.Tensor
    mask: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    time_mask_indices: torch.Tensor
    masked_loss: torch.Tensor
    unmasked_loss: torch.Tensor
    target_entropy: torch.Tensor
    unique_codes: torch.Tensor


__all__ = [
    "AudioBatch",
    "TokenLayout",
    "EncoderOutput",
    "ContrastiveAudioBatch",
    "ObjectiveOutput",
    "ContrastiveOutput",
    "BYOLOutput",
    "DINOOutput",
    "JEPAOutput",
    "MAEOutput",
    "UFOOutput",
    "Wav2Vec2Output",
    "HuBERTOutput",
]

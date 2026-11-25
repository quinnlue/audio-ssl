"""Reusable building blocks shared by objectives.

Teacher management, input validation, and the masked-token encoding path used
by wav2vec2-style objectives live here so each objective composes them instead
of re-implementing (and drifting from) private copies.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from audiossl.config.batch_dataclasses import AudioBatch, EncoderOutput

if TYPE_CHECKING:
    from ..model import AudioModel


def validate_waveform_batch(batch: Any, *, owner: str) -> None:
    """Raise unless ``batch`` is a raw waveform ``AudioBatch`` with ``(B, S)`` bool mask."""
    if not isinstance(batch, AudioBatch):
        raise TypeError(f"{owner}.forward expected AudioBatch, got {type(batch).__name__}")
    if not isinstance(batch.x, torch.Tensor) or batch.x.ndim != 2:
        shape = getattr(batch.x, "shape", None)
        description = tuple(shape) if shape is not None else type(batch.x).__name__
        raise ValueError(
            f"{owner}.forward expected batch.x to be rank-2 (B, T) waveform tensor, "
            f"got shape {description}"
        )
    if not isinstance(batch.mask, torch.Tensor) or batch.mask.dtype is not torch.bool:
        dtype = getattr(batch.mask, "dtype", type(batch.mask).__name__)
        raise ValueError(f"{owner}.forward expected boolean batch.mask, got {dtype}")
    if batch.mask.shape != batch.x.shape:
        raise ValueError(
            f"{owner}.forward expected batch.mask to match batch.x, "
            f"got {tuple(batch.mask.shape)} and {tuple(batch.x.shape)}"
        )


class FrozenTeacher(nn.Module):
    """Wrap a module so ``.train()`` cannot flip it out of eval mode."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: ARG002
        return super().train(False)

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def frozen_copy(module: nn.Module) -> FrozenTeacher:
    """Deep copy ``module`` as an eval-mode teacher with gradients disabled."""
    teacher = deepcopy(module)
    return FrozenTeacher(teacher)


def _unwrap_teacher(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, FrozenTeacher) else module


@torch.no_grad()
def ema_update_(teacher: nn.Module, student: nn.Module, *, decay: float) -> None:
    """In-place EMA of ``student`` into ``teacher``.

    Parameters follow ``teacher = decay * teacher + (1 - decay) * student``.
    Buffers (normalization statistics, STFT windows, mel filterbanks, caches)
    are hard-copied from the student: they are either constants or statistics
    the student already tracks, and averaging them would mix incompatible
    running estimates.
    """
    teacher = _unwrap_teacher(teacher)
    student = _unwrap_teacher(student)
    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    if teacher_parameters.keys() != student_parameters.keys():
        missing = sorted(teacher_parameters.keys() ^ student_parameters.keys())
        raise RuntimeError(f"EMA teacher and student parameter sets differ: {missing[:5]}")
    names = list(teacher_parameters)
    if names:
        teacher_list = [teacher_parameters[name] for name in names]
        student_list = [student_parameters[name].detach() for name in names]
        torch._foreach_mul_(teacher_list, decay)
        torch._foreach_add_(teacher_list, student_list, alpha=1.0 - decay)

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    if teacher_buffers.keys() != student_buffers.keys():
        missing = sorted(teacher_buffers.keys() ^ student_buffers.keys())
        raise RuntimeError(f"EMA teacher and student buffer sets differ: {missing[:5]}")
    for name, teacher_buffer in teacher_buffers.items():
        teacher_buffer.copy_(student_buffers[name].detach())
    teacher.eval()


class GradMultiply(torch.autograd.Function):
    """Identity forward; scales the gradient by ``scale`` in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad * ctx.scale, None


@dataclass
class MaskedTokenEncoding:
    """Result of :func:`encode_with_mask_embedding`.

    ``encoder_out`` has prefix tokens removed so ``z``/``mask`` align with
    ``time_mask`` on the token axis ``(B, N)``.
    """

    encoder_out: EncoderOutput
    time_mask: torch.Tensor
    prepared_tokens: torch.Tensor
    raw_features: torch.Tensor
    feature_mask: torch.Tensor


def encode_with_mask_embedding(
    model: AudioModel,
    batch: AudioBatch,
    *,
    mask_emb: torch.Tensor,
    time_masking: Any,
    feature_masking: Any | None,
    generator: torch.Generator | None,
    feature_grad_mult: float,
    training: bool,
    owner: str,
) -> MaskedTokenEncoding:
    """wav2vec2/HuBERT encoding: substitute ``mask_emb`` at masked projected tokens.

    Runs ``process -> tokenize -> prepare -> project``, samples a time mask on
    valid tokens (forcing at least one), applies optional channel masking and
    the mask embedding, then ``position -> finalize -> encode``.
    """
    features = model.process(batch)
    raw_features = features.x
    if feature_grad_mult != 1.0 and training:
        features = replace(features, x=GradMultiply.apply(raw_features, feature_grad_mult))
    layout = model.token_layout(features)
    tokens = model.tokenize(features)
    prepared = model.embedding.prepare(tokens)
    projected = model.embedding.project(prepared)

    time_mask = time_masking.compute_mask(
        shape=projected.x.shape[:2],
        valid_mask=projected.mask,
        generator=generator,
    ).to(projected.mask.device)
    if not time_mask.any():
        valid = torch.nonzero(projected.mask, as_tuple=False)
        if valid.numel() == 0:
            raise ValueError(f"{owner} requires at least one valid feature frame")
        first = valid[0]
        time_mask[first[0], first[1]] = True

    masked_x = projected.x
    if feature_masking is not None:
        channel_mask = feature_masking.compute_mask(
            shape=(masked_x.shape[0], masked_x.shape[2]),
            valid_mask=None,
            generator=generator,
        ).to(masked_x.device)
        masked_x = masked_x.masked_fill(channel_mask.unsqueeze(1), 0.0)
    masked_x = torch.where(
        time_mask.unsqueeze(-1),
        mask_emb.view(1, 1, -1).to(dtype=masked_x.dtype, device=masked_x.device),
        masked_x,
    )

    positioned = model.embedding.position(replace(projected, x=masked_x), layout=layout)
    embedded = model.embedding.finalize(positioned)
    num_prefix = embedded.num_prefix_tokens
    if embedded.x.shape[1] - num_prefix != time_mask.shape[1]:
        raise ValueError(
            f"{owner} needs an embedding whose finalize keeps one output per token; "
            f"got {embedded.x.shape[1] - num_prefix} outputs for {time_mask.shape[1]} tokens"
        )
    encoder_out = model.encode_embedded(embedded)
    if num_prefix:
        encoder_out = strip_prefix(encoder_out)
    return MaskedTokenEncoding(
        encoder_out=encoder_out,
        time_mask=time_mask,
        prepared_tokens=prepared.x,
        raw_features=raw_features,
        feature_mask=features.mask,
    )


def strip_prefix(output: EncoderOutput) -> EncoderOutput:
    """Drop prefix tokens (for example CLS) from an encoder output's token axis."""
    p = output.num_prefix_tokens
    if p == 0:
        return output
    mask = output.mask[:, p:]
    hidden = None
    if output.hidden_states is not None:
        hidden = tuple(h[:, p:] for h in output.hidden_states)
    return replace(
        output,
        z=output.z[:, p:],
        mask=mask,
        lengths=mask.sum(dim=1),
        hidden_states=hidden,
        num_prefix_tokens=0,
    )


__all__ = [
    "FrozenTeacher",
    "GradMultiply",
    "MaskedTokenEncoding",
    "ema_update_",
    "encode_with_mask_embedding",
    "frozen_copy",
    "strip_prefix",
    "validate_waveform_batch",
]

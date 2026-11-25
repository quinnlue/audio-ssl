"""Gumbel vector quantizer used by wav2vec 2.0 objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiossl.config import Wav2Vec2QuantizerConfig
from audiossl.modelling.core.initialization import initialize_linear
from .scalar_scheduler import ScalarScheduler


@dataclass(frozen=True)
class QuantizerOutput:
    """Outputs from the wav2vec 2.0 product quantizer."""

    quantized: torch.Tensor
    code_perplexity: torch.Tensor
    prob_perplexity: torch.Tensor
    num_vars: int
    temperature: float
    codebook_indices: torch.Tensor


class GumbelProductQuantizer(nn.Module):
    """Gumbel product quantizer used by the wav2vec 2.0 objective.

    Parameters
    ----------
    input_dim
        Feature dimension of incoming unmasked tokenizer features.
    config
        Quantizer config defining groups, variables, quantized width, and
        initialization.
    temperature_scheduler
        Trainer-step scheduler for Gumbel softmax temperature.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        config: Wav2Vec2QuantizerConfig,
        temperature_scheduler: ScalarScheduler,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.config = config
        self.num_groups = int(config.num_groups)
        self.num_vars = int(config.num_vars)
        self.vq_dim = int(config.vq_dim)
        self.var_dim = self.vq_dim // self.num_groups
        self.weight_proj = nn.Linear(self.input_dim, self.num_groups * self.num_vars)
        self.codevectors = nn.Parameter(
            torch.empty(1, self.num_groups * self.num_vars, self.var_dim)
        )
        self.temperature_scheduler = temperature_scheduler

        initialize_linear(self.weight_proj, config.initialization)
        nn.init.uniform_(self.codevectors)

    def forward(self, x: torch.Tensor) -> QuantizerOutput:
        """Quantize features with shape ``(B, N, input_dim)``."""

        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                "GumbelProductQuantizer expected input shape "
                f"(B, T, {self.input_dim}), got {tuple(x.shape)}"
            )
        batch, time, _ = x.shape
        logits = self.weight_proj(x).view(batch * time * self.num_groups, self.num_vars)
        temperature = self.temperature_scheduler.current_value()
        if self.training:
            probs = F.gumbel_softmax(
                logits.float(),
                tau=temperature,
                hard=self.config.hard,
                dim=-1,
            ).type_as(logits)
        else:
            indices = logits.argmax(dim=-1)
            probs = F.one_hot(indices, num_classes=self.num_vars).type_as(logits)

        soft_probs = torch.softmax(logits.float(), dim=-1)
        avg_probs = soft_probs.view(batch * time, self.num_groups, self.num_vars).mean(dim=0)
        prob_perplexity = _perplexity(avg_probs).sum()
        hard_probs = probs.view(batch * time, self.num_groups, self.num_vars).float().mean(dim=0)
        code_perplexity = _perplexity(hard_probs).sum()

        codevectors = self.codevectors.view(1, self.num_groups, self.num_vars, self.var_dim)
        quantized = probs.view(batch * time, self.num_groups, self.num_vars, 1) * codevectors
        quantized = quantized.sum(dim=2).reshape(batch, time, self.vq_dim)
        codebook_indices = probs.view(batch, time, self.num_groups, self.num_vars).argmax(dim=-1)

        return QuantizerOutput(
            quantized=quantized,
            code_perplexity=code_perplexity,
            prob_perplexity=prob_perplexity,
            num_vars=self.num_vars * self.num_groups,
            temperature=float(temperature),
            codebook_indices=codebook_indices,
        )

    def step(self) -> None:
        """Advance the quantizer temperature scheduler by one trainer step."""

        self.temperature_scheduler.step()


def _perplexity(probs: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.sum(probs * torch.log(probs.clamp_min(1e-7)), dim=-1))


__all__ = ["GumbelProductQuantizer", "QuantizerOutput"]

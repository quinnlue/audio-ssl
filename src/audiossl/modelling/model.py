"""Audio model orchestration for the modelling pipeline."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from audiossl.config import AudioModelConfig, validate_config
from audiossl.runtime import apply_runtime_to_module, configure_runtime, runtime_seed_scope

from ..config.batch_dataclasses import AudioBatch, EncoderOutput, TokenLayout
from .embeddings.embedding import Embedding
from .encoders.encoder import Encoder
from .processors.processor import Processor
from .tokenizers.tokenizer import Tokenizer
from .utils.pooling import masked_mean_pool

logger = logging.getLogger(__name__)


class AudioModel(nn.Module):
    """Composable audio representation model.

    The model owns the canonical inference pipeline:

    ``AudioBatch`` raw waveforms -> processor -> tokenizer -> embedding ->
    encoder. Processor, tokenizer, embedding, and encoder boundaries follow the
    modelling contracts: masks are boolean with ``True`` meaning valid, entry
    positional encoding is embedding-owned, and the encoder returns token states
    without owning pooling.

    Parameters
    ----------
    config
        Model config. It is deep-copied, validated, used to configure runtime,
        and then applied to the constructed module tree.
    """

    def __init__(self, config: AudioModelConfig):
        super().__init__()
        self.config = deepcopy(config)
        validate_config(self.config)
        self.applied_runtime = configure_runtime(self.config.runtime)
        self.cls_strategy = self.config.encoder.pooling

        self.tokenizer = Tokenizer(self.config)

        runtime = self.config.runtime
        with runtime_seed_scope(runtime, "model/processor"):
            self.processor = Processor(self.config)
        with runtime_seed_scope(runtime, "model/embedding"):
            self.embedding = Embedding(self.config)
        with runtime_seed_scope(runtime, "model/encoder"):
            self.encoder = Encoder(
                d_model=self.config.d_model,
                config=self.config.encoder,
            )
        if self.cls_strategy == "cls" and self.embedding.num_prefix_tokens < 1:
            raise ValueError(
                f"Encoder pooling='cls' requires an embedding with a prefix token; "
                f"{self.config.embedding.type!r} adds none"
            )

        apply_runtime_to_module(self, self.applied_runtime)
        logger.info("Built AudioModel:\n%s", self.describe())

    def describe(self) -> str:
        """One line per component: registered type, class, and parameter count."""
        lines = []
        for name in ("processor", "tokenizer", "embedding", "encoder"):
            module = getattr(self, name)
            component_config = getattr(self.config, name)
            inner = getattr(module, name, module)
            count = sum(p.numel() for p in module.parameters())
            lines.append(
                f"  {name:<10} type={getattr(component_config, 'type', '?')!s:<26} "
                f"class={type(inner).__name__:<28} params={count:,}"
            )
        lines.append(
            f"  d_model={self.config.d_model} pooling={self.cls_strategy} "
            f"prefix_tokens={self.embedding.num_prefix_tokens} "
            f"device={self.applied_runtime.device} dtype={self.applied_runtime.dtype} "
            f"seed={self.applied_runtime.seed} total_params={self.n_params:,}"
        )
        return "\n".join(lines)

    @property
    def n_params(self):
        """Return the total number of parameters in the model."""
        return sum([p.numel() for p in self.parameters()])

    def forward(self, batch: AudioBatch | Mapping[str, Any] | Sequence[Any]) -> EncoderOutput:
        """Run the full audio pipeline and return encoded token states.

        Parameters
        ----------
        batch
            Prefer an ``AudioBatch`` with waveform ``x`` and boolean mask shape
            ``(B, S)``. Dict, tensor, and sequence conveniences are accepted
            only when gradients are disabled.

        Returns
        -------
        EncoderOutput
            Encoded token states ``z`` with shape ``(B, N, D)`` and boolean mask
            ``(B, N)`` where ``True`` means valid.
        """
        return self.pipeline(batch)

    def pipeline(self, batch: AudioBatch | Mapping[str, Any] | Sequence[Any]) -> EncoderOutput:
        """Run process, tokenize, embed, and encode stages in contract order.

        The returned ``EncoderOutput.cls`` is populated through
        :meth:`get_cls` for legacy consumers. Encoders themselves do not own
        pooling or produce ``cls``.
        """
        normalized = self._normalize_pipeline_batch(batch)
        processed = self.process(normalized)
        layout = self.token_layout(processed)
        tokenized = self.tokenize(processed)
        embedded = self.embed(tokenized, layout=layout)
        output = self.encode_embedded(embedded)
        output.cls = self.get_cls(output)
        return output

    def process(self, batch: AudioBatch) -> AudioBatch:
        """Convert raw waveforms ``(B, S)`` into frame features.

        The processor recomputes a boolean feature mask using ``True`` for valid
        frames. Sample-rate compatibility is declared by config rather than
        validated at this boundary.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                f"AudioModel.process expected AudioBatch, got {type(batch).__name__}"
            )
        return self.processor(batch)

    def token_layout(self, batch: AudioBatch) -> TokenLayout:
        """Return tokenizer layout metadata for a feature batch.

        Layout-aware embeddings and objectives consume this public record
        instead of reconstructing tokenizer grid details from private state.
        """
        return self.tokenizer.layout(batch)

    def tokenize(self, batch: AudioBatch) -> AudioBatch:
        """Convert frame features ``(B, T, F)`` into tokens.

        Tokenizers own feature-to-token reshaping, token masks, and token
        layout. They do not project to ``d_model`` or apply positional encoding.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                f"AudioModel.tokenize expected AudioBatch, got {type(batch).__name__}"
            )
        return self.tokenizer(batch)

    def embed(self, batch: AudioBatch, *, layout: TokenLayout | None = None) -> AudioBatch:
        """Project tokenizer output to ``d_model`` and apply embedding PE.

        Parameters
        ----------
        batch
            Token batch with ``x`` shape ``(B, N, input_dim)`` and boolean mask
            ``(B, N)`` where ``True`` means valid.
        layout
            Optional tokenizer layout for layout-aware positional encodings.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                f"AudioModel.embed expected AudioBatch, got {type(batch).__name__}"
            )
        return self.embedding(batch, layout=layout)

    def encode_tokens(
        self,
        batch: AudioBatch,
        *,
        layout: TokenLayout | None = None,
        return_hidden_states: bool = False,
    ) -> EncoderOutput:
        """Embed and encode tokenizer output ``(B, N, input_dim)``.

        Objectives that build their own token batches (augmented, masked, or
        teacher views) use this instead of re-implementing embed + encode.
        """
        embedded = self.embed(batch, layout=layout)
        return self.encode_embedded(embedded, return_hidden_states=return_hidden_states)

    def encode_embedded(
        self,
        batch: AudioBatch,
        *,
        return_hidden_states: bool = False,
    ) -> EncoderOutput:
        """Encode embedded token states into an ``EncoderOutput``.

        The input must already have final dimension ``d_model``. The encoder
        preserves the token mask, ``positions`` and ``num_prefix_tokens``
        metadata, and leaves pooling to :meth:`get_cls`.
        """
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                f"AudioModel.encode_embedded expected AudioBatch, got {type(batch).__name__}"
            )
        if batch.x.ndim != 3:
            raise ValueError(
                "AudioModel.encode_embedded expected rank-3 embedded input "
                f"(B, T, d_model), got shape {tuple(batch.x.shape)}"
            )
        if batch.x.shape[-1] != self.config.d_model:
            raise ValueError(
                "AudioModel.encode_embedded expected final dimension "
                f"{self.config.d_model}, got {batch.x.shape[-1]}"
            )
        return self.encoder(batch, return_hidden_states=return_hidden_states)

    def _normalize_pipeline_batch(
        self,
        batch: AudioBatch | Mapping[str, Any] | Sequence[Any],
    ) -> AudioBatch:
        if not isinstance(batch, AudioBatch):
            if torch.is_grad_enabled():
                raise TypeError(
                    "AudioModel.pipeline expected AudioBatch while gradients are enabled; "
                    "list inputs and dict inputs with key 'x' are inference-only conveniences "
                    "and require torch.no_grad() or torch.inference_mode()"
                )
            return self._coerce_inference_batch(batch)
        return batch

    def _coerce_inference_batch(self, batch: Mapping[str, Any] | Sequence[Any]) -> AudioBatch:
        mask: Any | None = None
        if isinstance(batch, Mapping):
            if "x" not in batch:
                raise KeyError("AudioModel.pipeline dict input must contain key 'x'")
            mask = batch.get("mask")
            batch = batch["x"]

        if isinstance(batch, torch.Tensor):
            x = self._coerce_inference_tensor(batch, name="x")
            if x.ndim == 1:
                x = x.unsqueeze(0)
            elif x.ndim != 2:
                raise ValueError(
                    "AudioModel.pipeline convenience tensor input expected shape [T] or [B, T], "
                    f"got {tuple(x.shape)}"
                )
            return AudioBatch(
                x=x,
                mask=self._coerce_inference_mask(mask, shape=x.shape, device=x.device),
            )

        if isinstance(batch, Sequence) and not isinstance(batch, str | bytes | bytearray):
            return self._collate_inference_waveforms(batch, mask=mask)

        try:
            x = self._coerce_inference_tensor(batch, name="x")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "AudioModel.pipeline expected AudioBatch, a list of waveforms, "
                "or a dict with key 'x', "
                f"got {type(batch).__name__}"
            ) from exc
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(
                "AudioModel.pipeline convenience input expected shape [T] or [B, T], "
                f"got {tuple(x.shape)}"
            )
        return AudioBatch(
            x=x,
            mask=self._coerce_inference_mask(mask, shape=x.shape, device=x.device),
        )

    def _coerce_inference_tensor(self, value: Any, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if torch.is_complex(tensor):
            raise ValueError(f"AudioModel.pipeline convenience input {name} must not be complex")
        return tensor.detach().to(
            device=self.applied_runtime.device,
            dtype=torch.float32,
            non_blocking=True,
        )

    def _collate_inference_waveforms(
        self,
        waveforms: Sequence[Any],
        *,
        mask: Any | None,
    ) -> AudioBatch:
        if len(waveforms) == 0:
            raise ValueError("AudioModel.pipeline convenience waveform list must not be empty")

        items = [
            self._coerce_inference_tensor(item, name=f"x[{index}]")
            for index, item in enumerate(waveforms)
        ]
        for index, item in enumerate(items):
            if item.ndim != 1:
                raise ValueError(
                    "AudioModel.pipeline convenience waveform list expected rank-1 items, "
                    f"got x[{index}] with shape {tuple(item.shape)}"
                )

        lengths = torch.tensor(
            [item.shape[0] for item in items],
            device=self.applied_runtime.device,
            dtype=torch.long,
        )
        max_length = int(lengths.max().item())
        x = torch.zeros(
            (len(items), max_length),
            device=self.applied_runtime.device,
            dtype=torch.float32,
        )
        inferred_mask = torch.zeros(
            (len(items), max_length),
            device=self.applied_runtime.device,
            dtype=torch.bool,
        )
        for index, item in enumerate(items):
            length = int(lengths[index].item())
            x[index, :length] = item
            inferred_mask[index, :length] = True

        if mask is not None:
            inferred_mask = self._coerce_inference_mask(mask, shape=x.shape, device=x.device)
        return AudioBatch(x=x, mask=inferred_mask)

    def _coerce_inference_mask(
        self,
        mask: Any | None,
        *,
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(shape, device=device, dtype=torch.bool)
        mask_tensor = torch.as_tensor(mask)
        if mask_tensor.ndim == 1 and len(shape) == 2 and shape[0] == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        if tuple(mask_tensor.shape) != tuple(shape):
            raise ValueError(
                "AudioModel.pipeline convenience mask must match waveform shape, "
                f"got {tuple(mask_tensor.shape)} and {tuple(shape)}"
            )
        return mask_tensor.detach().to(device=device, dtype=torch.bool, non_blocking=True)

    def get_cls(self, output: EncoderOutput) -> torch.Tensor:
        """Return the pooled sequence representation ``(B, D)`` for an encoder output.

        ``"cls"`` pooling returns the first prefix token and requires
        ``output.num_prefix_tokens >= 1``. ``"masked_mean"`` averages valid
        non-prefix tokens.
        """
        if output.cls is not None:
            return output.cls
        num_prefix = output.num_prefix_tokens
        if self.cls_strategy == "cls":
            if num_prefix < 1:
                raise ValueError(
                    "AudioModel.get_cls: pooling='cls' requires a prefix token, but the "
                    "encoder output has num_prefix_tokens=0"
                )
            return output.z[:, 0]
        return masked_mean_pool(output.z[:, num_prefix:], output.mask[:, num_prefix:])

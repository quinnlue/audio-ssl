"""Open registries for modelling component dispatch and shared helpers.

Every component family (processors, tokenizers, embeddings, positional
encodings, encoder blocks, objectives, masking strategies) is resolved through
a :class:`ComponentRegistry` keyed by the config ``type`` discriminator.
Built-in entries are imported lazily; third-party components plug in with
``PROCESSORS.register("my_type")`` (usable as a decorator) without editing
library code. Register the matching config class with
:func:`audiossl.config.registry.register_config` so JSON loading and
validation can resolve it too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn

ActivationFactory = Callable[[], nn.Module]

ACTIVATION_REGISTRY: dict[str, ActivationFactory] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu_fast": partial(nn.GELU, approximate="tanh"),
}

INITIALIZATION_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
}


def build_activation(name: str) -> nn.Module:
    """Instantiate a fresh activation module for registry key ``name``."""
    try:
        factory = ACTIVATION_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unsupported activation {name!r}; expected one of {sorted(ACTIVATION_REGISTRY)}"
        ) from None
    module = factory()
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"ACTIVATION_REGISTRY[{name!r}] must build an nn.Module, got {type(module).__name__}"
        )
    return module


class ComponentRegistry(MutableMapping[str, Any]):
    """Name to component mapping with lazy built-ins and runtime registration.

    Lookups of unknown names raise ``KeyError`` listing every registered name.
    Registering an existing name raises unless ``overwrite=True`` so two
    plugins cannot silently shadow each other.
    """

    def __init__(self, kind: str, builtins: dict[str, Callable[[], Any]] | None = None) -> None:
        self.kind = kind
        self._loaders: dict[str, Callable[[], Any]] = dict(builtins or {})
        self._cache: dict[str, Any] = {}

    def register(self, name: str, component: Any = None, *, overwrite: bool = False) -> Any:
        """Register ``component`` under ``name``; returns a decorator when omitted."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self.kind} registry names must be non-empty strings, got {name!r}")
        if component is None:
            return partial(self.register, name, overwrite=overwrite)
        if name in self._loaders and not overwrite:
            raise ValueError(
                f"{self.kind} {name!r} is already registered; pass overwrite=True to replace it"
            )
        self._loaders[name] = lambda: component
        self._cache[name] = component
        return component

    def snapshot(self) -> tuple[dict[str, Callable[[], Any]], dict[str, Any]]:
        return dict(self._loaders), dict(self._cache)

    def restore(self, state: tuple[dict[str, Callable[[], Any]], dict[str, Any]]) -> None:
        loaders, cache = state
        self._loaders = dict(loaders)
        self._cache = dict(cache)

    def __getitem__(self, key: str) -> Any:
        if key not in self._cache:
            try:
                loader = self._loaders[key]
            except KeyError:
                raise KeyError(
                    f"Unknown {self.kind} {key!r}; registered: {sorted(self._loaders)}"
                ) from None
            self._cache[key] = loader()
        return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.register(key, value, overwrite=True)

    def __delitem__(self, key: str) -> None:
        del self._loaders[key]
        self._cache.pop(key, None)

    def __contains__(self, key: object) -> bool:
        return key in self._loaders

    def __iter__(self) -> Iterator[str]:
        return iter(self._loaders)

    def __len__(self) -> int:
        return len(self._loaders)

    def __repr__(self) -> str:
        return f"ComponentRegistry({self.kind!r}, {sorted(self._loaders)})"


def _lazy(module: str, name: str) -> Callable[[], Any]:
    def load() -> Any:
        from importlib import import_module

        return getattr(import_module(module, __package__), name)

    return load


PROCESSORS = ComponentRegistry(
    "processor",
    {
        "conv": _lazy(".processors.convolutional", "ConvolutionalProcessor"),
        "kaldi_fbank": _lazy(".processors.kaldi_fbank", "KaldiFbankProcessor"),
        "melspec": _lazy(".processors.melspec", "MelSpectrogramProcessor"),
        "wav2vec2_feature_extractor": _lazy(
            ".processors.wav2vec2", "Wav2Vec2FeatureExtractorProcessor"
        ),
    },
)

TOKENIZERS = ComponentRegistry(
    "tokenizer",
    {
        "none": _lazy(".tokenizers.noop", "NoOpTokenizer"),
        "patchify": _lazy(".tokenizers.patchify", "PatchifyTokenizer"),
    },
)

EMBEDDINGS = ComponentRegistry(
    "embedding",
    {
        "linear": _lazy(".embeddings.linear", "LinearEmbedding"),
        "conv": _lazy(".embeddings.conv", "ConvEmbedding"),
        "eat": _lazy(".embeddings.eat", "EATEmbedding"),
    },
)

EMBEDDING_POSITIONAL_ENCODINGS = ComponentRegistry(
    "positional_encoding",
    {
        "none": _lazy(".core.positional_encoding", "NoPositionalEncoding"),
        "learnable": _lazy(".core.positional_encoding", "LearnablePositionalEncoding"),
        "sinusoidal": _lazy(".core.positional_encoding", "SinusoidalPositionalEncoding"),
        "sinusoidal_2d": _lazy(".core.positional_encoding", "Sinusoidal2DPositionalEncoding"),
        "learnable_2d": _lazy(
            ".core.positional_encoding", "LearnableTimeFrequencyPositionalEncoding"
        ),
        "conv1d": _lazy(".core.positional_encoding", "ConvolutionalPositionalEncoding"),
    },
)

ENCODER_BLOCKS = ComponentRegistry(
    "encoder",
    {
        "transformer": _lazy(".encoders.transformer_block", "TransformerBlock"),
        "conformer": _lazy(".encoders.conformer_block", "ConformerBlock"),
        "branchformer": _lazy(".encoders.branchformer_block", "BranchformerBlock"),
        "ebranchformer": _lazy(".encoders.ebranchformer_block", "EBranchformerBlock"),
        "alt_transformer": _lazy(".encoders.alt_transformer_block", "AltTransformerBlock"),
    },
)

OBJECTIVES = ComponentRegistry(
    "objective",
    {
        "contrastive": _lazy(".objectives.contrastive.contrastive", "ContrastiveObjective"),
        "byol": _lazy(".objectives.byol.byol", "BYOLObjective"),
        "dino": _lazy(".objectives.dino.dino", "DINOObjective"),
        "jepa": _lazy(".objectives.jepa.jepa", "JEPAObjective"),
        "mae": _lazy(".objectives.mae.mae", "MAEObjective"),
        "ufo": _lazy(".objectives.ufo.ufo", "UFOObjective"),
        "wav2vec2": _lazy(".objectives.wav2vec2.wav2vec2", "Wav2Vec2Objective"),
        "hubert": _lazy(".objectives.hubert.hubert", "HuBERTObjective"),
    },
)

MASKING_STRATEGIES = ComponentRegistry(
    "masking strategy",
    {
        "channel": _lazy(".masking.channel", "ChannelMaskingStrategy"),
        "span": _lazy(".masking.span", "SpanMaskingStrategy"),
    },
)

ALL_COMPONENT_REGISTRIES: tuple[ComponentRegistry, ...] = (
    PROCESSORS,
    TOKENIZERS,
    EMBEDDINGS,
    EMBEDDING_POSITIONAL_ENCODINGS,
    ENCODER_BLOCKS,
    OBJECTIVES,
    MASKING_STRATEGIES,
)


def get_encoder_block_registry() -> ComponentRegistry:
    """Return encoder block classes keyed by encoder config ``type``."""
    return ENCODER_BLOCKS


def get_processor_registry() -> ComponentRegistry:
    """Return processors for raw waveform to feature conversion."""
    return PROCESSORS


def get_tokenizer_registry() -> ComponentRegistry:
    """Return tokenizers for feature to token conversion."""
    return TOKENIZERS


def get_embedding_registry() -> ComponentRegistry:
    """Return embeddings that project tokenizer output to ``d_model``."""
    return EMBEDDINGS


def get_embedding_positional_encoding_registry() -> ComponentRegistry:
    """Return embedding-owned positional encoding modules."""
    return EMBEDDING_POSITIONAL_ENCODINGS


def get_objective_registry() -> ComponentRegistry:
    """Return training objectives keyed by objective config ``type``."""
    return OBJECTIVES


def get_masking_strategy_registry() -> ComponentRegistry:
    """Return training-only masking strategies keyed by masking config ``type``."""
    return MASKING_STRATEGIES


__all__ = [
    "ACTIVATION_REGISTRY",
    "ALL_COMPONENT_REGISTRIES",
    "EMBEDDINGS",
    "EMBEDDING_POSITIONAL_ENCODINGS",
    "ENCODER_BLOCKS",
    "INITIALIZATION_REGISTRY",
    "MASKING_STRATEGIES",
    "OBJECTIVES",
    "PROCESSORS",
    "TOKENIZERS",
    "ComponentRegistry",
    "build_activation",
    "get_embedding_positional_encoding_registry",
    "get_embedding_registry",
    "get_encoder_block_registry",
    "get_masking_strategy_registry",
    "get_objective_registry",
    "get_processor_registry",
    "get_tokenizer_registry",
]

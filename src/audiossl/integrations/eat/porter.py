"""Weight porter: EAT Hugging Face checkpoint state_dict -> audiossl AudioModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.model import AudioModel

from .config import make_eat_base_config


def _convert_patch_embed_weight(conv_weight: torch.Tensor) -> torch.Tensor:
    """Convert EAT Conv2d patch weights into linear projection weights."""
    if conv_weight.ndim != 4 or conv_weight.shape[1] != 1:
        raise ValueError(
            "Expected EAT patch weight of shape (D, 1, kT, kF), "
            f"got {tuple(conv_weight.shape)}"
        )
    embed_dim, _, kernel_t, kernel_f = conv_weight.shape
    return conv_weight.squeeze(1).reshape(embed_dim, kernel_t * kernel_f)


def _block_key_pairs(depth: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index in range(depth):
        src = f"blocks.{index}"
        dst = f"encoder.blocks.{index}"
        pairs.extend(
            [
                (f"{src}.norm1.weight", f"{dst}.norm1.weight"),
                (f"{src}.norm1.bias", f"{dst}.norm1.bias"),
                (f"{src}.norm2.weight", f"{dst}.norm2.weight"),
                (f"{src}.norm2.bias", f"{dst}.norm2.bias"),
                (f"{src}.attn.qkv.weight", f"{dst}.attn.qkv.weight"),
                (f"{src}.attn.qkv.bias", f"{dst}.attn.qkv.bias"),
                (f"{src}.attn.proj.weight", f"{dst}.attn.proj.weight"),
                (f"{src}.attn.proj.bias", f"{dst}.attn.proj.bias"),
                (f"{src}.mlp.fc1.weight", f"{dst}.mlp.fc1.weight"),
                (f"{src}.mlp.fc1.bias", f"{dst}.mlp.fc1.bias"),
                (f"{src}.mlp.fc2.weight", f"{dst}.mlp.fc2.weight"),
                (f"{src}.mlp.fc2.bias", f"{dst}.mlp.fc2.bias"),
            ]
        )
    return pairs


def _normalize_eat_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        normalized[key] = value
    return normalized


def port_eat_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    model: AudioModel | None = None,
) -> dict[str, torch.Tensor]:
    """Map a Hugging Face EAT checkpoint into audiossl parameter names."""
    source = _normalize_eat_state_dict(state_dict)
    if model is None:
        model = AudioModel(make_eat_base_config())
    depth = len(model.encoder.blocks)
    target: dict[str, torch.Tensor] = {}

    patch_weight = source["local_encoder.proj.weight"]
    target["embedding.embedding.projection.weight"] = _convert_patch_embed_weight(patch_weight)
    if "local_encoder.proj.bias" in source:
        target["embedding.embedding.projection.bias"] = source["local_encoder.proj.bias"]

    target["embedding.embedding.positional_encoding.pe"] = source[
        "fixed_positional_encoder.positions"
    ]
    target["embedding.embedding.cls_token"] = source["extra_tokens"]
    target["embedding.embedding.entry_layer_norm.weight"] = source["pre_norm.weight"]
    target["embedding.embedding.entry_layer_norm.bias"] = source["pre_norm.bias"]

    for src_key, dst_key in _block_key_pairs(depth):
        target[dst_key] = source[src_key]

    return target


def load_eat_weights(
    checkpoint_path: Path | str,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> AudioModel:
    """Load an EAT checkpoint into an existing AudioModel."""
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path), device=str(map_location))
    else:
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload:
            state_dict = payload["model"]
        else:
            state_dict = payload

    ported = port_eat_state_dict(state_dict, model=model)
    incompatible = model.load_state_dict(ported, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "EAT load_state_dict reported incompatibilities: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model


def load_eat_base(
    checkpoint_path: Path | str,
    *,
    map_location: Any = "cpu",
    target_length: int = 1024,
) -> AudioModel:
    """Build EAT-base and load weights from a checkpoint path."""
    model = AudioModel(make_eat_base_config(target_length=target_length))
    return load_eat_weights(checkpoint_path, model, map_location=map_location)


def extract_features(model: AudioModel, mel: torch.Tensor) -> torch.Tensor:
    """Extract encoder features from normalized mel input ``(B, 1, T, F)``."""
    if mel.ndim != 4 or mel.shape[1] != 1:
        raise ValueError(
            "extract_features expected mel shape (B, 1, T, F), "
            f"got {tuple(mel.shape)}"
        )
    batch_size, _, time_frames, freq_bins = mel.shape
    features = mel.squeeze(1)
    mask = torch.ones(batch_size, time_frames, dtype=torch.bool, device=mel.device)
    feature_batch = AudioBatch(x=features, mask=mask)
    layout = model.token_layout(feature_batch)
    tokenized = model.tokenize(feature_batch)
    embedded = model.embed(tokenized, layout=layout)
    encoded = model.encode_embedded(embedded)
    return encoded.z


__all__ = [
    "extract_features",
    "load_eat_base",
    "load_eat_weights",
    "port_eat_state_dict",
]

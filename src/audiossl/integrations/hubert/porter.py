"""Weight porter: Hugging Face HuBERT state_dict -> audiossl AudioModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from audiossl.config import LinearEmbeddingConfig, Wav2Vec2FeatureExtractorProcessorConfig
from audiossl.modelling.model import AudioModel

from .config import make_hubert_base_ls960_config

_OPTIONAL_HUBERT_KEYS = frozenset({"masked_spec_embed"})
_OLD_WEIGHT_NORM_KEYS = (
    "encoder.pos_conv_embed.conv.weight_g",
    "encoder.pos_conv_embed.conv.weight_v",
)
_NEW_WEIGHT_NORM_KEYS = (
    "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
)


def _feature_extractor_key_pairs(
    depth: int,
    *,
    extractor_mode: str,
) -> list[tuple[str, str]]:
    """Map HuBERT feature-extractor keys to audiossl processor keys."""
    pairs: list[tuple[str, str]] = []
    for i in range(depth):
        pairs.append(
            (
                f"feature_extractor.conv_layers.{i}.conv.weight",
                f"processor.processor.conv_layers.{i}.0.weight",
            )
        )
        if extractor_mode == "default":
            if i == 0:
                pairs.extend(
                    [
                        (
                            "feature_extractor.conv_layers.0.layer_norm.weight",
                            "processor.processor.conv_layers.0.1.weight",
                        ),
                        (
                            "feature_extractor.conv_layers.0.layer_norm.bias",
                            "processor.processor.conv_layers.0.1.bias",
                        ),
                    ]
                )
        elif extractor_mode == "layer_norm":
            pairs.extend(
                [
                    (
                        f"feature_extractor.conv_layers.{i}.layer_norm.weight",
                        f"processor.processor.conv_layers.{i}.2.weight",
                    ),
                    (
                        f"feature_extractor.conv_layers.{i}.layer_norm.bias",
                        f"processor.processor.conv_layers.{i}.2.bias",
                    ),
                ]
            )
        else:
            raise ValueError(f"Unsupported HuBERT extractor_mode {extractor_mode!r}")
    return pairs


def _feature_projection_key_pairs(
    *,
    input_layer_norm: bool,
    post_pos_layer_norm: bool,
) -> list[tuple[str, str]]:
    """Map HuBERT feature projection and positional-encoding keys."""
    pairs = [
        (
            "feature_projection.projection.weight",
            "embedding.embedding.projector.projection.weight",
        ),
        (
            "feature_projection.projection.bias",
            "embedding.embedding.projector.projection.bias",
        ),
        ("encoder.pos_conv_embed.conv.bias", "embedding.embedding.positional_encoding.conv.bias"),
    ]
    if input_layer_norm:
        pairs.extend(
            [
                (
                    "feature_projection.layer_norm.weight",
                    "embedding.embedding.pre_layer_norm.weight",
                ),
                (
                    "feature_projection.layer_norm.bias",
                    "embedding.embedding.pre_layer_norm.bias",
                ),
            ]
        )
    if post_pos_layer_norm:
        pairs.extend(
            [
                ("encoder.layer_norm.weight", "embedding.embedding.post_layer_norm.weight"),
                ("encoder.layer_norm.bias", "embedding.embedding.post_layer_norm.bias"),
            ]
        )
    return pairs


def _encoder_key_pairs(depth: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i in range(depth):
        src = f"encoder.layers.{i}"
        dst = f"encoder.blocks.{i}"
        pairs.extend(
            [
                (
                    f"{src}.attention.out_proj.weight",
                    f"{dst}.mha.mha.out_proj.weight",
                ),
                (
                    f"{src}.attention.out_proj.bias",
                    f"{dst}.mha.mha.out_proj.bias",
                ),
                (f"{src}.layer_norm.weight", f"{dst}.norm1.weight"),
                (f"{src}.layer_norm.bias", f"{dst}.norm1.bias"),
                (
                    f"{src}.feed_forward.intermediate_dense.weight",
                    f"{dst}.mlp.fc1.weight",
                ),
                (
                    f"{src}.feed_forward.intermediate_dense.bias",
                    f"{dst}.mlp.fc1.bias",
                ),
                (
                    f"{src}.feed_forward.output_dense.weight",
                    f"{dst}.mlp.fc2.weight",
                ),
                (
                    f"{src}.feed_forward.output_dense.bias",
                    f"{dst}.mlp.fc2.bias",
                ),
                (f"{src}.final_layer_norm.weight", f"{dst}.norm2.weight"),
                (f"{src}.final_layer_norm.bias", f"{dst}.norm2.bias"),
            ]
        )
    return pairs


def _materialize_weight_norm(
    weight_g: torch.Tensor,
    weight_v: torch.Tensor,
) -> torch.Tensor:
    """Materialize Hugging Face positional conv weight norm into a dense weight."""
    if weight_g.ndim != 3 or weight_v.ndim != 3:
        raise ValueError(
            "Expected HuBERT positional conv weight norm tensors to be rank 3, "
            f"got {tuple(weight_g.shape)} and {tuple(weight_v.shape)}"
        )
    norm = weight_v.norm(dim=(0, 1), keepdim=True)
    return weight_g * (weight_v / norm)


def _pack_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.cat([q, k, v], dim=0)


def _pop_required(src: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in src:
        raise KeyError(f"HuBERT state_dict missing required key: {key!r}")
    return src.pop(key)


def _pop_pos_conv_weight(src: dict[str, torch.Tensor]) -> torch.Tensor:
    has_old = all(key in src for key in _OLD_WEIGHT_NORM_KEYS)
    has_new = all(key in src for key in _NEW_WEIGHT_NORM_KEYS)
    if has_old and has_new:
        raise ValueError(
            "HuBERT state_dict contains both old and new positional conv "
            "weight-norm key schemes"
        )
    if not has_old and not has_new:
        raise KeyError(
            "HuBERT state_dict missing positional conv weight-norm keys; "
            f"expected either {_OLD_WEIGHT_NORM_KEYS!r} or {_NEW_WEIGHT_NORM_KEYS!r}"
        )
    g_key, v_key = _OLD_WEIGHT_NORM_KEYS if has_old else _NEW_WEIGHT_NORM_KEYS
    return _materialize_weight_norm(src.pop(g_key), src.pop(v_key))


def _normalize_hubert_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Accept bare or ``hubert.``-prefixed Hugging Face checkpoint keys."""
    if not isinstance(state_dict, dict):
        raise TypeError(
            "hubert_state_dict must be dict[str, torch.Tensor], "
            f"got {type(state_dict).__name__}"
        )
    src = dict(state_dict)
    if not src:
        raise ValueError("HuBERT state_dict must not be empty")

    unprefixed = "feature_extractor.conv_layers.0.conv.weight" in src
    prefixed = "hubert.feature_extractor.conv_layers.0.conv.weight" in src
    if prefixed and not unprefixed:
        src = {
            key.removeprefix("hubert."): value
            for key, value in src.items()
            if key.startswith("hubert.")
        }
    return src


def port_hubert_state_dict(
    hubert_state_dict: dict[str, torch.Tensor],
    model: AudioModel,
) -> None:
    """Port a Hugging Face HuBERT backbone state_dict into ``model`` in place.

    Parameters
    ----------
    hubert_state_dict
        Checkpoint weights keyed like Hugging Face ``HubertModel``.
    model
        Target :class:`AudioModel` built from a compatible HuBERT config recipe.

    Notes
    -----
    Supports non-stable encoder layouts only. Positional conv weights may use
    either legacy or parametrization weight-norm key schemes.
    """

    src = _normalize_hubert_state_dict(hubert_state_dict)
    processor_config = model.config.processor
    embedding_config = model.config.embedding
    encoder_config = model.config.encoder
    if not isinstance(processor_config, Wav2Vec2FeatureExtractorProcessorConfig):
        raise TypeError(
            "port_hubert_state_dict requires Wav2Vec2FeatureExtractorProcessorConfig, "
            f"got {type(processor_config).__name__}"
        )
    if not isinstance(embedding_config, LinearEmbeddingConfig):
        raise TypeError(
            "port_hubert_state_dict requires LinearEmbeddingConfig, "
            f"got {type(embedding_config).__name__}"
        )
    if encoder_config.pre_norm or encoder_config.final_norm:
        raise ValueError(
            "port_hubert_state_dict supports non-stable HuBERT encoder layouts only"
        )

    pairs: list[tuple[str, str]] = []
    pairs.extend(
        _feature_extractor_key_pairs(
            len(processor_config.conv_dim),
            extractor_mode=processor_config.extractor_mode,
        )
    )
    pairs.extend(
        _feature_projection_key_pairs(
            input_layer_norm=embedding_config.input_layer_norm,
            post_pos_layer_norm=embedding_config.post_pos_layer_norm,
        )
    )
    pairs.extend(_encoder_key_pairs(encoder_config.num_layers))

    new_state: dict[str, torch.Tensor] = {}
    for src_key, dst_key in pairs:
        new_state[dst_key] = _pop_required(src, src_key)

    for i in range(encoder_config.num_layers):
        src_prefix = f"encoder.layers.{i}.attention"
        dst_prefix = f"encoder.blocks.{i}.mha.mha"
        q_w = _pop_required(src, f"{src_prefix}.q_proj.weight")
        k_w = _pop_required(src, f"{src_prefix}.k_proj.weight")
        v_w = _pop_required(src, f"{src_prefix}.v_proj.weight")
        q_b = _pop_required(src, f"{src_prefix}.q_proj.bias")
        k_b = _pop_required(src, f"{src_prefix}.k_proj.bias")
        v_b = _pop_required(src, f"{src_prefix}.v_proj.bias")
        new_state[f"{dst_prefix}.in_proj_weight"] = _pack_qkv(q_w, k_w, v_w)
        new_state[f"{dst_prefix}.in_proj_bias"] = _pack_qkv(q_b, k_b, v_b)

    new_state["embedding.embedding.positional_encoding.conv.weight"] = _pop_pos_conv_weight(src)

    for opt_key in list(src):
        if opt_key in _OPTIONAL_HUBERT_KEYS or opt_key.startswith(("lm_head.", "projector.")):
            src.pop(opt_key)
    if src:
        raise ValueError(
            "HuBERT state_dict contains unexpected keys after porting: "
            f"{sorted(src)}"
        )

    model_state = model.state_dict()
    for dst_key, tensor in new_state.items():
        if dst_key not in model_state:
            raise KeyError(f"Destination key {dst_key!r} not present in model.state_dict()")
        if tuple(model_state[dst_key].shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch at {dst_key!r}: "
                f"model expects {tuple(model_state[dst_key].shape)}, "
                f"porter produced {tuple(tensor.shape)}"
            )

    missing = sorted(set(model_state) - set(new_state))
    if missing:
        raise ValueError(
            "Port left model parameters uninitialised from the HuBERT checkpoint: "
            f"{missing}"
        )

    incompatible = model.load_state_dict(new_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "load_state_dict reported incompatibilities: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def load_hubert_base_ls960(
    ckpt_path: str | Path,
    *,
    map_location: Any = "cpu",
) -> AudioModel:
    """Build an :class:`AudioModel` and load ``facebook/hubert-base-ls960`` weights.

    Parameters
    ----------
    ckpt_path
        Path to a ``.bin`` or ``.safetensors`` Hugging Face checkpoint file.
    map_location
        Device or storage location passed to the checkpoint loader.
    """

    model = AudioModel(make_hubert_base_ls960_config())
    load_hubert_weights(ckpt_path, model, map_location=map_location)
    return model


def load_hubert_weights(
    ckpt_path: str | Path,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> None:
    """Load a Hugging Face HuBERT checkpoint into a compatible :class:`AudioModel`.

    Parameters
    ----------
    ckpt_path
        Path to a ``.bin`` or ``.safetensors`` Hugging Face checkpoint file.
    model
        Target model built from a compatible HuBERT config recipe.
    map_location
        Device or storage location passed to the checkpoint loader.
    """

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"HuBERT checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path), device=str(map_location))
    else:
        dump = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state_dict = _extract_state_dict(dump)
    port_hubert_state_dict(state_dict, model)


def _extract_state_dict(dump: Any) -> dict[str, torch.Tensor]:
    if isinstance(dump, dict) and all(isinstance(value, torch.Tensor) for value in dump.values()):
        return dump
    if isinstance(dump, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = dump.get(key)
            if isinstance(value, dict):
                return value
    raise ValueError("Unexpected HuBERT checkpoint structure")


__all__ = [
    "load_hubert_base_ls960",
    "load_hubert_weights",
    "port_hubert_state_dict",
]

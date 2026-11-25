"""Weight porter: Hugging Face wav2vec2 state_dict -> audiossl AudioModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from audiossl.config import Wav2Vec2FeatureExtractorProcessorConfig
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives.wav2vec2.wav2vec2 import Wav2Vec2Objective

from .config import make_wav2vec2_base_config

_OPTIONAL_W2V2_KEYS = frozenset({"masked_spec_embed"})
_OLD_WEIGHT_NORM_KEYS = (
    "encoder.pos_conv_embed.conv.weight_g",
    "encoder.pos_conv_embed.conv.weight_v",
)
_NEW_WEIGHT_NORM_KEYS = (
    "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
)


def _feature_extractor_key_pairs(depth: int) -> list[tuple[str, str]]:
    pairs = [
        (
            "feature_extractor.conv_layers.0.conv.weight",
            "processor.processor.conv_layers.0.0.weight",
        ),
        (
            "feature_extractor.conv_layers.0.layer_norm.weight",
            "processor.processor.conv_layers.0.1.weight",
        ),
        (
            "feature_extractor.conv_layers.0.layer_norm.bias",
            "processor.processor.conv_layers.0.1.bias",
        ),
    ]
    for i in range(1, depth):
        pairs.append(
            (
                f"feature_extractor.conv_layers.{i}.conv.weight",
                f"processor.processor.conv_layers.{i}.0.weight",
            )
        )
    return pairs


def _feature_projection_key_pairs() -> list[tuple[str, str]]:
    return [
        (
            "feature_projection.layer_norm.weight",
            "embedding.embedding.pre_layer_norm.weight",
        ),
        (
            "feature_projection.layer_norm.bias",
            "embedding.embedding.pre_layer_norm.bias",
        ),
        (
            "feature_projection.projection.weight",
            "embedding.embedding.projector.projection.weight",
        ),
        (
            "feature_projection.projection.bias",
            "embedding.embedding.projector.projection.bias",
        ),
        ("encoder.pos_conv_embed.conv.bias", "embedding.embedding.positional_encoding.conv.bias"),
        ("encoder.layer_norm.weight", "embedding.embedding.post_layer_norm.weight"),
        ("encoder.layer_norm.bias", "embedding.embedding.post_layer_norm.bias"),
    ]


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
    """Materialize `nn.utils.weight_norm(..., name="weight", dim=2)`."""

    if weight_g.ndim != 3 or weight_v.ndim != 3:
        raise ValueError(
            "Expected wav2vec2 positional conv weight norm tensors to be rank 3, "
            f"got {tuple(weight_g.shape)} and {tuple(weight_v.shape)}"
        )
    if weight_g.shape[-1] != weight_v.shape[-1]:
        raise ValueError(
            "wav2vec2 positional conv weight_g and weight_v must share kernel size, "
            f"got {tuple(weight_g.shape)} and {tuple(weight_v.shape)}"
        )
    try:
        torch.broadcast_shapes(weight_g.shape, weight_v.shape)
    except RuntimeError as exc:
        raise ValueError(
            "wav2vec2 positional conv weight_g must be broadcastable to weight_v, "
            f"got {tuple(weight_g.shape)} and {tuple(weight_v.shape)}"
        ) from exc

    norm = weight_v.norm(dim=(0, 1), keepdim=True)
    return weight_g * (weight_v / norm)


def _pack_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.cat([q, k, v], dim=0)


def _pack_qkv_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.cat([q, k, v], dim=0)


def _pop_required(src: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in src:
        raise KeyError(f"wav2vec2 state_dict missing required key: {key!r}")
    return src.pop(key)


def _pop_pos_conv_weight(src: dict[str, torch.Tensor]) -> torch.Tensor:
    has_old = all(key in src for key in _OLD_WEIGHT_NORM_KEYS)
    has_new = all(key in src for key in _NEW_WEIGHT_NORM_KEYS)
    if has_old and has_new:
        raise ValueError(
            "wav2vec2 state_dict contains both old and new positional conv "
            "weight-norm key schemes"
        )
    if not has_old and not has_new:
        raise KeyError(
            "wav2vec2 state_dict missing positional conv weight-norm keys; "
            f"expected either {_OLD_WEIGHT_NORM_KEYS!r} or {_NEW_WEIGHT_NORM_KEYS!r}"
        )

    g_key, v_key = _OLD_WEIGHT_NORM_KEYS if has_old else _NEW_WEIGHT_NORM_KEYS
    weight_g = src.pop(g_key)
    weight_v = src.pop(v_key)
    return _materialize_weight_norm(weight_g, weight_v)


def port_wav2vec2_state_dict(
    wav2vec2_state_dict: dict[str, torch.Tensor],
    model: AudioModel,
) -> None:
    """Port a Hugging Face wav2vec2 state_dict into ``model`` in place.

    Parameters
    ----------
    wav2vec2_state_dict
        Checkpoint weights keyed like Hugging Face ``Wav2Vec2Model``.
    model
        Target :class:`AudioModel` built from a compatible wav2vec2 config recipe.

    Notes
    -----
    Positional conv weights may use either legacy or parametrization weight-norm
    key schemes. Pretraining-only keys such as ``masked_spec_embed`` are ignored.
    """

    src = _normalize_wav2vec2_state_dict(wav2vec2_state_dict)
    processor_config = model.config.processor
    encoder_config = model.config.encoder
    if not isinstance(processor_config, Wav2Vec2FeatureExtractorProcessorConfig):
        raise TypeError(
            "port_wav2vec2_state_dict requires Wav2Vec2FeatureExtractorProcessorConfig, "
            f"got {type(processor_config).__name__}"
        )

    pairs: list[tuple[str, str]] = []
    pairs.extend(_feature_extractor_key_pairs(len(processor_config.conv_dim)))
    pairs.extend(_feature_projection_key_pairs())
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
        new_state[f"{dst_prefix}.in_proj_bias"] = _pack_qkv_bias(q_b, k_b, v_b)

    new_state["embedding.embedding.positional_encoding.conv.weight"] = _pop_pos_conv_weight(src)

    for opt_key in list(src):
        if opt_key in _OPTIONAL_W2V2_KEYS:
            src.pop(opt_key)

    if src:
        raise ValueError(
            "wav2vec2 state_dict contains unexpected keys after porting: "
            f"{sorted(src)}"
        )

    model_state = model.state_dict()
    for dst_key, tensor in new_state.items():
        if dst_key not in model_state:
            raise KeyError(
                f"Destination key {dst_key!r} not present in model.state_dict()"
            )
        if tuple(model_state[dst_key].shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch at {dst_key!r}: "
                f"model expects {tuple(model_state[dst_key].shape)}, "
                f"porter produced {tuple(tensor.shape)}"
            )

    missing = sorted(set(model_state) - set(new_state))
    if missing:
        raise ValueError(
            "Port left model parameters uninitialised from the wav2vec2 "
            f"checkpoint: {missing}"
        )

    incompatible = model.load_state_dict(new_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "load_state_dict reported incompatibilities: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def port_wav2vec2_pretraining_state_dict(
    wav2vec2_pretraining_state_dict: dict[str, torch.Tensor],
    objective: Wav2Vec2Objective,
) -> None:
    """Port a Hugging Face ``Wav2Vec2ForPreTraining`` state_dict into an objective.

    Parameters
    ----------
    wav2vec2_pretraining_state_dict
        Full pretraining checkpoint, optionally prefixed with ``wav2vec2.``.
    objective
        Target :class:`Wav2Vec2Objective` whose backbone and quantizer are filled.

    Notes
    -----
    The backbone is ported through :func:`port_wav2vec2_state_dict`. Quantizer,
    projection, and mask-embedding keys are mapped into the objective modules.
    """

    if not isinstance(objective, Wav2Vec2Objective):
        raise TypeError(
            "port_wav2vec2_pretraining_state_dict requires Wav2Vec2Objective, "
            f"got {type(objective).__name__}"
        )
    src = dict(wav2vec2_pretraining_state_dict)
    model_src = {
        key.removeprefix("wav2vec2."): value
        for key, value in src.items()
        if key.startswith("wav2vec2.")
    }
    if not model_src:
        model_src = {
            key: value
            for key, value in src.items()
            if key.startswith("feature_extractor.")
            or key.startswith("feature_projection.")
            or key.startswith("encoder.")
            or key == "masked_spec_embed"
        }
    port_wav2vec2_state_dict(model_src, objective.model)

    new_state: dict[str, torch.Tensor] = {}
    pairs = [
        ("quantizer.weight_proj.weight", "quantizer.weight_proj.weight"),
        ("quantizer.weight_proj.bias", "quantizer.weight_proj.bias"),
        ("quantizer.codevectors", "quantizer.codevectors"),
        ("project_hid.weight", "project_hid.weight"),
        ("project_hid.bias", "project_hid.bias"),
        ("project_q.weight", "project_q.weight"),
        ("project_q.bias", "project_q.bias"),
    ]
    for src_key, dst_key in pairs:
        new_state[dst_key] = _pop_required(src, src_key)
    mask_key = (
        "wav2vec2.masked_spec_embed"
        if "wav2vec2.masked_spec_embed" in src
        else "masked_spec_embed"
    )
    new_state["mask_emb"] = _pop_required(src, mask_key)

    for key in list(src):
        if key.startswith("wav2vec2."):
            src.pop(key)
    if src:
        raise ValueError(
            "wav2vec2 pretraining state_dict contains unexpected keys after porting: "
            f"{sorted(src)}"
        )

    objective_state = objective.state_dict()
    for dst_key, tensor in new_state.items():
        if dst_key not in objective_state:
            raise KeyError(
                f"Destination key {dst_key!r} not present in objective.state_dict()"
            )
        if tuple(objective_state[dst_key].shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch at {dst_key!r}: "
                f"objective expects {tuple(objective_state[dst_key].shape)}, "
                f"porter produced {tuple(tensor.shape)}"
            )
    objective.load_state_dict(new_state, strict=False)


def _normalize_wav2vec2_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Accept bare or ``wav2vec2.``-prefixed Hugging Face checkpoint keys."""
    if not isinstance(state_dict, dict):
        raise TypeError(
            "wav2vec2_state_dict must be dict[str, torch.Tensor], "
            f"got {type(state_dict).__name__}"
        )
    src = dict(state_dict)
    if not src:
        raise ValueError("wav2vec2 state_dict must not be empty")

    unprefixed = "feature_extractor.conv_layers.0.conv.weight" in src
    prefixed = "wav2vec2.feature_extractor.conv_layers.0.conv.weight" in src
    if prefixed and not unprefixed:
        src = {
            key.removeprefix("wav2vec2."): value
            for key, value in src.items()
            if key.startswith("wav2vec2.")
        }
    return src


def load_wav2vec2_base(
    ckpt_path: str | Path,
    *,
    map_location: Any = "cpu",
) -> AudioModel:
    """Build an :class:`AudioModel` and load ``facebook/wav2vec2-base`` weights.

    Parameters
    ----------
    ckpt_path
        Path to a ``.bin`` or ``.safetensors`` Hugging Face checkpoint file.
    map_location
        Device or storage location passed to the checkpoint loader.
    """

    model = AudioModel(make_wav2vec2_base_config())
    load_wav2vec2_weights(ckpt_path, model, map_location=map_location)
    return model


def load_wav2vec2_weights(
    ckpt_path: str | Path,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> None:
    """Load a Hugging Face wav2vec2 checkpoint into a compatible :class:`AudioModel`.

    Parameters
    ----------
    ckpt_path
        Path to a ``.bin`` or ``.safetensors`` Hugging Face checkpoint file.
    model
        Target model built from a compatible wav2vec2 config recipe.
    map_location
        Device or storage location passed to the checkpoint loader.
    """

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"wav2vec2 checkpoint not found: {ckpt_path}")

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path), device=str(map_location))
    else:
        dump = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state_dict = _extract_state_dict(dump)

    port_wav2vec2_state_dict(state_dict, model)


def _extract_state_dict(dump: Any) -> dict[str, torch.Tensor]:
    if isinstance(dump, dict) and all(isinstance(value, torch.Tensor) for value in dump.values()):
        return dump
    if isinstance(dump, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = dump.get(key)
            if isinstance(value, dict):
                return value
    raise ValueError("Unexpected wav2vec2 checkpoint structure")


__all__ = [
    "load_wav2vec2_base",
    "load_wav2vec2_weights",
    "port_wav2vec2_pretraining_state_dict",
    "port_wav2vec2_state_dict",
]

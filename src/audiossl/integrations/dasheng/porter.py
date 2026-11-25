"""Weight porter: Dasheng checkpoint state_dict -> audiossl AudioModel."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from audiossl.modelling.model import AudioModel

from .config import make_dasheng_base_config

if TYPE_CHECKING:
    from audiossl.modelling.objectives.mae import MAEObjective

# CLS / token positional entries are only present when `pooling='token'`,
# which the released variants don't use, but we list them defensively.
_OPTIONAL_DASHENG_KEYS = frozenset({"cls_token", "token_pos_embed"})


def _block_key_pairs(depth: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i in range(depth):
        src = f"blocks.{i}"
        dst = f"encoder.blocks.{i}"
        pairs.extend(
            [
                (f"{src}.norm1.weight", f"{dst}.norm1.weight"),
                (f"{src}.norm1.bias", f"{dst}.norm1.bias"),
                (f"{src}.norm2.weight", f"{dst}.norm2.weight"),
                (f"{src}.norm2.bias", f"{dst}.norm2.bias"),
                (f"{src}.attn.qkv.weight", f"{dst}.mha.mha.in_proj_weight"),
                (f"{src}.attn.qkv.bias", f"{dst}.mha.mha.in_proj_bias"),
                (f"{src}.attn.proj.weight", f"{dst}.mha.mha.out_proj.weight"),
                (f"{src}.attn.proj.bias", f"{dst}.mha.mha.out_proj.bias"),
                (f"{src}.mlp.fc1.weight", f"{dst}.mlp.fc1.weight"),
                (f"{src}.mlp.fc1.bias", f"{dst}.mlp.fc1.bias"),
                (f"{src}.mlp.fc2.weight", f"{dst}.mlp.fc2.weight"),
                (f"{src}.mlp.fc2.bias", f"{dst}.mlp.fc2.bias"),
            ]
        )
    return pairs


def _dasheng_decoder_block_key_pairs(depth: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i in range(depth):
        src = f"decoder.blocks.{i}"
        dst = f"decoder.blocks.blocks.{i}"
        pairs.extend(
            [
                (f"{src}.norm1.weight", f"{dst}.norm1.weight"),
                (f"{src}.norm1.bias", f"{dst}.norm1.bias"),
                (f"{src}.norm2.weight", f"{dst}.norm2.weight"),
                (f"{src}.norm2.bias", f"{dst}.norm2.bias"),
                (f"{src}.attn.qkv.weight", f"{dst}.mha.mha.in_proj_weight"),
                (f"{src}.attn.qkv.bias", f"{dst}.mha.mha.in_proj_bias"),
                (f"{src}.attn.proj.weight", f"{dst}.mha.mha.out_proj.weight"),
                (f"{src}.attn.proj.bias", f"{dst}.mha.mha.out_proj.bias"),
                (f"{src}.mlp.fc1.weight", f"{dst}.mlp.fc1.weight"),
                (f"{src}.mlp.fc1.bias", f"{dst}.mlp.fc1.bias"),
                (f"{src}.mlp.fc2.weight", f"{dst}.mlp.fc2.weight"),
                (f"{src}.mlp.fc2.bias", f"{dst}.mlp.fc2.bias"),
            ]
        )
    return pairs


def _front_end_key_pairs() -> list[tuple[str, str]]:
    # Buffers; copying is defensive (same params -> same buffers), but
    # makes the porter robust to mel-buffer differences if torchaudio
    # implementation details ever change.
    return [
        (
            "front_end.0.spectrogram.window",
            "processor.processor.melspec.spectrogram.window",
        ),
        (
            "front_end.0.mel_scale.fb",
            "processor.processor.melspec.mel_scale.fb",
        ),
    ]


def _init_bn_key_pairs() -> list[tuple[str, str]]:
    src = "init_bn.1"
    dst = "processor.processor.batch_norm.batch_norm"
    return [
        (f"{src}.weight", f"{dst}.weight"),
        (f"{src}.bias", f"{dst}.bias"),
        (f"{src}.running_mean", f"{dst}.running_mean"),
        (f"{src}.running_var", f"{dst}.running_var"),
        (f"{src}.num_batches_tracked", f"{dst}.num_batches_tracked"),
    ]


def _convert_patch_embed_weight(conv_weight: torch.Tensor) -> torch.Tensor:
    """Convert Dasheng patch weights into this repo's time-major token order.

    Dasheng stores the patch kernel as ``(D, 1, F, T)``. The Conv2d reads
    patches in ``(F, T)`` row-major order over ``(B, 1, F, T)``.

    Our :class:`PatchifyTokenizer` flattens patches as ``(T, F)`` row-major
    (time outer, frequency inner), then applies a linear map over
    ``T * F`` features. Permuting ``(F, T) -> (T, F)`` and row-major flatten
    aligns the two layouts.
    """
    if conv_weight.ndim != 4 or conv_weight.shape[1] != 1:
        raise ValueError(
            "Expected Dasheng patch_embed.proj.weight of shape (D, 1, F, T), "
            f"got {tuple(conv_weight.shape)}"
        )
    e, _, kf, kt = conv_weight.shape
    return conv_weight.squeeze(1).permute(0, 2, 1).contiguous().reshape(e, kt * kf)


def _convert_time_pos_embed(time_pos_embed: torch.Tensor) -> torch.Tensor:
    """Convert Dasheng time positional embeddings to ``(1, T, D)`` layout."""
    if time_pos_embed.ndim != 4 or time_pos_embed.shape[2] != 1:
        raise ValueError(
            "Expected Dasheng time_pos_embed of shape (1, D, 1, T), "
            f"got {tuple(time_pos_embed.shape)}"
        )
    return time_pos_embed.squeeze(2).transpose(1, 2).contiguous()


def _convert_freq_pos_embed(freq_pos_embed: torch.Tensor) -> torch.Tensor:
    """Convert Dasheng frequency positional embeddings to ``(1, F, D)`` layout."""
    if freq_pos_embed.ndim != 4 or freq_pos_embed.shape[3] != 1:
        raise ValueError(
            "Expected Dasheng freq_pos_embed of shape (1, D, F, 1), "
            f"got {tuple(freq_pos_embed.shape)}"
        )
    return freq_pos_embed.squeeze(3).transpose(1, 2).contiguous()


def port_dasheng_state_dict(
    dasheng_state_dict: dict[str, torch.Tensor], model: AudioModel
) -> None:
    """Port a Dasheng encoder state_dict into ``model`` in place.

    Parameters
    ----------
    dasheng_state_dict
        Encoder weights from a Dasheng training checkpoint ``model`` entry.
    model
        Target :class:`AudioModel` built from :func:`make_dasheng_base_config`
        or a compatible recipe.

    Notes
    -----
    After the call, every parameter in ``model`` is bound to a tensor derived
    from ``dasheng_state_dict``, and every expected source key has been
    consumed.
    """
    src = dict(dasheng_state_dict)
    encoder_config = model.config.encoder
    depth = encoder_config.num_layers

    pairs: list[tuple[str, str]] = []
    pairs.extend(_front_end_key_pairs())
    pairs.extend(_init_bn_key_pairs())
    pairs.extend(_block_key_pairs(depth))
    pairs.append(("norm.weight", "encoder.final_norm.weight"))
    pairs.append(("norm.bias", "encoder.final_norm.bias"))

    new_state: dict[str, torch.Tensor] = {}
    for src_key, dst_key in pairs:
        if src_key not in src:
            raise KeyError(f"Dasheng state_dict missing required key: {src_key!r}")
        new_state[dst_key] = src.pop(src_key)

    if "patch_embed.proj.weight" not in src:
        raise KeyError("Dasheng state_dict missing 'patch_embed.proj.weight'")
    if "patch_embed.proj.bias" not in src:
        raise KeyError("Dasheng state_dict missing 'patch_embed.proj.bias'")

    conv_weight = src.pop("patch_embed.proj.weight")
    conv_bias = src.pop("patch_embed.proj.bias")

    linear_weight_key = "embedding.embedding.projector.projection.weight"
    linear_bias_key = "embedding.embedding.projector.projection.bias"
    new_state[linear_weight_key] = _convert_patch_embed_weight(conv_weight)
    new_state[linear_bias_key] = conv_bias

    if "time_pos_embed" not in src:
        raise KeyError("Dasheng state_dict missing 'time_pos_embed'")
    if "freq_pos_embed" not in src:
        raise KeyError("Dasheng state_dict missing 'freq_pos_embed'")
    time_pe = _convert_time_pos_embed(src.pop("time_pos_embed"))
    freq_pe = _convert_freq_pos_embed(src.pop("freq_pos_embed"))
    new_state["embedding.embedding.positional_encoding.time_pe"] = time_pe
    new_state["embedding.embedding.positional_encoding.freq_pe"] = freq_pe

    for opt_key in list(src):
        if opt_key in _OPTIONAL_DASHENG_KEYS:
            src.pop(opt_key)

    if src:
        raise ValueError(
            "Dasheng state_dict contains unexpected keys after porting: "
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
            "Port left model parameters uninitialised from the Dasheng "
            f"checkpoint: {missing}"
        )

    incompatible = model.load_state_dict(new_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "load_state_dict reported incompatibilities: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def port_dasheng_mae_state_dict(
    state_dict: dict[str, torch.Tensor],
    objective: MAEObjective,
) -> None:
    """Port a Dasheng MAE decoder state_dict into :class:`MAEObjective`.

    Parameters
    ----------
    state_dict
        Full Dasheng MAE checkpoint or decoder-only weights.
    objective
        Target objective whose decoder blocks are filled from ``state_dict``.

    Notes
    -----
    The encoder side is covered by :func:`port_dasheng_state_dict`. Dasheng's
    1D ``decoder.pos_embed`` is intentionally skipped because audiossl MAE uses
    embedding-side 2D positional encodings.
    """

    src = dict(state_dict)
    depth = objective.config.decoder.num_layers

    pairs: list[tuple[str, str]] = [
        ("decoder.input_proj.weight", "decoder.input_proj.weight"),
        ("decoder.input_proj.bias", "decoder.input_proj.bias"),
        ("decoder.mask_token", "decoder.mask_token"),
    ]
    pairs.extend(_dasheng_decoder_block_key_pairs(depth))
    pairs.extend(
        [
            ("decoder.norm.weight", "decoder.blocks.final_norm.weight"),
            ("decoder.norm.bias", "decoder.blocks.final_norm.bias"),
            ("decoder.outputlayer.weight", "decoder.outputlayer.weight"),
            ("decoder.outputlayer.bias", "decoder.outputlayer.bias"),
        ]
    )

    new_state: dict[str, torch.Tensor] = {}
    for src_key, dst_key in pairs:
        if src_key not in src:
            raise KeyError(f"Dasheng MAE state_dict missing required key: {src_key!r}")
        new_state[dst_key] = src.pop(src_key)

    src.pop("decoder.pos_embed", None)
    if src:
        raise ValueError(
            "Dasheng MAE state_dict contains unexpected keys after porting: "
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
        objective_state[dst_key] = tensor

    incompatible = objective.load_state_dict(objective_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "load_state_dict reported incompatibilities: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def load_dasheng_base(
    ckpt_path: str | Path,
    *,
    target_length: int = 1008,
    map_location: Any = "cpu",
) -> AudioModel:
    """Build an :class:`AudioModel` and load a Dasheng ``dasheng_base`` checkpoint.

    Parameters
    ----------
    ckpt_path
        Path to a Dasheng training checkpoint with a top-level ``model`` entry.
    target_length
        Mel-frame count passed to :func:`make_dasheng_base_config`.
    map_location
        Device or storage location passed to :func:`torch.load`.

    Notes
    -----
    Checkpoint ``config`` metadata is ignored; the model is always built from
    :func:`make_dasheng_base_config`.
    """
    model = AudioModel(make_dasheng_base_config(target_length=target_length))
    load_dasheng_weights(ckpt_path, model, map_location=map_location)
    return model


def load_dasheng_weights(
    ckpt_path: str | Path,
    model: AudioModel,
    *,
    map_location: Any = "cpu",
) -> None:
    """Load a Dasheng-format checkpoint into a compatible :class:`AudioModel`.

    Parameters
    ----------
    ckpt_path
        Path to a Dasheng training checkpoint with a top-level ``model`` entry.
    model
        Target model built from a compatible Dasheng config recipe.
    map_location
        Device or storage location passed to :func:`torch.load`.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Dasheng checkpoint not found: {ckpt_path}")

    dump = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if not isinstance(dump, dict) or "model" not in dump:
        raise ValueError(
            f"Unexpected checkpoint structure at {ckpt_path}; expected "
            "a dict with a 'model' key"
        )
    port_dasheng_state_dict(dump["model"], model)


__all__ = [
    "load_dasheng_base",
    "load_dasheng_weights",
    "port_dasheng_mae_state_dict",
    "port_dasheng_state_dict",
]

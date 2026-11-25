"""JSON loading helpers for :class:`~audiossl.config.AudioSSLTrainingConfig`."""

from __future__ import annotations

import json
import typing
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Literal, cast, get_args, get_origin, get_type_hints

from .config import AudioSSLTrainingConfig
from .registry import CONFIG_CATEGORIES, resolve_registered_config
from .validate_config import validate_config


def load_config(src: Any, validate: bool = True) -> AudioSSLTrainingConfig:
    """Load an :class:`AudioSSLTrainingConfig` from JSON-like input.

    ``src`` may be a mapping, a JSON string/bytes payload, a path to a JSON
    file, a file-like object with ``read()``, or an existing
    ``AudioSSLTrainingConfig`` instance.
    """
    if isinstance(src, AudioSSLTrainingConfig):
        config = src
    else:
        data = _load_data(src)
        config = _build_dataclass(AudioSSLTrainingConfig, data, "AudioSSLTrainingConfig")

    if validate:
        validate_config(config)
    return config


def _load_data(src: Any) -> Any:
    if isinstance(src, Mapping):
        return dict(src)

    if hasattr(src, "read"):
        return _loads(src.read())

    if isinstance(src, Path):
        return _loads(src.read_text(encoding="utf-8"))

    if isinstance(src, str):
        path = Path(src)
        try:
            if path.exists():
                return _loads(path.read_text(encoding="utf-8"))
        except OSError:
            pass
        return _loads(src)

    if isinstance(src, bytes | bytearray):
        return _loads(bytes(src).decode("utf-8"))

    raise TypeError(
        "load_config expected mapping, JSON string/bytes, path, file-like object, "
        f"or AudioSSLTrainingConfig, got {type(src).__name__}"
    )


def _loads(payload: str | bytes) -> Any:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def _build_value(annotation: Any, value: Any, path: str) -> Any:
    if annotation is Any:
        return value

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        if value not in allowed:
            raise ValueError(f"{path} must be one of {allowed!r}, got {value!r}")
        return value

    if _is_union(annotation):
        return _build_union(annotation, value, path)

    if is_dataclass(annotation):
        return _build_dataclass(cast(type, annotation), value, path)

    if origin is tuple:
        return _build_tuple(annotation, value, path)

    if origin is list:
        (item_type,) = get_args(annotation) or (Any,)
        if not isinstance(value, list):
            raise TypeError(f"{path} must be list, got {type(value).__name__}")
        return [
            _build_value(item_type, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if origin is dict:
        key_type, value_type = get_args(annotation) or (Any, Any)
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be mapping, got {type(value).__name__}")
        return {
            _build_value(key_type, key, f"{path}.<key>"): _build_value(
                value_type,
                item,
                f"{path}.{key}",
            )
            for key, item in value.items()
        }

    return value


def _build_dataclass(cls: type, data: Any, path: str) -> Any:
    if isinstance(data, cls):
        return data
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be mapping, got {type(data).__name__}")

    field_names = {field.name for field in fields(cls)}
    extras = set(data) - field_names
    if extras:
        extra = sorted(extras)[0]
        raise TypeError(f"Unknown field {path}.{extra}")

    hints = get_type_hints(cls)
    kwargs = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        field_path = f"{path}.{field.name}"
        kwargs[field.name] = _build_value(hints[field.name], data[field.name], field_path)
    return cls(**kwargs)


def _build_union(annotation: Any, value: Any, path: str) -> Any:
    args = get_args(annotation)
    if value is None and NoneType in args:
        return None

    non_none_args = tuple(arg for arg in args if arg is not NoneType)
    dataclass_args = tuple(
        cast(type, arg) for arg in non_none_args if is_dataclass(arg)
    )
    if dataclass_args:
        if isinstance(value, tuple(dataclass_args)):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be mapping, got {type(value).__name__}")
        if len(dataclass_args) == 1 and not _literal_type_values(dataclass_args[0]):
            return _build_dataclass(dataclass_args[0], value, path)
        if "type" not in value:
            candidates = ", ".join(cls.__name__ for cls in dataclass_args)
            raise ValueError(f"{path} requires a 'type' field to choose one of {candidates}")
        discriminator = value["type"]
        for candidate in dataclass_args:
            if discriminator in _literal_type_values(candidate):
                return _build_dataclass(candidate, value, path)
        registered = resolve_registered_config(dataclass_args, discriminator)
        if registered is not None:
            return _build_dataclass(registered, value, path)
        candidates = ", ".join(cls.__name__ for cls in dataclass_args)
        raise ValueError(
            f"{path}.type={discriminator!r} does not match any of {candidates}"
        )

    last_error: Exception | None = None
    for candidate in non_none_args:
        try:
            return _build_value(candidate, value, path)
        except (TypeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return value


def _build_tuple(annotation: Any, value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{path} must be list or tuple, got {type(value).__name__}")

    args = get_args(annotation)
    if not args:
        return tuple(value)

    if len(args) == 2 and args[1] is Ellipsis:
        item_type = args[0]
        return tuple(
            _build_value(item_type, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )

    if len(value) != len(args):
        raise ValueError(f"{path} must contain exactly {len(args)} values")
    return tuple(
        _build_value(item_type, item, f"{path}[{index}]")
        for index, (item_type, item) in enumerate(zip(args, value, strict=True))
    )


def _literal_type_values(cls: type) -> tuple[Any, ...]:
    hints = get_type_hints(cls)
    type_hint = hints.get("type")
    if get_origin(type_hint) is not Literal:
        return ()
    return get_args(type_hint)


def _is_union(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin is typing.Union or origin is UnionType or isinstance(annotation, UnionType)


__all__ = ["load_config"]

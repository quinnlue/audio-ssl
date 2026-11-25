"""Open registration of component config classes and their validators.

Built-in configs are discriminated by their ``type: Literal[...]`` field inside
typed unions (``ProcessorConfig``, ``EncoderConfig``, ...). Plug-in components
register their config class under the same categories so that
:func:`~audiossl.config.load_config` can build them from JSON and
:func:`~audiossl.config.validate_config` can validate them::

    @register_config("processor")
    @dataclass(kw_only=True, frozen=True)
    class MyProcessorConfig:
        type: Literal["my_processor"] = "my_processor"
        ...

    @register_validator(MyProcessorConfig)
    def _validate_my_processor(config, **context):
        ...

Validators receive keyword context from the parent (encoders get
``d_model=...``) and must accept ``**context``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal, get_args, get_origin, get_type_hints

from . import config as _config
from . import logging as _logging

CONFIG_CATEGORIES: Mapping[str, Any] = MappingProxyType(
    {
        "processor": _config.ProcessorConfig,
        "tokenizer": _config.TokenizerConfig,
        "embedding": _config.EmbeddingConfig,
        "positional_encoding": _config.EmbeddingPositionalEncodingConfig,
        "per_block_positional_encoding": _config.PerBlockPositionalEncodingConfig,
        "encoder": _config.EncoderConfig,
        "objective": _config.ObjectiveConfig,
        "masking_strategy": _config.MaskingStrategyConfig,
        "scalar_scheduler": _config.ScalarSchedulerConfig,
        "augmentation": _config.AugmentationConfig,
        "optimizer": _config.OptimizerConfig,
        "scheduler": _config.LRSchedulerConfig,
        "logger": _logging.LoggerConfig,
    }
)

Validator = Callable[..., None]

_CONFIGS: dict[str, dict[str, type]] = {category: {} for category in CONFIG_CATEGORIES}
_VALIDATORS: dict[type, Validator] = {}


def _type_value(cls: type) -> str:
    hint = get_type_hints(cls).get("type")
    if get_origin(hint) is not Literal or len(get_args(hint)) != 1:
        raise TypeError(
            f"{cls.__name__} must declare a single-valued `type: Literal[...]` field "
            "to be registered"
        )
    return get_args(hint)[0]


def _union_members(union: Any) -> tuple[type, ...]:
    args = get_args(union)
    return args if args else (union,)


def register_config(category: str, *, overwrite: bool = False) -> Callable[[type], type]:
    """Class decorator registering a plug-in config under ``category``."""
    if category not in CONFIG_CATEGORIES:
        raise ValueError(
            f"Unknown config category {category!r}; expected one of {sorted(CONFIG_CATEGORIES)}"
        )

    def decorator(cls: type) -> type:
        name = _type_value(cls)
        builtin = {_type_value(member) for member in _union_members(CONFIG_CATEGORIES[category])}
        if not overwrite and (name in builtin or name in _CONFIGS[category]):
            raise ValueError(
                f"{category} config type {name!r} is already registered; "
                "pass overwrite=True to replace it"
            )
        _CONFIGS[category][name] = cls
        return cls

    return decorator


def register_validator(config_cls: type, validator: Validator | None = None) -> Any:
    """Register ``validator(config, **context)`` for ``config_cls`` (usable as decorator)."""
    if validator is None:
        return lambda fn: register_validator(config_cls, fn)
    _VALIDATORS[config_cls] = validator
    return validator


def registered_configs(category: str) -> Mapping[str, type]:
    """Plug-in config classes registered for ``category``, keyed by ``type``."""
    return MappingProxyType(_CONFIGS[category])


def resolve_registered_config(candidates: tuple[type, ...], type_name: str) -> type | None:
    """Find a plug-in config whose category contains any of the union ``candidates``."""
    for category, union in CONFIG_CATEGORIES.items():
        members = set(_union_members(union))
        if members & set(candidates) and type_name in _CONFIGS[category]:
            return _CONFIGS[category][type_name]
    return None


def is_registered_config(config: Any, category: str) -> bool:
    return any(isinstance(config, cls) for cls in _CONFIGS[category].values())


def in_category(config: Any, category: str) -> bool:
    """True for built-in members of ``category`` and registered plug-in configs."""
    union = CONFIG_CATEGORIES[category]
    return isinstance(config, _union_members(union)) or is_registered_config(config, category)


def validator_for(config: Any) -> Validator | None:
    for cls in type(config).__mro__:
        if cls in _VALIDATORS:
            return _VALIDATORS[cls]
    return None


def snapshot() -> tuple[dict[str, dict[str, type]], dict[type, Validator]]:
    return {k: dict(v) for k, v in _CONFIGS.items()}, dict(_VALIDATORS)


def restore(state: tuple[dict[str, dict[str, type]], dict[type, Validator]]) -> None:
    configs, validators = state
    for category in _CONFIGS:
        _CONFIGS[category] = dict(configs.get(category, {}))
    _VALIDATORS.clear()
    _VALIDATORS.update(validators)


__all__ = [
    "CONFIG_CATEGORIES",
    "in_category",
    "is_registered_config",
    "register_config",
    "register_validator",
    "registered_configs",
    "resolve_registered_config",
    "restore",
    "snapshot",
    "validator_for",
]

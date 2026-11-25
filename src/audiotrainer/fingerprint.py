"""Stable fingerprint of the experiment slice of a training config.

``AudioSSLTrainingConfig`` is already the experiment slice (model, objective,
trainer optimization). Operational settings live in ``RunConfig`` and never
enter the digest. ``data_fingerprint`` identifies the dataset/split.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from types import MappingProxyType
from typing import Any

from audiossl.config import AudioSSLTrainingConfig


def _to_plain(value: Any) -> Any:
    """Convert ``MappingProxyType`` -> ``dict`` and ``tuple`` -> ``list`` recursively."""

    if isinstance(value, MappingProxyType):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def experiment_payload(config: AudioSSLTrainingConfig) -> dict[str, Any]:
    """Return the JSON-serializable experiment slice of ``config``.

    Placement (``model.runtime.device``) is excluded so a run may resume on
    different hardware; the device is recorded in provenance instead.
    """
    if not isinstance(config, AudioSSLTrainingConfig):
        raise TypeError(
            "experiment_payload expected AudioSSLTrainingConfig, "
            f"got {type(config).__name__}"
        )
    payload = _to_plain(asdict(config))
    runtime = payload.get("model", {}).get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("device", None)
    return payload


def config_fingerprint(
    config: AudioSSLTrainingConfig,
    *,
    data_fingerprint: str,
) -> str:
    """Return a stable SHA-256 hex digest of the experiment config + data identity."""

    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise ValueError(
            "config_fingerprint requires a non-empty data_fingerprint string, "
            f"got {data_fingerprint!r}"
        )
    payload = experiment_payload(config)
    payload["data_fingerprint"] = data_fingerprint
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resume_compatible_payload(
    config: AudioSSLTrainingConfig | dict[str, Any],
    *,
    data_fingerprint: str | None = None,
) -> Any:
    """Normalize a config for resume comparison (horizon wild-carded)."""
    if isinstance(config, AudioSSLTrainingConfig):
        if data_fingerprint is None:
            raise ValueError(
                "resume_compatible_payload requires data_fingerprint for live configs"
            )
        plain = experiment_payload(config)
        plain["data_fingerprint"] = data_fingerprint
    else:
        plain = copy.deepcopy(_to_plain(config))
        if isinstance(plain, dict):
            runtime = plain.get("model", {}).get("runtime")
            if isinstance(runtime, dict):
                runtime.pop("device", None)
            if data_fingerprint is not None:
                plain["data_fingerprint"] = data_fingerprint
    if not isinstance(plain, dict):
        return plain
    trainer = plain.get("trainer")
    if isinstance(trainer, dict):
        loop = trainer.get("loop")
        if isinstance(loop, dict):
            loop["max_epochs"] = "<resume-target>"
            loop["max_steps"] = "<resume-target>"
    return plain


__all__ = [
    "config_fingerprint",
    "experiment_payload",
    "resume_compatible_payload",
]

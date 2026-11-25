"""Environment provenance stamped into checkpoints and the run log."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from audiossl.runtime import AppliedRuntimeConfig


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    dirty = bool(status.strip())
    result: dict[str, Any] = {"available": True, "commit": commit, "dirty": dirty}
    if dirty:
        try:
            diff = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=root, capture_output=True, timeout=10, check=True,
            ).stdout
            result["dirty_sha256"] = hashlib.sha256(diff).hexdigest()
        except (OSError, subprocess.SubprocessError):
            result["dirty_sha256"] = None
    return result


def collect_provenance(applied_runtime: AppliedRuntimeConfig) -> dict[str, Any]:
    """Return a JSON-serializable description of the software and hardware."""
    device = applied_runtime.device
    cuda_device = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(device)
    matmul_precision = None
    getter = getattr(torch, "get_float32_matmul_precision", None)
    if callable(getter):
        matmul_precision = getter()
    allow_tf32 = None
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
        allow_tf32 = bool(cuda_backend.matmul.allow_tf32)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "argv": list(sys.argv),
        "torch": torch.__version__,
        "torchaudio": _package_version("torchaudio"),
        "transformers": _package_version("transformers"),
        "audiossl": _package_version("audiossl") or _package_version("realssl"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "device_name": cuda_device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": matmul_precision,
        "allow_tf32": allow_tf32,
        "git": _git_state(),
    }


def assert_clean_git(provenance: dict[str, Any]) -> None:
    git = provenance.get("git") or {}
    if not git.get("available"):
        raise RuntimeError(
            "TrainerConfig/RunConfig.require_clean_git=True but this source tree "
            "is not a git repository (or git is unavailable)"
        )
    if git.get("dirty"):
        raise RuntimeError(
            "RunConfig.require_clean_git=True but the git working tree is dirty; "
            f"dirty_sha256={git.get('dirty_sha256')!r}"
        )


__all__ = ["assert_clean_git", "collect_provenance"]

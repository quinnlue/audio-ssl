"""Load the upstream EAT reference implementation without AutoModel.

Avoids ``transformers.AutoModel`` (version skew on remote-code models) and a
broken native ``torchaudio`` install.  We import the cached HF ``eat_model``
module and load weights from ``model.safetensors``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

_HF_REPO_ID = "worstchan/EAT-base_epoch30_pretrain"


def _hub_snapshot_dir() -> Path | None:
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub_root.is_dir():
        return None
    candidates = sorted(
        hub_root.glob(
            "models--worstchan--EAT-base_epoch30_pretrain/snapshots/*/config.json"
        )
    )
    if not candidates:
        return None
    return candidates[-1].parent


def _cached_module_dir() -> Path | None:
    modules_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
        / "worstchan"
    )
    if not modules_root.is_dir():
        return None
    candidates = sorted(modules_root.rglob("eat_model.py"))
    if not candidates:
        return None
    return candidates[-1].parent


def _module_dir(snapshot: Path | None) -> Path | None:
    cached = _cached_module_dir()
    if cached is not None:
        return cached
    if snapshot is not None and (snapshot / "eat_model.py").is_file():
        return snapshot
    return None


def _ensure_hf_assets(
    *,
    checkpoint_path: Path | str | None = None,
) -> tuple[Path, Path]:
    snapshot = _hub_snapshot_dir()
    module_dir = _module_dir(snapshot)

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"EAT checkpoint not found: {ckpt}")
        config_path = ckpt.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"EAT config not found next to checkpoint: {config_path}")
        if module_dir is None:
            raise FileNotFoundError(
                "HF EAT Python sources not cached; run "
                f"`transformers AutoModel.from_pretrained('{_HF_REPO_ID}', trust_remote_code=True)` "
                "once on a machine with network access."
            )
        return ckpt, module_dir

    if snapshot is None or module_dir is None:
        from huggingface_hub import snapshot_download

        snapshot_download(_HF_REPO_ID)
        snapshot = _hub_snapshot_dir()
        module_dir = _module_dir(snapshot)
        if snapshot is None or module_dir is None:
            raise FileNotFoundError(
                f"Unable to locate cached assets for {_HF_REPO_ID}"
            )

    ckpt = snapshot / "model.safetensors"
    if not ckpt.is_file():
        raise FileNotFoundError(f"EAT checkpoint not found: {ckpt}")
    return ckpt, module_dir


def _import_hf_eat_class(module_dir: Path) -> type[nn.Module]:
    pkg_name = "audiossl_hf_eat_reference"
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [str(module_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    for module_name in ("model_core", "eat_model"):
        qualified = f"{pkg_name}.{module_name}"
        if qualified in sys.modules:
            continue
        path = module_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(
            qualified,
            path,
            submodule_search_locations=[str(module_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load HF EAT module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)

    eat_module = sys.modules[f"{pkg_name}.eat_model"]
    eat_cls = getattr(eat_module, "EAT", None)
    if eat_cls is None:
        raise ImportError(f"No EAT class found in {module_dir / 'eat_model.py'}")
    return eat_cls


def _load_config(config_path: Path) -> SimpleNamespace:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("architectures", "auto_map", "model_type", "torch_dtype", "transformers_version"):
        payload.pop(key, None)
    return SimpleNamespace(**payload)


def load_hf_eat_reference(
    checkpoint_path: Path | str | None = None,
    *,
    map_location: Any = "cpu",
) -> nn.Module:
    """Instantiate upstream ``EAT`` with weights from the HF checkpoint."""
    ckpt_path, module_dir = _ensure_hf_assets(checkpoint_path=checkpoint_path)
    config = _load_config(ckpt_path.parent / "config.json")
    eat_cls = _import_hf_eat_class(module_dir)
    model = eat_cls(config)

    from safetensors.torch import load_file

    state_dict = load_file(str(ckpt_path), device=str(map_location))
    model_state = {
        key[len("model.") :]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    incompatible = model.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "HF EAT reference load_state_dict reported incompatibilities: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.eval()


__all__ = ["load_hf_eat_reference"]

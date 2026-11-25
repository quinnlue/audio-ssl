"""Tests for the Dasheng integration.

Uses the local reference implementation at
``eval/models/dasheng-inference/dasheng/dasheng/`` as the source of truth.
A small sys.modules stub is installed so the reference can be imported
without pip-installing the package (its top-level __init__ calls
``importlib.metadata.version('dasheng')`` which fails when the package is
not installed).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from testkit.external import load_reference

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DASHENG_PKG_DIR = (
    _REPO_ROOT / "eval" / "models" / "dasheng-inference" / "dasheng" / "dasheng"
)


def _install_dasheng_stub() -> None:
    if "dasheng" in sys.modules:
        return
    if not _DASHENG_PKG_DIR.is_dir():
        return
    pkg = types.ModuleType("dasheng")
    pkg.__path__ = [str(_DASHENG_PKG_DIR)]
    pkg.__version__ = "test"
    sys.modules["dasheng"] = pkg


_install_dasheng_stub()


@pytest.fixture(scope="session")
def dasheng_modules():
    """Reference Dasheng modules; skip the whole module if unavailable."""
    def load():
        from dasheng.pretrained import pretrained
        from dasheng.train import models

        return types.SimpleNamespace(models=models, pretrained=pretrained)

    return load_reference(load, "Dasheng reference")

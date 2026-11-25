"""Policy for tests that depend on external reference implementations.

A missing reference (package not installed, checkpoint not cached, no
network) is environmental and skips. Anything else raised while loading a
reference -- ``TypeError``, ``AttributeError``, a state-dict mismatch -- is a
bug in our integration code and must fail, so the skip net is deliberately
narrow.

Network downloads are opt-in: set ``AUDIOSSL_ALLOW_DOWNLOADS=1``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

import pytest

T = TypeVar("T")

DOWNLOADS_ENV = "AUDIOSSL_ALLOW_DOWNLOADS"
ENVIRONMENTAL_ERRORS: tuple[type[BaseException], ...] = (ImportError, OSError)


def downloads_allowed() -> bool:
    return os.environ.get(DOWNLOADS_ENV, "") == "1"


def require_downloads_or_cached(cached: bool, what: str) -> None:
    if not cached and not downloads_allowed():
        pytest.skip(f"{what} is not cached; set {DOWNLOADS_ENV}=1 to download it")


def load_reference(loader: Callable[[], T], what: str) -> T:
    try:
        return loader()
    except ENVIRONMENTAL_ERRORS as exc:
        pytest.skip(f"{what} not available: {type(exc).__name__}: {exc}")

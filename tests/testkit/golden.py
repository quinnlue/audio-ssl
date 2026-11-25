"""Golden-value storage for numerical regression tests.

Goldens pin the numerical behaviour of each case (initial parameters, forward
outputs, first-step loss) so that any change to modelling math, initialization
order, or RNG consumption is caught, reviewed, and deliberately re-baselined.

Workflow
--------
- Normal runs compare against ``tests/golden/<suite>.json``.
- A missing entry fails with instructions; it is never silently created.
- ``pytest --update-golden`` rewrites the entries exercised by the run. Commit
  the JSON diff alongside the change that caused it.

Values are compared with tolerance (not bitwise) because CPU BLAS kernels may
legitimately differ in the last bits across machines; bitwise determinism on a
single machine is asserted separately by the reproducibility suite.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"
RTOL = 1e-4
ATOL = 1e-5


def environment() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }


class GoldenStore:
    def __init__(self, suite: str, *, update: bool) -> None:
        self.path = GOLDEN_DIR / f"{suite}.json"
        self.update = update
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self._data = {"environment": environment(), "entries": {}}
        return self._data

    def check(
        self,
        key: str,
        values: dict[str, Any],
        *,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> None:
        data = self._load()
        entries: dict[str, Any] = data["entries"]
        if self.update:
            entries[key] = values
            data["environment"] = environment()
            self._write()
            return
        if key not in entries:
            raise AssertionError(
                f"No golden entry {key!r} in {self.path.name}. If this is a new case, run "
                "`uv run pytest tests/reproducibility --update-golden` and commit the JSON."
            )
        mismatches = _compare(entries[key], values, rtol=rtol, atol=atol)
        if mismatches:
            recorded = data.get("environment", {})
            note = ""
            if recorded != environment():
                note = f"\nGoldens recorded under {recorded}; running under {environment()}."
            raise AssertionError(
                f"Golden mismatch for {key!r} ({self.path.name}):\n  "
                + "\n  ".join(mismatches)
                + note
                + "\nIf the change is intentional, re-run with --update-golden and commit."
            )

    def _write(self) -> None:
        data = self._load()
        data["entries"] = dict(sorted(data["entries"].items()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def _compare(expected: Any, actual: Any, *, rtol: float, atol: float, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            sub = f"{path}.{key}" if path else key
            if key not in actual:
                out.append(f"{sub}: missing in actual")
            elif key not in expected:
                out.append(f"{sub}: not in golden")
            else:
                out.extend(_compare(expected[key], actual[key], rtol=rtol, atol=atol, path=sub))
        return out
    if isinstance(expected, list) and isinstance(actual, list | tuple):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)} != golden {len(expected)}"]
        return [
            m
            for i, (e, a) in enumerate(zip(expected, actual, strict=True))
            for m in _compare(e, a, rtol=rtol, atol=atol, path=f"{path}[{i}]")
        ]
    if isinstance(expected, float) or isinstance(actual, float):
        if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
            return [f"{path}: {actual!r} != golden {expected!r}"]
        return []
    if expected != actual:
        return [f"{path}: {actual!r} != golden {expected!r}"]
    return []

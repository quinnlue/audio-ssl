"""Reference model fixture for EAT parity tests."""

from __future__ import annotations

import pytest
from testkit.external import load_reference, require_downloads_or_cached

from audiossl.integrations.eat import reference


@pytest.fixture(scope="session")
def hf_eat_model():
    require_downloads_or_cached(
        reference._hub_snapshot_dir() is not None, "Hugging Face EAT checkpoint"
    )
    return load_reference(reference.load_hf_eat_reference, "Hugging Face EAT reference")

"""Shared builders, cases, and checks for the audiossl test suite.

Layout:

- ``configs``: tiny, CPU-fast config builders for every registered component.
- ``batches``: deterministic waveform batches that never touch global RNG.
- ``stubs``: trainer/teacher stand-ins that satisfy objective protocols.
- ``cases``: the model and objective case registries the contract suites
  parametrize over. Adding a registry variant without a case fails
  ``tests/contracts/test_registry_coverage.py``.
- ``checks``: assertion helpers for gradients, equality, and fingerprints.
"""

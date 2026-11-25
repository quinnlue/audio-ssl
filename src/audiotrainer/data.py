"""DataModule protocol: caller-owned datasets with a stable fingerprint."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader

from audiossl.config import BatchConfig
from audiossl.config.run import LoaderConfig


@runtime_checkable
class DataModule(Protocol):
    """Caller-owned data surface used by :func:`audiotrainer.build.build_trainer`.

    The library does not invent a dataset format. Callers implement this
    protocol so loaders and the run fingerprint stay under their control.
    """

    def fingerprint(self) -> str:
        """Stable identity of the dataset/split/crop policy for the run fingerprint."""

    def make_train_loader(
        self,
        batch: BatchConfig,
        loader: LoaderConfig,
        *,
        generator: torch.Generator,
    ) -> DataLoader:
        """Build the training DataLoader.

        ``generator`` is seeded by the trainer from ``RuntimeConfig.seed`` and
        must drive any shuffle/sampling randomness owned by the loader.
        """

    def make_val_loader(
        self,
        batch: BatchConfig,
        loader: LoaderConfig,
        *,
        generator: torch.Generator,
    ) -> DataLoader | None:
        """Build the validation DataLoader, or ``None`` when validation is unused."""


__all__ = ["DataModule"]

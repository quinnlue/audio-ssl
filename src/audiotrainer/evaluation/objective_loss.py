"""Held-out SSL objective loss evaluator with RNG isolation."""

from __future__ import annotations

import contextlib
from itertools import islice
from typing import Any

import torch
from torch.utils.data import DataLoader

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives._base import Objective
from audiossl.runtime import derive_seed

from ..rng import override_objective_generators
from ..state import TrainerState

_MP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class ObjectiveLossEvaluator:
    """Compute held-out SSL objective loss over an ``AudioBatch`` DataLoader."""

    def __init__(
        self,
        objective: Objective,
        loader: DataLoader,
        *,
        use_autocast: bool = False,
        autocast_dtype: str | None = None,
        max_batches: int | None = None,
        eval_seed: int | None = None,
    ) -> None:
        if not isinstance(objective, Objective):
            raise TypeError(
                "ObjectiveLossEvaluator.objective must be Objective, "
                f"got {type(objective).__name__}"
            )
        if not isinstance(loader, DataLoader):
            raise TypeError(
                "ObjectiveLossEvaluator.loader must be DataLoader, "
                f"got {type(loader).__name__}"
            )
        if not isinstance(use_autocast, bool):
            raise TypeError(
                "ObjectiveLossEvaluator.use_autocast must be bool, "
                f"got {type(use_autocast).__name__}"
            )
        if autocast_dtype is not None and autocast_dtype not in _MP_DTYPES:
            raise ValueError(
                "ObjectiveLossEvaluator.autocast_dtype must be one of "
                f"{sorted(_MP_DTYPES)}, got {autocast_dtype!r}"
            )
        if max_batches is not None:
            if isinstance(max_batches, bool) or not isinstance(max_batches, int):
                raise TypeError(
                    "ObjectiveLossEvaluator.max_batches must be int or None, "
                    f"got {type(max_batches).__name__}"
                )
            if max_batches < 1:
                raise ValueError(
                    "ObjectiveLossEvaluator.max_batches must be >= 1, "
                    f"got {max_batches}"
                )

        self.objective = objective
        self.loader = loader
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype
        self.max_batches = max_batches
        self.eval_seed = eval_seed

    def evaluate(self, model: AudioModel, state: TrainerState) -> dict[str, float]:
        if not isinstance(model, AudioModel):
            raise TypeError(
                "ObjectiveLossEvaluator.evaluate expected AudioModel, "
                f"got {type(model).__name__}"
            )

        was_training = self.objective.training
        if self.objective.eval_requires_train_mode:
            self.objective.train()
        else:
            self.objective.eval()

        seed = self.eval_seed
        if seed is None:
            seed = derive_seed(model.applied_runtime.seed, "validation")

        total_loss = 0.0
        num_batches = 0
        try:
            iterator = self.loader
            if self.max_batches is not None:
                iterator = islice(self.loader, self.max_batches)
            with torch.random.fork_rng(
                devices=list(range(torch.cuda.device_count()))
                if torch.cuda.is_available()
                else []
            ):
                torch.manual_seed(seed)
                with override_objective_generators(self.objective, seed=seed):
                    with torch.no_grad():
                        for batch in iterator:
                            batch = self._move_batch(batch, model)
                            with self._autocast_ctx(model):
                                output = self.objective(batch)
                            total_loss += float(output.loss.detach().item())
                            num_batches += 1
        finally:
            self.objective.train(was_training)

        if num_batches == 0:
            raise RuntimeError("ObjectiveLossEvaluator: loader produced zero batches")
        return {"loss": total_loss / num_batches}

    def _autocast_ctx(self, model: AudioModel):
        if not self.use_autocast:
            return contextlib.nullcontext()
        dtype = (
            _MP_DTYPES[self.autocast_dtype]
            if self.autocast_dtype is not None
            else None
        )
        return torch.autocast(
            device_type=model.applied_runtime.device.type,
            dtype=dtype,
        )

    @staticmethod
    def _move_batch(batch: Any, model: AudioModel) -> AudioBatch:
        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "ObjectiveLossEvaluator.loader must yield AudioBatch, "
                f"got {type(batch).__name__}"
            )
        return batch.to(model.applied_runtime.device, non_blocking=True)


__all__ = ["ObjectiveLossEvaluator"]

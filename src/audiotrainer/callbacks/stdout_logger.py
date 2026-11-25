"""Stdout / logging progress logger."""

from __future__ import annotations

import logging
from typing import ClassVar

from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput

from ..state import TrainerState
from ..step import StepContext
from .callback import Callback

logger = logging.getLogger("audiotrainer")


class StdoutLogger(Callback):
    phase: ClassVar = "log"
    name: ClassVar[str] = "StdoutLogger"

    def __init__(self, log_every_n_steps: int) -> None:
        super().__init__()
        if not isinstance(log_every_n_steps, int) or isinstance(log_every_n_steps, bool):
            raise TypeError(
                "StdoutLogger.log_every_n_steps must be int, "
                f"got {type(log_every_n_steps).__name__}"
            )
        if log_every_n_steps < 1:
            raise ValueError(
                f"StdoutLogger.log_every_n_steps must be >= 1, got {log_every_n_steps}"
            )
        self._log_every = log_every_n_steps
        self._last_logged_step = 0

    def on_train_start(self, state: TrainerState) -> None:
        max_epochs = state.train_config.trainer.loop.max_epochs
        max_steps = state.train_config.trainer.loop.max_steps
        logger.info(
            "start fingerprint=%s device=%s dtype=%s max_epochs=%s max_steps=%s",
            state.config_fingerprint[:8],
            state.applied_runtime.device,
            state.applied_runtime.requested_dtype,
            max_epochs,
            max_steps,
        )

    def on_optimizer_step(self, state: TrainerState, ctx: StepContext) -> None:
        del ctx
        step = state.optimizer_step
        if step == 0 or step == self._last_logged_step:
            return
        if step % self._log_every != 0:
            return
        loss = f"{state.last_loss:.4f}" if state.last_loss is not None else "nan"
        lr = f"{state.current_lr:.2e}" if state.current_lr is not None else "nan"
        logger.info("epoch=%s step=%s loss=%s lr=%s", state.epoch, step, loss, lr)
        self._last_logged_step = step

    def on_train_batch_end(
        self,
        state: TrainerState,
        batch: AudioBatch,
        output: ObjectiveOutput,
    ) -> None:
        del state, batch, output

    def on_validation_end(self, state: TrainerState) -> None:
        if not state.last_val_metrics:
            logger.info(
                "validation epoch=%s step=%s", state.epoch, state.optimizer_step
            )
            return
        keys = ["val/loss"] if "val/loss" in state.last_val_metrics else []
        keys.extend(sorted(key for key in state.last_val_metrics if key not in keys))
        metrics = " ".join(
            f"{key}={state.last_val_metrics[key]:.4f}" for key in keys
        )
        logger.info(
            "validation epoch=%s step=%s %s",
            state.epoch,
            state.optimizer_step,
            metrics,
        )


__all__ = ["StdoutLogger"]

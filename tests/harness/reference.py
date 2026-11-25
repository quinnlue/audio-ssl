"""Reference specification of a training run.

A deliberately naive loop written from the trainer's documented contract
(``Trainer`` docstring and callbacks ``CONTRACT.md``), not from ``trainer.py``.
On CPU the ``Trainer`` must reproduce it bit-for-bit. Specified behavior:

* the global RNG is seeded from ``model.config.runtime`` once, before training;
* each epoch sets ``objective.train()`` and visits batches in loader order;
* batches are grouped into accumulation windows of ``gradient_accumulation_steps``
  consecutive micro-batches, the last window of an epoch possibly shorter;
* every micro-batch loss is divided by the number of micro-batches in its
  window, so each optimizer step uses the window's mean gradient;
* per window: ``on_before_optimizer_step``, clip the global grad norm (if
  configured), set ``lr = base_lr * scale(k)`` for the k-th step (0-based, horizon
  ``ceil(len(loader) / accum) * max_epochs``), step, zero grads,
  ``on_after_optimizer_step``;
* objective hooks fire at their named boundaries.

The optimizer is built with ``build_optimizer`` and the schedule shape comes
from the ``*Scale`` classes; both are pinned against closed-form values in
``test_optim``. What this reference checks is the loop that drives them.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from audiossl.config import (
    ConstantSchedulerConfig,
    CosineAnnealingSchedulerConfig,
    LRSchedulerConfig,
    StepLRSchedulerConfig,
    WarmupCosineSchedulerConfig,
)
from audiossl.runtime import apply_global_runtime

from audiotrainer.optim import (
    ConstantScale,
    CosineAnnealingScale,
    StepLRScale,
    WarmupCosineScale,
    build_optimizer,
)

from .objectives import Run
from .recording import TrainingTrace


def lr_scale_schedule(config: LRSchedulerConfig, total_steps: int) -> Callable[[int], float]:
    if isinstance(config, ConstantSchedulerConfig):
        return ConstantScale()
    if isinstance(config, WarmupCosineSchedulerConfig):
        return WarmupCosineScale(
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr_scale=config.min_lr_scale,
        )
    if isinstance(config, CosineAnnealingSchedulerConfig):
        return CosineAnnealingScale(total_steps=total_steps, min_lr_scale=config.min_lr_scale)
    if isinstance(config, StepLRSchedulerConfig):
        return StepLRScale(step_size=config.step_size, gamma=config.gamma)
    raise TypeError(f"reference has no schedule for {type(config).__name__}")


def accumulation_windows(num_batches: int, accum: int) -> list[range]:
    return [range(start, min(start + accum, num_batches)) for start in range(0, num_batches, accum)]


def reference_fit(run: Run) -> TrainingTrace:
    """Train ``run`` with the reference loop and return its trace."""

    trainer_cfg = run.config.trainer
    if trainer_cfg.mixed_precision is not None:
        raise ValueError("reference_fit does not model mixed precision")
    objective = run.objective
    clip = trainer_cfg.loop.gradient_clip_norm
    max_epochs = trainer_cfg.loop.max_epochs
    windows = accumulation_windows(len(run.loader), trainer_cfg.batch.gradient_accumulation_steps)
    lr_scale = lr_scale_schedule(trainer_cfg.scheduler, total_steps=len(windows) * max_epochs)

    apply_global_runtime(run.model.config.runtime)
    optimizer = build_optimizer(objective, trainer_cfg.optimizer)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    trainable = [p for p in objective.parameters() if p.requires_grad]

    def fire(name: str, *args) -> None:
        getattr(objective, name)(*args)

    trace = TrainingTrace()
    global_step = 0
    optimizer_step = 0
    fire("on_train_start")
    for epoch in range(max_epochs):
        fire("on_epoch_start")
        objective.train()
        optimizer.zero_grad(set_to_none=True)
        batches = iter(run.loader)
        for window in windows:
            for _ in window:
                output = objective(next(batches))
                fire("on_after_forward", output)
                (output.loss / len(window)).backward()
                global_step += 1
                trace.losses[global_step] = float(output.loss.detach().item())

            fire("on_before_optimizer_step")
            optimizer_step += 1
            if clip is None:
                grad_norm = torch.nn.utils.get_total_norm(
                    [p.grad for p in trainable if p.grad is not None]
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=clip)
            trace.grad_norms[optimizer_step] = float(grad_norm)

            scale = lr_scale(optimizer_step - 1)
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base_lr * scale
            trace.applied_lrs[optimizer_step] = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            fire("on_after_optimizer_step")
        trace.epoch_end_losses[epoch] = trace.losses[global_step]
        fire("on_epoch_end")
    fire("on_train_end")
    return trace


__all__ = ["accumulation_windows", "lr_scale_schedule", "reference_fit"]

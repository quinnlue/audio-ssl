"""Base class for SSL training objectives.

Lives in its own module so that concrete objective implementations can import
:class:`Objective` without forming a circular dependency with the
``build_objective`` factory in :mod:`audiossl.modelling.objectives.objective`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from audiossl.config import OptimizerConfig, check_objective_compatibility
from audiossl.config.batch_dataclasses import AudioBatch, ObjectiveOutput

OBJECTIVE_HOOKS: tuple[str, ...] = (
    "on_train_start",
    "on_epoch_start",
    "on_epoch_end",
    "on_train_end",
    "on_after_forward",
    "on_before_optimizer_step",
    "on_after_optimizer_step",
)
"""Lifecycle methods the trainer calls on every objective (no-ops by default)."""


class Objective(nn.Module):
    """Base class for SSL training objectives.

    Subclasses implement :meth:`forward` returning an :class:`ObjectiveOutput`
    subclass that always carries a scalar ``loss``, and override the
    lifecycle hooks in :data:`OBJECTIVE_HOOKS` they need (for example EMA
    teachers update in :meth:`on_after_optimizer_step`).

    :meth:`metrics` returns objective-specific diagnostic scalars only.

    Class attributes ``uses_augmentor`` and ``uses_generator`` tell
    :func:`build_objective` which collaborators to pass. Register RNG streams
    with :meth:`register_generator` so the trainer checkpoints them.
    """

    uses_augmentor: ClassVar[bool] = False
    uses_generator: ClassVar[bool] = False
    eval_requires_train_mode: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self._registered_generators: dict[str, torch.Generator] = {}

    def register_generator(self, name: str, generator: torch.Generator) -> torch.Generator:
        """Register an owned RNG stream for checkpointing and resume.

        Mirrors :meth:`torch.nn.Module.register_buffer`: the trainer discovers
        every registered generator, saves its state, and raises on resume when
        the set of names disagrees with the checkpoint.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"register_generator name must be a non-empty str, got {name!r}")
        if not isinstance(generator, torch.Generator):
            raise TypeError(
                f"register_generator expected torch.Generator, got {type(generator).__name__}"
            )
        if name in self._registered_generators:
            raise ValueError(f"generator {name!r} is already registered on {type(self).__name__}")
        self._registered_generators[name] = generator
        return generator

    def named_generators(self) -> dict[str, torch.Generator]:
        """Return every registered generator, including nested modules'."""
        found: dict[str, torch.Generator] = dict(self._registered_generators)
        for module_name, module in self.named_modules():
            if module is self:
                continue
            nested = getattr(module, "_registered_generators", None)
            if not isinstance(nested, dict):
                continue
            for name, generator in nested.items():
                key = f"{module_name}.{name}" if module_name else name
                if key in found:
                    raise RuntimeError(
                        f"Duplicate registered generator name {key!r} under {type(self).__name__}"
                    )
                found[key] = generator
        # Augmentor is not an nn.Module; include its generator when present.
        augmentor = getattr(self, "augmentor", None)
        aug_gen = getattr(augmentor, "generator", None)
        if isinstance(aug_gen, torch.Generator):
            found.setdefault("augmentor.generator", aug_gen)
        return found

    def forward(self, batch: AudioBatch) -> ObjectiveOutput:
        """Run the objective on a raw waveform ``AudioBatch`` ``(B, S)``."""
        raise NotImplementedError

    def metrics(
        self,
        output: ObjectiveOutput,
        ctx: Any,
    ) -> dict[str, torch.Tensor]:
        """Return objective-specific diagnostics (trainer adds train/{type}/ prefix)."""
        del output, ctx
        return {}

    def param_groups(self, config: OptimizerConfig) -> list[dict]:
        """Build optimizer parameter groups for this objective."""
        from audiotrainer.optim import build_param_groups

        return build_param_groups(
            self,
            weight_decay=config.weight_decay,
            no_weight_decay_patterns=config.no_weight_decay_patterns,
        )

    # ----------------------------------------------------------- lifecycle ---

    def on_train_start(self, ctx: Any = None) -> None:
        """Called once before the first training epoch."""
        del ctx

    def on_epoch_start(self, ctx: Any = None) -> None:
        """Called at the start of every training epoch."""
        del ctx

    def on_epoch_end(self, ctx: Any = None) -> None:
        """Called at the end of every training epoch, before checkpoint callbacks."""
        del ctx

    def on_train_end(self, ctx: Any = None) -> None:
        """Called once after the last training epoch."""
        del ctx

    def on_after_forward(self, output: ObjectiveOutput, ctx: Any = None) -> None:
        """Called after every forward (micro-batch), before backward."""
        del output, ctx

    def on_before_optimizer_step(self, ctx: Any = None) -> None:
        """Called once per optimizer step, before gradient clipping and the step."""
        del ctx

    def on_after_optimizer_step(self, ctx: Any = None) -> None:
        """Called once per optimizer step, after the step and scheduler update."""
        del ctx

    # ------------------------------------------------------------- helpers ---

    def _check_model_compatibility(self, model: Any, config: Any) -> None:
        """Raise when ``model``'s components lack capabilities ``config`` requires."""
        check_objective_compatibility(model.config, config)

    def _require_generator(self) -> None:
        if self.uses_generator and not any(
            not name.startswith("augmentor.") for name in self.named_generators()
        ):
            raise ValueError(
                f"{type(self).__name__} declares uses_generator=True but no generator "
                "was registered; pass generator= to build_objective / the constructor "
                "and call self.register_generator('generator', generator)"
            )


__all__ = ["OBJECTIVE_HOOKS", "Objective"]

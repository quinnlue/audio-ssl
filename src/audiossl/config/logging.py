"""Logger backend configuration.

Logger settings are operational: changing the wandb project, run name, or log
cadence does not change the trained model, so they live in
:class:`~audiossl.config.RunConfig` and never enter ``config_fingerprint``.
Each backend config carries a ``type`` discriminator; third-party backends
register a config under the ``"logger"`` category and a factory in
``audiotrainer.registry.LOGGERS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, frozen=True)
class WandBConfig:
    """Wandb run configuration.

    ``mode`` matches ``wandb.init(mode=...)``: ``"online"`` streams to wandb
    cloud, ``"offline"`` writes a local run directory you can ``wandb sync``
    later, and ``"disabled"`` is a no-op runtime kill-switch that still
    lets the callback be attached.

    Every fresh run gets a new wandb run id, which is persisted in the
    checkpoint. With ``resume=True`` a resumed trainer continues that wandb run;
    with ``resume=False`` it starts a new one.
    """

    project: str
    type: Literal["wandb"] = "wandb"
    name: str | None = None
    entity: str | None = None
    tags: tuple[str, ...] = ()
    mode: Literal["online", "offline", "disabled"] = "online"
    notes: str = ""
    log_every_n_steps: int = 1
    aggregate_over_steps: bool = False
    upload_checkpoints: bool = False
    resume: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))


LoggerConfig = WandBConfig


__all__ = ["LoggerConfig", "WandBConfig"]

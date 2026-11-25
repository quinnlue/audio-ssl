"""Pretrained model metadata types and loader protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from audiossl.modelling.model import AudioModel


class PretrainedLoadFn(Protocol):
    """Callable that loads a checkpoint into an already-built model."""

    def __call__(
        self,
        checkpoint_path: Path,
        model: AudioModel,
        *,
        map_location: Any = "cpu",
    ) -> None: ...


@dataclass(frozen=True, kw_only=True)
class PretrainedModelSpec:
    """Registry entry describing one pretrained checkpoint source.

    ``architecture`` selects the preset passed to ``build_model``. Either
    ``hf_repo_id`` or ``checkpoint_url`` supplies the default remote source,
    and ``load_fn`` performs the architecture-specific weight import.
    """

    name: str
    aliases: tuple[str, ...] = ()
    architecture: str
    hf_repo_id: str | None = None
    checkpoint_url: str | None
    checkpoint_filename: str
    sha256: str | None
    load_fn: PretrainedLoadFn
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))


__all__ = ["PretrainedLoadFn", "PretrainedModelSpec"]

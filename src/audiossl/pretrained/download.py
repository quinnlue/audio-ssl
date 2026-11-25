"""Checkpoint download, cache layout, and SHA256 verification."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from .specs import PretrainedModelSpec

_URL_SCHEMES = {"http", "https"}


def is_url(value: str) -> bool:
    """Return whether a string looks like an HTTP(S) URL."""

    return urlparse(value).scheme.lower() in _URL_SCHEMES


def default_cache_dir() -> Path:
    """Return the default audiossl checkpoint cache directory."""

    root = os.environ.get("AUDIOSSL_CACHE")
    if root:
        return Path(root).expanduser() / "pretrained"
    return Path.home() / ".cache" / "audiossl" / "pretrained"


def resolve_checkpoint(
    spec: PretrainedModelSpec,
    *,
    source: str | Path | None,
    cache_dir: str | Path | None,
) -> Path:
    """Resolve a pretrained checkpoint from a local path, URL, or spec default.

    Parameters
    ----------
    spec
        Pretrained model metadata describing the default remote source.
    source
        Optional local checkpoint path or URL override.
    cache_dir
        Directory used when downloading remote checkpoints.
    """
    if source is None:
        if spec.hf_repo_id is not None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "Pretrained model "
                    f"{spec.name!r} is hosted on Hugging Face Hub; install "
                    "huggingface_hub or pass `source=` explicitly"
                ) from exc
            return Path(
                hf_hub_download(
                    repo_id=spec.hf_repo_id,
                    filename=spec.checkpoint_filename,
                    cache_dir=cache_dir,
                )
            )
        if spec.checkpoint_url is None:
            raise ValueError(
                f"Pretrained model {spec.name!r} does not define a checkpoint URL"
            )
        return download_url_to_cache(
            spec.checkpoint_url,
            filename=spec.checkpoint_filename,
            cache_dir=cache_dir,
            sha256=spec.sha256,
        )

    source_value = str(source)
    if is_url(source_value):
        return download_url_to_cache(
            source_value,
            filename=_filename_from_url(source_value) or spec.checkpoint_filename,
            cache_dir=cache_dir,
            sha256=spec.sha256,
        )

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    _verify_sha256(path, spec.sha256)
    return path


def download_url_to_cache(
    url: str,
    *,
    filename: str,
    cache_dir: str | Path | None,
    sha256: str | None,
) -> Path:
    """Download a URL to the checkpoint cache when it is not already present.

    Parameters
    ----------
    url
        Remote checkpoint URL.
    filename
        Destination filename inside ``cache_dir``.
    cache_dir
        Optional override for the default pretrained checkpoint cache.
    sha256
        Optional expected SHA256 digest used to validate cached downloads.
    """
    if not filename:
        raise ValueError(f"Cannot infer checkpoint filename for URL: {url}")

    cache_path = (
        Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    destination = cache_path / filename

    if destination.is_file():
        _verify_sha256(destination, sha256)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    _verify_sha256(partial, sha256)
    partial.replace(destination)
    return destination


def _filename_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or None


def _verify_sha256(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected!r}, got {actual!r}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "default_cache_dir",
    "download_url_to_cache",
    "is_url",
    "resolve_checkpoint",
]

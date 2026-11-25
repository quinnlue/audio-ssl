"""Teacher tokenizers used by HuBERT-style objectives."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import joblib
import torch
import torch.nn as nn

from audiossl.config import HuBERTTeacherConfig
from audiossl.config.batch_dataclasses import AudioBatch

logger = logging.getLogger(__name__)

CANONICAL_KM_URL = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin"
CANONICAL_KM_SHA256 = "4450dcbc90aa7d0553306a4b0fafef35c9b3c840468f02969273dfbb9db89a23"
CANONICAL_KM_FILENAME = "hubert_base_ls960_L9_km500.bin"


def _cache_dir() -> Path:
    return Path(
        os.getenv(
            "AUDIOSSL_CACHE_DIR",
            str(Path.home() / ".cache" / "audiossl" / "hubert"),
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_checksum(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()
    with urllib.request.urlopen(url, timeout=120) as response, part_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    actual = _sha256(part_path)
    if actual != expected_sha256:
        part_path.unlink(missing_ok=True)
        raise ValueError(
            "Downloaded HuBERT k-means artifact failed SHA256 verification: "
            f"expected {expected_sha256}, got {actual}"
        )
    part_path.replace(path)


def resolve_kmeans_path(config: HuBERTTeacherConfig) -> Path:
    """Return a verified local k-means artifact path for the HuBERT teacher.

    The artifact is identified only by the config (``km_url``/``km_sha256``,
    defaulting to the canonical fairseq release), so the training config fully
    determines the labels. ``AUDIOSSL_CACHE_DIR`` only chooses where verified
    files are cached.
    """

    if (config.km_url is None) != (config.km_sha256 is None):
        raise ValueError("HuBERTTeacherConfig.km_url and km_sha256 must be set together")
    url = config.km_url or CANONICAL_KM_URL
    sha256 = config.km_sha256 or CANONICAL_KM_SHA256
    filename = Path(url).name or CANONICAL_KM_FILENAME
    path = _cache_dir() / filename
    if path.exists():
        actual = _sha256(path)
        if actual != sha256:
            raise ValueError(
                f"Cached HuBERT k-means artifact at {path} failed SHA256 verification: "
                f"expected {sha256}, got {actual}"
            )
        return path
    _download_with_checksum(url, path, sha256)
    return path


class PretrainedHuBERTTokenizer(nn.Module):
    """Frozen HuBERT + k-means teacher.

    K-means artifacts are cached under ``AUDIOSSL_CACHE_DIR``. The artifact and
    encoder weights are fully identified by the config (``encoder_id``,
    ``encoder_revision``, ``km_url``, ``km_sha256``), so the frozen weights are
    excluded from ``state_dict`` and rebuilt from the config on load.
    """

    def __init__(self, config: HuBERTTeacherConfig) -> None:
        super().__init__()
        if not isinstance(config, HuBERTTeacherConfig):
            raise TypeError(
                "PretrainedHuBERTTokenizer expected HuBERTTeacherConfig, "
                f"got {type(config).__name__}"
            )
        try:
            from transformers import HubertModel
        except ImportError as exc:
            raise ImportError(
                "PretrainedHuBERTTokenizer requires transformers to load the HuBERT teacher"
            ) from exc

        self.config = config
        if config.encoder_revision is None:
            logger.warning(
                "HuBERT teacher %r is loaded without encoder_revision; pin a commit hash in "
                "HuBERTTeacherConfig.encoder_revision for reproducible pseudo-labels",
                config.encoder_id,
            )
        self.encoder = HubertModel.from_pretrained(
            config.encoder_id, revision=config.encoder_revision
        )
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        km_path = resolve_kmeans_path(config)
        kmeans = joblib.load(km_path)
        centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32)
        if centers.ndim != 2:
            raise ValueError(f"Expected k-means centers with rank 2, got {tuple(centers.shape)}")
        if centers.shape[0] != config.num_codes:
            raise ValueError(
                f"Expected {config.num_codes} HuBERT k-means centers, got {centers.shape[0]}"
            )
        self.register_buffer("cluster_centers", centers, persistent=False)
        self._register_state_dict_hook(_drop_own_state)
        self._register_load_state_dict_pre_hook(_fill_own_state, with_module=True)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Teacher label counts for prefix-valid raw lengths (HF conv extractor rule)."""
        return self.encoder._get_feat_extract_output_lengths(input_lengths).to(torch.long)

    def train(self, mode: bool = True):
        """Set module mode while keeping the frozen HuBERT encoder in eval mode."""

        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, batch: AudioBatch) -> torch.Tensor:
        """Return frame-level k-means labels with shape ``(B, N)``.

        Parameters
        ----------
        batch
            Raw waveform ``AudioBatch`` with ``x`` and boolean validity mask
            shape ``(B, S)``. ``True`` means valid.
        """

        if not isinstance(batch, AudioBatch):
            raise TypeError(
                "PretrainedHuBERTTokenizer.forward expected AudioBatch, "
                f"got {type(batch).__name__}"
            )
        if batch.x.ndim != 2 or batch.mask.shape != batch.x.shape:
            raise ValueError("PretrainedHuBERTTokenizer.forward expected waveform AudioBatch")
        outputs = self.encoder(
            input_values=batch.x,
            attention_mask=batch.mask.to(dtype=torch.long),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[self.config.feature_layer]
        centers = self.cluster_centers.to(device=hidden.device, dtype=hidden.dtype)
        distances = torch.cdist(hidden, centers.unsqueeze(0).expand(hidden.shape[0], -1, -1))
        return distances.argmin(dim=-1).to(dtype=torch.long)


def _drop_own_state(module: nn.Module, state_dict: dict, prefix: str, local_metadata) -> None:
    for key in [k for k in state_dict if k.startswith(prefix)]:
        del state_dict[key]


def _fill_own_state(
    module: nn.Module,
    state_dict: dict,
    prefix: str,
    local_metadata,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
) -> None:
    for name, tensor in module.named_parameters():
        state_dict.setdefault(prefix + name, tensor.detach())
    persistent = set()
    for submodule_name, submodule in module.named_modules():
        for buffer_name in submodule._buffers:
            if buffer_name not in submodule._non_persistent_buffers_set:
                persistent.add(f"{submodule_name}.{buffer_name}" if submodule_name else buffer_name)
    for name, tensor in module.named_buffers():
        if name in persistent:
            state_dict.setdefault(prefix + name, tensor.detach())


__all__ = [
    "CANONICAL_KM_FILENAME",
    "CANONICAL_KM_SHA256",
    "CANONICAL_KM_URL",
    "PretrainedHuBERTTokenizer",
    "resolve_kmeans_path",
]

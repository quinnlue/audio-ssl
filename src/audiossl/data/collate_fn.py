import torch

from audiossl.config.batch_dataclasses import AudioBatch


def validate_sample(sample: torch.Tensor) -> None:
    if not isinstance(sample, torch.Tensor):
        raise TypeError(f"sample must be torch.Tensor, got {type(sample).__name__}")
    if sample.ndim != 1:
        raise ValueError(f"sample must be rank-1 waveform tensor, got shape {tuple(sample.shape)}")
    if sample.dtype != torch.float32:
        raise ValueError(f"sample must have dtype torch.float32, got {sample.dtype}")


def validate_labels(labels: list[int]) -> None:
    if not all(isinstance(label, int) for label in labels):
        raise ValueError("Labels must be a list of integers")


def collate_fn(batch: dict[str, list[torch.Tensor]]) -> AudioBatch:
    if not isinstance(batch, dict):
        raise TypeError(f"batch must be dict[str, list[torch.Tensor]], got {type(batch).__name__}")

    if "x" not in batch:
        raise KeyError("batch must contain key 'x'")

    xs = batch["x"]

    if not isinstance(xs, list):
        raise TypeError(f"batch['x'] must be list[torch.Tensor], got {type(xs).__name__}")

    if not xs:
        raise ValueError("batch['x'] must contain at least one waveform")

    for item in xs:
        validate_sample(item)

    lengths = torch.tensor([item.shape[0] for item in xs], dtype=torch.long)
    max_length = int(lengths.max().item())

    x = torch.zeros((len(xs), max_length), dtype=torch.float32)
    mask = torch.zeros((len(xs), max_length), dtype=torch.bool)

    for i, item in enumerate(xs):
        length = int(lengths[i].item())
        x[i, :length] = item
        mask[i, :length] = True

    return AudioBatch(x=x, mask=mask)


__all__ = [
    "AudioBatch",
    "collate_fn",
    "validate_labels",
    "validate_sample",
]
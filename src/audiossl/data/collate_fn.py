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


def collate_labeled_fn(batch: dict[str, list]) -> AudioBatch:
    """Collate waveforms and integer class labels into an ``AudioBatch``."""

    if not isinstance(batch, dict):
        raise TypeError(f"batch must be dict, got {type(batch).__name__}")

    if "x" not in batch:
        raise KeyError("batch must contain key 'x'")
    if "y" not in batch:
        raise KeyError("batch must contain key 'y'")

    xs = batch["x"]
    ys = batch["y"]

    if not isinstance(xs, list):
        raise TypeError(f"batch['x'] must be list[torch.Tensor], got {type(xs).__name__}")
    if not isinstance(ys, list):
        raise TypeError(f"batch['y'] must be list[int], got {type(ys).__name__}")

    if not xs:
        raise ValueError("batch['x'] must contain at least one waveform")

    validate_labels(ys)

    padded = collate_fn({"x": xs})
    labels = torch.tensor(ys, dtype=torch.long)
    return AudioBatch(
        x=padded.x,
        mask=padded.mask,
        labels=labels,
    )


__all__ = [
    "AudioBatch",
    "collate_fn",
    "collate_labeled_fn",
    "validate_labels",
    "validate_sample",
]
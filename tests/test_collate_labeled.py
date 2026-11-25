"""Tests for labeled batch collation."""

from __future__ import annotations

import torch

from audiotrainer.collate_fn import collate_labeled_fn, validate_labels


def test_collate_labeled_fn_builds_batch_with_labels() -> None:
    xs = [torch.randn(100), torch.randn(80)]
    ys = [0, 1]
    validate_labels(ys)
    batch = collate_labeled_fn({"x": xs, "y": ys})

    assert batch.x.shape == (2, 100)
    assert batch.mask.shape == (2, 100)
    assert batch.labels is not None
    assert tuple(batch.labels.tolist()) == (0, 1)


def test_collate_labeled_fn_requires_y_key() -> None:
    try:
        collate_labeled_fn({"x": [torch.randn(4)]})
    except KeyError as exc:
        assert "y" in str(exc)
    else:
        raise AssertionError("expected KeyError")
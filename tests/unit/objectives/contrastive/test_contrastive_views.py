from __future__ import annotations

import unittest

import torch

from audiossl.config.batch_dataclasses import AudioBatch, ContrastiveAudioBatch
from audiossl.modelling.objectives.contrastive.views import create_contrastive_views


class _CountingAugmentor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, batch: AudioBatch) -> AudioBatch:
        self.calls += 1
        return AudioBatch(x=batch.x.clone() + self.calls, mask=batch.mask.clone())


def _waveform_batch(*, batch: int = 2, time: int = 4) -> AudioBatch:
    return AudioBatch(
        x=torch.randn(batch, time, dtype=torch.float32),
        mask=torch.ones(batch, time, dtype=torch.bool),
    )


class CreateContrastiveViewsTests(unittest.TestCase):
    def test_augmentation_strategy_calls_augmentor_twice(self) -> None:
        augmentor = _CountingAugmentor()
        batch = _waveform_batch()

        views = create_contrastive_views(
            batch,
            strategy="augmentation",
            augmentor=augmentor,
        )

        self.assertIsInstance(views, ContrastiveAudioBatch)
        self.assertEqual(augmentor.calls, 2)
        self.assertIsNot(views.batch_i, views.batch_j)
        self.assertIsInstance(views.batch_i, AudioBatch)
        self.assertIsInstance(views.batch_j, AudioBatch)

    def test_augmentation_strategy_requires_augmentor(self) -> None:
        with self.assertRaisesRegex(ValueError, "augmentor"):
            create_contrastive_views(
                _waveform_batch(),
                strategy="augmentation",
                augmentor=None,
            )

    def test_dropout_strategy_returns_views_sharing_the_immutable_batch(self) -> None:
        batch = _waveform_batch()

        views = create_contrastive_views(batch, strategy="dropout", augmentor=None)

        self.assertIs(views.batch_i.x, batch.x)
        self.assertIs(views.batch_j.x, batch.x)
        self.assertIs(views.batch_i.mask, batch.mask)
        self.assertIs(views.batch_j.mask, batch.mask)

    def test_dropout_strategy_forbids_augmentor(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbids"):
            create_contrastive_views(
                _waveform_batch(),
                strategy="dropout",
                augmentor=_CountingAugmentor(),
            )

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported contrastive view strategy"):
            create_contrastive_views(
                _waveform_batch(),
                strategy="random",  # type: ignore[arg-type]
                augmentor=None,
            )

    def test_rejects_wrong_rank_input_batch(self) -> None:
        bad_batch = AudioBatch(
            x=torch.randn(2, 3, 4, dtype=torch.float32),
            mask=torch.ones(2, 3, 4, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "rank-2"):
            create_contrastive_views(
                bad_batch,
                strategy="dropout",
                augmentor=None,
            )


if __name__ == "__main__":
    unittest.main()

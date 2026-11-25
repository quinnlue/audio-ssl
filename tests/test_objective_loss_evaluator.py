from __future__ import annotations

import math

from torch.utils.data import DataLoader, Dataset

from audiotrainer import ObjectiveLossEvaluator

from harness import CONTRASTIVE, collate_waveforms, waveform_loader


class _CountingDataset(Dataset):
    def __init__(self, loader: DataLoader) -> None:
        self._samples = list(loader.dataset)
        self.reads = 0

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        self.reads += 1
        return self._samples[index]


def _counting_loader(num_batches: int) -> tuple[DataLoader, _CountingDataset]:
    dataset = _CountingDataset(waveform_loader(num_batches=num_batches, batch_size=2, length=64))
    loader = DataLoader(dataset, batch_size=2, drop_last=True, collate_fn=collate_waveforms)
    return loader, dataset


def test_returns_finite_mean_loss() -> None:
    run = CONTRASTIVE.run(num_batches=1)
    loader, _ = _counting_loader(2)

    metrics = ObjectiveLossEvaluator(run.objective, loader).evaluate(run.model, run.trainer().state)

    assert set(metrics) == {"loss"}
    assert math.isfinite(metrics["loss"])


def test_respects_max_batches() -> None:
    run = CONTRASTIVE.run(num_batches=1)
    loader, dataset = _counting_loader(4)

    ObjectiveLossEvaluator(run.objective, loader, max_batches=2).evaluate(
        run.model, run.trainer().state
    )

    assert dataset.reads == 4


def test_restores_train_mode() -> None:
    run = CONTRASTIVE.run(num_batches=1)
    run.objective.train()
    loader, _ = _counting_loader(2)

    ObjectiveLossEvaluator(run.objective, loader).evaluate(run.model, run.trainer().state)

    assert run.objective.training

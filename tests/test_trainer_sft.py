"""Short supervised fine-tuning trainer integration tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from audiossl.config import BatchConfig, EvaluationConfig
from audiossl.config.run import LoaderConfig, RunConfig
from audiossl.modelling.model import AudioModel
from torch.utils.data import DataLoader

from audiotrainer import build_trainer, collate_labeled_fn
from testkit import configs as c
from testkit.cases import objective_case


@dataclass
class _LabeledWaveformModule:
    lengths: tuple[int, ...]
    labels: tuple[int, ...]
    seed: int = 0

    def fingerprint(self) -> str:
        return f"labeled-{self.lengths}-{self.labels}"

    def _loader(self, batch_size: int) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        xs = [0.1 * torch.randn(length, generator=generator) for length in self.lengths]
        records = [{"x": wave, "y": label} for wave, label in zip(xs, self.labels, strict=True)]

        def collate(items: list[dict]) -> object:
            return collate_labeled_fn({"x": [item["x"] for item in items], "y": [item["y"] for item in items]})

        return DataLoader(
            records,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate,
        )

    def make_train_loader(
        self,
        batch: BatchConfig,
        loader: LoaderConfig,
        *,
        generator: torch.Generator,
    ) -> DataLoader:
        del generator, loader
        return self._loader(batch.per_device_batch_size)

    def make_val_loader(
        self,
        batch: BatchConfig,
        loader: LoaderConfig,
        *,
        generator: torch.Generator,
    ) -> DataLoader:
        del generator, loader
        return self._loader(batch.per_device_batch_size)


def _sft_train_config():
    config = c.training(c.sft_objective(num_classes=2))
    loop = replace(config.trainer.loop, max_epochs=1, max_steps=2)
    return replace(config, trainer=replace(config.trainer, loop=loop))


def test_build_trainer_accepts_injected_pretrained_model() -> None:
    config = _sft_train_config()
    model = AudioModel(config.model)
    data = _LabeledWaveformModule((320, 256), (0, 1))
    trainer = build_trainer(config, data, model=model)
    assert trainer.state is not None


def test_sft_trainer_fit_logs_validation_accuracy() -> None:
    config = _sft_train_config()
    model = AudioModel(config.model)
    data = _LabeledWaveformModule((320, 256, 288), (0, 1, 0))
    run_config = RunConfig(evaluation=EvaluationConfig(every_n_steps=1, at_train_end=False))
    trainer = build_trainer(config, data, model=model, run_config=run_config)
    state = trainer.fit()
    assert state.optimizer_step >= 1
    if state.last_val_metrics:
        assert "val/accuracy" in state.last_val_metrics


def test_build_trainer_rejects_model_config_mismatch() -> None:
    config = _sft_train_config()
    other = AudioModel(
        replace(config.model, runtime=replace(config.model.runtime, seed=config.model.runtime.seed + 1))
    )
    data = _LabeledWaveformModule((320, 256), (0, 1))
    try:
        build_trainer(config, data, model=other)
    except ValueError as exc:
        assert "model.config does not match" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_sft_objective_case_runs_forward() -> None:
    case = objective_case("sft")
    objective = case.build()
    output = objective(case.batch())
    assert output.loss.ndim == 0
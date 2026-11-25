"""Third-party components plug in through registries without editing library code.

A processor defined entirely in this file is registered for config loading,
validation, and construction, then trained against through the public API.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

import pytest
import torch

from audiossl.config import load_config, register_config, register_validator, validate_config
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives import build_objective
from audiossl.modelling.processors.base import BaseProcessor
from audiossl.modelling.registry import PROCESSORS
from audiotrainer.fingerprint import _to_plain

from testkit import batches, configs

FRAME = 160


@dataclass(kw_only=True, frozen=True)
class FramedProcessorConfig:
    type: Literal["test_framed"] = "test_framed"
    output_dim: int = configs.N_MELS


class FramedProcessor(BaseProcessor):
    """Non-overlapping frames; a frame is valid only if all its samples are."""

    def __init__(self, *, config: FramedProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = torch.nn.Linear(FRAME, config.output_dim, bias=False)

    def output_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        frames = input_mask.shape[1] // FRAME
        return input_mask[:, : frames * FRAME].reshape(input_mask.shape[0], frames, FRAME).all(-1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return input_lengths // FRAME

    def forward(self, batch: AudioBatch) -> AudioBatch:
        self._validate_waveform_batch(batch)
        mask = self.output_mask(batch.mask)
        x = batch.x.masked_fill(~batch.mask, 0.0)[:, : mask.shape[1] * FRAME]
        x = self.proj(x.reshape(x.shape[0], mask.shape[1], FRAME))
        return AudioBatch(x=x.masked_fill(~mask[..., None], 0.0), mask=mask)


@pytest.fixture
def registered():
    validated: list[FramedProcessorConfig] = []
    register_config("processor")(FramedProcessorConfig)

    @register_validator(FramedProcessorConfig)
    def _validate(config, **context):
        if config.output_dim < 1:
            raise ValueError("FramedProcessorConfig.output_dim must be >= 1")
        validated.append(config)

    PROCESSORS.register("test_framed", FramedProcessor)
    return validated


def _model_config():
    return configs.model(processor=FramedProcessorConfig(), tokenizer=configs.noop_tokenizer())


def _training_config():
    return configs.training(configs.jepa_objective(teacher_mode="stop_grad"), _model_config())


def test_registered_processor_validates_builds_and_trains(registered):
    config = _training_config()
    assert validate_config(config)
    assert registered, "the registered validator must run"

    model = AudioModel(config.model)
    assert type(model.processor.processor) is FramedProcessor
    objective = build_objective(
        model, config.objective, generator=torch.Generator().manual_seed(0)
    ).train()
    out = objective(batches.waveform_batch(configs.DEFAULT_LENGTHS, seed=0))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert model.processor.processor.proj.weight.grad is not None


def test_registered_processor_loads_from_json(registered):
    payload = json.dumps(_to_plain(asdict(_training_config())))
    loaded = load_config(payload)
    assert isinstance(loaded.model.processor, FramedProcessorConfig)
    assert loaded == _training_config()


def test_registered_validator_failures_are_loud(registered):
    bad = configs.model(
        processor=FramedProcessorConfig(output_dim=0), tokenizer=configs.noop_tokenizer(0)
    )
    with pytest.raises(ValueError, match="output_dim"):
        validate_config(bad)


def test_unregistered_type_is_rejected_by_loader():
    payload = _to_plain(asdict(_training_config()))
    with pytest.raises((TypeError, ValueError), match="test_framed"):
        load_config(payload)


def test_duplicate_registration_is_rejected(registered):
    with pytest.raises(ValueError, match="already registered"):
        register_config("processor")(FramedProcessorConfig)
    with pytest.raises(ValueError, match="already registered"):
        PROCESSORS.register("test_framed", FramedProcessor)

"""MAE-specific behaviour beyond the generic objective contract."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from audiossl.config import (
    AdamWOptimizerConfig,
    AudioAugmentorConfig,
    AudioSSLTrainingConfig,
    BatchConfig,
    ConstantSchedulerConfig,
    MAEMaskingConfig,
    TrainerConfig,
    TrainingLoopConfig,
    validate_config,
)
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives.mae.mae import MAEObjective
from testkit import configs
from testkit.cases import objective_case

CASE = objective_case("mae")


def _with_objective(**overrides):
    return replace(CASE, objective=lambda: replace(CASE.objective(), **overrides))


@pytest.mark.parametrize(
    ("model_config", "match"),
    [
        (
            lambda: configs.mae_model(
                processor=configs.conv_processor(), tokenizer=configs.patchify(patch_shape=(2, 4))
            ),
            "reconstruction_target",
        ),
        (lambda: configs.mae_model(tokenizer=configs.noop_tokenizer()), "patch_grid"),
    ],
)
def test_requires_reconstruction_target_and_patch_grid_capabilities(model_config, match):
    with pytest.raises(ValueError, match=match):
        MAEObjective(AudioModel(model_config()), CASE.objective_config())


def test_output_layout():
    out = CASE.build().train()(CASE.batch())
    assert out.prediction.shape == out.target.shape
    assert out.target_mask.shape == out.ids_restore.shape == out.prediction.shape[:2]
    assert out.target_mask.dtype == torch.bool
    assert out.removal_mask.dtype == torch.bool
    assert not (out.target_mask & ~out.removal_mask).any(), "targets must be removed tokens"
    assert not out.target.requires_grad
    restored = torch.sort(out.ids_restore, dim=1).values
    assert torch.equal(restored, torch.arange(out.ids_restore.shape[1]).expand_as(restored))


def test_group_masking_path_runs():
    case = _with_objective(masking=MAEMaskingConfig(mask_ratio=0.5, group_factor=5))
    assert torch.isfinite(case.build().train()(case.batch()).loss)


def test_loss_stays_float32_under_cpu_bfloat16_autocast():
    objective = CASE.build().train()
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = objective(CASE.batch())
    assert out.loss.dtype == torch.float32


def _training_config(model=None, **objective_overrides) -> AudioSSLTrainingConfig:
    return AudioSSLTrainingConfig(
        model=model or CASE.model(),
        objective=replace(CASE.objective(), **objective_overrides),
        trainer=TrainerConfig(
            augmentor=AudioAugmentorConfig(augmentations=[]),
            optimizer=AdamWOptimizerConfig(lr=1e-3),
            scheduler=ConstantSchedulerConfig(),
            batch=BatchConfig(per_device_batch_size=2),
            loop=TrainingLoopConfig(max_epochs=1),
        ),
    )


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1])
def test_validator_rejects_invalid_mask_ratio(ratio):
    with pytest.raises(ValueError):
        validate_config(MAEMaskingConfig(mask_ratio=ratio))


@pytest.mark.parametrize("norm", ["random", "", 0])
def test_validator_rejects_invalid_norm_pix_loss(norm):
    with pytest.raises((ValueError, TypeError)):
        validate_config(replace(CASE.objective(), norm_pix_loss=norm))


def test_training_config_validates_and_checks_decoder_pe_grid():
    assert validate_config(_training_config())
    bad_pe = configs.pe_learnable_2d(max_t=configs.MAX_TIME_PATCHES - 1)
    with pytest.raises(ValueError, match="max_t"):
        validate_config(_training_config(decoder_positional_encoding=bad_pe))


def test_training_config_validator_rejects_processor_without_reconstruction_target():
    model = configs.mae_model(
        processor=configs.conv_processor(), tokenizer=configs.patchify(patch_shape=(2, 4))
    )
    with pytest.raises(ValueError, match="reconstruction_target"):
        validate_config(_training_config(model=model))


def test_padding_is_never_kept_or_scored():
    batch = CASE.batch()
    out = CASE.build().train()(batch)
    objective = CASE.build()
    tokens = objective.model.tokenize(objective.model.process(batch))
    valid = tokens.mask
    assert not valid.all(), "case must contain padded tokens"
    assert not (out.target_mask & ~valid).any(), "padding must not be scored"
    assert out.removal_mask[~valid].all(), "padding must never reach the encoder"
    kept_valid = (~out.removal_mask & valid).sum(dim=1).float()
    ratio = CASE.objective_config().masking.mask_ratio
    expected = torch.floor(valid.sum(dim=1).double() * (1 - ratio)).clamp(min=1).float()
    assert torch.equal(kept_valid, expected), "short clips must be masked at the configured ratio"

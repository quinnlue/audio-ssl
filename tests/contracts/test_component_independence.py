"""Components are seeded, described, and checked independently of each other."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from audiossl.config import validate_config
from audiossl.config.capabilities import check_objective_compatibility
from audiossl.modelling.model import AudioModel
from audiossl.runtime.runtime import derive_seed
from audiotrainer.fingerprint import config_fingerprint

from testkit import configs


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _assert_same_init(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = _state(a), _state(b)
    assert sa.keys() == sb.keys()
    for key in sa:
        torch.testing.assert_close(sa[key], sb[key], rtol=0, atol=0, msg=key)


def test_derive_seed_is_stable_and_component_specific():
    assert derive_seed(0, "model/encoder") == derive_seed(0, "model/encoder")
    assert derive_seed(0, "model/encoder") != derive_seed(0, "model/embedding")
    assert derive_seed(0, "model/encoder") != derive_seed(1, "model/encoder")
    assert 0 <= derive_seed(2**40, "x") < 2**63
    with pytest.raises(ValueError, match="non-empty component"):
        derive_seed(0, "")


def test_changing_one_component_leaves_other_initializations_unchanged():
    pytest.importorskip("torchaudio")
    shallow = AudioModel(configs.model(encoder=configs.transformer(num_layers=1)))
    deep = AudioModel(configs.model(encoder=configs.transformer(num_layers=3)))
    _assert_same_init(shallow.processor, deep.processor)
    _assert_same_init(shallow.embedding, deep.embedding)

    other_embedding = AudioModel(
        configs.model(embedding=configs.linear_embedding(configs.pe_learnable_2d()))
    )
    _assert_same_init(AudioModel(configs.model()).encoder, other_embedding.encoder)


def test_model_construction_does_not_consume_global_rng():
    pytest.importorskip("torchaudio")
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    AudioModel(configs.model())
    torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)


def test_describe_names_every_component():
    pytest.importorskip("torchaudio")
    model = AudioModel(configs.model())
    text = model.describe()
    for name, kind in (
        ("processor", "melspec"),
        ("tokenizer", "patchify"),
        ("embedding", "linear"),
        ("encoder", "transformer"),
    ):
        assert name in text and kind in text, text
    assert f"{sum(p.numel() for p in model.parameters()):,}" in text


def test_missing_capability_names_objective_and_components():
    with pytest.raises(ValueError) as info:
        check_objective_compatibility(configs.model(), configs.ufo_objective())
    message = str(info.value)
    assert "UFOObjectiveConfig" in message and "prefix_token" in message
    assert "embedding 'linear'" in message


def test_incompatible_capability_names_its_source():
    model = configs.mae_model(
        embedding=configs.linear_embedding(
            configs.pe_learnable_2d(), subsampling=configs.subsampling(2)
        )
    )
    with pytest.raises(ValueError, match=r"token_subsampling \(from embedding 'linear'\)"):
        check_objective_compatibility(model, configs.mae_objective())


def test_training_config_validation_runs_capability_checks():
    with pytest.raises(ValueError, match="requires model capabilities"):
        validate_config(configs.training(configs.ufo_objective(), configs.model()))


def test_fingerprint_ignores_device_but_not_seed():
    config = configs.training(configs.jepa_objective())

    def with_runtime(**changes):
        runtime = replace(config.model.runtime, **changes)
        return replace(config, model=replace(config.model, runtime=runtime))

    data_fp = "fixtures/audio"
    assert config_fingerprint(with_runtime(device="cpu"), data_fingerprint=data_fp) == (
        config_fingerprint(with_runtime(device="cuda"), data_fingerprint=data_fp)
    )
    assert config_fingerprint(with_runtime(seed=1), data_fingerprint=data_fp) != (
        config_fingerprint(with_runtime(seed=2), data_fingerprint=data_fp)
    )

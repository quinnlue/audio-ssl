"""Stage-boundary contracts for processors, tokenizers, masking strategies, and registries.

These run per registered component (via the model cases that exercise them),
so a new processor/tokenizer/masking strategy inherits the checks by being
added to ``testkit.cases`` or ``MASKING_CASES`` below.
"""

from __future__ import annotations

import pytest
import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.registry import (
    ACTIVATION_REGISTRY,
    INITIALIZATION_REGISTRY,
    build_activation,
    get_masking_strategy_registry,
)
from testkit.cases import MASKING_CASES, MODEL_CASES, model_params


def _first_case_per(kind: str):
    seen: dict[str, object] = {}
    for case in MODEL_CASES:
        key = dict(case.components())[kind]
        seen.setdefault(key, case)
    return list(seen.values())


def _boundary_violations(batch: AudioBatch) -> list[tuple[object, type[Exception]]]:
    return [
        (batch.x, TypeError),
        (AudioBatch(x=batch.x[..., None], mask=batch.mask[..., None]), ValueError),
        (AudioBatch(x=batch.x, mask=batch.mask.float()), ValueError),
        (AudioBatch(x=batch.x, mask=batch.mask[:, :-1]), ValueError),
    ]


# -------------------------------------------------------------- processors ---


@pytest.mark.parametrize("case", model_params("processor_boundary", _first_case_per("processor")))
def test_processor_boundary_validation(case):
    processor = case.build().processor.processor
    for bad, error in _boundary_violations(case.batch()):
        with pytest.raises(error):
            processor(bad)


@pytest.mark.parametrize("case", model_params("processor_output", _first_case_per("processor")))
def test_processor_output_is_rank3_features_with_bool_frame_mask(case):
    model = case.build().eval()
    batch = case.batch()
    with torch.no_grad():
        out = model.process(batch)
    assert out.x.ndim == 3 and out.x.shape[0] == batch.x.shape[0]
    assert out.mask.dtype == torch.bool and out.mask.shape == out.x.shape[:2]
    assert torch.isfinite(out.x).all()
    assert out.x.shape[-1] == model.config.processor.output_dim


# -------------------------------------------------------------- tokenizers ---


def _feature_batch(case, *, seed: int = 0) -> AudioBatch:
    model = case.build().eval()
    with torch.no_grad():
        processed = model.process(case.batch())
    generator = torch.Generator().manual_seed(seed)
    # Arbitrary (non-prefix) frame validity exercises the per-patch AND rule.
    mask = torch.rand(processed.mask.shape, generator=generator) > 0.2
    return AudioBatch(x=processed.x, mask=mask)


@pytest.mark.parametrize("case", model_params("tokenizer_boundary", _first_case_per("tokenizer")))
def test_tokenizer_boundary_validation(case):
    tokenizer = case.build().tokenizer.tokenizer
    for bad, error in _boundary_violations(_feature_batch(case)):
        with pytest.raises(error):
            tokenizer(bad)


@pytest.mark.parametrize(
    "case",
    model_params(
        "tokenizer_layout",
        [c for c in MODEL_CASES if dict(c.components())["tokenizer"] == "patchify"],
    ),
)
def test_patchify_tokens_follow_layout_time_major_with_and_reduced_mask(case):
    model = case.build()
    features = _feature_batch(case)
    layout = model.token_layout(features)
    tokens = model.tokenize(features)
    pt, pf = layout.patch_shape
    st, sf = layout.stride
    batch_size, frames, _ = features.x.shape
    x = features.x
    mask = features.mask
    if frames < (layout.time_patches - 1) * st + pt:
        pad = (layout.time_patches - 1) * st + pt - frames
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad), value=False)

    assert layout.num_tokens == tokens.x.shape[1] == layout.time_patches * layout.frequency_patches
    assert layout.output_dim == tokens.x.shape[-1] == pt * pf
    assert layout.flatten_order == "time_major"
    for k in range(layout.num_tokens):
        t, f = divmod(k, layout.frequency_patches)
        expected = x[:, t * st : t * st + pt, f * sf : f * sf + pf].reshape(batch_size, -1)
        assert torch.equal(tokens.x[:, k], expected), f"token {k} content/order"
        assert torch.equal(tokens.mask[:, k], mask[:, t * st : t * st + pt].all(dim=1))


@pytest.mark.parametrize("case", model_params("tokenizer_lengths", _first_case_per("tokenizer")))
def test_tokenizer_output_lengths_never_exceed_token_count(case):
    model = case.build().eval()
    with torch.no_grad():
        processed = model.process(case.batch())
    tokenizer = model.tokenizer.tokenizer
    tokens = model.tokenize(processed)
    lengths = tokenizer.output_lengths(processed.mask.sum(1))
    assert (lengths <= tokens.x.shape[1]).all()
    assert (lengths >= 0).all()


# ----------------------------------------------------------------- masking ---

def _valid_mask(shape=(4, 40)) -> torch.Tensor:
    lengths = torch.tensor([40, 31, 17, 9])
    return torch.arange(shape[1])[None, :] < lengths[:, None]


@pytest.mark.parametrize("name", sorted(MASKING_CASES))
def test_masking_strategy_contract(name):
    strategy = get_masking_strategy_registry()[name](config=MASKING_CASES[name])
    valid = _valid_mask()

    def run(seed):
        return strategy.compute_mask(
            shape=valid.shape, valid_mask=valid, generator=torch.Generator().manual_seed(seed)
        )

    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    first = run(0)
    assert torch.equal(torch.random.get_rng_state(), rng_before), "consumed global RNG"
    assert first.dtype == torch.bool and first.shape == valid.shape
    assert not (first & ~valid).any(), "selected invalid positions"
    assert first.any()
    assert torch.equal(run(0), first)
    assert any(not torch.equal(run(seed), first) for seed in range(1, 6))


@pytest.mark.parametrize("name", sorted(MASKING_CASES))
def test_masking_strategy_boundary_validation(name):
    strategy = get_masking_strategy_registry()[name](config=MASKING_CASES[name])
    valid = _valid_mask()
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        strategy.compute_mask(shape=(4, 40, 2), valid_mask=None, generator=generator)
    with pytest.raises(ValueError):
        strategy.compute_mask(shape=valid.shape, valid_mask=valid.float(), generator=generator)
    with pytest.raises(ValueError):
        strategy.compute_mask(shape=(4, 39), valid_mask=valid, generator=generator)


# ---------------------------------------------------------- shared registries ---


@pytest.mark.parametrize("name", sorted(ACTIVATION_REGISTRY))
def test_activations_are_elementwise_and_deterministic(name):
    activation = build_activation(name)
    assert isinstance(activation, torch.nn.Module)
    assert build_activation(name) is not activation, "each call must build a fresh module"
    x = torch.linspace(-4, 4, 33).reshape(3, 11)
    y = activation(x)
    assert torch.equal(activation(x), y)
    assert torch.equal(activation(x.flatten()).reshape(3, 11), y)
    assert torch.equal(activation(x[1:2]), y[1:2])


@pytest.mark.parametrize("name", sorted(INITIALIZATION_REGISTRY))
def test_initializers_are_seed_deterministic_and_in_place(name):
    init = INITIALIZATION_REGISTRY[name]

    def draw(seed):
        torch.manual_seed(seed)
        weight = torch.zeros(16, 8)
        returned = init(weight)
        assert returned is weight
        return weight

    first = draw(0)
    assert first.abs().sum() > 0
    assert torch.equal(draw(0), first)
    assert not torch.equal(draw(1), first)

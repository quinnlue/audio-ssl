"""Behavioural contracts every ``AudioModel`` configuration must satisfy.

Each test is a property over ``testkit.cases.MODEL_CASES``. The properties are
the ones research results silently depend on:

- the staged API composes to exactly the same computation as ``forward``;
- masks are truthful (lengths, prefix validity, declared ``output_lengths``);
- valid outputs do not depend on padding content, padding amount, or the other
  rows in the batch (eval mode);
- no gradient flows into padded waveform samples;
- outputs are deterministic and finite in float32 and bfloat16.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.model import AudioModel
from testkit import batches, checks
from testkit.cases import model_params

ATOL = 1e-5
RTOL = 1e-4


def _valid(z: torch.Tensor, mask: torch.Tensor, row: int) -> torch.Tensor:
    return z[row, mask[row]]


@pytest.mark.parametrize("case", model_params("output_contract"))
def test_output_shapes_dtypes_and_masks(case):
    model = case.build().eval()
    batch = case.batch()
    with torch.no_grad():
        out = model(batch)

    batch_size = batch.x.shape[0]
    assert out.z.ndim == 3 and out.z.shape[0] == batch_size
    assert out.z.shape[-1] == model.config.d_model
    assert out.z.dtype == torch.float32
    assert out.mask.dtype == torch.bool and out.mask.shape == out.z.shape[:2]
    assert torch.equal(out.lengths, out.mask.sum(dim=1))
    assert (out.lengths > 0).all(), "every non-empty clip must keep at least one token"
    assert out.cls is not None and out.cls.shape == (batch_size, model.config.d_model)
    assert torch.isfinite(out.z[out.mask]).all()
    assert torch.isfinite(out.cls).all()

    # Centered framing may invalidate leading frames, so validity is a single
    # contiguous run rather than a strict prefix.
    edges = out.mask.int().diff(dim=1, prepend=torch.zeros_like(out.mask[:, :1].int()))
    assert ((edges == 1).sum(dim=1) == 1).all(), "valid tokens must form one contiguous run"

    shorter_first = torch.argsort(batch.mask.sum(1))
    assert (out.lengths[shorter_first].diff() >= 0).all(), "longer input must not lose tokens"


@pytest.mark.parametrize("case", model_params("staged_api"))
def test_staged_api_composes_to_forward(case):
    model = case.build().eval()
    batch = case.batch()
    with torch.no_grad():
        full = model(batch)
        processed = model.process(batch)
        layout = model.token_layout(processed)
        tokens = model.tokenize(processed)
        embedded = model.embed(tokens, layout=layout)
        staged = model.encode_embedded(embedded)
        staged.cls = model.get_cls(staged)
        via_encode_tokens = model.encode_tokens(tokens, layout=layout)

    checks.assert_identical(staged, full, what="staged")
    checks.assert_identical(via_encode_tokens.z, full.z, what="encode_tokens.z")


@pytest.mark.parametrize("case", model_params("declared_lengths"))
def test_declared_output_lengths_match_masks(case):
    """``output_lengths`` is what objectives use to size targets; it must agree with forward."""
    model = case.build().eval()
    batch = case.batch()
    input_lengths = batch.mask.sum(dim=1)
    processor = model.processor.processor
    tokenizer = model.tokenizer.tokenizer
    with torch.no_grad():
        processed = model.process(batch)
        tokens = model.tokenize(processed)

    if hasattr(processor, "output_lengths"):
        assert torch.equal(processor.output_lengths(input_lengths), processed.mask.sum(1))
    if hasattr(processor, "output_mask"):
        assert torch.equal(processor.output_mask(batch.mask), processed.mask)
    if hasattr(tokenizer, "output_lengths"):
        assert torch.equal(tokenizer.output_lengths(processed.mask.sum(1)), tokens.mask.sum(1))


@pytest.mark.parametrize("case", model_params("padding_invariance"))
def test_valid_outputs_ignore_padding_content_and_amount(case):
    model = case.build().eval()
    batch = case.batch()
    noisy = batches.with_garbage_padding(batch, extra=97)
    with torch.no_grad():
        reference = model(batch)
        perturbed = model(noisy)

    for row in range(batch.x.shape[0]):
        assert int(perturbed.lengths[row]) == int(reference.lengths[row])
        torch.testing.assert_close(
            _valid(perturbed.z, perturbed.mask, row),
            _valid(reference.z, reference.mask, row),
            atol=ATOL,
            rtol=RTOL,
            msg=lambda m, r=row: f"row {r}: valid tokens changed with padding\n{m}",
        )
    torch.testing.assert_close(perturbed.cls, reference.cls, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("case", model_params("batch_independence"))
def test_rows_are_independent_in_eval(case):
    model = case.build().eval()
    batch = case.batch()
    with torch.no_grad():
        together = model(batch)
        for row in range(batch.x.shape[0]):
            alone = model(batches.row(batch, row))
            n = int(alone.lengths[0])
            assert int(together.lengths[row]) == n
            torch.testing.assert_close(
                together.z[row, :n], alone.z[0, :n], atol=ATOL, rtol=RTOL
            )
            torch.testing.assert_close(together.cls[row], alone.cls[0], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("case", model_params("input_gradient_isolation"))
def test_no_gradient_reaches_padded_samples(case):
    model = case.build().eval()
    batch = case.batch()
    x = batch.x.clone().requires_grad_(True)
    out = model(AudioBatch(x=x, mask=batch.mask))
    weights = torch.randn(out.z.shape, generator=torch.Generator().manual_seed(3))
    (out.z * weights * out.mask[..., None]).sum().backward()

    assert x.grad is not None
    padded_grad = x.grad[~batch.mask]
    assert torch.count_nonzero(padded_grad) == 0, (
        f"{torch.count_nonzero(padded_grad)} padded samples received gradient "
        f"(max |g| = {padded_grad.abs().max():.3e})"
    )
    for row in range(batch.x.shape[0]):
        assert x.grad[row, batch.mask[row]].abs().sum() > 0, f"row {row}: no input gradient"


@pytest.mark.parametrize("case", model_params("parameter_gradients"))
def test_every_trainable_parameter_receives_finite_gradient(case):
    model = case.build().train()
    out = model(case.batch())
    (out.z[out.mask].square().mean() + out.cls.square().mean()).backward()

    params = checks.trainable(model)
    assert params, "model has no trainable parameters"
    assert checks.missing_grads(params) == []
    assert checks.nonfinite_grads(params) == []


@pytest.mark.parametrize("case", model_params("eval_determinism"))
def test_eval_forward_is_bitwise_repeatable(case):
    model = case.build().eval()
    batch = case.batch()
    with torch.no_grad():
        first = model(batch)
        second = model(batches.clone(batch))
    checks.assert_identical(second, first, what="repeat forward")


@pytest.mark.parametrize("case", model_params("bfloat16"))
def test_bfloat16_runtime_produces_finite_bfloat16_outputs(case):
    config = case.model_config()
    config = replace(config, runtime=replace(config.runtime, dtype="bfloat16"))
    model = AudioModel(config).eval()
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    with torch.no_grad():
        out = model(case.batch())
    assert out.z.dtype == torch.bfloat16
    assert torch.isfinite(out.z[out.mask].float()).all()


@pytest.mark.parametrize("case", model_params("config_isolation"))
def test_construction_neither_mutates_nor_aliases_config(case):
    config = case.model_config()
    before = repr(config)
    model = AudioModel(config)
    assert repr(config) == before
    assert model.config == config
    assert model.config is not config

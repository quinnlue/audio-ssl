"""Independent PyTorch forward equations for the registered encoder stacks."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from audiossl.config import AudioBatch, BranchMergeConfig
from audiossl.modelling.encoders.encoder import Encoder
from audiossl.modelling.registry import get_encoder_block_registry

from testkit import configs
from testkit.parity import assert_training_parity


def _attention(module, config, x, mask, positions):
    batch, time, width = x.shape
    heads = config.mha.num_heads
    head_dim = width // heads
    rope = getattr(config, "positional_encoding", None)
    if config.type == "alt_transformer":
        projected = module.qkv(x)
        output = module.proj
    else:
        projected = F.linear(x, module.mha.in_proj_weight, module.mha.in_proj_bias)
        output = module.mha.out_proj

    q, k, v = [
        tensor.reshape(batch, time, heads, head_dim).transpose(1, 2)
        for tensor in projected.chunk(3, dim=-1)
    ]
    if rope is not None:
        inv = (rope.base ** (torch.arange(0, head_dim, 2).float() / head_dim)).reciprocal()
        phase = positions[:, None, :, None] * inv
        real, imag = phase.cos().to(x.dtype), phase.sin().to(x.dtype)

        def rotate(tensor):
            a, b = tensor[..., ::2], tensor[..., 1::2]
            return torch.stack((a * real - b * imag, a * imag + b * real), dim=-1).flatten(-2)

        q, k = rotate(q), rotate(k)
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) / head_dim**0.5
    weights = scores.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
    attended = torch.einsum("bhts,bhsd->bthd", weights, v)
    return output(attended.reshape(batch, time, width))


def _ffn(module, x):
    return module.fc2(F.silu(module.fc1(x)))


def _transformer(block, config, x, mask, pos):
    attention = block.attn if config.type == "alt_transformer" else block.mha
    pre_norm = config.layer_norm_first if config.type == "alt_transformer" else config.pre_norm
    if pre_norm:
        x = x + _attention(attention, config, block.norm1(x), mask, pos)
        return x + _ffn(block.mlp, block.norm2(x))
    x = block.norm1(x + _attention(attention, config, x, mask, pos))
    return block.norm2(x + _ffn(block.mlp, x))


def _conformer(block, config, x, mask, pos):
    x = x + 0.5 * _ffn(block.ffn1, block.norm_ffn1(x))
    x = x + _attention(block.attn, config, block.norm_attn(x), mask, pos)
    conv = block.conv
    channels = conv.norm(block.norm_conv(x)).transpose(1, 2)
    value, gate = conv.pw1(channels).chunk(2, dim=1)
    local = conv.depthwise((value * gate.sigmoid()).masked_fill(~mask[:, None, :], 0))
    local = conv.pw2(F.silu(conv.batch_norm(local))).transpose(1, 2)
    x = x + local.masked_fill(~mask[..., None], 0)
    return block.final_norm(x + 0.5 * _ffn(block.ffn2, block.norm_ffn2(x)))


def _branches(block, config, x, mask, pos):
    global_branch = _attention(block.attn, config, block.norm_attn(x), mask, pos)
    local = block.cgmlp
    value, gate = F.gelu(local.fc1(local.norm(x))).chunk(2, dim=-1)
    gate = local.gate_norm(gate).masked_fill(~mask[..., None], 0)
    gate = local.depthwise(gate.transpose(1, 2)).transpose(1, 2)
    local_branch = local.fc2(value * gate).masked_fill(~mask[..., None], 0)
    return global_branch, local_branch


def _branchformer(block, config, x, mask, pos):
    global_branch, local_branch = _branches(block, config, x, mask, pos)
    if config.merge.method == "concat":
        merged = block.merger.projection(torch.cat((global_branch, local_branch), dim=-1))
    else:
        ratio = (
            (block.merger.merge_weights[0] - block.merger.merge_weights[1]).sigmoid()
            if config.merge.method == "learned_ave" else block.merger.fixed_weight
        )
        merged = torch.lerp(local_branch, global_branch, ratio)
    return block.final_norm(x + merged)


def _ebranchformer(block, config, x, mask, pos):
    x = x + 0.5 * _ffn(block.ffn_macaron, block.norm_ff_macaron(x))
    global_branch, local_branch = _branches(block, config, x, mask, pos)
    joined = torch.cat((global_branch, local_branch), dim=-1).masked_fill(~mask[..., None], 0)
    refined = block.merger.depthwise(joined.transpose(1, 2)).transpose(1, 2)
    x = x + block.merger.projection(joined + refined)
    return block.final_norm(x + _ffn(block.ffn, block.norm_ff(x)))


ENCODERS = {
    "transformer": _transformer,
    "alt_transformer": _transformer,
    "conformer": _conformer,
    "branchformer": _branchformer,
    "ebranchformer": _ebranchformer,
}


def test_every_registered_encoder_has_a_training_reference():
    assert set(ENCODERS) == set(get_encoder_block_registry())


@pytest.mark.parametrize(
    ("make_config", "options"),
    [
        (configs.transformer, {}),
        (configs.transformer, {"pre_norm": False}),
        (configs.alt_transformer, {"layer_norm_first": True}),
        (configs.alt_transformer, {"layer_norm_first": False}),
        (configs.conformer, {"pre_norm": True}),
        (configs.branchformer, {"merge": BranchMergeConfig(method="concat")}),
        (configs.branchformer, {"merge": BranchMergeConfig(method="learned_ave")}),
        (configs.branchformer, {"merge": BranchMergeConfig(method="fixed_ave", fixed_weight=0.7)}),
        (configs.ebranchformer, {"pre_norm": True}),
    ],
    ids=["transformer-pre", "transformer-post", "alt-pre", "alt-post", "conformer",
         "branch-concat", "branch-learned", "branch-fixed", "ebranchformer"],
)
def test_encoder_training_matches_reference(make_config, options, parity_dtype, parity_precision):
    config = make_config(layer_norm_eps=parity_precision.eps, final_norm=True, **options)
    actual = Encoder(d_model=configs.D_MODEL, config=config).to(parity_dtype).train()
    reference = deepcopy(actual)
    block_forward = ENCODERS[config.type]
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    positions = torch.tensor([[0, 1, 4, 6, 8], [1, 2, 7, 0, 0]])

    def actual_forward(_step, x):
        return actual(AudioBatch(x=x, mask=mask, positions=positions)).z[mask]

    def reference_forward(_step, x):
        for block in reference.blocks:
            x = block_forward(block, config, x, mask, positions)
        return reference.final_norm(x)[mask]

    def compare_buffers():
        for name, value in actual.named_buffers():
            if name.endswith("num_batches_tracked"):
                assert torch.equal(value, reference.get_buffer(name)), name
            elif "running_" in name:
                parity_precision.close(value, reference.get_buffer(name), label=name)

    assert_training_parity(
        actual, reference,
        actual_forward=actual_forward,
        reference_forward=reference_forward,
        inputs=lambda step: torch.randn(
            (*mask.shape, configs.D_MODEL), dtype=parity_dtype,
            generator=torch.Generator().manual_seed(30 + step),
        ),
        precision=parity_precision,
        after_step=compare_buffers,
    )

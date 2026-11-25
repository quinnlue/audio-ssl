from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from audiossl.config import (
    AudioModelConfig,
    BranchMergeConfig,
    BranchformerEncoderConfig,
    CGMLPConfig,
    LinearEmbeddingConfig,
    MelSpectrogramProcessorConfig,
    MultiHeadAttentionConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    RopePositionalEncodingConfig,
    RuntimeConfig,
)
from audiossl.modelling.encoders.branchformer_block import BranchformerBlock


def _patchify_tokenizer_config(**overrides) -> PatchifyTokenizerConfig:
    values = {
        "patch_shape": (4, 1),
        "padding_strategy": "truncate",
    }
    values.update(overrides)
    return PatchifyTokenizerConfig(**values)


def _branchformer_config(
    *,
    merge_method: str = "concat",
    attention_branch_drop_p: float = 0.0,
    cgmlp_branch_drop_p: float = 0.0,
    pre_norm: bool = True,
) -> AudioModelConfig:
    return AudioModelConfig(
        d_model=8,
        processor=MelSpectrogramProcessorConfig(
            n_mels=4,
            n_fft=16,
            win_length=16,
            hop_length=8,
        ),
        tokenizer=_patchify_tokenizer_config(),
        embedding=LinearEmbeddingConfig(
            positional_encoding=NoPositionalEncodingConfig()
        ),
        encoder=BranchformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2, dropout_p=0.0),
            cgmlp=CGMLPConfig(
                linear_units=16,
                conv_kernel_size=3,
                activation="gelu",
                gate_activation="identity",
                dropout_p=0.0,
            ),
            merge=BranchMergeConfig(method=merge_method),  # type: ignore[arg-type]
            num_layers=1,
            pre_norm=pre_norm,
            residual_dropout_p=0.0,
            positional_encoding=RopePositionalEncodingConfig(max_len=64),
            attention_branch_drop_p=attention_branch_drop_p,
            cgmlp_branch_drop_p=cgmlp_branch_drop_p,
        ),
        runtime=RuntimeConfig(seed=0, device="cpu", dtype="float32"),
    )


class BranchformerBlockTests(unittest.TestCase):
    def test_forward_preserves_shape_pre_norm(self) -> None:
        config = _branchformer_config(pre_norm=True)
        block = BranchformerBlock(d_model=config.d_model, config=config.encoder)
        x = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.bool)

        output = block(x, mask)

        self.assertEqual(tuple(output.shape), (2, 5, config.d_model))

    def test_forward_preserves_shape_post_norm(self) -> None:
        config = _branchformer_config(pre_norm=False)
        block = BranchformerBlock(d_model=config.d_model, config=config.encoder)
        x = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.bool)

        output = block(x, mask)

        self.assertEqual(tuple(output.shape), (2, 5, config.d_model))

    def test_works_with_each_merge_method(self) -> None:
        for method in ("concat", "learned_ave", "fixed_ave"):
            with self.subTest(merge_method=method):
                config = _branchformer_config(merge_method=method)
                block = BranchformerBlock(d_model=config.d_model, config=config.encoder)
                output = block(
                    torch.randn(1, 3, config.d_model),
                    torch.ones(1, 3, dtype=torch.bool),
                )
                self.assertEqual(tuple(output.shape), (1, 3, config.d_model))

    def test_rejects_wrong_encoder_config_type(self) -> None:
        from audiossl.config import TransformerEncoderConfig, MLPConfig

        config = _branchformer_config()
        config = replace(
            config,
            encoder=TransformerEncoderConfig(
                mha=MultiHeadAttentionConfig(num_heads=2),
                mlp=MLPConfig(d_ff=16),
            ),
        )

        with self.assertRaisesRegex(TypeError, "BranchformerEncoderConfig"):
            BranchformerBlock(d_model=config.d_model, config=config.encoder)

    def test_attention_branch_drop_one_zeros_global_branch_in_training(self) -> None:
        torch.manual_seed(0)
        config = _branchformer_config(attention_branch_drop_p=1.0, merge_method="learned_ave")
        block = BranchformerBlock(d_model=config.d_model, config=config.encoder)
        block.train()

        x = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.bool)

        out_drop = block(x, mask)

        block.eval()
        config_off = _branchformer_config(attention_branch_drop_p=0.0, merge_method="learned_ave")
        block_off = BranchformerBlock(d_model=config_off.d_model, config=config_off.encoder)
        block_off.load_state_dict(block.state_dict())
        block_off.eval()
        with torch.no_grad():
            attn_off = block_off.attn(block_off.norm_attn(x), attention_mask=mask)
            cgmlp_only = block_off.cgmlp(x, attention_mask=mask)
            import torch.nn.functional as F

            w = F.softmax(block_off.merger.merge_weights, dim=0)
            merged_no_attn = w[0] * torch.zeros_like(attn_off) + w[1] * cgmlp_only
            expected = block_off.final_norm(x + merged_no_attn)

        torch.testing.assert_close(out_drop, expected, rtol=1e-5, atol=1e-6)

    def test_cgmlp_branch_drop_one_zeros_local_branch_in_training(self) -> None:
        torch.manual_seed(1)
        config = _branchformer_config(cgmlp_branch_drop_p=1.0, merge_method="learned_ave")
        block = BranchformerBlock(d_model=config.d_model, config=config.encoder)
        block.train()

        x = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.bool)
        out_drop = block(x, mask)

        block.eval()
        with torch.no_grad():
            attn_only = block.attn(block.norm_attn(x), attention_mask=mask)
            import torch.nn.functional as F

            w = F.softmax(block.merger.merge_weights, dim=0)
            merged_no_cgmlp = w[0] * attn_only + w[1] * torch.zeros_like(attn_only)
            expected = block.final_norm(x + merged_no_cgmlp)

        torch.testing.assert_close(out_drop, expected, rtol=1e-5, atol=1e-6)

    def test_branch_drop_is_noop_in_eval(self) -> None:
        config = _branchformer_config(
            attention_branch_drop_p=1.0,
            cgmlp_branch_drop_p=1.0,
        )
        block = BranchformerBlock(d_model=config.d_model, config=config.encoder)
        block.eval()

        x = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.bool)

        with torch.no_grad():
            out_eval = block(x, mask)

        block.train()
        out_train = block(x, mask)

        self.assertFalse(torch.allclose(out_eval, out_train))

    def test_padding_does_not_leak_into_valid_frames(self) -> None:
        torch.manual_seed(2026)
        config = _branchformer_config()
        block = BranchformerBlock(d_model=config.d_model, config=config.encoder)
        block.eval()

        valid_len = 4
        padded_len = 9
        sample = torch.randn(1, valid_len, config.d_model)
        sample_mask = torch.ones(1, valid_len, dtype=torch.bool)

        padded = torch.randn(2, padded_len, config.d_model)
        padded[0, :valid_len] = sample[0]
        padded[0, valid_len:] = 100.0 * torch.randn(padded_len - valid_len, config.d_model)
        padded[1] = -100.0 * torch.randn(padded_len, config.d_model)
        padded_mask = torch.zeros(2, padded_len, dtype=torch.bool)
        padded_mask[0, :valid_len] = True
        padded_mask[1, : padded_len - 1] = True

        with torch.no_grad():
            expected = block(sample, sample_mask)[0]
            actual = block(padded, padded_mask)[0, :valid_len]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

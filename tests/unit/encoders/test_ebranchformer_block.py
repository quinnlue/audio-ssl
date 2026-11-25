from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from audiossl.config import (
    AudioModelConfig,
    EBranchMergeConfig,
    EBranchformerEncoderConfig,
    CGMLPConfig,
    LinearEmbeddingConfig,
    MelSpectrogramProcessorConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    RopePositionalEncodingConfig,
    RuntimeConfig,
)
from audiossl.modelling.encoders.ebranchformer_block import EBranchformerBlock


def _patchify_tokenizer_config(**overrides) -> PatchifyTokenizerConfig:
    values = {
        "patch_shape": (4, 1),
        "padding_strategy": "truncate",
    }
    values.update(overrides)
    return PatchifyTokenizerConfig(**values)


def _ebranchformer_config(
    *,
    merge_kernel: int = 3,
    use_ffn: bool = False,
    macaron_ffn: bool = False,
    pre_norm: bool = True,
) -> AudioModelConfig:
    mlp = MLPConfig(d_ff=16, dropout_p=0.0) if (use_ffn or macaron_ffn) else None
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
        encoder=EBranchformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2, dropout_p=0.0),
            cgmlp=CGMLPConfig(
                linear_units=16,
                conv_kernel_size=3,
                activation="gelu",
                gate_activation="identity",
                dropout_p=0.0,
            ),
            merge=EBranchMergeConfig(conv_kernel_size=merge_kernel),
            mlp=mlp,
            use_ffn=use_ffn,
            macaron_ffn=macaron_ffn,
            num_layers=1,
            pre_norm=pre_norm,
            residual_dropout_p=0.0,
            positional_encoding=RopePositionalEncodingConfig(max_len=64),
        ),
        runtime=RuntimeConfig(seed=0, device="cpu", dtype="float32"),
    )


class EBranchformerBlockTests(unittest.TestCase):
    def test_forward_preserves_shape_pre_norm(self) -> None:
        config = _ebranchformer_config(pre_norm=True)
        block = EBranchformerBlock(d_model=config.d_model, config=config.encoder)
        x = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.bool)

        output = block(x, mask)

        self.assertEqual(tuple(output.shape), (2, 5, config.d_model))

    def test_forward_preserves_shape_post_norm(self) -> None:
        config = _ebranchformer_config(pre_norm=False)
        block = EBranchformerBlock(d_model=config.d_model, config=config.encoder)
        x = torch.randn(2, 5, config.d_model)
        mask = torch.ones(2, 5, dtype=torch.bool)

        output = block(x, mask)

        self.assertEqual(tuple(output.shape), (2, 5, config.d_model))

    def test_merge_only_mode(self) -> None:
        config = _ebranchformer_config(use_ffn=False, macaron_ffn=False)
        block = EBranchformerBlock(d_model=config.d_model, config=config.encoder)
        output = block(
            torch.randn(1, 3, config.d_model),
            torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (1, 3, config.d_model))
        self.assertFalse(hasattr(block, "ffn"))
        self.assertFalse(hasattr(block, "ffn_macaron"))

    def test_full_mode_with_macaron_and_post_ffn(self) -> None:
        config = _ebranchformer_config(use_ffn=True, macaron_ffn=True)
        block = EBranchformerBlock(d_model=config.d_model, config=config.encoder)
        output = block(
            torch.randn(1, 3, config.d_model),
            torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (1, 3, config.d_model))
        self.assertTrue(hasattr(block, "ffn"))
        self.assertTrue(hasattr(block, "ffn_macaron"))

    def test_rejects_wrong_encoder_config_type(self) -> None:
        from audiossl.config import TransformerEncoderConfig

        config = _ebranchformer_config()
        config = replace(
            config,
            encoder=TransformerEncoderConfig(
                mha=MultiHeadAttentionConfig(num_heads=2),
                mlp=MLPConfig(d_ff=16),
            ),
        )

        with self.assertRaisesRegex(TypeError, "EBranchformerEncoderConfig"):
            EBranchformerBlock(d_model=config.d_model, config=config.encoder)

    def test_works_with_merge_kernels(self) -> None:
        for kernel in (3, 31):
            with self.subTest(merge_kernel=kernel):
                config = _ebranchformer_config(merge_kernel=kernel)
                block = EBranchformerBlock(d_model=config.d_model, config=config.encoder)
                output = block(
                    torch.randn(1, 7, config.d_model),
                    torch.ones(1, 7, dtype=torch.bool),
                )
                self.assertEqual(tuple(output.shape), (1, 7, config.d_model))

    def test_padding_does_not_leak_into_valid_frames(self) -> None:
        torch.manual_seed(2026)
        config = _ebranchformer_config(use_ffn=True, macaron_ffn=True, merge_kernel=3)
        block = EBranchformerBlock(d_model=config.d_model, config=config.encoder)
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

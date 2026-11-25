from __future__ import annotations

import unittest
from dataclasses import replace

import torch
import torch.nn as nn

from audiossl.config import (
    AudioModelConfig,
    BranchMergeConfig,
    BranchformerEncoderConfig,
    CGMLPConfig,
    ConformerEncoderConfig,
    ConvModuleConfig,
    EBranchMergeConfig,
    EBranchformerEncoderConfig,
    LinearEmbeddingConfig,
    MelSpectrogramProcessorConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    RopePositionalEncodingConfig,
    RuntimeConfig,
    TransformerEncoderConfig,
)
from audiossl.config.batch_dataclasses import AudioBatch, EncoderOutput
from audiossl.modelling.encoders.encoder import Encoder


def _patchify_tokenizer_config(**overrides) -> PatchifyTokenizerConfig:
    values = {
        "patch_shape": (4, 1),
        "padding_strategy": "truncate",
    }
    values.update(overrides)
    return PatchifyTokenizerConfig(**values)


def _transformer_config() -> AudioModelConfig:
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
        encoder=TransformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2, dropout_p=0.0),
            mlp=MLPConfig(d_ff=16, dropout_p=0.0),
            num_layers=2,
            pre_norm=True,
            residual_dropout_p=0.0,
            positional_encoding=RopePositionalEncodingConfig(max_len=64),
        ),
        runtime=RuntimeConfig(seed=0, device="cpu", dtype="float32"),
    )


def _conformer_config() -> AudioModelConfig:
    base = _transformer_config()
    return replace(
        base,
        encoder=ConformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2, dropout_p=0.0),
            mlp=MLPConfig(d_ff=16, dropout_p=0.0),
            conv=ConvModuleConfig(
                kernel_size=3,
                expansion_factor=2,
                activation="gelu",
                dropout_p=0.0,
            ),
            num_layers=2,
            pre_norm=True,
            residual_dropout_p=0.0,
            positional_encoding=base.encoder.positional_encoding,
        ),
    )


def _branchformer_config() -> AudioModelConfig:
    base = _transformer_config()
    return replace(
        base,
        encoder=BranchformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2, dropout_p=0.0),
            cgmlp=CGMLPConfig(
                linear_units=16,
                conv_kernel_size=3,
                activation="gelu",
                gate_activation="identity",
                dropout_p=0.0,
            ),
            merge=BranchMergeConfig(method="concat"),
            num_layers=2,
            pre_norm=True,
            residual_dropout_p=0.0,
            positional_encoding=base.encoder.positional_encoding,
        ),
    )


def _ebranchformer_config() -> AudioModelConfig:
    base = _transformer_config()
    return replace(
        base,
        encoder=EBranchformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2, dropout_p=0.0),
            cgmlp=CGMLPConfig(
                linear_units=16,
                conv_kernel_size=3,
                activation="gelu",
                gate_activation="identity",
                dropout_p=0.0,
            ),
            merge=EBranchMergeConfig(conv_kernel_size=3),
            mlp=MLPConfig(d_ff=16, dropout_p=0.0),
            use_ffn=True,
            macaron_ffn=True,
            num_layers=2,
            pre_norm=True,
            residual_dropout_p=0.0,
            positional_encoding=base.encoder.positional_encoding,
        ),
    )


def _audio_batch(*, batch: int, time_steps: int, feature: int) -> AudioBatch:
    x = torch.randn(batch, time_steps, feature, dtype=torch.float32)
    mask = torch.ones(batch, time_steps, dtype=torch.bool)
    return AudioBatch(x=x, mask=mask)


def _encoder_from_model_config(config: AudioModelConfig) -> Encoder:
    return Encoder(
        d_model=config.d_model,
        config=config.encoder,
    )


class EncoderLayerLoopTests(unittest.TestCase):
    def test_encoder_returns_encoder_output_with_correct_shapes(self) -> None:
        config = _transformer_config()
        encoder = _encoder_from_model_config(config)
        batch = _audio_batch(batch=2, time_steps=5, feature=config.d_model)

        output = encoder(batch)

        self.assertIsInstance(output, EncoderOutput)
        self.assertEqual(tuple(output.z.shape), (2, 5, config.d_model))
        self.assertEqual(tuple(output.mask.shape), (2, 5))
        self.assertEqual(output.mask.dtype, torch.bool)
        self.assertEqual(tuple(output.lengths.shape), (2,))
        self.assertIsNone(output.cls)

    def test_layers_receive_tensor_and_attention_mask(self) -> None:
        config = _transformer_config()
        encoder = _encoder_from_model_config(config)
        recorded: list[tuple[type, type]] = []

        original_forwards = [layer.forward for layer in encoder.layers]

        def make_spy(layer: nn.Module, original):
            def spy(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                recorded.append((type(x), type(attention_mask)))
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"Expected x to be torch.Tensor, got {type(x).__name__}")
                if not isinstance(attention_mask, torch.Tensor):
                    raise TypeError(
                        f"Expected attention_mask to be torch.Tensor, got "
                        f"{type(attention_mask).__name__}"
                    )
                return original(x, attention_mask=attention_mask)

            return spy

        for layer, original in zip(encoder.layers, original_forwards, strict=True):
            layer.forward = make_spy(layer, original)  # type: ignore[method-assign]

        batch = _audio_batch(batch=2, time_steps=5, feature=config.d_model)
        encoder(batch)

        self.assertEqual(len(recorded), len(encoder.layers))
        for x_type, mask_type in recorded:
            self.assertIs(x_type, torch.Tensor)
            self.assertIs(mask_type, torch.Tensor)

    def test_encoder_does_not_populate_cls(self) -> None:
        torch.manual_seed(0)
        config = _transformer_config()
        encoder = _encoder_from_model_config(config)
        batch = _audio_batch(batch=2, time_steps=5, feature=config.d_model)

        output = encoder(batch)

        self.assertIsNone(output.cls)

    def test_works_with_transformer_encoder_config(self) -> None:
        config = _transformer_config()
        encoder = _encoder_from_model_config(config)
        batch = _audio_batch(batch=1, time_steps=3, feature=config.d_model)

        output = encoder(batch)

        self.assertEqual(tuple(output.z.shape), (1, 3, config.d_model))

    def test_works_with_conformer_encoder_config(self) -> None:
        config = _conformer_config()
        encoder = _encoder_from_model_config(config)
        batch = _audio_batch(batch=2, time_steps=4, feature=config.d_model)

        output = encoder(batch)

        self.assertEqual(tuple(output.z.shape), (2, 4, config.d_model))

    def test_works_with_branchformer_encoder_config(self) -> None:
        config = _branchformer_config()
        encoder = _encoder_from_model_config(config)
        batch = _audio_batch(batch=2, time_steps=4, feature=config.d_model)

        output = encoder(batch)

        self.assertEqual(tuple(output.z.shape), (2, 4, config.d_model))

    def test_works_with_ebranchformer_encoder_config(self) -> None:
        config = _ebranchformer_config()
        encoder = _encoder_from_model_config(config)
        batch = _audio_batch(batch=2, time_steps=4, feature=config.d_model)

        output = encoder(batch)

        self.assertEqual(tuple(output.z.shape), (2, 4, config.d_model))


if __name__ == "__main__":
    unittest.main()

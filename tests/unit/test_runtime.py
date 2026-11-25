"""Runtime config: seeding scope, dtype casting, determinism flags, initializers."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

import torch
import torch.nn as nn

from audiossl.config import (
    AudioModelConfig,
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
from audiossl.modelling.encoders.encoder import Encoder
from audiossl.modelling.model import AudioModel
from audiossl.modelling.registry import (
    INITIALIZATION_REGISTRY,
    get_encoder_block_registry,
    get_processor_registry,
)
from audiossl.runtime import apply_global_runtime
from audiossl.data import collate_fn

_CONSTANT_INIT = "__test_constant__"


def _constant_init(tensor: torch.Tensor) -> torch.Tensor:
    return nn.init.constant_(tensor, 0.125)


def _patchify_tokenizer_config(**overrides) -> PatchifyTokenizerConfig:
    values = {
        "patch_shape": (4, 1),
        "padding_strategy": "truncate",
    }
    values.update(overrides)
    return PatchifyTokenizerConfig(**values)


def _small_config(
    *,
    seed: int = 0,
    dtype: str = "float32",
    deterministic: bool = False,
    mha_initialization: str = "xavier_uniform",
    mlp_initialization: str = "xavier_uniform",
) -> AudioModelConfig:
    mha = MultiHeadAttentionConfig(
        num_heads=2,
        dropout_p=0.0,
        initialization=mha_initialization,
    )
    mlp = MLPConfig(
        d_ff=16,
        dropout_p=0.0,
        initialization=mlp_initialization,
    )
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
            mha=mha,
            mlp=mlp,
            num_layers=1,
            pre_norm=True,
            residual_dropout_p=0.0,
            positional_encoding=RopePositionalEncodingConfig(max_len=512),
        ),
        runtime=RuntimeConfig(
            seed=seed,
            device="cpu",
            dtype=dtype,  # type: ignore[arg-type]
            deterministic=deterministic,
        ),
    )


def _encoder_from_model_config(config: AudioModelConfig) -> Encoder:
    return Encoder(
        d_model=config.d_model,
        config=config.encoder,
    )


def _parameter_snapshots(module: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(param.detach().cpu().clone() for param in module.parameters())


class RuntimeTests(unittest.TestCase):
    def test_model_construction_preserves_caller_rng_state(self) -> None:
        random.seed(13579)
        torch.manual_seed(13579)
        expected_python_value = random.random()
        expected_torch_value = torch.rand(4)

        random.seed(13579)
        torch.manual_seed(13579)
        AudioModel(_small_config(seed=2024))

        self.assertEqual(random.random(), expected_python_value)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch_value))

    def test_encoder_initialization_uses_caller_rng(self) -> None:
        config = _small_config(seed=2024)
        torch.manual_seed(1)
        first = _encoder_from_model_config(config)
        torch.manual_seed(2)
        second = _encoder_from_model_config(config)

        self.assertTrue(
            any(
                not torch.equal(first_param, second_param)
                for first_param, second_param in zip(
                    _parameter_snapshots(first),
                    _parameter_snapshots(second),
                    strict=True,
                )
            )
        )

    def test_runtime_casts_model_parameters_to_requested_dtype(self) -> None:
        model = AudioModel(_small_config(seed=0, dtype="bfloat16"))

        self.assertEqual(next(model.parameters()).dtype, torch.bfloat16)
        self.assertEqual(model.applied_runtime.dtype, torch.bfloat16)

    def test_bfloat16_runtime_keeps_melspec_fp32_and_outputs_runtime_dtype(self) -> None:
        model = AudioModel(_small_config(seed=0, dtype="bfloat16"))
        processor = model.processor.processor

        floating_melspec_buffers = [
            buffer for buffer in processor.melspec.buffers() if torch.is_floating_point(buffer)
        ]
        self.assertTrue(floating_melspec_buffers)
        for buffer in floating_melspec_buffers:
            self.assertEqual(buffer.dtype, torch.float32)

        output = model(
            collate_fn(
                {
                    "x": [
                    torch.randn(96, dtype=torch.float32),
                    torch.randn(80, dtype=torch.float32),
                    ]
                }
            )
        )

        self.assertEqual(output.z.dtype, torch.bfloat16)
        self.assertEqual(output.z.device, model.applied_runtime.device)
        self.assertEqual(output.mask.dtype, torch.bool)
        self.assertEqual(output.mask.device, model.applied_runtime.device)

    def test_runtime_casts_melspec_batch_norm_but_keeps_torchaudio_fp32(self) -> None:
        config = replace(
            _small_config(seed=0, dtype="bfloat16"),
            processor=MelSpectrogramProcessorConfig(
                n_mels=4,
                n_fft=16,
                win_length=16,
                hop_length=8,
                batch_norm=True,
            ),
        )
        model = AudioModel(config)
        processor = model.processor.processor

        self.assertIsNotNone(processor.batch_norm)
        self.assertEqual(processor.batch_norm.batch_norm.weight.dtype, torch.bfloat16)
        self.assertEqual(processor.batch_norm.batch_norm.running_mean.dtype, torch.bfloat16)
        for buffer in processor.melspec.buffers():
            if torch.is_floating_point(buffer):
                self.assertEqual(buffer.dtype, torch.float32)

    def test_model_construction_does_not_disable_existing_determinism(self) -> None:
        torch.use_deterministic_algorithms(True)
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        if cudnn_backend is not None:
            cudnn_backend.deterministic = True
            cudnn_backend.benchmark = False

        model = AudioModel(_small_config(seed=0, deterministic=False))

        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(model.applied_runtime.deterministic)
        self.assertFalse(model.applied_runtime.global_runtime_applied)
        self.assertFalse(model.applied_runtime.deterministic_settings_applied)
        if cudnn_backend is not None:
            self.assertTrue(cudnn_backend.deterministic)
            self.assertFalse(cudnn_backend.benchmark)

    def test_model_construction_records_deterministic_runtime_without_applying_it(self) -> None:
        torch.use_deterministic_algorithms(False)
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        if cudnn_backend is not None:
            cudnn_backend.deterministic = False

        model = AudioModel(_small_config(seed=0, deterministic=True))

        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(model.applied_runtime.deterministic)
        self.assertFalse(model.applied_runtime.global_runtime_applied)
        self.assertFalse(model.applied_runtime.deterministic_settings_applied)
        self.assertFalse(model.applied_runtime.deterministic_algorithms_enabled)
        if cudnn_backend is not None:
            self.assertFalse(cudnn_backend.deterministic)

    def test_apply_global_runtime_applies_pytorch_settings(self) -> None:
        applied_runtime = apply_global_runtime(_small_config(seed=0, deterministic=True).runtime)

        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(applied_runtime.deterministic)
        self.assertTrue(applied_runtime.global_runtime_applied)
        self.assertTrue(applied_runtime.global_rng_seeded)
        self.assertTrue(applied_runtime.deterministic_settings_applied)
        self.assertTrue(applied_runtime.deterministic_algorithms_enabled)
        if applied_runtime.cudnn_deterministic is not None:
            self.assertTrue(applied_runtime.cudnn_deterministic)
            self.assertFalse(applied_runtime.cudnn_benchmark)

    def test_configured_initializer_is_applied_to_mha_and_mlp_weights(self) -> None:
        INITIALIZATION_REGISTRY[_CONSTANT_INIT] = _constant_init

        model = AudioModel(
            _small_config(
                seed=0,
                mha_initialization=_CONSTANT_INIT,
                mlp_initialization=_CONSTANT_INIT,
            )
        )
        block = model.encoder.layers[0]

        for weight in (
            block.mha.mha.in_proj_weight,
            block.mha.mha.out_proj.weight,
            block.mlp.fc1.weight,
            block.mlp.fc2.weight,
        ):
            self.assertTrue(torch.equal(weight, torch.full_like(weight, 0.125)))

        for bias in (
            block.mha.mha.in_proj_bias,
            block.mha.mha.out_proj.bias,
            block.mlp.fc1.bias,
            block.mlp.fc2.bias,
        ):
            self.assertIsNotNone(bias)
            self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def test_invalid_initializer_fails_early(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Unsupported \S+\.initialization"):
            AudioModel(_small_config(mha_initialization="not_an_initializer"))

        with self.assertRaisesRegex(ValueError, r"Unsupported \S+\.initialization"):
            AudioModel(_small_config(mlp_initialization="not_an_initializer"))

    def test_invalid_runtime_config_fails_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "RuntimeConfig.seed"):
            AudioModel(_small_config(seed=-1))

        with self.assertRaisesRegex(ValueError, "Unsupported RuntimeConfig.dtype"):
            AudioModel(_small_config(dtype="float64"))

        config = replace(
            _small_config(),
            runtime=RuntimeConfig(
                seed=0,
                device="definitely-not-a-device",
                dtype="float32",
                deterministic=True,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "Invalid RuntimeConfig.device"):
            AudioModel(config)

    def test_block_registry_exposes_transformer_and_conformer(self) -> None:
        registry = get_encoder_block_registry()
        self.assertIn("transformer", registry)
        self.assertIn("conformer", registry)

    def test_processor_registry_exposes_melspec_and_conv(self) -> None:
        registry = get_processor_registry()
        self.assertIn("melspec", registry)
        self.assertIn("conv", registry)

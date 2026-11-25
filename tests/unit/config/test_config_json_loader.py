from __future__ import annotations

import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from audiossl.config import (
    AdamWOptimizerConfig,
    AudioSSLTrainingConfig,
    BYOLObjectiveConfig,
    ConstantSchedulerConfig,
    ContrastiveObjectiveConfig,
    ConvolutionalProcessorConfig,
    DINOObjectiveConfig,
    EmbeddingSubsamplingConfig,
    JEPAObjectiveConfig,
    LinearEmbeddingConfig,
    MAEObjectiveConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    TransformerEncoderConfig,
    load_config,
)


def _training_payload() -> dict:
    return {
        "model": {
            "d_model": 8,
            "processor": {
                "type": "conv",
                "in_channels": 1,
                "out_channels": 4,
                "kernel_sizes": [3, 3],
                "strides": [2, 2],
                "groups": 1,
                "padding": 0,
                "dilation": 1,
                "bias": True,
                "activation": "relu",
                "initialization": "kaiming_uniform",
            },
            "tokenizer": {
                "type": "patchify",
                "patch_shape": [4, 1],
                "padding_strategy": "truncate",
            },
            "embedding": {
                "type": "linear",
                "positional_encoding": {
                    "type": "none",
                },
                "dropout_p": 0.0,
                "initialization": "xavier_uniform",
                "subsampling": None,
            },
            "encoder": {
                "type": "transformer",
                "mha": {
                    "num_heads": 2,
                    "dropout_p": 0.0,
                    "initialization": "xavier_uniform",
                },
                "mlp": {
                    "d_ff": 16,
                    "dropout_p": 0.0,
                    "activation": "silu",
                    "initialization": "xavier_uniform",
                },
                "num_layers": 1,
                "pre_norm": True,
                "residual_dropout_p": 0.0,
                "pooling": "masked_mean",
                "positional_encoding": {
                    "type": "rope",
                    "max_len": 128,
                    "base": 10000.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "seed": 0,
                "dtype": "float32",
                "deterministic": False,
            },
        },
        "objective": {
            "type": "contrastive",
            "view_strategy": "augmentation",
            "projection": {
                "depth": 2,
                "hidden_dim": 16,
                "output_dim": 8,
                "activation": "gelu",
                "final_l2_normalize": True,
                "dropout_p": 0.0,
                "initialization": "xavier_uniform",
            },
            "temperature": 0.2,
            "symmetric": True,
            "compute_stats": True,
        },
        "trainer": {
            "augmentor": {
                "augmentations": [
                    {
                        "type": "loudness",
                        "p": 1.0,
                        "min_gain_db": -6.0,
                        "max_gain_db": 6.0,
                    }
                ]
            },
            "optimizer": {
                "type": "adamw",
                "lr": 0.001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "scheduler": {
                "type": "constant",
            },
            "batch": {
                "per_device_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "drop_last": True,
            },
            "loop": {
                "max_epochs": 1,
                "gradient_clip_norm": None,
            },
        },
    }


class ConfigJsonLoaderTests(unittest.TestCase):
    def test_loads_full_training_config_from_dict(self) -> None:
        config = load_config(_training_payload())

        self.assertIsInstance(config, AudioSSLTrainingConfig)
        self.assertIsInstance(config.model.processor, ConvolutionalProcessorConfig)
        self.assertIsInstance(config.model.tokenizer, PatchifyTokenizerConfig)
        self.assertIsInstance(config.model.embedding, LinearEmbeddingConfig)
        self.assertIsInstance(
            config.model.embedding.positional_encoding,
            NoPositionalEncodingConfig,
        )
        self.assertIsInstance(config.model.encoder, TransformerEncoderConfig)
        self.assertIsInstance(config.objective, ContrastiveObjectiveConfig)
        self.assertIsInstance(config.trainer.optimizer, AdamWOptimizerConfig)
        self.assertIsInstance(config.trainer.scheduler, ConstantSchedulerConfig)
        self.assertIsNone(config.model.embedding.subsampling)

    def test_converts_json_lists_to_tuples(self) -> None:
        config = load_config(_training_payload())

        self.assertEqual(config.model.processor.kernel_sizes, (3, 3))
        self.assertEqual(config.model.processor.strides, (2, 2))
        self.assertEqual(config.model.tokenizer.patch_shape, (4, 1))
        self.assertIsNone(config.model.tokenizer.stride)
        self.assertEqual(config.model.tokenizer.effective_stride, (4, 1))
        self.assertEqual(config.trainer.optimizer.betas, (0.9, 0.999))
        self.assertIsInstance(config.trainer.augmentor.augmentations, tuple)

    def test_loads_patchify_stride_from_json_list(self) -> None:
        payload = _training_payload()
        payload["model"]["tokenizer"]["stride"] = [2, 1]

        config = load_config(payload)

        self.assertEqual(config.model.tokenizer.stride, (2, 1))

    def test_loads_embedding_subsampling_config(self) -> None:
        payload = _training_payload()
        payload["model"]["embedding"] = {
            "type": "linear",
            "positional_encoding": {
                "type": "none",
            },
            "dropout_p": 0.0,
            "initialization": "xavier_uniform",
            "subsampling": {
                "type": "strided_select",
                "factor": 2,
            },
        }

        config = load_config(payload)

        self.assertIsInstance(config.model.embedding, LinearEmbeddingConfig)
        self.assertIsInstance(config.model.embedding.subsampling, EmbeddingSubsamplingConfig)
        self.assertEqual(config.model.embedding.subsampling.factor, 2)

    def test_rejects_malformed_embedding_subsampling_config(self) -> None:
        payload = _training_payload()
        payload["model"]["embedding"] = {
            "type": "linear",
            "positional_encoding": {
                "type": "none",
            },
            "subsampling": {
                "type": "strided_select",
                "factor": 0,
            },
        }

        with self.assertRaisesRegex(ValueError, "LinearEmbeddingConfig.subsampling.factor"):
            load_config(payload)

    def test_rejects_checkpoint_settings_in_training_config(self) -> None:
        payload = _training_payload()
        payload["trainer"]["checkpoint"] = {
            "save_dir": "checkpoints/run-1",
            "save_every_n_epochs": 1,
            "keep_last_n": 1,
        }

        with self.assertRaisesRegex(TypeError, r"Unknown field .*trainer\.checkpoint"):
            load_config(payload)

    def test_loads_from_json_string_bytes_file_path_and_file_like(self) -> None:
        payload = _training_payload()
        text = json.dumps(payload)

        self.assertIsInstance(load_config(text), AudioSSLTrainingConfig)
        self.assertIsInstance(load_config(text.encode("utf-8")), AudioSSLTrainingConfig)
        self.assertIsInstance(load_config(io.StringIO(text)), AudioSSLTrainingConfig)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(text, encoding="utf-8")
            self.assertIsInstance(load_config(path), AudioSSLTrainingConfig)
            self.assertIsInstance(load_config(str(path)), AudioSSLTrainingConfig)

    def test_passes_existing_training_config_through(self) -> None:
        config = load_config(_training_payload())

        self.assertIs(load_config(config), config)

    def test_loads_jepa_objective_variant(self) -> None:
        payload = _training_payload()
        payload["objective"] = {
            "type": "jepa",
            "loss_style": "mse",
            "mask_token_init_scale": 0.1,
            "teacher_mode": "ema",
            "output_norm_type": "l2",
            "teacher_output_norm": True,
            "student_output_norm": True,
            "teacher_ema_decay": 0.999,
            "predictor_config": {
                "d_ff": 16,
                "dropout_p": 0.0,
                "activation": "silu",
                "initialization": "xavier_uniform",
            },
            "masking": {
                "mask_ratio": 0.5,
                "min_span_len": 1,
                "max_span_len": 10,
                "allow_overlap": True,
            },
        }

        config = load_config(payload)

        self.assertIsInstance(config.objective, JEPAObjectiveConfig)

    def test_loads_byol_objective_variant(self) -> None:
        payload = _training_payload()
        payload["objective"] = {
            "type": "byol",
            "projector": {
                "depth": 2,
                "hidden_dim": 16,
                "output_dim": 8,
                "activation": "gelu",
                "final_l2_normalize": False,
            },
            "predictor": {
                "depth": 2,
                "hidden_dim": 16,
                "output_dim": 8,
                "activation": "gelu",
                "final_l2_normalize": False,
            },
        }

        config = load_config(payload)

        self.assertIsInstance(config.objective, BYOLObjectiveConfig)

    def test_loads_dino_objective_variant(self) -> None:
        payload = _training_payload()
        payload["objective"] = {
            "type": "dino",
            "head": {
                "depth": 2,
                "hidden_dim": 16,
                "bottleneck_dim": 8,
                "num_prototypes": 11,
                "activation": "gelu",
            },
        }

        config = load_config(payload)

        self.assertIsInstance(config.objective, DINOObjectiveConfig)

    def test_loads_mae_objective_variant(self) -> None:
        payload = _training_payload()
        payload["model"]["processor"] = {
            "type": "melspec",
            "n_mels": 4,
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
        }
        payload["model"]["tokenizer"] = {
            "type": "patchify",
            "patch_shape": [4, 1],
            "padding_strategy": "truncate",
        }
        payload["model"]["embedding"]["positional_encoding"] = {
            "type": "learnable_2d",
            "max_t": 8,
            "max_f": 1,
        }
        payload["objective"] = {
            "type": "mae",
            "decoder_d_model": 8,
            "decoder": {
                "type": "transformer",
                "mha": {
                    "num_heads": 2,
                    "dropout_p": 0.0,
                    "initialization": "xavier_uniform",
                },
                "mlp": {
                    "d_ff": 16,
                    "dropout_p": 0.0,
                    "activation": "gelu",
                    "initialization": "xavier_uniform",
                },
                "num_layers": 1,
                "pre_norm": True,
                "residual_dropout_p": 0.0,
                "pooling": "masked_mean",
                "positional_encoding": None,
                "layer_norm_eps": 1e-6,
                "final_norm": True,
            },
            "decoder_positional_encoding": {
                "type": "learnable_2d",
                "max_t": 8,
                "max_f": 1,
            },
            "mask_token_init_scale": 0.02,
            "norm_pix_loss": "per_patch",
            "norm_pix_eps": 1e-6,
            "masking": {
                "mask_ratio": 0.75,
                "group_factor": None,
            },
        }

        config = load_config(payload)

        self.assertIsInstance(config.objective, MAEObjectiveConfig)

    def test_unknown_field_raises_with_path(self) -> None:
        payload = _training_payload()
        payload["model"]["encoder"]["extra_field"] = True

        with self.assertRaisesRegex(TypeError, "AudioSSLTrainingConfig.model.encoder.extra_field"):
            load_config(payload)

    def test_unknown_type_raises(self) -> None:
        payload = _training_payload()
        payload["objective"]["type"] = "made_up"

        with self.assertRaisesRegex(ValueError, "objective.type='made_up'"):
            load_config(payload)

    def test_malformed_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_config("{")

    def test_validate_true_rejects_invalid_config(self) -> None:
        payload = _training_payload()
        payload["objective"]["temperature"] = 0.0

        with self.assertRaisesRegex(ValueError, "temperature"):
            load_config(payload)

    def test_validate_false_allows_semantically_invalid_config(self) -> None:
        payload = deepcopy(_training_payload())
        payload["model"]["d_model"] = 0

        config = load_config(payload, validate=False)

        self.assertEqual(config.model.d_model, 0)


if __name__ == "__main__":
    unittest.main()

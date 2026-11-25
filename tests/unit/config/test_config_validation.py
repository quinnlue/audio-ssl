from __future__ import annotations

import unittest
from typing import Any

from audiossl.config import (
    AudioAugmentorConfig,
    AudioModelConfig,
    AudioSSLTrainingConfig,
    BranchformerEncoderConfig,
    BranchMergeConfig,
    BYOLObjectiveConfig,
    CGMLPConfig,
    ConformerEncoderConfig,
    ContrastiveObjectiveConfig,
    ConvolutionalPositionalEncodingConfig,
    ConvEmbeddingConfig,
    ConvModuleConfig,
    ConvolutionalProcessorConfig,
    DINOHeadConfig,
    DINOObjectiveConfig,
    EBranchformerEncoderConfig,
    EBranchMergeConfig,
    EmbeddingSubsamplingConfig,
    JEPAObjectiveConfig,
    LearnablePositionalEncodingConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    LinearEmbeddingConfig,
    MelSpectrogramProcessorConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoOpTokenizerConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    ProjectionHeadConfig,
    RopePositionalEncodingConfig,
    RuntimeConfig,
    SinusoidalPositionalEncodingConfig,
    TemporalMaskingConfig,
    TrainerConfig,
    TransformerEncoderConfig,
    Wav2Vec2FeatureExtractorProcessorConfig,
    WandBConfig,
    validate_config,
)
from audiossl.config.run import CheckpointConfig, EvaluationConfig, LoaderConfig, RunConfig
from audiossl.config.config import (
    AdamWOptimizerConfig,
    BatchConfig,
    ConstantSchedulerConfig,
    CosineAnnealingSchedulerConfig,
    LARSOptimizerConfig,
    SGDOptimizerConfig,
    StepLRSchedulerConfig,
    TrainingLoopConfig,
    WarmupCosineSchedulerConfig,
)


def _projection_head_config(**overrides: Any) -> ProjectionHeadConfig:
    values: dict[str, Any] = {
        "depth": 2,
        "hidden_dim": 16,
        "output_dim": 8,
        "activation": "gelu",
        "final_l2_normalize": True,
    }
    values.update(overrides)
    return ProjectionHeadConfig(**values)


def _contrastive_objective_config(**overrides: Any) -> ContrastiveObjectiveConfig:
    values: dict[str, Any] = {
        "view_strategy": "augmentation",
        "projection": _projection_head_config(),
        "temperature": 0.2,
    }
    values.update(overrides)
    return ContrastiveObjectiveConfig(**values)


def _jepa_objective_config(**overrides: Any) -> JEPAObjectiveConfig:
    values: dict[str, Any] = {
        "loss_style": "mse",
        "mask_token_init_scale": 0.1,
        "teacher_mode": "ema",
        "output_norm_type": "l2",
        "teacher_output_norm": True,
        "student_output_norm": True,
        "teacher_ema_decay": 0.999,
        "predictor_config": MLPConfig(d_ff=16),
        "masking": TemporalMaskingConfig(mask_ratio=0.5),
    }
    values.update(overrides)
    return JEPAObjectiveConfig(**values)


def _byol_objective_config(**overrides: Any) -> BYOLObjectiveConfig:
    head = _projection_head_config(final_l2_normalize=False)
    values: dict[str, Any] = {
        "projector": head,
        "predictor": head,
        "teacher_ema_decay": 0.99,
    }
    values.update(overrides)
    return BYOLObjectiveConfig(**values)


def _dino_head_config(**overrides: Any) -> DINOHeadConfig:
    values: dict[str, Any] = {
        "depth": 2,
        "hidden_dim": 16,
        "bottleneck_dim": 8,
        "num_prototypes": 32,
        "activation": "gelu",
    }
    values.update(overrides)
    return DINOHeadConfig(**values)


def _dino_objective_config(**overrides: Any) -> DINOObjectiveConfig:
    values: dict[str, Any] = {
        "head": _dino_head_config(),
    }
    values.update(overrides)
    return DINOObjectiveConfig(**values)


def _processor_config(**overrides: Any) -> MelSpectrogramProcessorConfig:
    values: dict[str, Any] = {
        "n_mels": 4,
        "n_fft": 16,
        "win_length": 16,
        "hop_length": 8,
    }
    values.update(overrides)
    return MelSpectrogramProcessorConfig(**values)


def _patchify_tokenizer_config(**overrides: Any) -> PatchifyTokenizerConfig:
    values: dict[str, Any] = {
        "patch_shape": (4, 1),
        "padding_strategy": "truncate",
    }
    values.update(overrides)
    return PatchifyTokenizerConfig(**values)


def _noop_tokenizer_config(**overrides: Any) -> NoOpTokenizerConfig:
    values: dict[str, Any] = {
        "_output_dim": 4,
    }
    values.update(overrides)
    return NoOpTokenizerConfig(**values)


def _conv_processor_config(**overrides: Any) -> ConvolutionalProcessorConfig:
    values: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": 4,
        "kernel_sizes": [3, 3],
        "strides": [2, 1],
    }
    values.update(overrides)
    return ConvolutionalProcessorConfig(**values)


def _model_config() -> AudioModelConfig:
    return AudioModelConfig(
        d_model=8,
        processor=_processor_config(),
        tokenizer=_patchify_tokenizer_config(),
        embedding=LinearEmbeddingConfig(
            positional_encoding=NoPositionalEncodingConfig()
        ),
        encoder=TransformerEncoderConfig(
            mha=MultiHeadAttentionConfig(num_heads=2),
            mlp=MLPConfig(d_ff=16),
            positional_encoding=RopePositionalEncodingConfig(max_len=128),
        ),
        runtime=RuntimeConfig(device="cpu"),
    )


def _training_config(**overrides: Any) -> AudioSSLTrainingConfig:
    values: dict[str, Any] = {
        "model": _model_config(),
        "objective": _contrastive_objective_config(),
        "trainer": _trainer_config(
            optimizer=AdamWOptimizerConfig(lr=1e-3),
            batch=BatchConfig(per_device_batch_size=4),
        ),
    }
    values.update(overrides)
    return AudioSSLTrainingConfig(**values)


def _trainer_config(**overrides: Any) -> TrainerConfig:
    from audiossl.config import LoudnessAugmentationConfig

    values: dict[str, Any] = {
        "augmentor": AudioAugmentorConfig(
            augmentations=[
                LoudnessAugmentationConfig(p=1.0, min_gain_db=-6.0, max_gain_db=6.0)
            ]
        ),
        "optimizer": AdamWOptimizerConfig(lr=1e-3),
        "scheduler": ConstantSchedulerConfig(),
        "batch": BatchConfig(per_device_batch_size=4),
        "loop": TrainingLoopConfig(max_epochs=1),
    }
    values.update(overrides)
    return TrainerConfig(**values)


class ConfigValidationTests(unittest.TestCase):
    def test_valid_model_and_training_configs_pass(self) -> None:
        model_config = _model_config()
        training_config = _training_config()

        self.assertTrue(validate_config(model_config))
        self.assertTrue(validate_config(training_config))

    def test_missing_or_wrong_nested_config_fails(self) -> None:
        config = _model_config()
        object.__setattr__(config, "processor", object())

        with self.assertRaisesRegex(TypeError, "AudioModelConfig.processor"):
            validate_config(config)

    def test_invalid_numeric_fields_fail(self) -> None:
        for config, pattern in (
            (_model_config(), "d_model"),
            (_model_config(), "num_heads"),
            (_model_config(), "d_ff"),
            (_model_config(), "num_layers"),
            (_model_config(), "dropout_p"),
        ):
            if pattern == "d_model":
                object.__setattr__(config, "d_model", 0)
            elif pattern == "num_heads":
                object.__setattr__(config.encoder.mha, "num_heads", 0)
            elif pattern == "d_ff":
                object.__setattr__(config.encoder.mlp, "d_ff", 0)
            elif pattern == "num_layers":
                object.__setattr__(config.encoder, "num_layers", 0)
            else:
                object.__setattr__(config.encoder.mha, "dropout_p", 1.5)

            with self.assertRaisesRegex((TypeError, ValueError), pattern):
                validate_config(config)

    def test_invalid_temperature_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            validate_config(_contrastive_objective_config(temperature=0.0))

    def test_invalid_view_strategy_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "view_strategy"):
            validate_config(_contrastive_objective_config(view_strategy="random"))

    def test_invalid_projection_head_depth_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth"):
            validate_config(
                _contrastive_objective_config(projection=_projection_head_config(depth=0))
            )

    def test_invalid_projection_head_hidden_dim_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            validate_config(
                _contrastive_objective_config(projection=_projection_head_config(hidden_dim=0))
            )

    def test_invalid_projection_head_output_dim_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_dim"):
            validate_config(
                _contrastive_objective_config(projection=_projection_head_config(output_dim=0))
            )

    def test_invalid_projection_head_activation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation"):
            validate_config(
                _contrastive_objective_config(
                    projection=_projection_head_config(activation="not_an_activation")
                )
            )

    def test_invalid_projection_head_initialization_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "initialization"):
            validate_config(
                _contrastive_objective_config(
                    projection=_projection_head_config(initialization="not_an_initializer")
                )
            )

    def test_valid_jepa_objective_config_passes(self) -> None:
        self.assertTrue(validate_config(_jepa_objective_config()))

    def test_valid_byol_objective_config_passes(self) -> None:
        self.assertTrue(validate_config(_byol_objective_config()))

    def test_byol_rejects_l2_normalized_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_l2_normalize"):
            validate_config(
                _byol_objective_config(
                    projector=_projection_head_config(final_l2_normalize=True)
                )
            )

    def test_valid_dino_objective_config_passes(self) -> None:
        self.assertTrue(validate_config(_dino_objective_config()))

    def test_invalid_dino_numeric_fields_fail(self) -> None:
        for overrides, pattern in (
            ({"teacher_temperature": 0.0}, "teacher_temperature"),
            ({"student_temperature": 0.0}, "student_temperature"),
            ({"teacher_ema_decay": 1.0}, "teacher_ema_decay"),
            ({"center_momentum": 0.0}, "center_momentum"),
            ({"head": _dino_head_config(num_prototypes=0)}, "num_prototypes"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_config(_dino_objective_config(**overrides))

    def test_invalid_jepa_loss_style_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss_style"):
            validate_config(_jepa_objective_config(loss_style="huber"))

    def test_invalid_jepa_teacher_mode_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "teacher_mode"):
            validate_config(_jepa_objective_config(teacher_mode="momentum"))

    def test_invalid_jepa_output_norm_type_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_norm_type"):
            validate_config(_jepa_objective_config(output_norm_type="batchnorm"))

    def test_invalid_jepa_ema_decay_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "teacher_ema_decay"):
            validate_config(_jepa_objective_config(teacher_ema_decay=1.0))

    def test_invalid_jepa_predictor_config_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "d_ff"):
            validate_config(_jepa_objective_config(predictor_config=MLPConfig(d_ff=0)))

    def test_invalid_temporal_mask_ratio_fails_at_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "mask_ratio"):
            validate_config(TemporalMaskingConfig(mask_ratio=1.0))

    def test_invalid_discriminators_fail(self) -> None:
        config = _model_config()
        object.__setattr__(config.processor, "type", "spectrogram")

        with self.assertRaisesRegex(ValueError, "MelSpectrogramProcessorConfig.type"):
            validate_config(config)

        config = _model_config()
        object.__setattr__(config, "processor", _conv_processor_config())
        object.__setattr__(config.processor, "type", "waveform")

        with self.assertRaisesRegex(ValueError, "ConvolutionalProcessorConfig.type"):
            validate_config(config)

        config = _model_config()
        object.__setattr__(config.encoder, "type", "stacked")

        with self.assertRaisesRegex(ValueError, "TransformerEncoderConfig.type"):
            validate_config(config)

    def test_d_model_must_be_divisible_by_num_heads(self) -> None:
        config = _model_config()
        object.__setattr__(config.encoder.mha, "num_heads", 3)

        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_config(config)

    def test_conformer_config_validates_conv_fields(self) -> None:
        config = _model_config()
        object.__setattr__(
            config,
            "encoder",
            ConformerEncoderConfig(
                mha=MultiHeadAttentionConfig(num_heads=2),
                mlp=MLPConfig(d_ff=16),
                conv=ConvModuleConfig(kernel_size=30),
            ),
        )

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            validate_config(config)

    def test_conv_module_config_validates_standalone(self) -> None:
        self.assertTrue(validate_config(ConvModuleConfig(kernel_size=3)))

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            validate_config(ConvModuleConfig(kernel_size=4))

        with self.assertRaisesRegex(ValueError, "expansion_factor"):
            validate_config(ConvModuleConfig(kernel_size=3, expansion_factor=0))

        with self.assertRaisesRegex(ValueError, "expansion_factor"):
            validate_config(ConvModuleConfig(kernel_size=3, expansion_factor=3))

        with self.assertRaisesRegex(ValueError, "dropout_p"):
            validate_config(ConvModuleConfig(kernel_size=3, dropout_p=1.5))

    def test_rope_base_must_be_positive_finite(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.encoder,
            "positional_encoding",
            RopePositionalEncodingConfig(max_len=128, base=0.0),
        )
        with self.assertRaisesRegex(ValueError, "RopePositionalEncodingConfig.base"):
            validate_config(config)

        config = _model_config()
        object.__setattr__(
            config.encoder,
            "positional_encoding",
            RopePositionalEncodingConfig(max_len=128, base=float("inf")),
        )
        with self.assertRaisesRegex(ValueError, "RopePositionalEncodingConfig.base"):
            validate_config(config)

    def test_learnable_positional_encoding_validates(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.embedding,
            "positional_encoding",
            LearnablePositionalEncodingConfig(max_len=64),
        )
        self.assertTrue(validate_config(config))

    def test_sinusoidal_positional_encoding_validates(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.embedding,
            "positional_encoding",
            SinusoidalPositionalEncodingConfig(max_len=64),
        )
        self.assertTrue(validate_config(config))

    def test_no_positional_encoding_validates(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.embedding,
            "positional_encoding",
            NoPositionalEncodingConfig(),
        )
        self.assertTrue(validate_config(config))

    def test_learnable_time_frequency_positional_encoding_validates(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.embedding,
            "positional_encoding",
            LearnableTimeFrequencyPositionalEncodingConfig(max_t=8, max_f=2),
        )
        self.assertTrue(validate_config(config))

    def test_encoder_rejects_embedding_only_positional_encoding(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.encoder,
            "positional_encoding",
            LearnablePositionalEncodingConfig(max_len=64),
        )
        with self.assertRaisesRegex(TypeError, "per-block positional encoding"):
            validate_config(config)

    def test_sinusoidal_base_must_be_positive_finite(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.embedding,
            "positional_encoding",
            SinusoidalPositionalEncodingConfig(max_len=128, base=0.0),
        )
        with self.assertRaisesRegex(ValueError, "SinusoidalPositionalEncodingConfig.base"):
            validate_config(config)

        config = _model_config()
        object.__setattr__(
            config.embedding,
            "positional_encoding",
            SinusoidalPositionalEncodingConfig(max_len=128, base=float("inf")),
        )
        with self.assertRaisesRegex(ValueError, "SinusoidalPositionalEncodingConfig.base"):
            validate_config(config)

    def test_sinusoidal_invalid_discriminator_fails(self) -> None:
        config = _model_config()
        object.__setattr__(
            config.embedding,
            "positional_encoding",
            SinusoidalPositionalEncodingConfig(max_len=64),
        )
        object.__setattr__(config.embedding.positional_encoding, "type", "sin")

        with self.assertRaisesRegex(ValueError, "SinusoidalPositionalEncodingConfig.type"):
            validate_config(config)

    def test_melspectrogram_processor_accepts_explicit_preprocessing_defaults(self) -> None:
        config = _processor_config(
            f_max=8000.0,
            pad=2,
            power=1.0,
            normalized="window",
            center=False,
            pad_mode="constant",
            norm="slaney",
            mel_scale="slaney",
            log_mel=True,
            log_mel_top_db=120.0,
            batch_norm=True,
            batch_norm_momentum=0.01,
            batch_norm_eps=1e-5,
        )

        self.assertTrue(validate_config(config))

    def test_melspectrogram_processor_validates_preprocessing_constraints(self) -> None:
        invalid_cases = (
            ({"f_min": 8000.0}, "f_min"),
            ({"f_max": 8001.0}, "f_max"),
            ({"win_length": 17}, "win_length"),
            ({"pad": -1}, "pad"),
            ({"window_fn": "hamming"}, "window_fn"),
            ({"power": 0.0}, "power"),
            ({"normalized": "loudness"}, "normalized"),
            ({"center": "yes"}, "center"),
            ({"pad_mode": "edge"}, "pad_mode"),
            ({"norm": "htk"}, "norm"),
            ({"mel_scale": "linear"}, "mel_scale"),
            ({"log_mel": "yes"}, "log_mel"),
            ({"log_mel_top_db": 0.0}, "log_mel_top_db"),
            ({"log_mel_top_db": float("inf")}, "log_mel_top_db"),
            ({"batch_norm": "yes"}, "batch_norm"),
            ({"batch_norm_momentum": -0.1}, "batch_norm_momentum"),
            ({"batch_norm_momentum": 1.1}, "batch_norm_momentum"),
            ({"batch_norm_momentum": float("inf")}, "batch_norm_momentum"),
            ({"batch_norm_eps": 0.0}, "batch_norm_eps"),
            ({"batch_norm_eps": float("inf")}, "batch_norm_eps"),
        )

        for overrides, pattern in invalid_cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex((TypeError, ValueError), pattern):
                    validate_config(_processor_config(**overrides))

    def test_convolutional_processor_accepts_valid_config(self) -> None:
        processor = _conv_processor_config(
            padding=1,
            dilation=2,
            bias=False,
            activation="gelu",
        )
        model = _model_config()
        object.__setattr__(model, "processor", processor)

        self.assertTrue(validate_config(processor))
        self.assertTrue(validate_config(model))

    def test_convolutional_processor_has_no_d_model(self) -> None:
        with self.assertRaisesRegex(TypeError, "d_model"):
            _conv_processor_config(d_model=8)

    def test_convolutional_processor_validates_stack_shape(self) -> None:
        invalid_cases = (
            ({"kernel_sizes": []}, "kernel_sizes"),
            ({"kernel_sizes": [3], "strides": [2, 1]}, "same length"),
            ({"kernel_sizes": [0, 3]}, "kernel_sizes"),
            ({"strides": [0, 1]}, "strides"),
        )

        for overrides, pattern in invalid_cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex((TypeError, ValueError), pattern):
                    validate_config(_conv_processor_config(**overrides))

    def test_convolutional_processor_validates_scalar_fields(self) -> None:
        invalid_cases = (
            ({"in_channels": 2}, "in_channels"),
            ({"out_channels": 0}, "out_channels"),
            ({"groups": 0}, "groups"),
            ({"padding": -1}, "padding"),
            ({"dilation": 0}, "dilation"),
            ({"bias": "yes"}, "bias"),
            ({"activation": "tanh"}, "activation"),
        )

        for overrides, pattern in invalid_cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex((TypeError, ValueError), pattern):
                    validate_config(_conv_processor_config(**overrides))

    def test_convolutional_processor_groups_must_divide_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "groups"):
            validate_config(_conv_processor_config(groups=2))

    def test_convolutional_processor_validates_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "initialization"):
            validate_config(_conv_processor_config(initialization="not_an_initializer"))

    def test_wav2vec2_feature_extractor_processor_validates(self) -> None:
        config = Wav2Vec2FeatureExtractorProcessorConfig(
            conv_dim=(16, 16),
            conv_kernel=(3, 3),
            conv_stride=(2, 2),
            group_norm_num_groups=16,
        )

        self.assertTrue(validate_config(config))

        with self.assertRaisesRegex(ValueError, "same length"):
            validate_config(
                Wav2Vec2FeatureExtractorProcessorConfig(
                    conv_dim=(16, 16),
                    conv_kernel=(3,),
                    conv_stride=(2, 2),
                    group_norm_num_groups=16,
                )
            )

        with self.assertRaisesRegex(ValueError, "group_norm_num_groups"):
            validate_config(
                Wav2Vec2FeatureExtractorProcessorConfig(
                    conv_dim=(16, 16),
                    conv_kernel=(3, 3),
                    conv_stride=(2, 2),
                    group_norm_num_groups=10,
                )
            )

    def test_model_rejects_string_tokenizer(self) -> None:
        config = _model_config()
        object.__setattr__(config, "tokenizer", "none")
        with self.assertRaisesRegex(TypeError, "AudioModelConfig.tokenizer"):
            validate_config(config)

    def test_valid_patchify_tokenizer_config_passes(self) -> None:
        self.assertTrue(validate_config(_patchify_tokenizer_config()))

    def test_patchify_tokenizer_validates_patch_shape(self) -> None:
        invalid_cases = (
            ("1,1", "patch_shape"),
            ((1,), "patch_shape"),
            ((1, 1, 1), "patch_shape"),
            ((0, 1), "patch_shape"),
            ((1, -1), "patch_shape"),
            ((1.5, 1), "patch_shape"),
        )

        for patch_shape, pattern in invalid_cases:
            with self.subTest(patch_shape=patch_shape):
                with self.assertRaisesRegex((TypeError, ValueError), pattern):
                    validate_config(_patchify_tokenizer_config(patch_shape=patch_shape))

    def test_patchify_tokenizer_validates_stride(self) -> None:
        invalid_cases = (
            ("1,1", "stride"),
            ((1,), "stride"),
            ((1, 1, 1), "stride"),
            ((0, 1), "stride"),
            ((1, -1), "stride"),
            ((1.5, 1), "stride"),
        )

        for stride, pattern in invalid_cases:
            with self.subTest(stride=stride):
                with self.assertRaisesRegex((TypeError, ValueError), pattern):
                    validate_config(_patchify_tokenizer_config(stride=stride))

    def test_patchify_tokenizer_validates_padding_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "padding_strategy"):
            validate_config(_patchify_tokenizer_config(padding_strategy="extend"))

    def test_valid_noop_tokenizer_config_passes(self) -> None:
        self.assertTrue(validate_config(_noop_tokenizer_config()))

    def test_noop_tokenizer_validates_output_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "_output_dim"):
            validate_config(_noop_tokenizer_config(_output_dim=0))

    def test_model_accepts_matching_noop_tokenizer(self) -> None:
        config = _model_config()
        object.__setattr__(config, "tokenizer", _noop_tokenizer_config(_output_dim=4))

        self.assertTrue(validate_config(config))

    def test_model_rejects_noop_tokenizer_output_dim_mismatch(self) -> None:
        config = _model_config()
        object.__setattr__(config, "tokenizer", _noop_tokenizer_config(_output_dim=5))

        with self.assertRaisesRegex(ValueError, "NoOpTokenizerConfig.output_dim"):
            validate_config(config)

    def test_model_rejects_unknown_tokenizer_type(self) -> None:
        config = _model_config()
        tokenizer = _patchify_tokenizer_config()
        object.__setattr__(tokenizer, "type", "patches")
        object.__setattr__(config, "tokenizer", tokenizer)
        with self.assertRaisesRegex(ValueError, "Unsupported PatchifyTokenizerConfig.type"):
            validate_config(config)

    def test_embedding_subsampling_config_validates(self) -> None:
        self.assertTrue(validate_config(EmbeddingSubsamplingConfig(factor=2)))

        with self.assertRaisesRegex(ValueError, "factor"):
            validate_config(EmbeddingSubsamplingConfig(factor=0))

        with self.assertRaisesRegex(TypeError, "factor"):
            validate_config(EmbeddingSubsamplingConfig(factor=True))  # type: ignore[arg-type]

        config = EmbeddingSubsamplingConfig(factor=2)
        object.__setattr__(config, "type", "window")
        with self.assertRaisesRegex(ValueError, "type"):
            validate_config(config)

    def test_model_accepts_embedding_subsampling_configs(self) -> None:
        for embedding in (
            LinearEmbeddingConfig(
                positional_encoding=NoPositionalEncodingConfig(),
                subsampling=EmbeddingSubsamplingConfig(factor=2),
            ),
            ConvEmbeddingConfig(
                positional_encoding=NoPositionalEncodingConfig(),
                subsampling=EmbeddingSubsamplingConfig(factor=2),
            ),
        ):
            with self.subTest(embedding=embedding.type):
                config = _model_config()
                object.__setattr__(config, "embedding", embedding)
                self.assertTrue(validate_config(config))

    def test_convolutional_positional_encoding_validates(self) -> None:
        self.assertTrue(
            validate_config(ConvolutionalPositionalEncodingConfig(kernel_size=3))
        )

        with self.assertRaisesRegex(ValueError, "kernel_size"):
            validate_config(ConvolutionalPositionalEncodingConfig(kernel_size=0))

        config = _model_config()
        object.__setattr__(
            config,
            "embedding",
            LinearEmbeddingConfig(
                positional_encoding=ConvolutionalPositionalEncodingConfig(
                    kernel_size=3,
                    groups=3,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "groups"):
            validate_config(config)

    def test_cgmlp_config_validates_standalone(self) -> None:
        self.assertTrue(
            validate_config(CGMLPConfig(linear_units=16, conv_kernel_size=3))
        )

        with self.assertRaisesRegex(ValueError, "linear_units"):
            validate_config(CGMLPConfig(linear_units=15, conv_kernel_size=3))

        with self.assertRaisesRegex(ValueError, "linear_units"):
            validate_config(CGMLPConfig(linear_units=0, conv_kernel_size=3))

        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            validate_config(CGMLPConfig(linear_units=16, conv_kernel_size=4))

        with self.assertRaisesRegex(ValueError, "gate_activation"):
            validate_config(
                CGMLPConfig(
                    linear_units=16,
                    conv_kernel_size=3,
                    gate_activation="not_an_activation",
                )
            )

        with self.assertRaisesRegex(ValueError, "dropout_p"):
            validate_config(
                CGMLPConfig(linear_units=16, conv_kernel_size=3, dropout_p=1.5)
            )

    def test_branch_merge_config_validates_standalone(self) -> None:
        self.assertTrue(validate_config(BranchMergeConfig(method="concat")))
        self.assertTrue(validate_config(BranchMergeConfig(method="learned_ave")))
        self.assertTrue(
            validate_config(BranchMergeConfig(method="fixed_ave", fixed_weight=0.25))
        )

        with self.assertRaisesRegex(ValueError, "method"):
            validate_config(BranchMergeConfig(method="weighted"))  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "fixed_weight"):
            validate_config(
                BranchMergeConfig(method="fixed_ave", fixed_weight=1.5)
            )

        with self.assertRaisesRegex(ValueError, "fixed_weight"):
            validate_config(
                BranchMergeConfig(method="fixed_ave", fixed_weight=-0.1)
            )

    def test_branchformer_encoder_config_validates_against_model(self) -> None:
        def _replace_encoder(branch_overrides: dict) -> AudioModelConfig:
            config = _model_config()
            object.__setattr__(
                config,
                "encoder",
                BranchformerEncoderConfig(
                    mha=MultiHeadAttentionConfig(num_heads=2),
                    cgmlp=branch_overrides.get(
                        "cgmlp",
                        CGMLPConfig(linear_units=16, conv_kernel_size=3),
                    ),
                    merge=branch_overrides.get("merge", BranchMergeConfig(method="concat")),
                    num_layers=branch_overrides.get("num_layers", 1),
                    pre_norm=branch_overrides.get("pre_norm", True),
                    residual_dropout_p=branch_overrides.get("residual_dropout_p", 0.0),
                    attention_branch_drop_p=branch_overrides.get(
                        "attention_branch_drop_p", 0.0
                    ),
                    cgmlp_branch_drop_p=branch_overrides.get("cgmlp_branch_drop_p", 0.0),
                ),
            )
            return config

        self.assertTrue(validate_config(_replace_encoder({})))

        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            validate_config(
                _replace_encoder({"cgmlp": CGMLPConfig(linear_units=16, conv_kernel_size=4)})
            )

        with self.assertRaisesRegex(ValueError, "method"):
            validate_config(
                _replace_encoder({"merge": BranchMergeConfig(method="weighted")})  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ValueError, "attention_branch_drop_p"):
            validate_config(_replace_encoder({"attention_branch_drop_p": 1.5}))

        with self.assertRaisesRegex(ValueError, "cgmlp_branch_drop_p"):
            validate_config(_replace_encoder({"cgmlp_branch_drop_p": -0.1}))

        config = _replace_encoder({})
        object.__setattr__(config.encoder, "type", "not_a_type")
        with self.assertRaisesRegex(ValueError, "BranchformerEncoderConfig.type"):
            validate_config(config)

    def test_ebranch_merge_config_validates_standalone(self) -> None:
        self.assertTrue(validate_config(EBranchMergeConfig(conv_kernel_size=31)))

        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            validate_config(EBranchMergeConfig(conv_kernel_size=4))

        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            validate_config(EBranchMergeConfig(conv_kernel_size=0))

    def test_ebranchformer_encoder_config_validates_against_model(self) -> None:
        def _replace_encoder(branch_overrides: dict) -> AudioModelConfig:
            config = _model_config()
            object.__setattr__(
                config,
                "encoder",
                EBranchformerEncoderConfig(
                    mha=MultiHeadAttentionConfig(num_heads=2),
                    cgmlp=branch_overrides.get(
                        "cgmlp",
                        CGMLPConfig(linear_units=16, conv_kernel_size=3),
                    ),
                    merge=branch_overrides.get(
                        "merge",
                        EBranchMergeConfig(conv_kernel_size=3),
                    ),
                    mlp=branch_overrides.get("mlp", None),
                    use_ffn=branch_overrides.get("use_ffn", False),
                    macaron_ffn=branch_overrides.get("macaron_ffn", False),
                    num_layers=branch_overrides.get("num_layers", 1),
                    pre_norm=branch_overrides.get("pre_norm", True),
                    residual_dropout_p=branch_overrides.get("residual_dropout_p", 0.0),
                ),
            )
            return config

        self.assertTrue(validate_config(_replace_encoder({})))

        self.assertTrue(
            validate_config(
                _replace_encoder(
                    {
                        "mlp": MLPConfig(d_ff=16),
                        "use_ffn": True,
                        "macaron_ffn": True,
                    }
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "mlp must be set"):
            validate_config(_replace_encoder({"macaron_ffn": True}))

        with self.assertRaisesRegex(ValueError, "mlp must be set"):
            validate_config(_replace_encoder({"use_ffn": True}))

        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            validate_config(
                _replace_encoder({"merge": EBranchMergeConfig(conv_kernel_size=2)})
            )

        config = _replace_encoder({})
        object.__setattr__(config.encoder, "type", "not_a_type")
        with self.assertRaisesRegex(ValueError, "EBranchformerEncoderConfig.type"):
            validate_config(config)

    def test_training_config_accepts_all_optimizer_variants(self) -> None:
        optimizers = (
            AdamWOptimizerConfig(lr=1e-3),
            SGDOptimizerConfig(lr=1e-2, momentum=0.9),
            LARSOptimizerConfig(lr=0.5, eta=0.001),
        )
        for optimizer in optimizers:
            with self.subTest(optimizer=optimizer.type):
                self.assertTrue(
                    validate_config(_training_config(trainer=_trainer_config(optimizer=optimizer)))
                )

    def test_training_config_accepts_all_scheduler_variants(self) -> None:
        schedulers = (
            ConstantSchedulerConfig(),
            WarmupCosineSchedulerConfig(warmup_steps=100, min_lr_scale=0.1),
            CosineAnnealingSchedulerConfig(min_lr_scale=0.0),
            StepLRSchedulerConfig(step_size=10, gamma=0.5),
        )
        for scheduler in schedulers:
            with self.subTest(scheduler=scheduler.type):
                self.assertTrue(
                    validate_config(_training_config(trainer=_trainer_config(scheduler=scheduler)))
                )

    def test_run_config_accepts_optional_checkpoint(self) -> None:
        self.assertTrue(
            validate_config(
                RunConfig(checkpoint=CheckpointConfig(save_dir="checkpoints/run-1"))
            )
        )

    def test_invalid_optimizer_lr_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "AdamWOptimizerConfig.lr"):
            validate_config(AdamWOptimizerConfig(lr=0.0))

    def test_invalid_adamw_betas_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "betas"):
            validate_config(AdamWOptimizerConfig(lr=1e-3, betas=(0.0, 0.999)))

        with self.assertRaisesRegex(ValueError, "betas"):
            validate_config(AdamWOptimizerConfig(lr=1e-3, betas=(0.9, 1.0)))

    def test_invalid_sgd_momentum_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "momentum"):
            validate_config(SGDOptimizerConfig(lr=1e-3, momentum=1.5))

    def test_invalid_lars_eta_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "eta"):
            validate_config(LARSOptimizerConfig(lr=1e-3, eta=0.0))

    def test_invalid_scheduler_fields_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            validate_config(WarmupCosineSchedulerConfig(warmup_steps=-1))

        with self.assertRaisesRegex(ValueError, "min_lr_scale"):
            validate_config(CosineAnnealingSchedulerConfig(min_lr_scale=1.5))

        with self.assertRaisesRegex(ValueError, "step_size"):
            validate_config(StepLRSchedulerConfig(step_size=0))

        with self.assertRaisesRegex(ValueError, "gamma"):
            validate_config(StepLRSchedulerConfig(step_size=10, gamma=0.0))

    def test_invalid_batch_fields_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_device_batch_size"):
            validate_config(BatchConfig(per_device_batch_size=0))

        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            validate_config(BatchConfig(per_device_batch_size=4, gradient_accumulation_steps=0))

        with self.assertRaisesRegex(ValueError, "prefetch_factor"):
            validate_config(LoaderConfig(num_workers=1, prefetch_factor=0))

        with self.assertRaisesRegex(ValueError, "persistent_workers"):
            validate_config(LoaderConfig(persistent_workers=True, num_workers=0))

    def test_invalid_training_loop_fields_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_epochs"):
            validate_config(TrainingLoopConfig(max_epochs=0))

        with self.assertRaisesRegex(ValueError, "gradient_clip_norm"):
            validate_config(TrainingLoopConfig(max_epochs=1, gradient_clip_norm=0.0))

    def test_invalid_evaluation_fields_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "every_n_epochs"):
            validate_config(EvaluationConfig(every_n_epochs=0))

        with self.assertRaisesRegex(ValueError, "every_n_steps"):
            validate_config(EvaluationConfig(every_n_steps=0))

        with self.assertRaisesRegex(ValueError, "cannot both"):
            validate_config(EvaluationConfig(every_n_epochs=1, every_n_steps=1))

        with self.assertRaisesRegex(ValueError, "at least one validation trigger"):
            validate_config(EvaluationConfig(at_train_end=False))

    def test_invalid_checkpoint_fields_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "save_dir"):
            validate_config(CheckpointConfig(save_dir=""))

        with self.assertRaisesRegex(ValueError, "save_every_n_epochs"):
            validate_config(CheckpointConfig(save_dir="checkpoints", save_every_n_epochs=0))

        with self.assertRaisesRegex(ValueError, "keep_last_n"):
            validate_config(CheckpointConfig(save_dir="checkpoints", keep_last_n=0))

    def test_invalid_wandb_fields_fail(self) -> None:
        config = WandBConfig(project="audiossl-test")
        object.__setattr__(config, "resume", "no")

        with self.assertRaisesRegex(TypeError, "WandBConfig.resume"):
            validate_config(config)

    def test_invalid_training_config_nested_fields_fail(self) -> None:
        config = _training_config()
        object.__setattr__(config.trainer, "optimizer", AdamWOptimizerConfig(lr=0.0))

        with self.assertRaisesRegex(ValueError, "AdamWOptimizerConfig.lr"):
            validate_config(config)

        config = _training_config()
        object.__setattr__(
            config.trainer,
            "loop",
            TrainingLoopConfig(max_epochs=1, gradient_clip_norm=0.0),
        )

        with self.assertRaisesRegex(ValueError, "gradient_clip_norm"):
            validate_config(config)

        with self.assertRaisesRegex(ValueError, "keep_last_n"):
            validate_config(
                RunConfig(
                    checkpoint=CheckpointConfig(save_dir="checkpoints", keep_last_n=0)
                )
            )


if __name__ == "__main__":
    unittest.main()

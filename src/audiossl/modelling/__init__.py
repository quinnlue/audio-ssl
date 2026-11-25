"""Public modelling facade for audio model components and configs."""

from importlib import import_module

from audiossl.config import (
    AudioModelConfig,
    BranchformerEncoderConfig,
    BranchMergeConfig,
    CGMLPConfig,
    ConformerEncoderConfig,
    ConvolutionalPositionalEncodingConfig,
    ConvModuleConfig,
    ConvolutionalProcessorConfig,
    EBranchformerEncoderConfig,
    EBranchMergeConfig,
    EmbeddingPositionalEncodingConfig,
    EncoderConfig,
    LearnablePositionalEncodingConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    MelSpectrogramProcessorConfig,
    MLPConfig,
    MultiHeadAttentionConfig,
    NoPositionalEncodingConfig,
    PerBlockPositionalEncodingConfig,
    PositionalEncodingConfig,
    ProcessorConfig,
    RopePositionalEncodingConfig,
    SinusoidalPositionalEncodingConfig,
    TransformerEncoderConfig,
    Wav2Vec2FeatureExtractorProcessorConfig,
    validate_config,
)

_LAZY_EXPORTS = {
    "AudioModel": (".model", "AudioModel"),
    "ContrastiveObjective": (
        ".objectives.contrastive.contrastive",
        "ContrastiveObjective",
    ),
    "JEPAObjective": (".objectives.jepa.jepa", "JEPAObjective"),
    "MAEObjective": (".objectives.mae.mae", "MAEObjective"),
    "HuBERTObjective": (".objectives.hubert.hubert", "HuBERTObjective"),
    "build_objective": (".objectives.objective", "build_objective"),
    "ConvolutionalProcessor": (".processors.convolutional", "ConvolutionalProcessor"),
    "MelSpectrogramProcessor": (".processors.melspec", "MelSpectrogramProcessor"),
    "Wav2Vec2FeatureExtractorProcessor": (
        ".processors.wav2vec2",
        "Wav2Vec2FeatureExtractorProcessor",
    ),
    "Encoder": (".encoders.encoder", "Encoder"),
    "masked_mean_pool": (".utils.pooling", "masked_mean_pool"),
}


def __getattr__(name: str):
    """Lazily import heavyweight modelling exports on first access."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AudioModelConfig",
    "BranchMergeConfig",
    "BranchformerEncoderConfig",
    "CGMLPConfig",
    "ConformerEncoderConfig",
    "ConvModuleConfig",
    "ContrastiveObjective",
    "EBranchMergeConfig",
    "EBranchformerEncoderConfig",
    "JEPAObjective",
    "MAEObjective",
    "HuBERTObjective",
    "build_objective",
    "ConvolutionalProcessor",
    "ConvolutionalProcessorConfig",
    "ConvolutionalPositionalEncodingConfig",
    "EmbeddingPositionalEncodingConfig",
    "EncoderConfig",
    "ProcessorConfig",
    "MLPConfig",
    "MultiHeadAttentionConfig",
    "NoPositionalEncodingConfig",
    "PerBlockPositionalEncodingConfig",
    "PositionalEncodingConfig",
    "LearnablePositionalEncodingConfig",
    "LearnableTimeFrequencyPositionalEncodingConfig",
    "RopePositionalEncodingConfig",
    "SinusoidalPositionalEncodingConfig",
    "TransformerEncoderConfig",
    "Wav2Vec2FeatureExtractorProcessor",
    "Wav2Vec2FeatureExtractorProcessorConfig",
    "validate_config",
    "Encoder",
    "MelSpectrogramProcessor",
    "MelSpectrogramProcessorConfig",
    "AudioModel",
    "masked_mean_pool",
]

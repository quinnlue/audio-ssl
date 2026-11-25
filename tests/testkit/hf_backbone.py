"""Training adapter for the HF base speech backbones."""

import torch
from torch.nn.utils import parametrize

from audiossl.config import AudioBatch
from audiossl.integrations.wav2vec2.porter import (
    _encoder_key_pairs,
    _feature_extractor_key_pairs,
    _feature_projection_key_pairs,
)

from .parity import assert_training_parity


def compare_hf_backbone(ours, reference, config, *, samples, dtype, precision):
    # The porter folds weight normalization into a dense convolution. Train both
    # sides in those same coordinates so an SGD update has the same meaning.
    conv = reference.encoder.pos_conv_embed.conv
    if parametrize.is_parametrized(conv, "weight"):
        parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    else:
        torch.nn.utils.remove_weight_norm(conv)
    if hasattr(reference, "masked_spec_embed"):
        # The bare representation API does not include the objective's mask token.
        reference.masked_spec_embed.requires_grad_(False)

    ours.to(dtype).train()
    reference.to(dtype).train()
    # HF force-enables input grads in training mode for gradient checkpointing.
    # The extractor has no training-mode layers here, and our input is already a leaf.
    reference.feature_extractor.eval()
    pairs = [
        *_feature_extractor_key_pairs(len(config.conv_dim)),
        *_feature_projection_key_pairs(),
        *_encoder_key_pairs(config.num_hidden_layers),
        (
            "encoder.pos_conv_embed.conv.weight",
            "embedding.embedding.positional_encoding.conv.weight",
        ),
    ]

    def map_reference(values):
        mapped = {destination: values[source] for source, destination in pairs}
        remaining = set(values) - {source for source, _ in pairs}
        for layer in range(config.num_hidden_layers):
            src = f"encoder.layers.{layer}.attention"
            dst = f"encoder.blocks.{layer}.mha.mha"
            for kind in ("weight", "bias"):
                keys = [f"{src}.{part}_proj.{kind}" for part in "qkv"]
                mapped[f"{dst}.in_proj_{kind}"] = torch.cat([values[key] for key in keys])
                remaining.difference_update(keys)
        assert not remaining, f"unmapped reference parameters: {remaining}"
        return mapped

    mask = torch.arange(samples)[None, :] < torch.tensor([[samples], [samples - samples // 4]])
    width = int(reference._get_feat_extract_output_lengths(samples))
    valid_lengths = reference._get_feat_extract_output_lengths(mask.sum(dim=1))
    token_mask = torch.arange(width)[None, :] < valid_lengths[:, None]

    def actual_forward(_step, waveform):
        result = ours(AudioBatch(x=waveform, mask=mask))
        assert torch.equal(result.mask, token_mask)
        return result.z[token_mask]

    def reference_forward(_step, waveform):
        # The library zeros padding before the GroupNorm extractor.
        result = reference(waveform.masked_fill(~mask, 0), attention_mask=mask.long())
        return result.last_hidden_state[token_mask]

    assert_training_parity(
        ours,
        reference,
        actual_forward=actual_forward,
        reference_forward=reference_forward,
        inputs=lambda step: torch.randn(
            mask.shape, dtype=dtype, generator=torch.Generator().manual_seed(18 + step)
        ).masked_fill(~mask, 0),
        precision=precision,
        map_reference=map_reference,
    )

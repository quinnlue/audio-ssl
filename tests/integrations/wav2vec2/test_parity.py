"""Offline training parity for the wav2vec 2.0 backbone and quantizer."""

from copy import deepcopy

import torch

from audiossl.integrations.wav2vec2 import make_wav2vec2_base_config
from audiossl.integrations.wav2vec2.porter import port_wav2vec2_state_dict
from audiossl.modelling.model import AudioModel

from testkit.cases import objective_case
from testkit.hf_backbone import compare_hf_backbone
from testkit.parity import assert_training_parity


def _our_config_from_hf(hf_config):
    return make_wav2vec2_base_config(
        d_model=hf_config.hidden_size,
        depth=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        ffn_dim=hf_config.intermediate_size,
        conv_dim=tuple(hf_config.conv_dim),
        conv_kernel=tuple(hf_config.conv_kernel),
        conv_stride=tuple(hf_config.conv_stride),
        conv_pos_kernel_size=hf_config.num_conv_pos_embeddings,
        conv_pos_groups=hf_config.num_conv_pos_embedding_groups,
        layer_norm_eps=hf_config.layer_norm_eps,
        feature_projection_dropout=hf_config.feat_proj_dropout,
        hidden_dropout=hf_config.hidden_dropout,
        attention_dropout=hf_config.attention_dropout,
    )


def test_backbone_training_matches_hf(
    hf_wav2vec2, hf_w2v2_base_config, parity_dtype, parity_precision,
):
    torch.manual_seed(8)
    hf_w2v2_base_config = deepcopy(hf_w2v2_base_config)
    hf_w2v2_base_config.layer_norm_eps = parity_precision.eps
    ref_model = hf_wav2vec2.modeling.Wav2Vec2Model(hf_w2v2_base_config).to(parity_dtype)
    our_model = AudioModel(_our_config_from_hf(hf_w2v2_base_config)).to(parity_dtype)
    port_wav2vec2_state_dict(ref_model.state_dict(), our_model)
    compare_hf_backbone(
        our_model, ref_model, hf_w2v2_base_config, samples=3000,
        dtype=parity_dtype, precision=parity_precision,
    )


def test_gumbel_quantizer_training_matches_hf(hf_wav2vec2, parity_precision):
    objective = objective_case("wav2vec2").build()
    actual = objective.quantizer.train()
    config = actual.config
    hf_config = hf_wav2vec2.config.Wav2Vec2Config(
        conv_dim=(actual.input_dim,),
        conv_stride=(1,),
        conv_kernel=(3,),
        num_codevector_groups=config.num_groups,
        num_codevectors_per_group=config.num_vars,
        codevector_dim=config.vq_dim,
    )
    reference = hf_wav2vec2.modeling.Wav2Vec2GumbelVectorQuantizer(hf_config).train()
    reference.load_state_dict({
        name: param.detach() for name, param in actual.named_parameters()
    }, strict=True)
    schedule = config.temperature_scheduler

    def actual_forward(step, x):
        torch.manual_seed(70 + step)
        out = actual(x)
        assert out.temperature == reference_temperature(step)
        return torch.cat((out.quantized.flatten(), out.prob_perplexity.view(1)))

    def reference_temperature(step):
        return max(schedule.start * schedule.decay**step, schedule.end)

    def reference_forward(step, x):
        torch.manual_seed(70 + step)
        reference.temperature = reference_temperature(step)
        values, perplexity = reference(x)
        return torch.cat((values.flatten(), perplexity.view(1)))

    assert_training_parity(
        actual, reference,
        actual_forward=actual_forward,
        reference_forward=reference_forward,
        inputs=lambda step: torch.randn(
            2, 5, actual.input_dim, generator=torch.Generator().manual_seed(step)
        ),
        precision=parity_precision,
        after_step=objective.on_after_optimizer_step,
    )

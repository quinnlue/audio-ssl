"""Waveform-to-embedding parity using PyTorch convolution and patch extraction."""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import fbank

from audiossl.config import AudioBatch
from audiossl.modelling.model import AudioModel

from testkit import configs
from testkit.parity import assert_training_parity


@pytest.mark.parametrize("embedding", ["conv", "learned-2d", "learned-1d", "sine-1d"])
def test_convolutional_frontend_training_matches_reference(
    embedding, parity_dtype, parity_precision,
):
    subsampling = configs.subsampling(3)
    if embedding != "conv":
        tokenizer = configs.patchify(patch_shape=(3, 4), stride=(2, 4), padding_strategy="pad")
        positional_encoding = {
            "learned-2d": configs.pe_learnable_2d,
            "learned-1d": configs.pe_learnable,
            "sine-1d": configs.pe_sinusoidal,
        }[embedding]()
        embedding_config = configs.linear_embedding(
            positional_encoding,
            subsampling=subsampling,
            input_layer_norm=True,
            input_layer_norm_eps=parity_precision.eps,
            post_pos_layer_norm=True,
            post_pos_layer_norm_eps=parity_precision.eps,
        )
    else:
        tokenizer = configs.noop_tokenizer()
        embedding_config = configs.conv_embedding(subsampling=subsampling)
    model = AudioModel(configs.model(
        processor=configs.conv_processor(), tokenizer=tokenizer, embedding=embedding_config,
    ))
    actual = nn.ModuleDict({
        "processor": model.processor,
        "tokenizer": model.tokenizer,
        "embedding": model.embedding,
    }).to(parity_dtype).train()
    reference = deepcopy(actual)
    mask = torch.arange(199)[None, :] < torch.tensor([[199], [147]])
    observed_mask = None

    def reference_forward(_step, waveform):
        x = waveform.masked_fill(~mask, 0).unsqueeze(1)
        valid = mask
        conv_layers = reference["processor"].processor.conv_layers
        for conv in conv_layers[::2]:
            x = F.relu(conv(x))
            valid = valid.unfold(-1, conv.kernel_size[0], conv.stride[0]).all(-1)
        x = x.transpose(1, 2)
        ref = reference["embedding"].embedding
        if embedding != "conv":
            pt, pf = tokenizer.patch_shape
            st, sf = tokenizer.effective_stride
            pad = (pt - valid.shape[-1]) % st
            x = F.pad(x, (0, 0, 0, pad))
            valid = F.pad(valid, (0, pad)).unfold(-1, pt, st).all(-1)
            f_count = (x.shape[-1] - pf) // sf + 1
            patches = F.unfold(x.unsqueeze(1), kernel_size=(pt, pf), stride=(st, sf))
            patches = patches.transpose(1, 2)
            x = ref.projection(ref.pre_layer_norm(patches))
            pe = ref.positional_encoding
            t_count = valid.shape[1]
            valid = valid.repeat_interleave(f_count, dim=-1)
            if embedding == "learned-2d":
                table = pe.time_pe[0, :t_count, None, :] + pe.freq_pe[0, None, :f_count, :]
                table = table.flatten(0, 1) * valid.unsqueeze(-1)
            elif embedding == "learned-1d":
                table = pe.pe[:, :x.shape[1]]
            else:
                bands = torch.arange(0, configs.D_MODEL, 2).float() / configs.D_MODEL
                angles = torch.arange(x.shape[1])[:, None] / positional_encoding.base**bands
                table = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(x.dtype)
            x = ref.post_layer_norm(x + table)
        else:
            x = ref.conv(x.masked_fill(~valid[..., None], 0).transpose(1, 2)).transpose(1, 2)
            x = ref.norm(x)
        x = x[:, ::subsampling.factor]
        valid = valid[:, ::subsampling.factor]
        assert torch.equal(observed_mask, valid)
        return x[valid]

    def actual_forward(_step, waveform):
        nonlocal observed_mask
        features = model.process(AudioBatch(x=waveform, mask=mask))
        result = model.embed(model.tokenize(features), layout=model.token_layout(features))
        observed_mask = result.mask
        return result.x[result.mask]

    assert_training_parity(
        actual, reference,
        actual_forward=actual_forward,
        reference_forward=reference_forward,
        inputs=lambda step: torch.randn(
            mask.shape, dtype=parity_dtype, generator=torch.Generator().manual_seed(step)
        ),
        precision=parity_precision,
    )


def test_masked_log_mel_batch_norm_matches_torchaudio_and_pytorch(
    parity_dtype, parity_precision,
):
    config = configs.melspec(log_mel=True, batch_norm=True, batch_norm_eps=parity_precision.eps)
    actual = AudioModel(configs.model(processor=config)).processor.processor.to(parity_dtype)
    reference = deepcopy(actual)
    mask = torch.arange(241)[None, :] < torch.tensor([[241], [163]])
    pad = config.pad + (config.n_fft // 2 if config.center else 0)
    frames = F.pad(mask, (pad, pad)).unfold(-1, config.n_fft, config.hop_length).all(-1)

    def actual_forward(_step, x):
        out = actual(AudioBatch(x=x, mask=mask))
        assert torch.equal(out.mask, frames)
        return out.x

    def reference_forward(_step, x):
        power = reference.melspec(x.float().masked_fill(~mask, 0)).to(x.dtype)
        log_mel = reference.amplitude_to_db(power).transpose(1, 2)
        output = torch.zeros_like(log_mel)
        output[frames] = reference.batch_norm.batch_norm(log_mel[frames])
        return output

    def compare_statistics():
        left = actual.batch_norm.batch_norm
        right = reference.batch_norm.batch_norm
        assert torch.equal(left.num_batches_tracked, right.num_batches_tracked)
        for name in ("running_mean", "running_var"):
            parity_precision.close(left.get_buffer(name), right.get_buffer(name), label=name)

    assert_training_parity(
        actual, reference,
        actual_forward=actual_forward,
        reference_forward=reference_forward,
        inputs=lambda step: torch.randn(
            mask.shape, dtype=parity_dtype, generator=torch.Generator().manual_seed(step)
        ),
        precision=parity_precision,
        after_step=compare_statistics,
    )


def test_eat_frontend_training_matches_kaldi_and_conv2d(parity_dtype, parity_precision):
    processor = configs.kaldi_fbank(target_length=16, mask_all_frames=True)
    tokenizer = configs.patchify()
    model = AudioModel(configs.model(
        processor=processor,
        tokenizer=tokenizer,
        embedding=configs.eat_embedding(entry_layer_norm_eps=parity_precision.eps),
    ))
    actual = model.embedding.to(parity_dtype).train()
    reference = deepcopy(actual)
    lengths = (2200, 1360)
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    pt, pf = tokenizer.patch_shape
    times = torch.arange(processor.target_length // pt)
    frequencies = torch.arange(processor.num_mel_bins // pf)
    bands = configs.D_MODEL // 4
    inverse = (10000 ** (torch.arange(bands).float() / bands)).reciprocal()

    def sincos(positions):
        phase = positions[:, None] * inverse
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    frequency_pe = sincos(frequencies).expand(len(times), -1, -1)
    time_pe = sincos(times)[:, None, :].expand(-1, len(frequencies), -1)
    position_encoding = torch.cat((frequency_pe, time_pe), dim=-1).flatten(0, 1).to(parity_dtype)

    def actual_forward(_step, x):
        features = model.process(AudioBatch(x=x, mask=mask))
        out = model.embed(model.tokenize(features), layout=model.token_layout(features))
        assert out.num_prefix_tokens == 1 and out.mask.all()
        return out.x

    def reference_forward(_step, x):
        rows = []
        for row, length in zip(x, lengths, strict=True):
            waveform = row[None, :length]
            if processor.remove_waveform_mean:
                waveform = waveform - waveform.mean()
            mel = fbank(
                waveform, sample_frequency=processor.sr, frame_shift=processor.frame_shift_ms,
                num_mel_bins=processor.num_mel_bins, dither=0, htk_compat=processor.htk_compat,
                use_energy=processor.use_energy, window_type=processor.window_type,
            )[:processor.target_length]
            padded = F.pad(mel, (0, 0, 0, processor.target_length - mel.shape[0]))
            rows.append((padded - processor.norm_mean) / (2 * processor.norm_std))
        embedding = reference.embedding
        proj = embedding.projection
        x = F.conv2d(
            torch.stack(rows).unsqueeze(1),
            proj.weight.view(configs.D_MODEL, 1, pt, pf), proj.bias, stride=(pt, pf),
        )
        patches = x.flatten(2).transpose(1, 2) + position_encoding
        cls = embedding.cls_token.expand(len(lengths), -1, -1)
        return embedding.entry_layer_norm(torch.cat((cls, patches), dim=1))

    assert_training_parity(
        actual, reference,
        actual_forward=actual_forward,
        reference_forward=reference_forward,
        inputs=lambda step: torch.randn(
            mask.shape, dtype=parity_dtype, generator=torch.Generator().manual_seed(step)
        ),
        precision=parity_precision,
    )

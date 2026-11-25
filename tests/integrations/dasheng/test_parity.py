"""Component-level parity tests between our AudioModel pieces and the
reference Dasheng modules.

Tests are ordered from finest grain to coarsest:
  1. mel front-end + init BN
  2. patch embedding (weight reshape correctness)
  3. separable positional embedding
  4. single Attention layer
  5. single Block (catches LayerNorm eps regressions)
  6. full 12-block encoder stack + final norm
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn as nn

from audiossl.config import (
    AudioModelConfig,
    LearnableTimeFrequencyPositionalEncodingConfig,
    LinearEmbeddingConfig,
    MLPConfig,
    MelSpectrogramProcessorConfig,
    MultiHeadAttentionConfig,
    NoPositionalEncodingConfig,
    PatchifyTokenizerConfig,
    RuntimeConfig,
    TransformerEncoderConfig,
)
from audiossl.config.batch_dataclasses import AudioBatch, TokenLayout
from audiossl.integrations.dasheng import make_dasheng_base_config
from audiossl.integrations.dasheng.porter import (
    _convert_freq_pos_embed,
    _convert_patch_embed_weight,
    _convert_time_pos_embed,
)
from audiossl.modelling.core.positional_encoding import LearnableTimeFrequencyPositionalEncoding
from audiossl.modelling.core.mha import MultiHeadAttention
from audiossl.modelling.core.mlp import MLP
from audiossl.modelling.encoders.encoder import Encoder
from audiossl.modelling.encoders.transformer_block import TransformerBlock
from audiossl.modelling.model import AudioModel
from audiossl.modelling.processors.melspec import MelSpectrogramProcessor
from audiossl.modelling.tokenizers.patchify import PatchifyTokenizer


# --- Test 1 -----------------------------------------------------------------

def test_mel_front_end_parity(dasheng_modules):
    """Mel + AmplitudeToDB + BatchNorm produce identical features."""
    torch.manual_seed(0)
    audio_transforms = pytest.importorskip("torchaudio.transforms")
    from einops.layers.torch import Rearrange

    # Build reference: front_end (MelSpec + AmplitudeToDB) + init_bn (BN2d).
    n_mels, n_fft, hop, win = 64, 512, 160, 512
    front_end = nn.Sequential(
        audio_transforms.MelSpectrogram(
            f_min=0,
            sample_rate=16000,
            win_length=win,
            center=True,
            n_fft=n_fft,
            f_max=8000,
            hop_length=hop,
            n_mels=n_mels,
        ),
        audio_transforms.AmplitudeToDB(top_db=120),
    ).eval()
    init_bn = nn.Sequential(
        Rearrange("b c f t -> b f c t"),
        nn.BatchNorm2d(n_mels, momentum=0.01),
        Rearrange("b f c t -> b c f t"),
    ).eval()
    # Give BN non-trivial running stats so the comparison isn't trivially zero.
    with torch.no_grad():
        init_bn[1].running_mean.copy_(torch.randn(n_mels))
        init_bn[1].running_var.copy_(torch.rand(n_mels) + 0.5)
        init_bn[1].weight.copy_(torch.randn(n_mels))
        init_bn[1].bias.copy_(torch.randn(n_mels))

    # Build ours via the Dasheng recipe.
    cfg = make_dasheng_base_config()
    proc = MelSpectrogramProcessor(config=cfg.processor)
    proc.eval()

    # Copy BN params from reference to ours.
    with torch.no_grad():
        proc.batch_norm.batch_norm.running_mean.copy_(init_bn[1].running_mean)
        proc.batch_norm.batch_norm.running_var.copy_(init_bn[1].running_var)
        proc.batch_norm.batch_norm.weight.copy_(init_bn[1].weight)
        proc.batch_norm.batch_norm.bias.copy_(init_bn[1].bias)
        # Mel buffers should already match by construction (same params), but
        # be defensive.
        proc.melspec.spectrogram.window.copy_(front_end[0].spectrogram.window)
        proc.melspec.mel_scale.fb.copy_(front_end[0].mel_scale.fb)

    waveform = torch.randn(2, 16000)
    mask = torch.ones_like(waveform, dtype=torch.bool)

    ref = init_bn(front_end(waveform).unsqueeze(1))  # [B, 1, F, T]
    ours = proc(AudioBatch(x=waveform, mask=mask)).x  # [B, T, F]

    # Reshape ref to match ours: [B, T, F].
    ref_btf = ref.squeeze(1).transpose(1, 2)
    assert ref_btf.shape == ours.shape, (ref_btf.shape, ours.shape)
    torch.testing.assert_close(ours, ref_btf, atol=1e-5, rtol=1e-5)


# --- Test 2 -----------------------------------------------------------------

def test_patch_embed_weight_transform_parity():
    """Conv2d patch embed == our Patchify + Linear with the converted weight."""
    torch.manual_seed(1)
    from einops import rearrange

    embed_dim = 32  # small for speed
    n_mels = 64
    patch_freq = 64
    patch_time = 4
    t_frames = 4 * 7  # exactly 7 patches, no truncation

    conv = nn.Conv2d(
        1,
        embed_dim,
        kernel_size=(patch_freq, patch_time),
        stride=(patch_freq, patch_time),
    )

    # Build our pipeline: PatchifyTokenizer + LinearEmbedding (input_dim=256).
    tok_cfg = PatchifyTokenizerConfig(
        patch_shape=(patch_time, patch_freq),
        padding_strategy="truncate",
    )
    tokenizer = PatchifyTokenizer(config=tok_cfg)

    from audiossl.modelling.embeddings.linear import LinearEmbedding

    emb = LinearEmbedding(
        input_dim=patch_time * patch_freq,
        d_model=embed_dim,
        config=LinearEmbeddingConfig(
            positional_encoding=NoPositionalEncodingConfig()
        ),
    )
    emb.eval()

    # Port conv weights into emb's Linear via the porter's helper.
    with torch.no_grad():
        emb.projector.projection.weight.copy_(
            _convert_patch_embed_weight(conv.weight)
        )
        emb.projector.projection.bias.copy_(conv.bias)

    # Input shaped for both paths.
    mel_bft = torch.randn(2, n_mels, t_frames)  # [B, F, T] for Dasheng
    mel_btf = mel_bft.transpose(1, 2).contiguous()  # [B, T, F] for ours

    # Reference path.
    ref = conv(mel_bft.unsqueeze(1))  # [B, E, F_p=1, T_p=7]
    ref = rearrange(ref, "b c f t -> b (f t) c")  # [B, 7, E]

    # Ours.
    mask = torch.ones((2, t_frames), dtype=torch.bool)
    patches = tokenizer(AudioBatch(x=mel_btf, mask=mask))
    ours = emb(patches).x  # [B, 7, E]

    assert ref.shape == ours.shape, (ref.shape, ours.shape)
    torch.testing.assert_close(ours, ref, atol=1e-6, rtol=1e-6)


# --- Test 3 -----------------------------------------------------------------

def test_positional_embedding_2d_parity():
    """Embedding-owned 2D PE reproduces Dasheng's time + freq broadcast."""
    torch.manual_seed(2)
    from einops import rearrange

    embed_dim = 32
    t_patches = 7
    f_patches = 1  # released variants

    time_pos = torch.randn(1, embed_dim, 1, t_patches) * 0.02
    freq_pos = torch.randn(1, embed_dim, f_patches, 1) * 0.02
    # Pretend patch_embed output (freq=1).
    patches = torch.randn(3, embed_dim, f_patches, t_patches)

    # Reference path.
    ref = patches + time_pos[:, :, :, :t_patches]
    ref = ref + freq_pos[:, :, :, :]
    ref = rearrange(ref, "b c f t -> b (f t) c")  # [B, T, E]

    # Our path: flatten patches first, then add embedding-owned 2D PE.
    flat = rearrange(patches, "b c f t -> b (f t) c")  # [B, T, E]
    pe = LearnableTimeFrequencyPositionalEncoding(
        LearnableTimeFrequencyPositionalEncodingConfig(
            max_t=t_patches,
            max_f=f_patches,
        ),
        d_model=embed_dim,
    )
    with torch.no_grad():
        pe.time_pe.copy_(_convert_time_pos_embed(time_pos))
        pe.freq_pe.copy_(_convert_freq_pos_embed(freq_pos))
    layout = TokenLayout(
        num_tokens=t_patches * f_patches,
        time_patches=t_patches,
        frequency_patches=f_patches,
        patch_shape=(1, 1),
        stride=(1, 1),
        flatten_order="time_major",
        output_dim=1,
        source_shape=(t_patches, f_patches),
    )
    ours = pe(flat, layout=layout)

    torch.testing.assert_close(ours, ref, atol=1e-6, rtol=1e-6)


# --- Test 4 -----------------------------------------------------------------

def test_single_attention_layer_parity(dasheng_modules):
    """Dasheng Attention(qkv_bias=True) == our MultiHeadAttention with ported weights."""
    torch.manual_seed(3)
    embed_dim, num_heads, seq_len, batch_size = 64, 4, 11, 2

    ref_attn = dasheng_modules.models.Attention(
        dim=embed_dim, num_heads=num_heads, qkv_bias=True
    ).eval()

    ours = MultiHeadAttention(
        d_model=embed_dim,
        config=MultiHeadAttentionConfig(num_heads=num_heads),
        positional_encoding=None,
    ).eval()

    with torch.no_grad():
        ours.mha.in_proj_weight.copy_(ref_attn.qkv.weight)
        ours.mha.in_proj_bias.copy_(ref_attn.qkv.bias)
        ours.mha.out_proj.weight.copy_(ref_attn.proj.weight)
        ours.mha.out_proj.bias.copy_(ref_attn.proj.bias)

    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)  # all valid

    ref_out = ref_attn(x)
    our_out = ours(x, attention_mask=mask)

    torch.testing.assert_close(our_out, ref_out, atol=1e-5, rtol=1e-5)


# --- Test 5 -----------------------------------------------------------------

def test_single_block_parity(dasheng_modules):
    """Dasheng Block (eps=1e-6) == our TransformerBlock with ported weights.

    This is the test that catches LayerNorm eps regressions.
    """
    torch.manual_seed(4)
    embed_dim, num_heads, seq_len, batch_size = 64, 4, 11, 2
    mlp_ratio = 4

    ref_block = dasheng_modules.models.Block(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
    ).eval()

    enc_cfg = TransformerEncoderConfig(
        mha=MultiHeadAttentionConfig(num_heads=num_heads),
        mlp=MLPConfig(d_ff=embed_dim * mlp_ratio, activation="gelu"),
        num_layers=1,
        pre_norm=True,
        positional_encoding=None,
        layer_norm_eps=1e-6,
        final_norm=False,
    )
    our_block = TransformerBlock(d_model=embed_dim, config=enc_cfg).eval()

    with torch.no_grad():
        our_block.norm1.weight.copy_(ref_block.norm1.weight)
        our_block.norm1.bias.copy_(ref_block.norm1.bias)
        our_block.norm2.weight.copy_(ref_block.norm2.weight)
        our_block.norm2.bias.copy_(ref_block.norm2.bias)
        our_block.mha.mha.in_proj_weight.copy_(ref_block.attn.qkv.weight)
        our_block.mha.mha.in_proj_bias.copy_(ref_block.attn.qkv.bias)
        our_block.mha.mha.out_proj.weight.copy_(ref_block.attn.proj.weight)
        our_block.mha.mha.out_proj.bias.copy_(ref_block.attn.proj.bias)
        our_block.mlp.fc1.weight.copy_(ref_block.mlp.fc1.weight)
        our_block.mlp.fc1.bias.copy_(ref_block.mlp.fc1.bias)
        our_block.mlp.fc2.weight.copy_(ref_block.mlp.fc2.weight)
        our_block.mlp.fc2.bias.copy_(ref_block.mlp.fc2.bias)

    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    ref_out = ref_block(x)
    our_out = our_block(x, attention_mask=mask)

    torch.testing.assert_close(our_out, ref_out, atol=1e-5, rtol=1e-5)


def test_single_block_with_wrong_eps_fails(dasheng_modules):
    """Sanity check: with eps=1e-5 (default) instead of 1e-6, parity breaks.

    This guards against a regression where the eps plumbing silently reverts
    to PyTorch's default.
    """
    torch.manual_seed(5)
    embed_dim, num_heads, seq_len, batch_size = 64, 4, 11, 2

    ref_block = dasheng_modules.models.Block(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).eval()

    # Build our block with the WRONG eps (PyTorch default).
    enc_cfg = TransformerEncoderConfig(
        mha=MultiHeadAttentionConfig(num_heads=num_heads),
        mlp=MLPConfig(d_ff=embed_dim * 4, activation="gelu"),
        num_layers=1,
        positional_encoding=None,
        layer_norm_eps=1e-5,
        final_norm=False,
    )
    our_block = TransformerBlock(d_model=embed_dim, config=enc_cfg).eval()

    with torch.no_grad():
        our_block.norm1.weight.copy_(ref_block.norm1.weight)
        our_block.norm1.bias.copy_(ref_block.norm1.bias)
        our_block.norm2.weight.copy_(ref_block.norm2.weight)
        our_block.norm2.bias.copy_(ref_block.norm2.bias)
        our_block.mha.mha.in_proj_weight.copy_(ref_block.attn.qkv.weight)
        our_block.mha.mha.in_proj_bias.copy_(ref_block.attn.qkv.bias)
        our_block.mha.mha.out_proj.weight.copy_(ref_block.attn.proj.weight)
        our_block.mha.mha.out_proj.bias.copy_(ref_block.attn.proj.bias)
        our_block.mlp.fc1.weight.copy_(ref_block.mlp.fc1.weight)
        our_block.mlp.fc1.bias.copy_(ref_block.mlp.fc1.bias)
        our_block.mlp.fc2.weight.copy_(ref_block.mlp.fc2.weight)
        our_block.mlp.fc2.bias.copy_(ref_block.mlp.fc2.bias)

    # Use inputs scaled so the eps difference matters (small variance ->
    # the eps term dominates the LN denominator).
    x = torch.randn(batch_size, seq_len, embed_dim) * 1e-3
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    ref_out = ref_block(x)
    our_out = our_block(x, attention_mask=mask)

    # Outputs should NOT match within the strict tolerance from the previous
    # test (else our eps plumbing isn't actually being used).
    with pytest.raises(AssertionError):
        torch.testing.assert_close(our_out, ref_out, atol=1e-5, rtol=1e-5)


# --- Test 6 -----------------------------------------------------------------

def test_full_encoder_stack_parity(dasheng_modules):
    """12 random blocks + final norm: ours == Dasheng's block stack."""
    torch.manual_seed(6)
    embed_dim, num_heads, depth, seq_len, batch_size = 64, 4, 12, 9, 2
    mlp_ratio = 4

    # Build reference: 12 Blocks + final LayerNorm(eps=1e-6).
    ref_blocks = nn.ModuleList(
        [
            dasheng_modules.models.Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(depth)
        ]
    ).eval()
    ref_final_norm = nn.LayerNorm(embed_dim, eps=1e-6).eval()
    with torch.no_grad():
        ref_final_norm.weight.copy_(torch.randn(embed_dim))
        ref_final_norm.bias.copy_(torch.randn(embed_dim))

    # Build our Encoder with matching config and no entry PE.
    enc_cfg = TransformerEncoderConfig(
        mha=MultiHeadAttentionConfig(num_heads=num_heads),
        mlp=MLPConfig(d_ff=embed_dim * mlp_ratio, activation="gelu"),
        num_layers=depth,
        positional_encoding=None,
        layer_norm_eps=1e-6,
        final_norm=True,
    )
    our_enc = Encoder(d_model=embed_dim, config=enc_cfg).eval()

    # Zero the PE so it doesn't contribute (we test the stack, not PE).
    with torch.no_grad():
        our_enc.final_norm.weight.copy_(ref_final_norm.weight)
        our_enc.final_norm.bias.copy_(ref_final_norm.bias)
        for i in range(depth):
            rb = ref_blocks[i]
            ob = our_enc.blocks[i]
            ob.norm1.weight.copy_(rb.norm1.weight)
            ob.norm1.bias.copy_(rb.norm1.bias)
            ob.norm2.weight.copy_(rb.norm2.weight)
            ob.norm2.bias.copy_(rb.norm2.bias)
            ob.mha.mha.in_proj_weight.copy_(rb.attn.qkv.weight)
            ob.mha.mha.in_proj_bias.copy_(rb.attn.qkv.bias)
            ob.mha.mha.out_proj.weight.copy_(rb.attn.proj.weight)
            ob.mha.mha.out_proj.bias.copy_(rb.attn.proj.bias)
            ob.mlp.fc1.weight.copy_(rb.mlp.fc1.weight)
            ob.mlp.fc1.bias.copy_(rb.mlp.fc1.bias)
            ob.mlp.fc2.weight.copy_(rb.mlp.fc2.weight)
            ob.mlp.fc2.bias.copy_(rb.mlp.fc2.bias)

    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    ref = x
    for blk in ref_blocks:
        ref = blk(ref)
    ref = ref_final_norm(ref)

    ours = our_enc(AudioBatch(x=x, mask=mask)).z

    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)

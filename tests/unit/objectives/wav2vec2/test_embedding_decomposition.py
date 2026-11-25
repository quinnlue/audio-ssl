import torch

from audiossl.config.batch_dataclasses import AudioBatch


def test_staged_embedding_matches_forward(tiny_wav2vec2_model):
    embedding = tiny_wav2vec2_model.embedding
    embedding.eval()
    x = torch.randn(2, 13, tiny_wav2vec2_model.config.tokenizer.output_dim)
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[1, 9:] = False
    batch = AudioBatch(x=x, mask=mask)

    with torch.inference_mode():
        direct = embedding(batch)
        staged = embedding.finalize(
            embedding.position(embedding.project(embedding.prepare(batch)))
        )

    torch.testing.assert_close(staged.x, direct.x, atol=0.0, rtol=0.0)
    torch.testing.assert_close(staged.mask, direct.mask)
    assert staged.num_prefix_tokens == direct.num_prefix_tokens

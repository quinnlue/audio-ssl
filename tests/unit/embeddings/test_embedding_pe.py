from __future__ import annotations

import unittest

import torch

from audiossl.config import (
    ConvEmbeddingConfig,
    EmbeddingSubsamplingConfig,
    LearnablePositionalEncodingConfig,
    LinearEmbeddingConfig,
    NoPositionalEncodingConfig,
)
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.core.positional_encoding import NoPositionalEncoding
from audiossl.modelling.embeddings.conv import ConvEmbedding
from audiossl.modelling.embeddings.linear import LinearEmbedding


class EmbeddingPositionalEncodingTests(unittest.TestCase):
    def test_linear_embedding_applies_pe_before_subsampling(self) -> None:
        embedding = LinearEmbedding(
            input_dim=2,
            d_model=2,
            config=LinearEmbeddingConfig(
                positional_encoding=LearnablePositionalEncodingConfig(max_len=4),
                subsampling=EmbeddingSubsamplingConfig(factor=2),
            ),
        )
        with torch.no_grad():
            embedding.projector.projection.weight.copy_(torch.eye(2))
            embedding.projector.projection.bias.zero_()
            embedding.positional_encoding.pe.copy_(
                torch.tensor([[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]])
            )
        batch = AudioBatch(
            x=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
            mask=torch.tensor([[True, False, True, True]], dtype=torch.bool),
        )

        out = embedding(batch)
        projected_plus_pe = batch.x + embedding.positional_encoding.pe

        torch.testing.assert_close(out.x, projected_plus_pe[:, ::2])
        self.assertEqual(out.mask.tolist(), [[True, True]])

    def test_conv_embedding_applies_pe_before_subsampling(self) -> None:
        embedding = ConvEmbedding(
            input_dim=2,
            d_model=2,
            config=ConvEmbeddingConfig(
                positional_encoding=LearnablePositionalEncodingConfig(max_len=4),
                kernel_size=1,
                subsampling=EmbeddingSubsamplingConfig(factor=2),
            ),
        )
        with torch.no_grad():
            embedding.conv.weight.copy_(torch.eye(2).reshape(2, 2, 1))
            embedding.conv.bias.zero_()
            embedding.positional_encoding.pe.copy_(
                torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])
            )
        batch = AudioBatch(
            x=torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
            mask=torch.ones(1, 4, dtype=torch.bool),
        )

        out = embedding(batch)
        projected_plus_pe = batch.x + embedding.positional_encoding.pe

        torch.testing.assert_close(out.x, projected_plus_pe[:, ::2])
        self.assertEqual(out.mask.tolist(), [[True, True]])

    def test_mask_is_unchanged_without_subsampling(self) -> None:
        embedding = LinearEmbedding(
            input_dim=2,
            d_model=2,
            config=LinearEmbeddingConfig(
                positional_encoding=LearnablePositionalEncodingConfig(max_len=3),
            ),
        )
        batch = AudioBatch(
            x=torch.randn(1, 3, 2),
            mask=torch.tensor([[True, False, True]], dtype=torch.bool),
        )

        out = embedding(batch)

        self.assertTrue(torch.equal(out.mask, batch.mask))

    def test_no_positional_encoding_returns_identity_tensor(self) -> None:
        pe = NoPositionalEncoding(NoPositionalEncodingConfig(), d_model=2)
        x = torch.randn(1, 3, 2)

        self.assertIs(pe(x), x)


if __name__ == "__main__":
    unittest.main()

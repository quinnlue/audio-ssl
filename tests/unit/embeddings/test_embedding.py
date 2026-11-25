from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from audiossl.config import (
    ConvEmbeddingConfig,
    EmbeddingSubsamplingConfig,
    LinearEmbeddingConfig,
    NoPositionalEncodingConfig,
)
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.embeddings.base import BaseEmbedding
from audiossl.modelling.embeddings.conv import ConvEmbedding
from audiossl.modelling.embeddings.linear import LinearEmbedding


class LinearEmbeddingTests(unittest.TestCase):
    def _make(
        self,
        *,
        input_dim: int = 4,
        d_model: int = 8,
        dropout_p: float = 0.0,
        subsampling: EmbeddingSubsamplingConfig | None = None,
    ) -> LinearEmbedding:
        return LinearEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            config=LinearEmbeddingConfig(
                positional_encoding=NoPositionalEncodingConfig(),
                dropout_p=dropout_p,
                subsampling=subsampling,
            ),
        )

    def _batch(
        self,
        *,
        batch_size: int = 2,
        time: int = 6,
        input_dim: int = 4,
    ) -> AudioBatch:
        return AudioBatch(
            x=torch.randn(batch_size, time, input_dim),
            mask=torch.ones(batch_size, time, dtype=torch.bool),
        )

    def test_forward_changes_feature_dim_to_d_model(self) -> None:
        embedding = self._make(input_dim=4, d_model=8)
        out = embedding(self._batch(input_dim=4))

        self.assertIsInstance(out, AudioBatch)
        self.assertEqual(out.x.shape, (2, 6, 8))
        self.assertIsInstance(embedding, BaseEmbedding)

    def test_forward_passes_mask_through_unchanged(self) -> None:
        embedding = self._make()
        batch = self._batch()
        mask = batch.mask.clone()

        out = embedding(batch)

        self.assertTrue(torch.equal(out.mask, mask))
        self.assertEqual(out.mask.dtype, torch.bool)

    def test_rejects_non_audiobatch(self) -> None:
        embedding = self._make()
        with self.assertRaisesRegex(TypeError, "LinearEmbedding.forward expected AudioBatch"):
            embedding(object())

    def test_rejects_wrong_rank_x(self) -> None:
        embedding = self._make()
        batch = AudioBatch(x=torch.randn(2, 4), mask=torch.ones(2, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "expected batch.x to have shape"):
            embedding(batch)

    def test_rejects_input_dim_mismatch(self) -> None:
        embedding = self._make(input_dim=4)
        bad = self._batch(input_dim=5)
        with self.assertRaisesRegex(ValueError, "last dimension must match input_dim"):
            embedding(bad)

    def test_rejects_non_bool_mask(self) -> None:
        embedding = self._make()
        batch = AudioBatch(
            x=torch.randn(2, 6, 4),
            mask=torch.ones(2, 6, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "torch.bool"):
            embedding(batch)

    def test_rejects_non_linearembeddingconfig(self) -> None:
        with self.assertRaisesRegex(TypeError, "LinearEmbeddingConfig"):
            LinearEmbedding(input_dim=4, d_model=8, config=object())  # type: ignore[arg-type]

    def test_projector_shapes_match_dims(self) -> None:
        embedding = self._make(input_dim=4, d_model=8)
        linear = embedding.projector.projection
        self.assertEqual(linear.in_features, 4)
        self.assertEqual(linear.out_features, 8)

    def test_subsampling_strided_selects_tokens_and_mask(self) -> None:
        embedding = self._make(subsampling=EmbeddingSubsamplingConfig(factor=2))
        batch = replace(
            self._batch(time=7),
            mask=torch.tensor(
                [
                    [True, False, True, True, False, True, True],
                    [True, True, False, True, True, False, False],
                ],
                dtype=torch.bool,
            ),
        )

        out = embedding(batch)

        self.assertEqual(tuple(out.x.shape), (2, 4, 8))
        self.assertEqual(out.mask.tolist(), batch.mask[:, ::2].tolist())

    def test_subsampling_factor_one_is_noop(self) -> None:
        embedding = self._make(subsampling=EmbeddingSubsamplingConfig(factor=1))
        batch = self._batch(time=5)

        out = embedding(batch)

        self.assertEqual(tuple(out.x.shape), (2, 5, 8))
        self.assertTrue(torch.equal(out.mask, batch.mask))


class ConvEmbeddingTests(unittest.TestCase):
    def _make(
        self,
        *,
        input_dim: int = 4,
        d_model: int = 8,
        kernel_size: int = 1,
        dropout_p: float = 0.0,
        layer_norm: bool = False,
        subsampling: EmbeddingSubsamplingConfig | None = None,
    ) -> ConvEmbedding:
        return ConvEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            config=ConvEmbeddingConfig(
                positional_encoding=NoPositionalEncodingConfig(),
                kernel_size=kernel_size,
                dropout_p=dropout_p,
                layer_norm=layer_norm,
                subsampling=subsampling,
            ),
        )

    def _batch(
        self,
        *,
        batch_size: int = 2,
        time: int = 6,
        input_dim: int = 4,
    ) -> AudioBatch:
        return AudioBatch(
            x=torch.randn(batch_size, time, input_dim),
            mask=torch.ones(batch_size, time, dtype=torch.bool),
        )

    def test_forward_changes_feature_dim_to_d_model(self) -> None:
        embedding = self._make(input_dim=4, d_model=8, kernel_size=3)
        out = embedding(self._batch(input_dim=4))

        self.assertIsInstance(out, AudioBatch)
        self.assertEqual(out.x.shape, (2, 6, 8))
        self.assertIsInstance(embedding, BaseEmbedding)

    def test_token_count_preserved_for_odd_kernel(self) -> None:
        for kernel_size in (1, 3, 5, 7):
            with self.subTest(kernel_size=kernel_size):
                embedding = self._make(kernel_size=kernel_size)
                out = embedding(self._batch(time=10))
                self.assertEqual(out.x.shape[1], 10)

    def test_forward_passes_mask_through_unchanged(self) -> None:
        embedding = self._make(kernel_size=3)
        batch = self._batch()
        mask = batch.mask.clone()
        out = embedding(batch)
        self.assertTrue(torch.equal(out.mask, mask))
        self.assertEqual(out.mask.dtype, torch.bool)

    def test_layer_norm_applies_when_enabled(self) -> None:
        embedding = self._make(layer_norm=True)
        self.assertIsNotNone(embedding.norm)
        out = embedding(self._batch())
        self.assertEqual(out.x.shape, (2, 6, 8))

    def test_layer_norm_absent_by_default(self) -> None:
        embedding = self._make()
        self.assertIsNone(embedding.norm)

    def test_masked_positions_do_not_leak_into_valid_outputs(self) -> None:
        # Two batches identical at the unmasked positions but different at
        # masked positions. The conv output at unmasked positions must match
        # because masked content is zeroed before the conv.
        torch.manual_seed(0)
        embedding = self._make(kernel_size=3)
        a = torch.randn(1, 6, 4)
        b = a.clone()
        mask = torch.tensor([[True, True, True, True, False, False]])
        b[0, 4:] = 1234.0  # differ only at masked positions
        out_a = embedding(AudioBatch(x=a, mask=mask)).x
        out_b = embedding(AudioBatch(x=b, mask=mask)).x
        # At the valid prefix, outputs must be identical
        self.assertTrue(torch.allclose(out_a[:, :4], out_b[:, :4]))

    def test_kernel_size_one_equivalent_to_linear_projection(self) -> None:
        # Conv1d with kernel_size=1 is mathematically a per-token Linear.
        torch.manual_seed(0)
        embedding = self._make(kernel_size=1)
        batch = self._batch(input_dim=4)
        # Recompute manually via einsum on conv.weight (shape: d_model x input_dim x 1)
        w = embedding.conv.weight.squeeze(-1)  # (d_model, input_dim)
        b = embedding.conv.bias  # (d_model,)
        expected = torch.einsum("bni,oi->bno", batch.x, w) + b
        out = embedding(batch).x
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_rejects_non_audiobatch(self) -> None:
        embedding = self._make()
        with self.assertRaisesRegex(TypeError, "ConvEmbedding.forward expected AudioBatch"):
            embedding(object())

    def test_rejects_input_dim_mismatch(self) -> None:
        embedding = self._make(input_dim=4)
        bad = self._batch(input_dim=5)
        with self.assertRaisesRegex(ValueError, "last dimension must match input_dim"):
            embedding(bad)

    def test_rejects_non_bool_mask(self) -> None:
        embedding = self._make()
        batch = AudioBatch(
            x=torch.randn(2, 6, 4),
            mask=torch.ones(2, 6, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "torch.bool"):
            embedding(batch)

    def test_rejects_non_convembeddingconfig(self) -> None:
        with self.assertRaisesRegex(TypeError, "ConvEmbeddingConfig"):
            ConvEmbedding(input_dim=4, d_model=8, config=object())  # type: ignore[arg-type]

    def test_conv_shapes_match_dims(self) -> None:
        embedding = self._make(input_dim=4, d_model=8, kernel_size=3)
        conv = embedding.conv
        self.assertEqual(conv.in_channels, 4)
        self.assertEqual(conv.out_channels, 8)
        self.assertEqual(conv.kernel_size, (3,))
        self.assertEqual(conv.stride, (1,))
        self.assertEqual(conv.padding, (1,))

    def test_subsampling_strided_selects_tokens_and_mask(self) -> None:
        embedding = self._make(
            kernel_size=3,
            subsampling=EmbeddingSubsamplingConfig(factor=3),
        )
        batch = replace(
            self._batch(time=8),
            mask=torch.tensor(
                [
                    [True, False, True, True, False, True, True, False],
                    [True, True, False, True, True, False, False, True],
                ],
                dtype=torch.bool,
            ),
        )

        out = embedding(batch)

        self.assertEqual(tuple(out.x.shape), (2, 3, 8))
        self.assertEqual(out.mask.tolist(), batch.mask[:, ::3].tolist())

    def test_subsampling_factor_one_is_noop(self) -> None:
        embedding = self._make(subsampling=EmbeddingSubsamplingConfig(factor=1))
        batch = self._batch(time=5)

        out = embedding(batch)

        self.assertEqual(tuple(out.x.shape), (2, 5, 8))
        self.assertTrue(torch.equal(out.mask, batch.mask))


class BaseEmbeddingSubsamplingTests(unittest.TestCase):
    def test_subsample_batch_factor_one_returns_original_batch(self) -> None:
        batch = AudioBatch(
            x=torch.randn(1, 4, 3),
            mask=torch.ones(1, 4, dtype=torch.bool),
        )

        out = BaseEmbedding.subsample_batch(batch, EmbeddingSubsamplingConfig(factor=1))

        self.assertIs(out, batch)

    def test_scatter_subsampled_mask_restores_strided_positions(self) -> None:
        mask = torch.tensor([[True, False, True]], dtype=torch.bool)

        scattered = BaseEmbedding.scatter_subsampled_mask(
            mask,
            original_length=6,
            config=EmbeddingSubsamplingConfig(factor=2),
        )

        self.assertEqual(scattered.tolist(), [[True, False, False, False, True, False]])


if __name__ == "__main__":
    unittest.main()

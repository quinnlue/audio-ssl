from __future__ import annotations

import unittest

import torch

from audiossl.config import PatchifyTokenizerConfig
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.tokenizers.patchify import PatchifyTokenizer


class PatchifyTokenizerGridTests(unittest.TestCase):
    def test_patch_grid_matches_forward_for_representative_shapes(self) -> None:
        cases = (
            ((2, 4), None, "pad", 3, 8),
            ((2, 4), None, "truncate", 3, 8),
            ((3, 2), (2, 1), "pad", 8, 5),
            ((3, 2), (2, 1), "truncate", 8, 5),
        )
        for patch_shape, stride, padding_strategy, t_feat, f_in in cases:
            with self.subTest(
                patch_shape=patch_shape,
                stride=stride,
                padding_strategy=padding_strategy,
                t_feat=t_feat,
                f_in=f_in,
            ):
                tokenizer = PatchifyTokenizer(
                    config=PatchifyTokenizerConfig(
                        patch_shape=patch_shape,
                        stride=stride,
                        padding_strategy=padding_strategy,
                    )
                )
                grid = tokenizer.patch_grid(T_feat=t_feat, F_in=f_in)
                batch = AudioBatch(
                    x=torch.randn(2, t_feat, f_in),
                    mask=torch.ones(2, t_feat, dtype=torch.bool),
                )

                output = tokenizer(batch)

                self.assertEqual(output.mask.shape[1], grid[0] * grid[1])
                self.assertEqual(output.mask.shape[1] // grid[1], grid[0])

    def test_patch_grid_shares_feature_validation_with_forward(self) -> None:
        tokenizer = PatchifyTokenizer(
            config=PatchifyTokenizerConfig(
                patch_shape=(2, 3),
                padding_strategy="truncate",
            )
        )

        with self.assertRaisesRegex(ValueError, "Feature dimension"):
            tokenizer.patch_grid(T_feat=4, F_in=4)

    def test_layout_matches_forward_token_grid(self) -> None:
        tokenizer = PatchifyTokenizer(
            config=PatchifyTokenizerConfig(
                patch_shape=(3, 2),
                stride=(2, 1),
                padding_strategy="truncate",
            )
        )
        batch = AudioBatch(
            x=torch.randn(2, 8, 5),
            mask=torch.ones(2, 8, dtype=torch.bool),
        )

        layout = tokenizer.layout(T_feat=8, F_in=5)
        output = tokenizer(batch)

        self.assertEqual(layout.time_patches, 3)
        self.assertEqual(layout.frequency_patches, 4)
        self.assertEqual(layout.num_tokens, output.x.shape[1])
        self.assertEqual(layout.patch_shape, (3, 2))
        self.assertEqual(layout.stride, (2, 1))
        self.assertEqual(layout.flatten_order, "time_major")

    def test_reorder_patch_targets_supports_dasheng_frequency_time_order(self) -> None:
        tokenizer = PatchifyTokenizer(
            config=PatchifyTokenizerConfig(
                patch_shape=(2, 3),
                padding_strategy="truncate",
            )
        )
        layout = tokenizer.layout(T_feat=4, F_in=3)
        target = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

        reordered = tokenizer.reorder_patch_targets(
            target,
            layout=layout,
            order="frequency_time",
        )

        expected = (
            target.view(1, 2, 2, 3)
            .transpose(-2, -1)
            .reshape(1, 2, 6)
        )
        torch.testing.assert_close(reordered, expected)
        self.assertIs(
            tokenizer.reorder_patch_targets(target, layout=layout, order="time_frequency"),
            target,
        )


if __name__ == "__main__":
    unittest.main()

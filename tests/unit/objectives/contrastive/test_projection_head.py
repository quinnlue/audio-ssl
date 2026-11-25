from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from audiossl.config import ProjectionHeadConfig
from audiossl.modelling.objectives.contrastive.projection import ProjectionHead


def _config(**overrides) -> ProjectionHeadConfig:
    values = {
        "depth": 2,
        "hidden_dim": 16,
        "output_dim": 8,
        "activation": "gelu",
        "final_l2_normalize": False,
    }
    values.update(overrides)
    return ProjectionHeadConfig(**values)


class ProjectionHeadTests(unittest.TestCase):
    def test_depth_one_is_single_linear(self) -> None:
        head = ProjectionHead(input_dim=4, config=_config(depth=1))

        modules = list(head.layers)

        self.assertEqual(len(modules), 1)
        self.assertIsInstance(modules[0], nn.Linear)
        self.assertEqual(modules[0].in_features, 4)
        self.assertEqual(modules[0].out_features, 8)

    def test_depth_two_is_linear_activation_dropout_linear(self) -> None:
        head = ProjectionHead(input_dim=4, config=_config(depth=2, dropout_p=0.5))

        modules = list(head.layers)

        self.assertEqual(len(modules), 4)
        self.assertIsInstance(modules[0], nn.Linear)
        self.assertEqual(modules[0].in_features, 4)
        self.assertEqual(modules[0].out_features, 16)
        self.assertNotIsInstance(modules[1], nn.Linear)
        self.assertNotIsInstance(modules[1], nn.Dropout)
        self.assertIsInstance(modules[2], nn.Dropout)
        self.assertEqual(modules[2].p, 0.5)
        self.assertIsInstance(modules[3], nn.Linear)
        self.assertEqual(modules[3].in_features, 16)
        self.assertEqual(modules[3].out_features, 8)

    def test_final_l2_normalize_true_yields_unit_norm_rows(self) -> None:
        torch.manual_seed(0)
        head = ProjectionHead(input_dim=4, config=_config(final_l2_normalize=True))
        head.eval()
        z = torch.randn(5, 4)

        out = head(z)

        norms = out.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

    def test_final_l2_normalize_false_does_not_unit_normalize(self) -> None:
        torch.manual_seed(0)
        head = ProjectionHead(input_dim=4, config=_config(final_l2_normalize=False))
        head.eval()
        z = torch.randn(5, 4)

        out = head(z)

        norms = out.norm(dim=-1)
        self.assertFalse(torch.allclose(norms, torch.ones_like(norms)))

    def test_invalid_depth_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth"):
            ProjectionHead(input_dim=4, config=_config(depth=0))

    def test_invalid_hidden_dim_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            ProjectionHead(input_dim=4, config=_config(hidden_dim=0))

    def test_invalid_output_dim_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_dim"):
            ProjectionHead(input_dim=4, config=_config(output_dim=0))

    def test_unknown_activation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation"):
            ProjectionHead(input_dim=4, config=_config(activation="not_an_activation"))

    def test_unknown_initialization_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported initialization"):
            ProjectionHead(
                input_dim=4,
                config=_config(initialization="not_an_initializer"),
            )

    def test_forward_rejects_wrong_rank(self) -> None:
        head = ProjectionHead(input_dim=4, config=_config())

        with self.assertRaisesRegex(ValueError, "rank-2"):
            head(torch.randn(2, 3, 4))

    def test_forward_rejects_wrong_input_dim(self) -> None:
        head = ProjectionHead(input_dim=4, config=_config())

        with self.assertRaisesRegex(ValueError, "feature dim"):
            head(torch.randn(2, 5))


if __name__ == "__main__":
    unittest.main()

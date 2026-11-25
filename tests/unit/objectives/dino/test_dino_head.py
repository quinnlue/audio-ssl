from __future__ import annotations

import unittest

import torch

from audiossl.config import DINOHeadConfig
from audiossl.modelling.objectives.dino.head import DINOHead


def _config(**overrides) -> DINOHeadConfig:
    values = {
        "depth": 2,
        "hidden_dim": 16,
        "bottleneck_dim": 6,
        "num_prototypes": 13,
        "activation": "gelu",
    }
    values.update(overrides)
    return DINOHeadConfig(**values)


class DINOHeadTests(unittest.TestCase):
    def test_shape_contract(self) -> None:
        torch.manual_seed(0)
        head = DINOHead(input_dim=8, config=_config())

        out = head(torch.randn(5, 8))

        self.assertEqual(out.shape, (5, 13))

    def test_pre_prototype_features_are_l2_normalized(self) -> None:
        torch.manual_seed(0)
        head = DINOHead(input_dim=8, config=_config())

        features = head.forward_features(torch.randn(5, 8))

        torch.testing.assert_close(features.norm(dim=-1), torch.ones(5))

    def test_weight_norm_scale_is_frozen(self) -> None:
        head = DINOHead(input_dim=8, config=_config())

        self.assertFalse(head.prototypes.weight_g.requires_grad)
        self.assertTrue(head.prototypes.weight_v.requires_grad)

    def test_invalid_input_rank_fails(self) -> None:
        head = DINOHead(input_dim=8, config=_config())

        with self.assertRaisesRegex(ValueError, "rank-2"):
            head(torch.randn(2, 3, 8))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch
from audiossl.config import AudioAugmentorConfig, LoudnessAugmentationConfig

from audiotrainer.augment.augmentor import AugmentationPipeline
from audiotrainer.augment.waveform import LoudnessAugmentation
from audiotrainer.collate_fn import collate_fn


def _loudness_config(**overrides) -> AudioAugmentorConfig:
    values = {"p": 1.0, "min_gain_db": -6.0, "max_gain_db": 6.0}
    values.update(overrides)
    return AudioAugmentorConfig(
        augmentations=(LoudnessAugmentationConfig(**values),)
    )


class LoudnessAugmentationTests(unittest.TestCase):
    def test_loudness_requires_explicit_research_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_gain_db"):
            LoudnessAugmentation({"p": 1.0, "min_gain_db": -6.0})

    def test_pipeline_rejects_invalid_gain_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_gain_db"):
            AugmentationPipeline(
                AudioAugmentorConfig(
                    augmentations=(
                        LoudnessAugmentationConfig(
                            p=1.0, min_gain_db=-6.0, max_gain_db=-7.0
                        ),
                    )
                )
            )

    def test_fixed_gain_is_deterministic_and_does_not_consume_rng(self) -> None:
        augmentation = LoudnessAugmentation(
            {"p": 1.0, "min_gain_db": 6.0, "max_gain_db": 6.0}
        )
        waveform = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()

        first = augmentation([waveform])[0]
        second = augmentation([waveform])[0]

        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first, waveform * (10.0 ** (6.0 / 20.0))))

    def test_zero_probability_is_identity_and_does_not_consume_rng(self) -> None:
        augmentation = LoudnessAugmentation(
            {"p": 0.0, "min_gain_db": -20.0, "max_gain_db": 20.0}
        )
        waveform = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()

        output = augmentation([waveform])[0]

        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertIs(output, waveform)

    def test_random_gain_uses_generator_without_consuming_global_rng(self) -> None:
        augmentation = LoudnessAugmentation(
            {"p": 1.0, "min_gain_db": -6.0, "max_gain_db": 6.0}
        )
        waveform = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        generator = torch.Generator().manual_seed(123)
        expected_generator = torch.Generator().manual_seed(123)
        expected_gain_db = torch.empty((), dtype=waveform.dtype).uniform_(
            -6.0, 6.0, generator=expected_generator
        )
        expected = waveform * (10.0 ** (expected_gain_db / 20.0))
        torch.manual_seed(4321)
        before = torch.random.get_rng_state()

        output = augmentation([waveform], generator=generator)[0]

        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.allclose(output, expected))

    def test_probability_gate_uses_generator_without_consuming_global_rng(self) -> None:
        augmentation = LoudnessAugmentation(
            {"p": 0.5, "min_gain_db": -6.0, "max_gain_db": 6.0}
        )
        waveform = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        generator = torch.Generator().manual_seed(3)
        expected_generator = torch.Generator().manual_seed(3)
        gate_sample = torch.rand((), generator=expected_generator)
        self.assertLess(float(gate_sample), 0.5)
        expected_gain_db = torch.empty((), dtype=waveform.dtype).uniform_(
            -6.0, 6.0, generator=expected_generator
        )
        expected = waveform * (10.0 ** (expected_gain_db / 20.0))
        torch.manual_seed(4321)
        before = torch.random.get_rng_state()

        output = augmentation([waveform], generator=generator)[0]

        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.allclose(output, expected))

    def test_pipeline_generator_controls_randomness(self) -> None:
        config = _loudness_config()
        first_pipeline = AugmentationPipeline(
            config, generator=torch.Generator().manual_seed(77)
        )
        second_pipeline = AugmentationPipeline(
            config, generator=torch.Generator().manual_seed(77)
        )
        waveform = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        torch.manual_seed(4321)
        before = torch.random.get_rng_state()

        first = first_pipeline(collate_fn({"x": [waveform]})).x[0]
        second = second_pipeline(collate_fn({"x": [waveform]})).x[0]

        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.allclose(first, second))

    def test_pipeline_call_time_generator_overrides_constructor_generator(self) -> None:
        config = _loudness_config()
        pipeline = AugmentationPipeline(
            config, generator=torch.Generator().manual_seed(1)
        )
        waveform = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        call_generator = torch.Generator().manual_seed(77)
        expected_generator = torch.Generator().manual_seed(77)
        expected = LoudnessAugmentation(
            {"p": 1.0, "min_gain_db": -6.0, "max_gain_db": 6.0}
        )([waveform], generator=expected_generator)[0]

        output = pipeline(
            collate_fn({"x": [waveform]}), generator=call_generator
        ).x[0]
        self.assertTrue(torch.allclose(output, expected))

    def test_pipeline_keeps_batch_device(self) -> None:
        pipeline = AugmentationPipeline(_loudness_config(p=0.0))
        waveform = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
        batch = collate_fn({"x": [waveform]})
        output = pipeline(batch)
        self.assertEqual(output.x.device, batch.x.device)
        self.assertEqual(output.x.dtype, batch.x.dtype)


if __name__ == "__main__":
    unittest.main()

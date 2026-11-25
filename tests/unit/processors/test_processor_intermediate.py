from __future__ import annotations

import copy
import unittest

import torch

from audiossl.config import MelSpectrogramProcessorConfig
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.processors.melspec import MelSpectrogramProcessor


class MelSpectrogramProcessorIntermediateTests(unittest.TestCase):
    def _batch(self) -> AudioBatch:
        torch.manual_seed(0)
        x = torch.randn(2, 96, dtype=torch.float32)
        mask = torch.ones(2, 96, dtype=torch.bool)
        mask[1, 72:] = False
        return AudioBatch(x=x, mask=mask)

    def test_intermediate_methods_compose_to_forward_bit_exactly(self) -> None:
        for log_mel in (False, True):
            for batch_norm in (False, True):
                for frame_validity in ("overlap_any", "strict_window"):
                    with self.subTest(
                        log_mel=log_mel,
                        batch_norm=batch_norm,
                        frame_validity=frame_validity,
                    ):
                        config = MelSpectrogramProcessorConfig(
                            n_mels=4,
                            n_fft=16,
                            win_length=16,
                            hop_length=8,
                            center=False,
                            log_mel=log_mel,
                            batch_norm=batch_norm,
                            frame_validity=frame_validity,
                        )
                        forward_processor = MelSpectrogramProcessor(
                            config=config,
                        )
                        composed_processor = copy.deepcopy(forward_processor)
                        batch = self._batch()

                        forward_out = forward_processor(batch)
                        composed = composed_processor.compute_melspec(batch)
                        composed = composed_processor.compute_amplitude_to_db(composed)
                        composed = composed_processor.compute_batch_norm(composed)

                        self.assertTrue(torch.equal(composed.x, forward_out.x))
                        self.assertTrue(torch.equal(composed.mask, forward_out.mask))

    def test_amplitude_to_db_identity_when_disabled(self) -> None:
        processor = MelSpectrogramProcessor(
            config=MelSpectrogramProcessorConfig(
                n_mels=2,
                n_fft=16,
                win_length=16,
                hop_length=8,
                center=False,
                log_mel=False,
            ),
        )
        mel = processor.compute_melspec(self._batch())

        self.assertIs(processor.compute_amplitude_to_db(mel), mel)

    def test_batch_norm_identity_when_disabled(self) -> None:
        processor = MelSpectrogramProcessor(
            config=MelSpectrogramProcessorConfig(
                n_mels=2,
                n_fft=16,
                win_length=16,
                hop_length=8,
                center=False,
                batch_norm=False,
            ),
        )
        mel = processor.compute_melspec(self._batch())

        self.assertIs(processor.compute_batch_norm(mel), mel)

    def test_frame_validity_policy_controls_mask_leniency(self) -> None:
        batch = AudioBatch(
            x=torch.ones(1, 32, dtype=torch.float32),
            mask=torch.tensor([[False, False, *([True] * 28), False, False]], dtype=torch.bool),
        )
        strict = MelSpectrogramProcessor(
            config=MelSpectrogramProcessorConfig(
                n_mels=4,
                n_fft=8,
                win_length=8,
                hop_length=4,
                center=False,
                frame_validity="strict_window",
            ),
        )
        overlap = MelSpectrogramProcessor(
            config=MelSpectrogramProcessorConfig(
                n_mels=4,
                n_fft=8,
                win_length=8,
                hop_length=4,
                center=False,
                frame_validity="overlap_any",
            ),
        )

        self.assertLess(
            int(strict.output_mask(batch.mask).sum().item()),
            int(overlap.output_mask(batch.mask).sum().item()),
        )


if __name__ == "__main__":
    unittest.main()

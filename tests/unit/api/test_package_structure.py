from __future__ import annotations

import importlib
import unittest


class ModuleStructureBreakTests(unittest.TestCase):
    def test_old_modelling_subpackages_are_removed(self) -> None:
        for module_name in (
            "audiossl.modelling.runtime",
            "audiossl.modelling.augment.waveform",
            "audiossl.training",
        ):
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)

    def test_modelling_does_not_export_contrastive_symbols(self) -> None:
        modelling = importlib.import_module("audiossl.modelling")

        self.assertFalse(hasattr(modelling, "ContrastiveObjectiveOutput"))
        self.assertFalse(hasattr(modelling, "ContrastiveObjectiveConfig"))
        self.assertNotIn("ContrastiveObjectiveOutput", getattr(modelling, "__all__", ()))
        self.assertNotIn("ContrastiveObjectiveConfig", getattr(modelling, "__all__", ()))

    def test_new_paths_are_importable(self) -> None:
        importlib.import_module("audiossl.runtime")
        importlib.import_module("audiossl.modelling.objectives.contrastive.contrastive")
        importlib.import_module("audiossl.data")

    def test_top_level_public_api_exports_are_truthful(self) -> None:
        audiossl = importlib.import_module("audiossl")

        self.assertIn("AudioBatch", audiossl.__all__)
        self.assertIn("AudioModel", audiossl.__all__)
        self.assertNotIn("Backbone", audiossl.__all__)
        self.assertFalse(hasattr(audiossl, "Backbone"))
        self.assertNotIn("AudioModel", audiossl.__dict__)
        self.assertNotIn("Trainer", audiossl.__all__)

        modelling = importlib.import_module("audiossl.modelling")
        self.assertIs(audiossl.AudioModel, modelling.AudioModel)
        for name in audiossl.__all__:
            with self.subTest(name=name):
                getattr(audiossl, name)


if __name__ == "__main__":
    unittest.main()
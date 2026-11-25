from __future__ import annotations

import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audiossl.config import AudioModelConfig
from audiossl.modelling.model import AudioModel
from audiossl.presets import build_model, get_config, get_preset_spec, list_presets
from audiossl.pretrained import get_pretrained_spec, list_pretrained, load_pretrained
from audiossl.pretrained.download import resolve_checkpoint
from audiossl.pretrained.specs import PretrainedModelSpec


class PresetApiTests(unittest.TestCase):
    def test_dasheng_preset_aliases_return_valid_config(self) -> None:
        for name in ("dasheng-base", "dasheng_base", "dasheng/base"):
            with self.subTest(name=name):
                spec = get_preset_spec(name)
                config = get_config(name)

                self.assertEqual(spec.name, "dasheng-base")
                self.assertIsInstance(config, AudioModelConfig)

    def test_wav2vec2_preset_aliases_return_valid_config(self) -> None:
        for name in (
            "wav2vec2-base",
            "wav2vec2_base",
            "wav2vec2/base",
            "facebook/wav2vec2-base",
        ):
            with self.subTest(name=name):
                spec = get_preset_spec(name)
                config = get_config(name)

                self.assertEqual(spec.name, "wav2vec2-base")
                self.assertIsInstance(config, AudioModelConfig)

    def test_hubert_preset_aliases_return_valid_config(self) -> None:
        for name in (
            "hubert-base-ls960",
            "hubert_base_ls960",
            "hubert/base-ls960",
            "facebook/hubert-base-ls960",
        ):
            with self.subTest(name=name):
                spec = get_preset_spec(name)
                config = get_config(name)

                self.assertEqual(spec.name, "hubert-base-ls960")
                self.assertIsInstance(config, AudioModelConfig)

    def test_eat_preset_aliases_return_valid_config(self) -> None:
        for name in ("eat-base", "eat_base", "eat/base"):
            with self.subTest(name=name):
                spec = get_preset_spec(name)
                config = get_config(name)

                self.assertEqual(spec.name, "eat-base")
                self.assertIsInstance(config, AudioModelConfig)

    def test_list_presets_returns_canonical_names(self) -> None:
        self.assertEqual(
            list_presets(),
            ("dasheng-base", "eat-base", "hubert-base-ls960", "wav2vec2-base"),
        )

    def test_build_model_returns_audio_model(self) -> None:
        self.assertIsInstance(build_model("dasheng-base"), AudioModel)

    def test_unknown_preset_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown model preset"):
            get_config("missing-model")


class PretrainedApiTests(unittest.TestCase):
    def test_dasheng_pretrained_aliases_return_spec(self) -> None:
        for name in ("dasheng-base", "dasheng_base", "dasheng/base"):
            with self.subTest(name=name):
                spec = get_pretrained_spec(name)

                self.assertEqual(spec.name, "dasheng-base")
                self.assertEqual(spec.architecture, "dasheng-base")
                self.assertEqual(spec.checkpoint_filename, "dasheng_base.pt")

    def test_wav2vec2_pretrained_aliases_return_spec(self) -> None:
        for name in (
            "wav2vec2-base",
            "wav2vec2_base",
            "wav2vec2/base",
            "facebook/wav2vec2-base",
        ):
            with self.subTest(name=name):
                spec = get_pretrained_spec(name)

                self.assertEqual(spec.name, "wav2vec2-base")
                self.assertEqual(spec.architecture, "wav2vec2-base")
                self.assertEqual(spec.hf_repo_id, "facebook/wav2vec2-base")
                self.assertEqual(spec.checkpoint_filename, "model.safetensors")

    def test_hubert_pretrained_aliases_return_spec(self) -> None:
        for name in (
            "hubert-base-ls960",
            "hubert_base_ls960",
            "hubert/base-ls960",
            "facebook/hubert-base-ls960",
        ):
            with self.subTest(name=name):
                spec = get_pretrained_spec(name)

                self.assertEqual(spec.name, "hubert-base-ls960")
                self.assertEqual(spec.architecture, "hubert-base-ls960")
                self.assertEqual(spec.hf_repo_id, "facebook/hubert-base-ls960")
                self.assertEqual(spec.checkpoint_filename, "pytorch_model.bin")

    def test_eat_pretrained_aliases_return_spec(self) -> None:
        for name in ("eat-base", "eat_base", "eat/base"):
            with self.subTest(name=name):
                spec = get_pretrained_spec(name)

                self.assertEqual(spec.name, "eat-base")
                self.assertEqual(spec.architecture, "eat-base")
                self.assertEqual(spec.hf_repo_id, "worstchan/EAT-base_epoch30_pretrain")
                self.assertEqual(spec.checkpoint_filename, "model.safetensors")

    def test_list_pretrained_returns_canonical_names(self) -> None:
        self.assertEqual(
            list_pretrained(),
            ("dasheng-base", "eat-base", "hubert-base-ls960", "wav2vec2-base"),
        )

    def test_unknown_pretrained_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown pretrained model"):
            get_pretrained_spec("missing-model")

    def test_load_pretrained_builds_model_and_invokes_registered_loader(self) -> None:
        calls = []
        checkpoint_path = Path("checkpoint.pt")
        model = object()

        def load_fn(path, loaded_model, *, map_location="cpu") -> None:
            calls.append((path, loaded_model, map_location))

        spec = PretrainedModelSpec(
            name="fake-pretrained",
            aliases=(),
            architecture="fake-architecture",
            checkpoint_url=None,
            checkpoint_filename="fake.pt",
            sha256=None,
            load_fn=load_fn,
            description="test spec",
        )

        with (
            mock.patch("audiossl.pretrained.load.get_pretrained_spec", return_value=spec),
            mock.patch("audiossl.pretrained.load.build_model", return_value=model) as build,
            mock.patch(
                "audiossl.pretrained.load.resolve_checkpoint",
                return_value=checkpoint_path,
            ) as resolve,
        ):
            loaded_model, returned_spec = load_pretrained(
                "fake-pretrained",
                source=checkpoint_path,
                map_location="meta",
                config_kwargs={"target_length": 12},
            )

        self.assertIs(loaded_model, model)
        self.assertIs(returned_spec, spec)
        build.assert_called_once_with("fake-architecture", target_length=12)
        resolve.assert_called_once_with(spec, source=checkpoint_path, cache_dir=None)
        self.assertEqual(calls, [(checkpoint_path, model, "meta")])

    def test_resolve_checkpoint_accepts_local_path_and_verifies_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "weights.pt"
            checkpoint_bytes = b"pretend checkpoint"
            checkpoint_path.write_bytes(checkpoint_bytes)
            expected_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
            spec = _fake_spec(sha256=expected_sha)

            resolved = resolve_checkpoint(spec, source=checkpoint_path, cache_dir=None)

        self.assertEqual(resolved, checkpoint_path)

    def test_resolve_checkpoint_downloads_default_url_without_network_in_test(self) -> None:
        spec = get_pretrained_spec("dasheng-base")
        downloaded = Path("dasheng_base.pt")

        with mock.patch(
            "audiossl.pretrained.download.download_url_to_cache",
            return_value=downloaded,
        ) as download:
            resolved = resolve_checkpoint(spec, source=None, cache_dir="cache")

        self.assertEqual(resolved, downloaded)
        download.assert_called_once_with(
            spec.checkpoint_url,
            filename="dasheng_base.pt",
            cache_dir="cache",
            sha256=None,
        )

    def test_resolve_checkpoint_downloads_url_override_without_network_in_test(self) -> None:
        spec = _fake_spec()
        downloaded = Path("custom.pt")
        url = "https://example.test/checkpoints/custom.pt?download=1"

        with mock.patch(
            "audiossl.pretrained.download.download_url_to_cache",
            return_value=downloaded,
        ) as download:
            resolved = resolve_checkpoint(spec, source=url, cache_dir="cache")

        self.assertEqual(resolved, downloaded)
        download.assert_called_once_with(
            url,
            filename="custom.pt",
            cache_dir="cache",
            sha256=None,
        )

    def test_legacy_modelling_pretrained_shim_points_to_public_api(self) -> None:
        legacy = importlib.import_module("audiossl.modelling.pretrained")

        self.assertIs(legacy.load_pretrained, load_pretrained)

    def test_legacy_modelling_pretrained_models_do_not_import_placeholders(self) -> None:
        legacy_config = importlib.import_module(
            "audiossl.modelling.pretrained.models.config"
        )
        legacy_dasheng = importlib.import_module(
            "audiossl.modelling.pretrained.models.dasheng"
        )

        self.assertIsInstance(legacy_config.dasheng_base, AudioModelConfig)
        self.assertTrue(callable(legacy_dasheng.dasheng_base))


def _fake_spec(*, sha256: str | None = None) -> PretrainedModelSpec:
    def load_fn(path, model, *, map_location="cpu") -> None:
        del path, model, map_location

    return PretrainedModelSpec(
        name="fake-pretrained",
        aliases=(),
        architecture="fake-architecture",
        checkpoint_url="https://example.test/fake.pt",
        checkpoint_filename="fake.pt",
        sha256=sha256,
        load_fn=load_fn,
        description="test spec",
    )


if __name__ == "__main__":
    unittest.main()

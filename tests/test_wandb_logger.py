"""Offline-mode end-to-end tests for the WandB logger callback."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import wandb
from dataclasses import replace

from audiossl.config import CheckpointConfig, WandBConfig

from audiotrainer import Callback, WandBLogger

from harness import CONTRASTIVE


def _reset_wandb_module_state() -> None:
    """Force-finish any active wandb run and tear down the module singleton.

    Without this, state from a prior test (settings cache, internal subprocess
    handle, run id) leaks into the next test and the new ``wandb.init`` call
    silently no-ops or files end up in the wrong directory.
    """
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
    teardown = getattr(wandb, "teardown", None)
    if teardown is not None:
        try:
            teardown()
        except Exception:
            pass


class _OfflineWandBFixture:
    """Context manager that isolates wandb state inside a temp dir.

    Wandb keeps module-level state across `init`/`finish`; we run each test in
    a fresh temp dir with ``WANDB_MODE=offline`` so nothing reaches the
    network and the offline run files are scoped to one test. Cleanup uses
    ``ignore_cleanup_errors=True`` because wandb's background subprocess may
    still hold a log file open on Windows when the test body returns.
    """

    def __init__(self) -> None:
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._env: dict[str, str | None] = {}

    def __enter__(self) -> Path:
        _reset_wandb_module_state()
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        path = Path(self._tmpdir.name)
        for key, value in {
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(path),
            "WANDB_SILENT": "true",
            "WANDB_CONSOLE": "off",
            "WANDB_DISABLE_GIT": "true",
        }.items():
            self._env[key] = os.environ.get(key)
            os.environ[key] = value
        return path

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _reset_wandb_module_state()
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


class WandBLoggerConstructionTests(unittest.TestCase):
    def test_rejects_non_wandb_config(self) -> None:
        cfg = CONTRASTIVE.training_config()
        with self.assertRaisesRegex(TypeError, "WandBConfig"):
            WandBLogger(config=object(), train_config=cfg)  # type: ignore[arg-type]

    def test_rejects_non_training_config(self) -> None:
        with self.assertRaisesRegex(TypeError, "AudioSSLTrainingConfig"):
            WandBLogger(
                config=WandBConfig(project="x", mode="disabled"),
                train_config=object(),  # type: ignore[arg-type]
            )

    def test_resume_enabled_allocates_a_fresh_run_id(self) -> None:
        logger = WandBLogger(
            config=WandBConfig(project="x", mode="disabled"),
            train_config=CONTRASTIVE.training_config(),
        )
        state = SimpleNamespace(
            config_fingerprint="abcdef123456", data_fingerprint="harness/waveforms"
        )

        with patch("wandb.init") as init:
            logger.on_train_start(state)

        run_id = init.call_args.kwargs["id"]
        self.assertIsInstance(run_id, str)
        self.assertEqual(len(run_id), 32)
        self.assertNotIn("resume", init.call_args.kwargs)

    def test_resume_disabled_lets_wandb_create_fresh_run(self) -> None:
        logger = WandBLogger(
            config=WandBConfig(project="x", mode="disabled", resume=False),
            train_config=CONTRASTIVE.training_config(),
        )
        state = SimpleNamespace(
            config_fingerprint="abcdef123456",
            data_fingerprint="harness/waveforms",
        )

        with patch("wandb.init") as init:
            logger.on_train_start(state)

        run_id = init.call_args.kwargs["id"]
        self.assertIsInstance(run_id, str)
        self.assertEqual(len(run_id), 32)


class _RunCapture(Callback):
    """Capture wandb state from inside the run.

    ``wandb.init`` rebinds the module-level ``wandb.log`` to a new closure
    after init runs, so any pre-init patch is lost. Patching the
    module attributes from inside ``on_train_start`` (which fires AFTER the
    auto-attached ``WandBLogger.on_train_start``) is the only reliable way to
    intercept the calls.
    """

    def __init__(self) -> None:
        self.run_id_at_train_start: str | None = None
        self.log_calls: list[tuple[dict, int | None]] = []
        self.artifact_calls: list = []
        self._original_log = None
        self._original_log_artifact = None

    def on_train_start(self, state) -> None:
        if wandb.run is not None:
            self.run_id_at_train_start = wandb.run.id
        self._original_log = wandb.log
        captured_log = self.log_calls

        def _capture_log(*args, **kwargs):
            data = args[0] if args else kwargs.get("data")
            step = kwargs.get("step")
            captured_log.append((dict(data), step))
            return self._original_log(*args, **kwargs)

        wandb.log = _capture_log

        self._original_log_artifact = wandb.log_artifact
        captured_artifact = self.artifact_calls

        def _capture_artifact(artifact, *args, **kwargs):
            captured_artifact.append((artifact, args, kwargs))
            return None

        wandb.log_artifact = _capture_artifact

    def on_train_end(self, state) -> None:
        if self._original_log is not None:
            wandb.log = self._original_log
            self._original_log = None
        if self._original_log_artifact is not None:
            wandb.log_artifact = self._original_log_artifact
            self._original_log_artifact = None


class WandBLoggerOfflineEndToEndTests(unittest.TestCase):
    def test_run_initializes_with_fresh_uuid_id(self) -> None:
        with _OfflineWandBFixture():
            capture = _RunCapture()
            run = CONTRASTIVE.run(num_batches=2)
            run.run_config = replace(
                run.run_config,
                loggers=(WandBConfig(project="audiossl-test", mode="offline"),),
            )
            trainer = run.trainer(capture)
            trainer.fit()

            self.assertIsNotNone(capture.run_id_at_train_start)
            self.assertEqual(len(capture.run_id_at_train_start), 32)

    def test_aggregate_over_steps_emits_one_log_per_window(self) -> None:
        with _OfflineWandBFixture():
            capture = _RunCapture()
            run = CONTRASTIVE.run(num_batches=4)
            run.run_config = replace(
                run.run_config,
                loggers=(
                    WandBConfig(
                        project="audiossl-test",
                        mode="offline",
                        log_every_n_steps=2,
                        aggregate_over_steps=True,
                    ),
                ),
            )
            # 4 batches -> 4 optimizer steps -> 2 log emissions at steps 2 and 4.
            run.trainer(capture).fit()

            steps = [step for _, step in capture.log_calls]
            self.assertEqual(steps, [2, 4])
            for payload, _ in capture.log_calls:
                self.assertIn("train/loss", payload)
                self.assertIn("train/contrastive/positive_loss", payload)


class WandBLoggerCheckpointUploadTests(unittest.TestCase):
    def _fit_with_checkpoints(self, checkpoint_dir: Path, *, upload: bool) -> _RunCapture:
        capture = _RunCapture()
        run = CONTRASTIVE.run(
            num_batches=2,
            checkpoint=CheckpointConfig(save_dir=str(checkpoint_dir), save_every_n_epochs=1),
        )
        run.run_config = replace(
            run.run_config,
            loggers=(
                WandBConfig(
                    project="audiossl-test",
                    mode="offline",
                    upload_checkpoints=upload,
                ),
            ),
        )
        run.trainer(capture).fit()
        return capture

    def test_upload_checkpoints_invokes_artifact_log(self) -> None:
        with _OfflineWandBFixture() as tmp:
            checkpoint_dir = tmp / "ckpt"
            capture = self._fit_with_checkpoints(checkpoint_dir, upload=True)

            # Checkpoint writes last.pt and epoch_NNNN.pt per epoch.
            self.assertEqual(len(capture.artifact_calls), 2)
            for name in ("last.pt", "epoch_0001.pt"):
                self.assertTrue((checkpoint_dir / name).exists(), msg=name)

    def test_no_upload_when_flag_is_off(self) -> None:
        with _OfflineWandBFixture() as tmp:
            capture = self._fit_with_checkpoints(tmp / "ckpt", upload=False)

            self.assertEqual(capture.artifact_calls, [])


if __name__ == "__main__":
    unittest.main()

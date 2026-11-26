from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch

from audiossl.config import WandBConfig
from audiotrainer import Callback, Trainer, WandBLogger

from harness import assert_trees_equal
from harness.objectives import WAV2VEC2


def make_trainer(run, *callbacks):
    return Trainer(
        run.model,
        run.objective,
        None,
        replace(
            run.config,
            trainer=replace(run.config.trainer, loop=replace(run.config.trainer.loop, max_steps=3)),
        ),
        run_config=run.run_config,
        data_fingerprint=run.data_fingerprint,
        callbacks=callbacks,
    )


def masked_frames(output):
    return output.time_mask_indices.sum().item()


def test_update_uses_masked_frame_weighted_gradients_and_metrics():
    run = WAV2VEC2.run(num_batches=2)
    first, second = run.loader
    # A smaller second microbatch exercises unequal denominators.
    batches = [first, replace(second, x=second.x[:1, :3200], mask=second.mask[:1, :3200])]
    trainer = make_trainer(run)
    with trainer.update_session():
        initial = trainer.snapshot()
        outputs = [trainer.objective(batch) for batch in batches]
        counts = [masked_frames(output) for output in outputs]
        assert counts[0] != counts[1]
        loss = sum(
            output.loss * count for output, count in zip(outputs, counts, strict=True)
        ) / sum(counts)
        contrastive = sum(
            output.contrastive_loss.item() * count
            for output, count in zip(outputs, counts, strict=True)
        ) / sum(counts)
        params = [param for param in trainer.objective.parameters() if param.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        trainer.restore(initial)

        def verify_gradients(*_):
            for param, expected in zip(params, grads, strict=True):
                torch.testing.assert_close(param.grad, expected, atol=1e-6, rtol=1e-5)

        with trainer.optimizer.register_step_pre_hook(verify_gradients):
            assert trainer.train_update(batches, loss_weight=masked_frames, learning_rate=2e-3)
        assert trainer.state.optimizer_step == 1
        assert trainer.state.global_step == 2
        assert trainer.state.last_loss == pytest.approx(loss.item())
        assert trainer.state.last_metrics["train/wav2vec2/contrastive_loss"] == pytest.approx(
            contrastive
        )
        assert trainer.state.last_metrics["train/lr"] == 2e-3


def test_snapshot_restores_branch_cursor_optimizer_and_rng():
    run = WAV2VEC2.run(num_batches=3)
    first, stable, decay = run.loader
    trainer = make_trainer(run)
    with trainer.update_session():
        trainer.train_update([first])
        cursor = {"positions": [1]}
        fork = trainer.snapshot(data_state=cursor)
        cursor["positions"].append(2)
        trainer.train_update([stable, first], loss_weight=masked_frames)
        expected = trainer.snapshot()

        assert trainer.restore(fork) == {"positions": [1]}
        trainer.train_update([decay], learning_rate=0)
        restored_cursor = trainer.restore(fork)
        restored_cursor["positions"].append(3)
        assert fork["data_state"] == {"positions": [1]}
        trainer.train_update([stable, first], loss_weight=masked_frames)
        actual = trainer.snapshot()

    for key in ("objective_state", "optimizer_state", "scheduler_state", "rng"):
        assert_trees_equal(actual[key], expected[key], label=key)
    assert actual["optimizer_step"] == 2
    assert actual["global_step"] == 3
    assert actual["update_state"]["last_loss"] == expected["update_state"]["last_loss"]


def test_branch_artifact_and_metrics_use_caller_owned_run(tmp_path):
    class Observe(Callback):
        def on_checkpoint_written(self, state, path):
            events.append((state.optimizer_step, path))

    events = []
    run = WAV2VEC2.run(num_batches=1)
    (batch,) = run.loader
    handle = Mock(id="existing-run")
    logger = WandBLogger(
        config=WandBConfig(project="test", upload_checkpoints=True),
        train_config=run.config,
        run=handle,
    )
    trainer = make_trainer(run, logger, Observe())
    bundle = tmp_path / "branch-0.125"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}", encoding="utf-8")

    with patch("wandb.init") as init, patch("wandb.Artifact") as artifact:
        with trainer.update_session():
            fork = trainer.snapshot()
            trainer.train_update([batch])
            trainer.announce_artifact(bundle)
            trainer.restore(fork)
            trainer.train_update([batch])

    assert events == [(1, bundle)]
    artifact.return_value.add_dir.assert_called_once_with(str(bundle))
    handle.log_artifact.assert_called_once_with(
        artifact.return_value, aliases=[bundle.name, "step-1"]
    )
    assert [call.args[0]["train/optimizer_step"] for call in handle.log.call_args_list] == [1, 1]
    assert all(not call.kwargs for call in handle.log.call_args_list)
    init.assert_not_called()
    handle.finish.assert_not_called()

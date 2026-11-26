# realssl

One installable library with two import packages:

- **`audiossl`** — encoders, processors, objectives, data, and shared config types
- **`audiotrainer`** — training loop, callbacks, optimizers, and augmentation

Imports stay separate (`import audiossl`, `import audiotrainer`). Modelling code does not depend on the trainer at import time; the trainer depends on `audiossl` at runtime.

## Install

```bash
cd real-ssl
uv sync
```

## Layout

```text
src/
  audiossl/
  audiotrainer/
tests/          # audiossl + audiotrainer suites
docs/
```

## Supervised fine-tuning (SFT)

SSL training uses waveform-only batches and objectives such as `contrastive` or `jepa`. For **classification SFT**, use the `sft` objective with labeled batches (`collate_labeled_fn` or `AudioBatch.labels`).

You are responsible for loading a pretrained backbone before training (for example with `audiossl.pretrained.load_pretrained`). Pass the built model into the trainer:

```python
from audiossl.config import load_config
from audiossl.pretrained import load_pretrained
from audiotrainer import build_trainer, collate_labeled_fn

train_config = load_config("sft.json")
model, _spec = load_pretrained("dasheng-base")  # architecture must match train_config.model
trainer = build_trainer(train_config, data_module, model=model)
trainer.fit()
```

Set `objective.freeze_backbone: true` for linear probe training, or tune `backbone_lr_scale` for differential learning rates. Validation logs `val/loss` and `val/accuracy` when `RunConfig.evaluation` is enabled.

## Caller-defined training updates

Use `Trainer` with `train_loader=None` and set `train_config.trainer.loop.max_steps` to the scheduler horizon. Within one `update_session()`, pass each optimizer update as a group of microbatches. The objective supplies a mean loss per microbatch. For wav2vec2, weight that mean by its masked-frame count.

```python
from audiotrainer import Trainer


def masked_frames(output):
    return output.time_mask_indices.sum().item()


trainer = Trainer(model, objective, None, train_config, data_fingerprint=data_revision)
with trainer.update_session():
    trainer.train_update(microbatches, loss_weight=masked_frames)
    fork = trainer.snapshot(data_state=stream.state())
    trainer.train_update(decay_microbatches, loss_weight=masked_frames, learning_rate=decay_lr)
    export_branch(trainer.state, branch_dir)
    trainer.announce_artifact(branch_dir)
    stream.load_state_dict(trainer.restore(fork))
```

The caller controls update windows, stopping, and the data cursor. Snapshots restore optimizer, scheduler, and RNG state between updates. Artifact notifications go to `on_checkpoint_written` after the caller finalizes the file or directory.

Supply an existing W&B run as `WandBLogger(..., run=wandb_run)` to keep control of its lifecycle. The logger records optimizer step as a metric so custom chart axes can show repeated steps after a restore.

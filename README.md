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
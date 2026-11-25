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
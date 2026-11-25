# Augmentation contract

An augmentation transforms raw waveform values during training. Augmentations are
composed sequentially by `AugmentationPipeline` and are not part of the inference
pipeline.

There are two boundaries:

- `AugmentationPipeline` accepts and returns `AudioBatch`.
- Individual augmentations accept and return `list[torch.Tensor]`.

The pipeline owns conversion between padded `AudioBatch` tensors and variable
length waveform lists. Individual augmentations do not accept `AudioBatch`,
modify the batch mask, or operate on padded 2-D tensors directly.

This contract currently covers length-preserving mono waveform augmentations. An
augmentation must not add samples, remove samples, crop, resample, time-stretch,
or otherwise change waveform length unless this contract is updated first.

---

## 1. Class shape

```python
class MyAugmentation:
    required_parameters: tuple[str, ...] = ("p", ...)

    def __init__(
        self,
        config: Mapping[str, Any],
        generator: torch.Generator | None = None,
    ) -> None: ...

    def __call__(
        self,
        x: list[torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> list[torch.Tensor]: ...
```

`AugmentationPipeline` accepts `AudioAugmentorConfig` whose entries are typed
augmentation configs (e.g. `LoudnessAugmentationConfig`). Individual
augmentation classes may still accept a mapping of research parameters when
constructed directly. Pipeline construction dispatches through the
`AUGMENTATIONS` registry keyed by config `type`.

### Constructor rules

- Validate all required parameters are present; raise `ValueError` listing the
  missing names.
- Validate parameter types, numeric ranges, finite-value requirements, and
  cross-field constraints; raise `ValueError` with the offending field and
  constraint.
- Copy or extract only the values the augmentation needs. Do not retain a
  mutable config object and depend on later external mutation.
- If a constructor `generator` is accepted, store it as a per-instance fallback.
  `__call__` must prefer its `generator` argument over the constructor fallback.
  Pipeline-created augmentations normally receive randomness through
  `__call__`, because the pipeline instantiates registry entries as
  `registry[key](params_mapping)`.

### `__call__` rules

- Input must be `list[torch.Tensor]`; raise `TypeError` if `x` is not a list or
  any element is not a `torch.Tensor`.
- Each tensor must have `ndim == 1`; raise `ValueError` if a waveform has any
  other rank.
- Output list length and each tensor's shape must equal the input.
- Output dtype and device must equal the corresponding input dtype and device.
- Do not mutate input tensors in-place. Returning the original tensor object is
  allowed only when the sample is intentionally unchanged, such as a probability
  skip or `p == 0.0`.
- Apply the augmentation independently per waveform. Do not read, aggregate, or
  mix information across samples.
- Respect `p` as an independent application probability per waveform. `p == 0.0`
  means the probability gate never applies the augmentation. `p == 1.0` means the
  probability gate never skips it; the mathematical transform may still be an
  identity for degenerate parameters or inputs.
- All randomness must pass through the supplied `torch.Generator` when one is
  provided. With no supplied or stored generator, PyTorch's default RNG may be
  used.

---

## 2. Pipeline boundary

`AugmentationPipeline.__call__` accepts `AudioBatch` and returns a newly collated
`AudioBatch`.

Pipeline responsibilities:

- Validate that the top-level input is an `AudioBatch`.
- Use `batch.mask` to recover each unpadded waveform from `batch.x`.
- Apply configured augmentations sequentially in config order.
- Pass the same effective call generator to each augmentation in that pipeline
  call. A call-time generator overrides the pipeline constructor generator.
- Re-collate augmented waveforms into an `AudioBatch`.

Current `collate_fn` behavior is CPU `torch.float32` output. Until collation is
changed, the pipeline-level batch output should be treated as CPU/float32 even
though individual augmentation outputs must preserve their own input dtype and
device.

The pipeline must not rely on augmentations mutating their inputs. Contrastive
training may call the same pipeline repeatedly on the same source batch to
produce independent views.

---

## 3. Parameter spec schema

Every registered augmentation must have a matching entry in
`_AUGMENTATION_PARAMETERS` in [`augmentor.py`](augmentor.py). Each parameter is
described by a spec dict:

```python
{
    "required": bool,          # whether the key must appear in config
    "min": float | int,        # inclusive lower bound (optional)
    "max": float | int,        # inclusive upper bound (optional)
    "choices": tuple[...],     # allowed discrete values (optional)
}
```

At minimum, every augmentation must declare a `"p"` spec with `min: 0.0`,
`max: 1.0`, and `required: True`.

`AugmentationPipeline._validate_augmentations` validates registry-level specs
before construction, including unknown augmentation names, unknown parameters,
missing required parameters, range checks, and discrete choices. Constructors
remain responsible for validation that the schema cannot express, including
parameter type checks, NaN/finite policy, and cross-field constraints such as
`min_gain_db <= max_gain_db`.

`required_parameters` and the spec entries marked `"required": True` must stay
in sync for every augmentation.

---

## 4. Registry registration

To make an augmentation available to `AugmentationPipeline`:

1. Add the class to `_get_augmentation_registry()` in [`augmentor.py`](augmentor.py)
   under the lowercase string key used in configs.
2. Add the parameter spec under the same key in `_AUGMENTATION_PARAMETERS`.
3. Export the class from [`__init__.py`](__init__.py).
4. Add focused tests for constructor validation, probability behavior,
   generator-driven determinism, shape preservation, dtype/device preservation
   where applicable, and malformed input errors.

Augmentation config shape is:

```python
[
    {"loudness": {"p": 1.0, "min_gain_db": -6.0, "max_gain_db": 6.0}},
]
```

Each mapping entry is one augmentation registration key mapped to that
augmentation's parameter mapping. Augmentations run in the order produced by the
outer sequence and then each mapping's insertion order. Prefer one augmentation
per mapping for readability and deterministic review.

The pipeline instantiates augmentations as `registry[key](params_mapping)`. Do
not require positional constructor arguments beyond the config mapping.

---

## 5. Invariants

| # | Invariant |
|---|-----------|
| 1 | `len(output) == len(input)` always. |
| 2 | `output[i].shape == input[i].shape` for all `i`. |
| 3 | `output[i].dtype == input[i].dtype` for all `i` at the augmentation boundary. |
| 4 | `output[i].device == input[i].device` for all `i` at the augmentation boundary. |
| 5 | Inputs are not modified in-place. |
| 6 | When `p == 0.0`, all output tensors are value-identical to their inputs and no random numbers are consumed. Returning the original tensor object is allowed. |
| 7 | When `p == 1.0`, the probability gate always applies the transform path. |
| 8 | Given equal input tensors and equal initial `torch.Generator` states, output values are deterministic for the same device/backend. |

---

## 6. Error contract

| Condition | Exception | Message must include |
|---|---|---|
| Missing required parameter | `ValueError` | parameter name(s) |
| Unknown augmentation name | `ValueError` | augmentation name and available names |
| Unknown parameter | `ValueError` | parameter name and available parameters |
| Parameter has wrong type | `ValueError` | field name and expected type |
| Parameter out of range | `ValueError` | field name and constraint |
| Parameter violates finite/NaN policy | `ValueError` | field name and finite/NaN constraint |
| Cross-field inconsistency | `ValueError` | both field names |
| Pipeline input is not `AudioBatch` | `TypeError` | received type |
| Augmentation input `x` is not a list | `TypeError` | received type |
| Augmentation list element is not `torch.Tensor` | `TypeError` | element index and received type |
| Waveform `ndim != 1` | `ValueError` | element index and actual shape |

---

## Cross-references

- Pipeline: [`augmentor.py`](augmentor.py)
- Existing implementation: [`waveform.py`](waveform.py) (`LoudnessAugmentation`)
- Batch type: [`src/audiossl/config/batch_dataclasses.py`](../../config/batch_dataclasses.py) (`AudioBatch`)

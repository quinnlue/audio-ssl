# Modelling Layer Contract

This is the cross-component contract for `src/audiossl/modelling/`. Category contracts add shape details:

| Component | Contract |
| --- | --- |
| Processor | [processors/CONTRACT.md](processors/CONTRACT.md) |
| Tokenizer | [tokenizers/CONTRACT.md](tokenizers/CONTRACT.md) |
| Embedding | [embeddings/CONTRACT.md](embeddings/CONTRACT.md) |
| Encoder | [encoders/CONTRACT.md](encoders/CONTRACT.md) |
| Objective | [objectives/CONTRACT.md](objectives/CONTRACT.md) |

## Pipeline

```text
AudioBatch (raw waveform)
  -> Processor   -> AudioBatch (features)
  -> Tokenizer   -> AudioBatch (tokens, input_dim)
  -> Embedding   -> AudioBatch (tokens, d_model)
  -> Encoder     -> EncoderOutput
```

Training calls `objective.forward(batch)`; objectives own training-specific masking, views, targets, losses, and teacher logic. Inference may call `AudioModel.forward`, which delegates to `pipeline`.

## Responsibility Table

| Layer | Owns | Does Not Own |
| --- | --- | --- |
| Processor | waveform-to-feature conversion, feature masks, mel intermediate APIs | tokenization, PE, `d_model` projection, loss |
| Tokenizer | feature-to-token reshape, token masks, `TokenLayout`, patch target order | PE, projection, attention |
| Embedding | `tokenizer.output_dim -> d_model` projection, embedding-side PE, optional embedding subsampling | encoder blocks, pooling, objective masks |
| Encoder | block stack, optional per-block RoPE, final norm, `EncoderOutput` sequence states | entry/absolute PE, pooling, loss |
| Objective | pretraining or downstream objective orchestration | component construction beyond using `AudioModel` |

## Positional Encoding

Absolute/entry positional encoding lives on embedding configs:

```python
EmbeddingPositionalEncodingConfig = (
    NoPositionalEncodingConfig
    | LearnablePositionalEncodingConfig
    | SinusoidalPositionalEncodingConfig
    | LearnableTimeFrequencyPositionalEncodingConfig
    | ConvolutionalPositionalEncodingConfig
)
```

Every `LinearEmbeddingConfig` and `ConvEmbeddingConfig` must explicitly set `positional_encoding`. Use `NoPositionalEncodingConfig()` when PE is disabled.
Special architecture stacks such as wav2vec 2.0 must still use explicit generic
embedding fields and PE variants; they must not hide PE in a private embedding
variant.

Encoder configs only accept optional per-block PE:

```python
PerBlockPositionalEncodingConfig = RopePositionalEncodingConfig
```

There is no `pe_placement` field. The encoder wrapper must not instantiate or apply entry PE.

## Core Batch Types

`AudioBatch`:

| Field | Contract |
| --- | --- |
| `x` | rank 2 `(B, T_raw)` at pipeline entry; rank 3 after processor |
| `mask` | `torch.bool`, same time axis as `x`; `True` means valid |
| `positions` | optional integer tensor shaped like `mask`; index of each token in the full undropped sequence, prefix tokens included (`None` means `arange`). Set when tokens were dropped or reordered; consumed by RoPE |
| `num_prefix_tokens` | number of leading non-content tokens (for example CLS) added by the embedding |

`AudioBatch` is frozen. Derive modified batches with `dataclasses.replace`;
never mutate `x` or `mask` in place.

`EncoderOutput`:

| Field | Contract |
| --- | --- |
| `z` | `(B, N_tokens, d_model)` |
| `mask` | `(B, N_tokens)`, boolean |
| `lengths` | `mask.sum(dim=1)` |
| `cls` | optional legacy cache; consumers should call `AudioModel.get_cls(output)` |

`TokenLayout` is the public tokenizer layout record for layout-aware PE and
objectives. It records token count, time/frequency patch counts, patch shape,
stride, flatten order, output dim, and source feature shape.

## AudioModel Stages

`AudioModel` exposes the canonical staged API:

```python
process(batch) -> AudioBatch
token_layout(features) -> TokenLayout
tokenize(features) -> AudioBatch
embed(tokens, *, layout=None) -> AudioBatch
encode_tokens(tokens, *, layout=None, return_hidden_states=False) -> EncoderOutput
encode_embedded(embedded, *, return_hidden_states=False) -> EncoderOutput
describe() -> str
```

`model.embedding` additionally exposes `prepare`, `project`, `position`, and
`finalize` (see [embeddings/CONTRACT.md](embeddings/CONTRACT.md)), plus
`num_prefix_tokens`, `subsampling_factor`, and `output_mask`.

Objectives may orchestrate these stages, but must not reach through wrapper
internals to access nested component attributes.

`pooling="cls"` requires an embedding that adds a prefix token;
`AudioModel.__init__` raises `ValueError` otherwise.

## Config Discipline

- Variant configs are frozen kw-only dataclasses with `type: Literal[...]`.
- Union types use PEP 604 syntax.
- Components with variable output feature dimensions expose `output_dim` as a property.
- Cross-component invariants are enforced in `validate_config`. Objective/model
  compatibility is expressed as capabilities (`audiossl.config.capabilities`):
  component configs declare `capabilities`, objective configs declare
  `required_capabilities` / `incompatible_capabilities`.
- Components are built from their own config section only; a component never
  reads another component's config (for example the processor has no `d_model`).
- New variants require config, validator, registry entry, implementation, and
  tests. They can live outside the library: `register_config(category)`,
  `register_validator(cls)` (from `audiossl.config`), and
  `<REGISTRY>.register(type, cls)` (from `audiossl.modelling.registry`) make a
  plug-in loadable from JSON, validated, and buildable. See
  `tests/contracts/test_plugin_registration.py`.

## Runtime

`AudioModel.__init__`:

1. Deep-copies and validates config.
2. Configures runtime.
3. Constructs processor, embedding, and encoder each inside
   `runtime_seed_scope(runtime, "model/<component>")`. The scope seeds from
   `derive_seed(seed, component)`, so changing one component's config never
   changes another component's initialization, and construction leaves the
   global RNG untouched. Objectives use `"objective/<type>"` scopes.
4. Applies runtime dtype/device to the full module tree.
5. Logs `describe()` (per-component type, class, parameter count) at `INFO`.

Components do not validate device placement at boundaries. Processor sample rate is a config declaration; `AudioBatch` carries no sample-rate field.

## Registries

`audiossl.modelling.registry` owns one `ComponentRegistry` per category:

| Registry | Accessor | Dispatch |
| --- | --- | --- |
| `PROCESSORS` | `get_processor_registry()` | processor classes |
| `TOKENIZERS` | `get_tokenizer_registry()` | tokenizer classes |
| `EMBEDDINGS` | `get_embedding_registry()` | embedding classes |
| `EMBEDDING_POSITIONAL_ENCODINGS` | `get_embedding_positional_encoding_registry()` | embedding PE modules |
| `ENCODER_BLOCKS` | `get_encoder_block_registry()` | encoder block classes |
| `OBJECTIVES` | `get_objective_registry()` | objective classes |
| `MASKING_STRATEGIES` | `get_masking_strategy_registry()` | masking strategies |

Built-ins load lazily. `register(name, cls)` (also usable as a decorator)
raises `ValueError` on a duplicate name unless `overwrite=True`; unknown
lookups raise `KeyError` listing every registered name.

Activation and initialization names must route through `ACTIVATION_REGISTRY`
(name to module factory; `build_activation(name)` returns a fresh module) and
`INITIALIZATION_REGISTRY`.

## Closed Objective Hook List

Objectives may expose only the hook names documented here and in
[`objectives/CONTRACT.md`](objectives/CONTRACT.md):

| Hook | Signature | Owner rule |
| --- | --- | --- |
| `metrics` | `(output, ctx) -> dict[str, torch.Tensor]` | Returns objective-specific diagnostics. The trainer supplies the namespace and training telemetry. |
| `OBJECTIVE_HOOKS` lifecycle methods | see objectives contract §4.5 | No-ops on the base class; override as needed. `on_after_forward` only for forward-dependent running statistics such as DINO centering. |
| `update_teacher` | `() -> None` | Only objectives with an EMA teacher path; called from `on_after_optimizer_step`. |

## Validation Checklist

1. Component subclasses `nn.Module`.
2. Config is a frozen kw-only variant dataclass with literal `type`.
3. Validator covers static config fields and cross-component invariants.
4. Registry entry is keyed by `config.type`.
5. `forward` boundary checks validate type, rank, shape, and mask dtype.
6. Masks use `True == valid`.
7. Encoder does not pool; `AudioModel.get_cls` owns pooling.
8. Entry PE is embedding-owned; encoder PE is per-block only.
9. Objective hook surface remains explicit in the closed objective hook list.

## Research regression tests

The suite checks the numerical trajectory of small seeded runs. Training
parity executes three SGD steps on both paths and compares the forward output
or loss, every trainable parameter gradient, the input gradient, and the
updated parameters. The input and probe vary across steps. A missing gradient
or a non-finite value fails with the parameter name and step.

| Test | Reference and coverage |
| --- | --- |
| `tests/contracts/test_encoder_parity.py` | Independent PyTorch equations for all registered encoder families, with ragged masks, explicit RoPE positions, and all three Branchformer merge modes |
| `tests/contracts/test_frontend_parity.py` | PyTorch convolution, patch extraction, time-frequency embeddings, and masked BatchNorm, plus torchaudio Kaldi filterbanks for the EAT front end |
| `tests/integrations/{hubert,wav2vec2}/` | Randomly initialized Hugging Face speech backbones and the wav2vec 2.0 Gumbel quantizer. These cases run offline with the locked dependencies |
| `tests/contracts/test_objective_parity.py` | Reference forward graphs and criteria for every registered SSL objective, plus DINO center, EMA teacher, and quantizer temperature updates |
| `tests/reproducibility/` | Seed isolation, numerical fingerprints, and exact continuation after saving and reloading objective, optimizer, and RNG state |

The objective references reuse the component modules for representation
learning, heads, and discrete samplers. They replay JEPA and MAE mask choices
and rebuild decoder inputs by scattering visible tokens into the original
sequence. The wav2vec 2.0 objective reference uses within-example negatives.
Checkpoint-level EAT and Dasheng comparisons require the external reference
or checkpoint. The Dasheng-style `overlap_any` frame-count discrepancy is
tracked by a strict expected failure.

Run the modelling suites with:

```sh
uv run --locked pytest tests/contracts tests/unit tests/reproducibility tests/integrations
```

The FP32 parity defaults are `rtol=1e-4`, `atol=1e-6`, and `eps=1e-5`. For
FP64 they are `1e-8`, `1e-10`, and `1e-6`. Use `--parity-rtol`,
`--parity-atol`, and `--parity-eps` to override these values. Epsilon sets
the normalization constant on both paths where the test config exposes it.
Resume checks compare exact values. Golden fingerprints run with the four CPU
threads used for the checked-in baseline.

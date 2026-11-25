# Embedding Component Contract

Embeddings bridge tokenizer output to encoder hidden size. They are the first parametric token step and own embedding-side positional encoding.

## Pipeline Position

```text
AudioBatch (tokens, input_dim) -> Embedding -> AudioBatch (tokens, d_model) -> Encoder
```

Input:

| Field | Shape | Contract |
| --- | --- | --- |
| `batch.x` | `(B, N_tokens, input_dim)` | floating tensor |
| `batch.mask` | `(B, N_tokens)` | `torch.bool` |

Output:

| Field | Shape | Contract |
| --- | --- | --- |
| `batch.x` | `(B, num_prefix_tokens + N_out, d_model)`; `N_out = ceil(N_tokens / subsampling_factor)` | projected token states |
| `batch.mask` | same token axis as output | unchanged by PE; subsampling selects strided rows; prefix tokens are valid |
| `batch.positions` | same shape as `batch.mask`, or `None` (meaning `arange`) | index of each token in the full undropped sequence, prefix tokens included |
| `batch.num_prefix_tokens` | `int` | leading non-content tokens (CLS for EAT) |

## Staged API

`BaseEmbedding` splits `forward` into four public stages, and
`forward(batch, layout=...)` is exactly
`finalize(position(project(prepare(batch)), layout=layout))`:

| Stage | Contract |
| --- | --- |
| `prepare(batch)` | boundary checks (below) and optional input LayerNorm; shape-preserving |
| `project(batch)` | `(B, N, input_dim) -> (B, N, d_model)` |
| `position(batch, *, layout=None)` | positional encoding plus any post-PE norm; requires the full token grid |
| `finalize(batch)` | dropout, subsampling, prefix tokens; safe on a token subset when `batch.positions` gives the kept indices |

Masking objectives substitute mask tokens before `position` and drop tokens
between `position` and `finalize`. Properties `num_prefix_tokens`,
`subsampling_factor`, and `output_mask(token_mask)` let callers map token-axis
masks onto the output axis without knowing the concrete embedding. New
embeddings implement `project` and `finalize` (and override `prepare` /
`position` only when they add stages).

## Config

```python
EmbeddingConfig = (
    LinearEmbeddingConfig
    | ConvEmbeddingConfig
)
EmbeddingPositionalEncodingConfig = (
    NoPositionalEncodingConfig
    | LearnablePositionalEncodingConfig
    | SinusoidalPositionalEncodingConfig
    | LearnableTimeFrequencyPositionalEncodingConfig
    | ConvolutionalPositionalEncodingConfig
)
```

General-purpose embedding configs must explicitly provide:

```python
positional_encoding: EmbeddingPositionalEncodingConfig
```

No default PE is implied. Use `NoPositionalEncodingConfig()` when PE is intentionally disabled.
Special-purpose architecture stacks must be represented by generic embedding
configuration fields plus an explicit `positional_encoding` variant.

## Positional Encoding

Embedding PE is applied after the projection has produced `(B, N_tokens, d_model)` and before dropout/subsampling. The mask is not mutated by PE.

Every PE module has the signature `forward(x, layout=None, *, mask=None)`.

Supported embedding PE modules:

| Config `type` | Module | Contract |
| --- | --- | --- |
| `"none"` | `NoPositionalEncoding` | identity |
| `"learnable"` | `LearnablePositionalEncoding` | 1D absolute PE |
| `"sinusoidal"` | `SinusoidalPositionalEncoding` | 1D sinusoidal PE |
| `"sinusoidal_2d"` | `Sinusoidal2DPositionalEncoding` | fixed 2D PE; **requires** `layout` |
| `"learnable_2d"` | `LearnableTimeFrequencyPositionalEncoding` | separable time + frequency PE on a time-major patch grid; **requires** `layout` |
| `"conv1d"` | `ConvolutionalPositionalEncoding` | grouped Conv1d positional residual used by wav2vec 2.0; zero-fills invalid tokens (`mask`) before the convolution so padding never reaches valid outputs |

Layout-aware PEs raise `ValueError` when `layout` is missing or does not match
the token count, rather than guessing the grid from configured maxima.

`LearnableTimeFrequencyPositionalEncoding` stores:

| Parameter | Shape |
| --- | --- |
| `time_pe` | `(1, max_t, d_model)` |
| `freq_pe` | `(1, max_f, d_model)` |

With `F = layout.frequency_patches` (at most `max_f`), it requires
`N % F == 0`, derives `T_patches = N // F` (which must equal
`layout.time_patches` and be at most `max_t`), and applies time-major broadcast:

```python
x + time_pe[:, :T_patches].repeat_interleave(F, dim=1)
  + freq_pe[:, :F].repeat(1, T_patches, 1)
```

## Concrete Embeddings

`LinearEmbedding.__init__(*, input_dim: int, d_model: int, config: LinearEmbeddingConfig)`:

- Builds `Projector(ProjectionConfig(input_dim=input_dim, output_dim=d_model, ...))`.
- Builds `self.positional_encoding` from `get_embedding_positional_encoding_registry()`.
- Forward order: linear projection -> positional encoding -> dropout -> optional subsampling.

`ConvEmbedding.__init__(*, input_dim: int, d_model: int, config: ConvEmbeddingConfig)`:

- Builds `nn.Conv1d(input_dim, d_model, kernel_size=config.kernel_size, stride=1, same-padding)`.
- Requires odd `kernel_size`.
- Zero-fills masked input tokens before convolution so invalid tokens do not leak into valid local windows.
- Builds `self.positional_encoding` from `get_embedding_positional_encoding_registry()`.
- Forward order: conv projection -> positional encoding -> optional layer norm -> dropout -> optional subsampling.

`LinearEmbeddingConfig` may also enable input LayerNorm and post-positional
LayerNorm. This models wav2vec 2.0 as generic embedding stages:
pre-LN -> projection -> conv1d PE residual -> post-LN -> dropout.

Layout-aware PE (`learnable_2d`, `sinusoidal_2d`) requires `TokenLayout` when
invoked through an embedding. `AudioModel.embed(..., layout=...)` is the
canonical path.

## Boundary Checks

`prepare` (and therefore `forward`) validates:

1. `batch` is `AudioBatch`.
2. `batch.x.ndim == 3`.
3. `batch.x.shape[-1] == self.input_dim`.
4. `batch.mask.ndim == 2`.
5. `batch.mask.shape == batch.x.shape[:2]`.
6. `batch.mask.dtype == torch.bool`.

## Wrapper

`Embedding(AudioModelConfig)` dispatches via `get_embedding_registry()[config.embedding.type]` and constructs:

```python
embedding_cls(
    input_dim=config.tokenizer.output_dim,
    d_model=config.d_model,
    config=config.embedding,
)
```

The wrapper forwards the staged API and the `num_prefix_tokens`,
`subsampling_factor`, and `output_mask` properties of the inner embedding.

Register new embeddings with `EMBEDDINGS.register(type, cls)` and new embedding
PE modules with `EMBEDDING_POSITIONAL_ENCODINGS.register(type, cls)`, plus
`register_config` / `register_validator` for their configs. An embedding that
adds prefix tokens or subsamples declares `prefix_token` / `token_subsampling`
in its config `capabilities` so objectives can check compatibility.

# Encoder Component Contract

Encoders consume `d_model` token sequences and produce `EncoderOutput`. Entry positional encoding is not an encoder responsibility; embeddings own absolute/token-grid PE. Encoders may optionally apply per-block positional encoding inside attention, currently RoPE only.

## Pipeline Position

```text
AudioBatch (tokens, d_model) -> Encoder -> EncoderOutput
```

Input:

| Field | Shape | Contract |
| --- | --- | --- |
| `batch.x` | `(B, N_tokens, d_model)` | floating tensor |
| `batch.mask` | `(B, N_tokens)` | `torch.bool`, `True` means valid |

Output:

| Field | Shape | Contract |
| --- | --- | --- |
| `z` | `(B, N_tokens, d_model)` | encoded token states |
| `mask` | `(B, N_tokens)` | identical/equal to input mask |
| `lengths` | `(B,)` | `batch.mask.sum(dim=1)` |
| `cls` | `None` | model wrapper owns pooling via `AudioModel.get_cls` |

## Config

Each encoder config has:

| Field | Contract |
| --- | --- |
| `type` | one of `"transformer"`, `"conformer"`, `"branchformer"`, `"ebranchformer"` |
| `num_layers` | `int >= 1` |
| `pre_norm` | `bool` |
| `residual_dropout_p` | float in `[0, 1]` |
| `pooling` | currently `"masked_mean"`; consumed by `AudioModel`, not the encoder |
| `positional_encoding` | `PerBlockPositionalEncodingConfig | None`; currently `RopePositionalEncodingConfig | None` |
| `layer_norm_eps` | positive finite float |
| `final_norm` | `bool` |

There is no `pe_placement` field. Absolute/entry PE is configured on embedding configs through `EmbeddingPositionalEncodingConfig`.

## Positional Encoding

- Entry PE is the embedding's responsibility.
- Per-block PE is optional and legal only through `config.positional_encoding`.
- The only per-block PE type currently supported by validators is `RopePositionalEncodingConfig`.
- Blocks pass `config.positional_encoding` directly into `MultiHeadAttention`; `None` means no per-block PE.
- The encoder wrapper must not instantiate or apply positional encoding before the block loop.

## Encoder Wrapper

`Encoder.__init__(*, d_model: int, config: EncoderConfig)`:

1. Stores `self.d_model` and `self.config`.
2. Validates `d_model % config.mha.num_heads == 0`.
3. Looks up the block class via `get_encoder_block_registry()[config.type]`.
4. Builds exactly `config.num_layers` blocks.
5. Optionally builds `final_norm` when `config.final_norm` is true.

`Encoder.forward(batch: AudioBatch) -> EncoderOutput` validates:

1. `batch` is `AudioBatch`.
2. `batch.x.ndim == 3`.
3. `batch.x.shape[-1] == d_model`.
4. `batch.mask.ndim == 2`.
5. `batch.mask.shape == batch.x.shape[:2]`.
6. `batch.mask.dtype == torch.bool`.

Forward order:

1. `x = batch.x`.
2. Apply each block as `x = block(x, attention_mask=batch.mask)`.
3. Apply `final_norm` when configured.
4. Return `EncoderOutput(z=x, mask=batch.mask, lengths=batch.mask.sum(dim=1), cls=None)`.

## Block Contract

Every block:

- Subclasses `nn.Module`.
- Has `__init__(*, d_model: int, config: <matching encoder config>)`.
- Raises `TypeError` when passed the wrong encoder config variant.
- Accepts `forward(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor`.
- Preserves `(B, N_tokens, d_model)`.
- Uses `attention_mask` for attention and local branches where applicable.
- Does not pool, create `EncoderOutput`, mutate masks, or apply entry PE.

Register new block variants in `get_encoder_block_registry()` and validate their config in `validate_config.py`.

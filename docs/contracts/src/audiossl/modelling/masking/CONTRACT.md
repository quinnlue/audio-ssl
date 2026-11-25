# Masking Strategies

Masking strategies are training-only modules invoked by objectives. They are
not part of the inference pipeline and do not change tensor values or shape by
themselves.

Each strategy implements:

```python
compute_mask(shape, valid_mask, generator) -> torch.Tensor
```

The returned tensor is boolean and has exactly `shape`. A provided `valid_mask`
must be boolean and shape-compatible. Strategy randomness is driven by the
injected `torch.Generator` whenever one is supplied, so seeded objective runs
are reproducible.


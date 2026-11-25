"""Pooling helpers shared by modelling components."""

import torch


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid token states with a boolean validity mask.

    `AudioModel.get_cls` owns sequence pooling for encoder outputs; encoders
    should leave `EncoderOutput.cls` empty and preserve token states.

    Parameters
    ----------
    x
        Token states with shape `(B, N, D)`.
    attention_mask
        Boolean validity mask with shape `(B, N)`, where `True` means valid.

    Returns
    -------
    torch.Tensor
        Mean-pooled states with shape `(B, D)`, computed only over valid token
        states.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError(
            f"attention_mask must be a torch.Tensor, got {type(attention_mask).__name__}"
        )
    if x.ndim != 3:
        raise ValueError(f"x must be rank-3 [B, T, D], got shape {tuple(x.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be rank-2 [B, T], got shape {tuple(attention_mask.shape)}"
        )
    if attention_mask.dtype != torch.bool:
        raise ValueError(f"attention_mask must be boolean, got dtype {attention_mask.dtype}")
    if x.shape[:2] != attention_mask.shape:
        raise ValueError(
            "x and attention_mask must agree on batch/time dimensions, "
            f"got {tuple(x.shape[:2])} and {tuple(attention_mask.shape)}"
        )
    valid_counts = attention_mask.sum(dim=1)
    if torch.any(valid_counts == 0):
        raise ValueError("Cannot masked-mean pool a sample with zero valid timesteps")

    mask = attention_mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(dim=1) / valid_counts.unsqueeze(-1).to(x.dtype)

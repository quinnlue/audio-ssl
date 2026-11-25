"""MAE random masking helpers for token removal and restoration."""

from __future__ import annotations

import torch

from audiossl.rng import generator_device


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token random masking.

    Mirrors XiaoMi/dasheng `random_masking` (models.py:371-399) and adds
    padding awareness.

    Parameters
    ----------
    x
        Token embeddings with shape ``(B, N, D)``.
    mask_ratio
        Fraction of tokens to remove.
    generator
        Optional RNG supplied by the trainer.
    valid_mask
        Optional boolean ``(B, N)`` validity mask. Invalid tokens sort after
        every valid token, and each row keeps
        ``max(1, floor(valid_b * (1 - mask_ratio)))`` valid tokens, so short
        clips are masked at the configured ratio instead of keeping padding.
        With an all-valid mask the result equals the unmasked call bit for bit.

    Returns
    -------
    x_masked
        Gathered tokens with shape ``(B, len_keep, D)`` in shuffle order,
        ``len_keep = int(N * (1 - mask_ratio))``. With ``valid_mask``, slots
        beyond a row's keep count are padding for the encoder; recover them
        with :func:`kept_token_mask`.
    mask
        Float mask with shape ``(B, N)``, where ``1.0`` means "not encoded"
        (removed or padding). MAE-specific semantics, not a validity mask.
    ids_restore
        Inverse permutation with shape ``(B, N)`` for decoder unshuffle.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"random_masking expected torch.Tensor, got {type(x).__name__}")
    if x.ndim != 3:
        raise ValueError(
            f"random_masking expected x rank-3 [B, N, D], got shape {tuple(x.shape)}"
        )
    _check_ratio("random_masking", mask_ratio)

    batch_size, num_tokens, _ = x.shape
    len_keep = _len_keep("random_masking", num_tokens, mask_ratio)
    _check_valid_mask("random_masking", valid_mask, (batch_size, num_tokens))

    rng_device = generator_device(generator)
    noise = torch.rand(batch_size, num_tokens, device=rng_device, generator=generator)
    if noise.device != x.device:
        noise = noise.to(x.device)
    if valid_mask is not None:
        noise = noise + (~valid_mask).to(noise.dtype)
    ids_shuffle = torch.argsort(noise, dim=1)
    return _apply_shuffle(x, ids_shuffle, len_keep, mask_ratio, valid_mask)


def random_masking_group(
    x: torch.Tensor,
    mask_ratio: float,
    *,
    group_factor: int,
    generator: torch.Generator | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise random masking.

    Mirrors XiaoMi/dasheng `random_masking_group` (models.py:337-369).

    Noise is drawn at group resolution (N // group_factor). Group indices are
    then expanded into contiguous-N blocks via `indices[ids_shuffle].flatten(-2)`,
    so each group of `group_factor` consecutive tokens shares a shuffle key.
    `len_keep = int(N * (1 - mask_ratio))` is not rounded to a multiple of
    group_factor, matching dasheng; the kept/masked boundary may split a group.

    With ``valid_mask``, groups containing any invalid token sort last and the
    per-row keep count follows :func:`random_masking`. Returns the same
    ``(x_masked, mask, ids_restore)`` triple.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"random_masking_group expected torch.Tensor, got {type(x).__name__}"
        )
    if x.ndim != 3:
        raise ValueError(
            "random_masking_group expected x rank-3 [B, N, D], "
            f"got shape {tuple(x.shape)}"
        )
    if isinstance(group_factor, bool) or not isinstance(group_factor, int):
        raise TypeError(
            "random_masking_group expected int group_factor, "
            f"got {type(group_factor).__name__}"
        )
    if group_factor < 1:
        raise ValueError(
            f"random_masking_group expected group_factor >= 1, got {group_factor!r}"
        )
    _check_ratio("random_masking_group", mask_ratio)

    batch_size, num_tokens, _ = x.shape
    if num_tokens % group_factor != 0:
        raise ValueError(
            "random_masking_group expected N to be a multiple of "
            f"group_factor={group_factor}, got N={num_tokens}"
        )
    len_keep = _len_keep("random_masking_group", num_tokens, mask_ratio)
    _check_valid_mask("random_masking_group", valid_mask, (batch_size, num_tokens))

    num_groups = num_tokens // group_factor
    rng_device = generator_device(generator)
    noise = torch.rand(batch_size, num_groups, device=rng_device, generator=generator)
    if noise.device != x.device:
        noise = noise.to(x.device)
    if valid_mask is not None:
        group_invalid = (~valid_mask).view(batch_size, num_groups, group_factor).any(dim=-1)
        noise = noise + group_invalid.to(noise.dtype)
    indices = torch.arange(num_tokens, device=x.device).view(num_groups, group_factor)

    ids_shuffle_groups = torch.argsort(noise, dim=1)
    ids_shuffle = indices[ids_shuffle_groups].flatten(-2)
    return _apply_shuffle(x, ids_shuffle, len_keep, mask_ratio, valid_mask)


def kept_token_mask(mask: torch.Tensor, ids_restore: torch.Tensor, len_keep: int) -> torch.Tensor:
    """Boolean ``(B, len_keep)`` validity of the gathered slots returned by masking."""
    ids_keep = torch.argsort(ids_restore, dim=1)[:, :len_keep]
    return torch.gather(mask < 0.5, dim=1, index=ids_keep)


def _apply_shuffle(
    x: torch.Tensor,
    ids_shuffle: torch.Tensor,
    len_keep: int,
    mask_ratio: float,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_tokens, dim = x.shape
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))

    if valid_mask is None:
        mask = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.get_default_dtype())
        mask[:, :len_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    valid_counts = valid_mask.sum(dim=1, keepdim=True)
    row_keep = torch.floor(valid_counts.to(torch.float64) * (1.0 - float(mask_ratio)))
    row_keep = row_keep.clamp(min=1).to(torch.long).clamp(max=len_keep)
    kept = (ids_restore < row_keep) & valid_mask
    mask = (~kept).to(torch.get_default_dtype())
    return x_masked, mask, ids_restore


def _check_ratio(owner: str, mask_ratio: float) -> None:
    if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, int | float):
        raise TypeError(f"{owner} expected float mask_ratio, got {type(mask_ratio).__name__}")
    if not (0.0 < float(mask_ratio) < 1.0):
        raise ValueError(f"{owner} expected mask_ratio in (0, 1), got {mask_ratio!r}")


def _len_keep(owner: str, num_tokens: int, mask_ratio: float) -> int:
    len_keep = int(num_tokens * (1.0 - float(mask_ratio)))
    if len_keep <= 0 or len_keep >= num_tokens:
        raise ValueError(
            f"{owner} with N={num_tokens}, mask_ratio={mask_ratio} yields "
            f"len_keep={len_keep}; needs 0 < len_keep < N"
        )
    return len_keep


def _check_valid_mask(owner: str, valid_mask: torch.Tensor | None, shape: tuple[int, int]) -> None:
    if valid_mask is None:
        return
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError(f"{owner} expected boolean valid_mask tensor")
    if tuple(valid_mask.shape) != shape:
        raise ValueError(
            f"{owner} expected valid_mask shape {shape}, got {tuple(valid_mask.shape)}"
        )


__all__ = ["kept_token_mask", "random_masking", "random_masking_group"]

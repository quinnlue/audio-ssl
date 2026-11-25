"""Channel-axis masking strategy used by training objectives."""

from __future__ import annotations

from audiossl.config import ChannelMaskingConfig, SpanMaskingConfig

from .span import SpanMaskingStrategy


class ChannelMaskingStrategy(SpanMaskingStrategy):
    """Sample contiguous channel masks with span masking semantics.

    The strategy adapts `ChannelMaskingConfig` to `SpanMaskingStrategy` and is
    invoked only by objectives during training. Returned masks are boolean and
    have exactly the requested shape, usually `(B, F)`.

    Parameters
    ----------
    config
        Channel masking configuration. Its probability, length, distribution,
        spacing, and overlap fields are forwarded to the span sampler.
    """

    def __init__(self, *, config: ChannelMaskingConfig) -> None:
        """Initialize channel masking from a channel masking configuration."""
        span_config = SpanMaskingConfig(
            mask_prob=config.mask_prob,
            mask_length=config.mask_length,
            mask_selection=config.mask_selection,
            mask_other=config.mask_other,
            min_space=config.min_space,
            no_overlap=config.no_overlap,
            require_same_masks=False,
        )
        super().__init__(config=span_config)
        self.config = config


__all__ = ["ChannelMaskingStrategy"]

"""Training-only masking strategies consumed by objectives."""

from ._base import MaskingStrategy
from .channel import ChannelMaskingStrategy
from .span import SpanMaskingStrategy

__all__ = ["ChannelMaskingStrategy", "MaskingStrategy", "SpanMaskingStrategy"]

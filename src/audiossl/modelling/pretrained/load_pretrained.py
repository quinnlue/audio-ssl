"""Legacy import path for public pretrained checkpoint helpers.

The functions exported here delegate to :mod:`audiossl.pretrained`. They do
not own preset lookup, checkpoint resolution, or
:class:`audiossl.modelling.model.AudioModel` runtime construction.
"""

from audiossl.pretrained import get_pretrained_spec, list_pretrained, load_pretrained


__all__ = ["get_pretrained_spec", "list_pretrained", "load_pretrained"]

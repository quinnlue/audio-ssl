"""Legacy modelling namespace for public pretrained checkpoint helpers.

The canonical pretrained registry and loaders live in :mod:`audiossl.pretrained`.
This package re-exports that API for older callers without changing model
construction, registry lookup, or checkpoint loading behavior.
"""

from audiossl.pretrained import get_pretrained_spec, list_pretrained, load_pretrained


__all__ = ["get_pretrained_spec", "list_pretrained", "load_pretrained"]

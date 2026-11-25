"""Legacy modelling factories for registered pretrained-compatible presets.

The canonical preset registry is :mod:`audiossl.presets`, and pretrained weight
loading is owned by :mod:`audiossl.pretrained`. These exports preserve the older
``audiossl.modelling.pretrained.models`` import path.
"""

from .dasheng import dasheng_base, dasheng_base_config, dasheng_base_pretrained


__all__ = [
    "dasheng_base",
    "dasheng_base_config",
    "dasheng_base_pretrained",
]

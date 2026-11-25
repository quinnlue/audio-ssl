"""Legacy config objects for registered architecture presets.

New code should request configs through :func:`audiossl.presets.get_config` so
that preset names, aliases, and config overrides stay centralized in the preset
registry.
"""

from audiossl.presets import get_config


dasheng_base = get_config("dasheng-base")


__all__ = ["dasheng_base"]

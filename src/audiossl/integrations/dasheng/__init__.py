"""Dasheng encoder integration: config recipes and weight porters."""

from .config import make_dasheng_base_config
from .porter import (
    load_dasheng_base,
    load_dasheng_weights,
    port_dasheng_mae_state_dict,
    port_dasheng_state_dict,
)

__all__ = [
    "load_dasheng_base",
    "load_dasheng_weights",
    "make_dasheng_base_config",
    "port_dasheng_mae_state_dict",
    "port_dasheng_state_dict",
]

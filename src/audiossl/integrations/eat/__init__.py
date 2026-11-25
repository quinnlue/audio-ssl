"""EAT encoder integration: config recipes and weight porters."""

from .config import make_eat_500m_config, make_eat_base_config, make_eat_scaled_config
from .porter import (
    extract_features,
    load_eat_base,
    load_eat_weights,
    port_eat_state_dict,
)
from .reference import load_hf_eat_reference

__all__ = [
    "extract_features",
    "load_eat_base",
    "load_eat_weights",
    "load_hf_eat_reference",
    "make_eat_500m_config",
    "make_eat_base_config",
    "make_eat_scaled_config",
    "port_eat_state_dict",
]

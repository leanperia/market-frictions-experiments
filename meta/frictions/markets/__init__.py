"""Market profiles for country-specific friction layers."""

from __future__ import annotations

from .nyse import LAYER_BUILDERS as NYSE_LAYER_BUILDERS
from .pse import LAYER_BUILDERS as PSE_LAYER_BUILDERS

MARKET_LAYER_BUILDERS = {
    "nyse": NYSE_LAYER_BUILDERS,
    "pse": PSE_LAYER_BUILDERS,
}


def get_market_layer_builders(market: str):
    market_key = str(market).lower()
    if market_key not in MARKET_LAYER_BUILDERS:
        raise ValueError(f"Unknown market profile: {market}")
    return MARKET_LAYER_BUILDERS[market_key]

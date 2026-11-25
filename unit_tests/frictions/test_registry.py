"""Registry checks for market-specific friction adapters."""

from __future__ import annotations

import numpy as np
import pytest

from meta.frictions import build_pipeline_from_config


def _env_config():
    return {
        "date_list": np.array(["2024-01-02", "2024-01-03"], dtype=object),
        "high_array": np.array([[101.0], [102.0]], dtype=np.float64),
        "low_array": np.array([[99.0], [100.0]], dtype=np.float64),
        "adv20_array": np.array([[1_000_000.0], [1_000_000.0]], dtype=np.float64),
    }


def test_market_profile_resolves_generic_fees_layer():
    nyse = build_pipeline_from_config({"layers": [{"name": "fees"}]}, _env_config(), market="nyse")
    pse = build_pipeline_from_config({"layers": [{"name": "fees"}]}, _env_config(), market="pse")

    assert nyse.layers[0].name == "fees_nyse"
    assert pse.layers[0].name == "fees_pse"


def test_market_profile_rejects_unsupported_layer():
    with pytest.raises(ValueError, match="unsupported friction layer"):
        build_pipeline_from_config({"layers": [{"name": "board_lots"}]}, _env_config(), market="nyse")

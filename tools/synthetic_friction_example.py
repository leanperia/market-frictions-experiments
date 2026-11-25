#!/usr/bin/env python3
"""Demonstrate supported friction-aware construction without private data or RL.

Prices and allocations are invented. The explicit fees, spread and impact
settings are modelling assumptions, not broker calibration or trading advice.
This runs the supplied modified Gym environment, not an upstream FinRL adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta.env_market_impact.envs.env_portfolio_optimization_impact import (  # noqa: E402
    build_portfolio_env,
)
from meta.frictions.diagnostics import flatten_step_info  # noqa: E402


def run_example(market: str) -> dict:
    """Return intended allocations, realised trades, costs and cash diagnostics."""
    if market not in {"nyse", "pse"}:
        raise ValueError("supported example markets are nyse and pse")
    dates = np.array(
        [
            "2024-05-23",
            "2024-05-24",
            "2024-05-28",
            "2024-05-29",
            "2024-05-30",
            "2024-05-31",
            "2024-06-03",
            "2024-06-04",
        ],
        dtype="datetime64[ns]",
    )
    prices = np.array(
        [
            [25, 100],
            [25.5, 102],
            [26, 101],
            [25, 103],
            [25.75, 102],
            [26.5, 104],
            [27, 101],
            [27.5, 103],
        ],
        dtype=float,
    )
    opens = np.array(
        [
            [24.975, 100],
            [25, 101],
            [26.125, 102],
            [25.5, 101],
            [25.25, 102],
            [25.75, 103],
            [26.5, 104],
            [27, 101],
        ]
    )
    volume = np.array(
        [
            [10000, 20000],
            [12000, 18000],
            [0, 19000],
            [13000, 21000],
            [10000, 20000],
            [14000, 16000],
            [15000, 17000],
            [16000, 19000],
        ],
        dtype=float,
    )
    # Lagged liquidity uses preceding observations only. An observed zero-volume
    # session is not silently filled or classified as a fabricated halt.
    frame = pd.DataFrame(volume)
    adv = frame.rolling(20, min_periods=1).mean().shift().ffill().fillna(frame)
    share_factors = np.tile([0.5, 1.0], (len(dates), 1))
    dividends = np.zeros_like(prices)
    dividends[3, 0] = 0.75
    inputs = {
        "date_list": dates,
        "tic_list": np.array(["SYN-A", "SYN-B"]),
        "price_array": prices,
        "open_array": opens,
        "high_array": np.maximum(prices, opens) * 1.02,
        "low_array": np.minimum(prices, opens) * 0.98,
        "quote_price_array": prices / share_factors,
        "quote_open_array": opens / share_factors,
        "share_factor_array": share_factors,
        "dividend_array": dividends,
        "tech_array": np.zeros_like(prices),
        "volatility_array": np.full_like(prices, 0.02),
        "volume_array": volume,
        "adv20_array": adv.to_numpy(),
        "synthetic_halt_array": np.zeros_like(prices, dtype=bool),
    }
    settings = {
        "market": market,
        "environment": {"initial_amount": 100_000.0},
        "friction": {
            "cost_buffer_pct": 0.007 if market == "pse" else 0.003,
            "max_trade_volume_pct": 1.0,
            "impact_model": {"name": "sqrt", "Y": 0.6, "perm_fraction": 0.25},
            "layers": [
                {"name": "execution_timing"},
                {"name": "adv_cap", "params": {"max_pov": 0.10}},
                {
                    "name": "fees",
                    "params": {"commission_bps": 25.0 if market == "pse" else 0.0},
                },
                {
                    "name": "spread",
                    "params": {"flat_spread": 0.0012 if market == "pse" else 0.0004},
                },
                {
                    "name": "settlement",
                    "params": {
                        "cycle_days": 2,
                        **(
                            {"t1_effective_date": "2024-05-28"}
                            if market == "nyse"
                            else {}
                        ),
                    },
                },
                *([{"name": "board_lots"}] if market == "pse" else []),
            ],
        },
    }
    # A deterministic non-RL schedule alternates allocations and cash. Actions
    # are logits in cash/SYN-A/SYN-B order, not guaranteed executable holdings.
    allocations = [
        [-8, 8, -8],
        [-8, 8, -8],
        [-8, -8, 8],
        [8, -8, -8],
        [-8, 5, 5],
        [-8, -8, 8],
        [-8, 8, -8],
    ]
    env = build_portfolio_env(inputs, settings, seed=7)
    rows = []
    try:
        for action in allocations:
            _, _, terminated, truncated, info = env.step(np.array(action, dtype=float))
            diagnostics = flatten_step_info(info)
            rows.append(
                {
                    "decision_date": str(info["decision_date"]),
                    "execution_date": str(info["execution_date"]),
                    "intended_action_logits": action,
                    "intended_weights": info["target_weights"].tolist(),
                    "realized_weights": info["realized_weights"].tolist(),
                    "holdings": env.holdings.tolist(),
                    "wealth": float(env.total_asset),
                    "cash": float(env.cash),
                    "settled_cash": diagnostics["settled_cash"],
                    "in_transit_cash": diagnostics["in_transit_cash"],
                    "trades": info["trades"],
                    "costs": {
                        "fees": diagnostics["fees_explicit"],
                        "spread": diagnostics["spread_cost"],
                        "impact": diagnostics["impact_cost"],
                    },
                    "blocked": {
                        "adv": diagnostics["blocked_by_adv"],
                        "lots": diagnostics["blocked_by_lot"],
                        "cash": diagnostics["blocked_by_cash"],
                    },
                }
            )
        if not terminated or truncated:
            raise RuntimeError("synthetic example did not complete")
        return {
            "market": market,
            "campaign_evidence": False,
            "purpose": "synthetic non-RL integration example; not calibrated trading evidence",
            "initial_amount": 100_000.0,
            "weight_order": ["cash", "SYN-A", "SYN-B"],
            "final_equity": float(env.total_asset),
            "steps": rows,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("nyse", "pse"), default="pse")
    args = parser.parse_args()

    def numpy_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not a JSON scalar: {type(value).__name__}")

    print(
        json.dumps(
            run_example(args.market), indent=2, allow_nan=False, default=numpy_scalar
        )
    )


if __name__ == "__main__":
    main()

"""Direction checks for SpreadHaircut + Corwin-Schultz estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.pipeline import TradeContext
from meta.frictions.spread import SpreadHaircut, corwin_schultz_spread


def _ctx(t, trades, prices=None):
    n = len(trades)
    return TradeContext(
        time=t,
        date=pd.Timestamp("2024-01-15"),
        prices=np.array(prices or [100.0] * n, dtype=np.float64),
        volume=np.array([1e7] * n),
        volatility=np.array([0.02] * n),
        holdings=np.zeros(n),
        cash=1_000_000.0,
        target_weights=np.zeros(n + 1),
        trades=np.array(trades, dtype=np.int64),
    )


def test_cs_first_row_uses_floor():
    high = np.array([[101.0], [102.0], [103.0]])
    low = np.array([[99.0], [100.0], [101.0]])
    spread = corwin_schultz_spread(high, low, spread_floor=0.0005)
    assert spread[0, 0] == 0.0005


def test_cs_clipped_above_floor():
    rng = np.random.default_rng(0)
    high = 100 + rng.uniform(0, 1, size=(50, 3))
    low = 100 - rng.uniform(0, 1, size=(50, 3))
    spread = corwin_schultz_spread(high, low, spread_floor=0.0010)
    assert (spread >= 0.0010 - 1e-12).all()


def test_haircut_charges_proportional_to_notional():
    spread_arr = np.array([[0.001, 0.001]])  # 10 bps spread → 5 bps half
    layer = SpreadHaircut(spread_arr)
    ctx = _ctx(0, [1000, -2000])  # notional 100k + 200k = 300k, half-spread 0.0005
    layer.after_trade(ctx)
    # cost = 300_000 × 0.0005 = 150
    assert abs(ctx.diagnostics["spread_cost"] - 150.0) < 1e-9


def test_higher_spread_yields_higher_cost():
    layer_lo = SpreadHaircut(np.array([[0.0005]]))
    layer_hi = SpreadHaircut(np.array([[0.003]]))
    c1 = _ctx(0, [1000]); layer_lo.after_trade(c1)
    c2 = _ctx(0, [1000]); layer_hi.after_trade(c2)
    assert c2.diagnostics["spread_cost"] > c1.diagnostics["spread_cost"]

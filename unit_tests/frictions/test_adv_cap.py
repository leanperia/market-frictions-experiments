"""Direction check for AdvCap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.adv_cap import AdvCap
from meta.frictions.pipeline import TradeContext


def _ctx(time, trades, prices=None):
    n = len(trades)
    return TradeContext(
        time=time,
        date=pd.Timestamp("2024-01-15"),
        prices=np.array(prices or [100.0] * n),
        volume=np.array([1e6] * n),
        volatility=np.array([0.02] * n),
        holdings=np.zeros(n),
        cash=1e7,
        target_weights=np.zeros(n + 1),
        trades=np.array(trades, dtype=np.int64),
    )


def test_caps_trades_above_pov_limit():
    adv = np.array([[1_000_000, 1_000_000]])  # 1 day × 2 stocks
    layer = AdvCap(adv, max_pov=0.10)  # cap = 100k each
    ctx = _ctx(0, trades=[250_000, -50_000])  # buy way above cap
    out = layer.before_trade(ctx)
    assert out.trades[0] == 100_000
    assert out.trades[1] == -50_000
    np.testing.assert_array_equal(out.diagnostics["blocked_by_adv"], [150_000, 0])


def test_pov_diagnostic_reflects_capped_size():
    adv = np.array([[1_000_000, 1_000_000]])
    layer = AdvCap(adv, max_pov=0.10)
    ctx = _ctx(0, trades=[250_000, 50_000])
    out = layer.before_trade(ctx)
    np.testing.assert_allclose(out.diagnostics["participation_pct"], [0.10, 0.05])


def test_passthrough_within_cap():
    adv = np.array([[10_000_000]])
    layer = AdvCap(adv, max_pov=0.10)  # cap = 1M
    ctx = _ctx(0, trades=[123])
    out = layer.before_trade(ctx)
    assert out.trades[0] == 123
    assert out.diagnostics["blocked_by_adv"][0] == 0

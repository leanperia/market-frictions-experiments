"""Direction checks for NextDayExecution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.execution_timing import NextDayExecution
from meta.frictions.pipeline import TradeContext


def _ctx(t, trades, open_prices=None, close_prices=None):
    n = len(trades)
    close = np.array(close_prices or [100.0] * n, dtype=np.float64)
    open_ = np.array(open_prices or [99.0] * n, dtype=np.float64)
    return TradeContext(
        time=t,
        date=pd.Timestamp("2024-01-15"),
        prices=close.copy(),
        volume=np.array([1e6] * n),
        volatility=np.array([0.02] * n),
        holdings=np.zeros(n),
        cash=1e6,
        target_weights=np.zeros(n + 1),
        trades=np.array(trades, dtype=np.int64),
        close_prices=close,
        open_prices=open_,
    )


def test_marks_next_open_without_overwriting_effective_prices():
    layer = NextDayExecution()
    ctx = _ctx(0, [100, -50, 30], open_prices=[90.0, 91.0, 92.0], close_prices=[90.5, 91.5, 92.5])
    out = layer.before_trade(ctx)
    np.testing.assert_array_equal(out.trades, [100, -50, 30])
    np.testing.assert_allclose(out.prices, [90.5, 91.5, 92.5])
    assert out.diagnostics["execution_lag_days"] == 1
    assert out.diagnostics["execution_price"] == "next_open"


def test_layer_does_not_buffer_stale_share_deltas():
    layer = NextDayExecution()
    layer.before_trade(_ctx(0, [100, -50]))
    ctx = _ctx(1, [777, 888])
    out = layer.before_trade(ctx)
    np.testing.assert_array_equal(out.trades, [777, 888])

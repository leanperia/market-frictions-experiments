"""Direction check for OutageFilter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.outages import OutageFilter
from meta.frictions.pipeline import TradeContext


def _ctx(date, trades):
    return TradeContext(
        time=10,
        date=pd.Timestamp(date),
        prices=np.array([100.0] * len(trades)),
        volume=np.array([1e6] * len(trades)),
        volatility=np.array([0.02] * len(trades)),
        holdings=np.zeros(len(trades)),
        cash=1e6,
        target_weights=np.zeros(len(trades) + 1),
        trades=np.array(trades, dtype=np.int64),
    )


def test_full_day_cancellation_blocks_all_trades():
    layer = OutageFilter(outages=[{"date": "2022-01-04", "nature": "full_day_cancellation", "duration_minutes": 330}])
    ctx = _ctx("2022-01-04", [100, -50, 0])
    out = layer.before_trade(ctx)
    assert (out.trades == 0).all()
    assert out.diagnostics["outage_scale"] == 0.0
    assert np.array_equal(out.diagnostics["blocked_by_outage"], np.array([100, 50, 0]))


def test_partial_halt_scales_proportionally():
    # 165-min halt of a 330-min session: half of trades survive.
    layer = OutageFilter(outages=[{"date": "2024-01-03", "nature": "mid_session_halt", "duration_minutes": 165}])
    ctx = _ctx("2024-01-03", [200, -200])
    out = layer.before_trade(ctx)
    assert out.diagnostics["outage_scale"] == 0.5
    np.testing.assert_array_equal(out.trades, np.array([100, -100]))


def test_non_outage_day_is_passthrough():
    layer = OutageFilter(outages=[{"date": "2022-01-04", "nature": "full_day_cancellation", "duration_minutes": 330}])
    ctx = _ctx("2023-06-01", [100, -50])
    out = layer.before_trade(ctx)
    np.testing.assert_array_equal(out.trades, np.array([100, -50]))
    assert "blocked_by_outage" not in out.diagnostics


def test_block_full_day_only_skips_partial_halts():
    layer = OutageFilter(
        outages=[{"date": "2024-01-03", "nature": "mid_session_halt", "duration_minutes": 150}],
        block_full_day_only=True,
    )
    ctx = _ctx("2024-01-03", [100, -100])
    out = layer.before_trade(ctx)
    np.testing.assert_array_equal(out.trades, np.array([100, -100]))

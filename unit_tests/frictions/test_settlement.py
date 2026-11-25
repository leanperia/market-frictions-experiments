"""Direction checks for SettlementLag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.pipeline import TradeContext
from meta.frictions.settlement import SettlementLag


def _ctx(date, trades, cash=1_000_000.0, prices=None):
    n = len(trades)
    return TradeContext(
        time=0,
        date=pd.Timestamp(date),
        prices=np.array(prices or [100.0] * n, dtype=np.float64),
        volume=np.array([1e7] * n),
        volatility=np.array([0.02] * n),
        holdings=np.zeros(n),
        cash=float(cash),
        target_weights=np.zeros(n + 1),
        trades=np.array(trades, dtype=np.int64),
    )


def test_first_buy_uses_full_cash_no_in_transit():
    layer = SettlementLag(cycle_days=2)
    ctx = _ctx("2024-01-02", [1000], cash=200_000)
    out = layer.before_trade(ctx)
    assert out.trades[0] == 1000  # 100 × 1000 = 100k ≤ 200k cash
    assert out.diagnostics["in_transit_cash"] == 0.0


def test_sell_proceeds_locked_until_settlement():
    layer = SettlementLag(cycle_days=2)
    # Day 0: sell 500 @ 100 → 50k in transit
    s0 = _ctx("2024-01-02", [-500], cash=200_000)
    layer.before_trade(s0)
    layer.after_trade(s0)
    state = layer.get_state()
    assert state["pending"][0]["amount"] == 50_000

    # Day 1 (T+1): cash still locked, attempt to buy 200k worth.
    s1 = _ctx("2024-01-03", [2000], cash=200_000)
    out1 = layer.before_trade(s1)
    # settled = 200k − 50k = 150k → scale = 0.75 → 1500 shares kept, 500 blocked
    assert out1.trades[0] == 1500
    assert out1.diagnostics["blocked_by_cash"][0] == 500
    assert out1.diagnostics["settled_cash"] == 150_000


def test_proceeds_mature_on_settlement_day():
    layer = SettlementLag(cycle_days=2)
    s0 = _ctx("2024-01-02", [-500], cash=200_000)
    layer.before_trade(s0)
    layer.after_trade(s0)
    # Day +2 (T+2): proceeds mature, full cash available.
    s2 = _ctx("2024-01-04", [2000], cash=200_000)
    out2 = layer.before_trade(s2)
    assert out2.trades[0] == 2000
    assert out2.diagnostics["in_transit_cash"] == 0.0


def test_t1_effective_date_flips_cycle():
    layer = SettlementLag(cycle_days=2, t1_effective_date="2024-05-28")
    # Pre-flip: T+2
    pre = _ctx("2024-05-01", [-100])
    layer.before_trade(pre)
    layer.after_trade(pre)
    pending_pre = layer.get_state()["pending"][-1]
    assert (pd.Timestamp(pending_pre["settle_date"]) - pd.Timestamp("2024-05-01")).days == 2
    # Post-flip: T+1
    post = _ctx("2024-06-03", [-100])
    layer.before_trade(post)
    layer.after_trade(post)
    pending_post = layer.get_state()["pending"][-1]
    assert (pd.Timestamp(pending_post["settle_date"]) - pd.Timestamp("2024-06-03")).days == 1


def test_trading_calendar_skips_weekends_for_settlement():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])  # Fri, Mon, Tue
    layer = SettlementLag(cycle_days=2, trading_dates=dates)
    s0 = _ctx("2024-01-05", [-100])
    layer.before_trade(s0)
    layer.after_trade(s0)
    pending = layer.get_state()["pending"][-1]
    assert pd.Timestamp(pending["settle_date"]) == pd.Timestamp("2024-01-09")


def test_reset_clears_in_transit_cash():
    layer = SettlementLag(cycle_days=2)
    s0 = _ctx("2024-01-02", [-500], cash=200_000)
    layer.before_trade(s0)
    layer.after_trade(s0)
    assert layer.get_state()["pending"]
    layer.reset()
    assert layer.get_state()["pending"] == []


def test_state_roundtrip_restores_pending_settlement_queue():
    layer = SettlementLag(cycle_days=2)
    s0 = _ctx("2024-01-02", [-500], cash=200_000)
    layer.before_trade(s0)
    layer.after_trade(s0)
    state = layer.get_state()

    restored = SettlementLag(cycle_days=2)
    restored.set_state(state)
    assert restored.get_state() == state


# ---------------------------------------------------------------------------
# Regression tests for SETTLE-END-CLAMP: tail-of-calendar settlement boundary.
#
# Calendar: [Jan 8, Jan 9, Jan 10, Jan 11]  (4 trading dates), cycle_days=2.
# BDay(2) from Jan 10 is Jan 14; BDay(2) from Jan 11 is Jan 15.
# The old code clamped both to Jan 11 (the last calendar date), giving T+0/T+1
# semantics at the tail end and letting the agent spend unsettled proceeds.
# ---------------------------------------------------------------------------

def _four_date_layer():
    dates = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"])
    return SettlementLag(cycle_days=2, trading_dates=dates)


def test_settle_end_clamp_penultimate_date_settle_date_after_last_calendar_date():
    """Sell on second-to-last date: settle_date must be AFTER the last calendar date.

    With cycle=2 and calendar [Jan8..Jan11], a sell on Jan10 (pos=2) needs
    pos+2=4 which is out of range.  The fix falls back to BDay(2) from Jan10
    = Jan14, which is strictly after Jan11.  The old code clamped to Jan11,
    yielding premature settlement.
    """
    layer = _four_date_layer()
    sell = _ctx("2024-01-10", [-500], cash=0.0)
    layer.before_trade(sell)
    layer.after_trade(sell)
    pending = layer.get_state()["pending"]
    assert len(pending) == 1
    settle_date = pd.Timestamp(pending[0]["settle_date"])
    last_calendar_date = pd.Timestamp("2024-01-11")
    assert settle_date > last_calendar_date, (
        f"Expected settle_date > {last_calendar_date}, got {settle_date}; "
        "proceeds must NOT be accessible on the final calendar day"
    )


def test_settle_end_clamp_penultimate_proceeds_still_locked_on_last_date():
    """After a sell on the penultimate date, proceeds must remain in-transit on the last date.

    before_trade on the last date calls _mature; if settle_date > Jan11 the
    pending item should NOT be popped, so in_transit_cash > 0.
    """
    layer = _four_date_layer()
    # Sell on Jan10 (penultimate)
    sell = _ctx("2024-01-10", [-500], cash=100_000.0)
    layer.before_trade(sell)
    layer.after_trade(sell)
    # Now before_trade on Jan11 (last date) — must NOT mature the pending item
    buy = _ctx("2024-01-11", [1], cash=100_000.0)
    out = layer.before_trade(buy)
    assert out.diagnostics["in_transit_cash"] > 0, (
        "Sell proceeds from the penultimate date must still be in transit on the last date"
    )


def test_settle_end_clamp_last_date_settle_date_after_last_calendar_date():
    """Sell on the last calendar date: settle_date must be strictly after that date.

    With the old code pos+2 was clamped to pos (T+0 effective).
    The fix returns BDay(2) from Jan11 = Jan15.
    """
    layer = _four_date_layer()
    sell = _ctx("2024-01-11", [-500], cash=0.0)
    layer.before_trade(sell)
    layer.after_trade(sell)
    pending = layer.get_state()["pending"]
    assert len(pending) == 1
    settle_date = pd.Timestamp(pending[0]["settle_date"])
    last_calendar_date = pd.Timestamp("2024-01-11")
    assert settle_date > last_calendar_date, (
        f"Expected settle_date > {last_calendar_date}, got {settle_date}; "
        "a sell on the last trading day must not settle on that same day"
    )

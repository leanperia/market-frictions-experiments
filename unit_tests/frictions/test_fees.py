"""Direction checks for NYSE and PSE fee schedules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.fees import NyseFeeSchedule, PseFeeSchedule
from meta.frictions.pipeline import TradeContext


def _ctx(date, trades, prices=None):
    n = len(trades)
    return TradeContext(
        time=0,
        date=pd.Timestamp(date),
        prices=np.array(prices or [100.0] * n, dtype=np.float64),
        volume=np.array([1e7] * n),
        volatility=np.array([0.02] * n),
        holdings=np.zeros(n),
        cash=1_000_000.0,
        target_weights=np.zeros(n + 1),
        trades=np.array(trades, dtype=np.int64),
    )


def test_nyse_sells_pay_sec_and_taf_buys_only_commission():
    layer = NyseFeeSchedule(commission_bps=1.0)
    ctx_buy = _ctx("2024-06-01", [1000])
    layer.after_trade(ctx_buy)
    ctx_sell = _ctx("2024-06-01", [-1000])
    layer.after_trade(ctx_sell)
    assert ctx_sell.diagnostics["fees_explicit"] > ctx_buy.diagnostics["fees_explicit"]
    assert ctx_buy.diagnostics["fees_by_type"]["sec"] == 0
    assert ctx_buy.diagnostics["fees_by_type"]["finra_taf"] == 0
    assert ctx_sell.diagnostics["fees_by_type"]["sec"] > 0
    assert ctx_sell.diagnostics["fees_by_type"]["finra_taf"] > 0


def test_pse_stt_drops_after_cmepa_effective_date():
    layer = PseFeeSchedule()
    pre = _ctx("2025-06-30", [-1000])  # day before CMEPA
    layer.after_trade(pre)
    post = _ctx("2025-07-01", [-1000])  # CMEPA day
    layer.after_trade(post)
    assert post.diagnostics["fees_by_type"]["stock_transaction_tax"] < pre.diagnostics["fees_by_type"]["stock_transaction_tax"]
    assert pre.diagnostics["stt_rate_today"] == 0.006
    assert post.diagnostics["stt_rate_today"] == 0.001


def test_pse_fees_strictly_greater_than_nyse_for_same_trade():
    nyse = NyseFeeSchedule()
    pse = PseFeeSchedule()
    n_ctx = _ctx("2024-06-01", [1000])
    nyse.after_trade(n_ctx)
    p_ctx = _ctx("2024-06-01", [1000])
    pse.after_trade(p_ctx)
    assert p_ctx.diagnostics["fees_explicit"] > n_ctx.diagnostics["fees_explicit"]


def test_cash_is_deducted_by_total_fee():
    layer = NyseFeeSchedule(commission_bps=1.0)
    ctx = _ctx("2024-06-01", [1000])
    starting_cash = ctx.cash
    layer.after_trade(ctx)
    assert abs((starting_cash - ctx.cash) - ctx.diagnostics["fees_explicit"]) < 1e-9

"""Settlement composition checks (tests-review FRIC-1).

SettlementLag queues NET sell proceeds: gross sell notional minus the
``sell_side_costs`` that fee/spread layers (which must run earlier in
the pipeline) accumulate into ``ctx.diagnostics``. These tests pin down
that composition at the layer level and then check full-stack cash
conservation through PortfolioOptimizationImpactEnv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
    PortfolioOptimizationImpactEnv,
)
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.frictions.fees import PseFeeSchedule
from meta.frictions.pipeline import FrictionPipeline, TradeContext
from meta.frictions.settlement import SettlementLag
from meta.frictions.spread import SpreadHaircut


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


def test_settlement_queues_net_proceeds_after_sell_side_costs():
    """Sell 1000 @ 100 → gross 100_000; sell_side_costs 2500 → net 97_500.

    The fee/spread layers accumulate their sell-side share via
    ``diagnostics.add_scalar("sell_side_costs", ...)``; settlement must
    queue gross − costs, and the ``in_transit_cash`` diagnostic must
    reflect the same net amount.
    """
    layer = SettlementLag(cycle_days=2)
    ctx = _ctx("2024-01-02", [-1000], cash=200_000)
    layer.before_trade(ctx)
    # Same diagnostics API the fee/spread layers use to record costs.
    ctx.diagnostics.add_scalar("sell_side_costs", 2500.0)
    out = layer.after_trade(ctx)

    pending = layer.get_state()["pending"]
    assert len(pending) == 1
    assert pending[0]["amount"] == 97_500.0
    assert out.diagnostics["in_transit_cash"] == 97_500.0


def test_settlement_clamps_over_cost_sell_to_zero_not_negative():
    """Degenerate case: sell_side_costs exceed gross proceeds.

    Sell 10 @ 100 → gross 1000; costs 2500 → net = max(0, −1500) = 0.
    Characterizes the silent max(0, ...) clamp in
    SettlementLag.after_trade: nothing is queued, nothing negative
    enters the pending queue, and no warning/diagnostic flags the
    discrepancy (the excess cost simply never transits).
    """
    layer = SettlementLag(cycle_days=2)
    ctx = _ctx("2024-01-02", [-10], cash=200_000)
    layer.before_trade(ctx)
    ctx.diagnostics.add_scalar("sell_side_costs", 2500.0)
    out = layer.after_trade(ctx)

    pending = layer.get_state()["pending"]
    assert pending == []  # net clamped to 0 → nothing queued
    assert all(p["amount"] >= 0.0 for p in pending)
    assert out.diagnostics["in_transit_cash"] == 0.0


def test_full_stack_cash_conservation_on_flat_prices():
    """One rebalancing step on flat prices: money is neither created nor lost.

    Setup: 2 stocks at a flat 100.0, start with holdings = [50, 0] and
    cash = 5000 → initial total = 10_000. Action [-20, -20, 20] targets
    ~100% weight on stock 1; with cost_buffer_pct = 0.02 the sizing
    value is 9800, so the env sells 50 sh of S0 (notional 5000) and
    buys 98 sh of S1 (notional 9800); total traded notional 14_800.

    Hand-computed costs (PSE schedule, 2024-01-02 → pre-CMEPA STT 0.6%):
    * impact (Baseline 10 bps): 14_800 × 0.001            =  14.80
    * fees_explicit: commission 14_800 × 0.0025 = 37.00
                     + VAT 37.00 × 0.12         =  4.44
                     + SCCP 14_800 × 0.0001     =  1.48
                     + PSE tx 14_800 × 0.00005  =  0.74
                     + STT 5000 × 0.006         = 30.00   →  73.66
    * spread (10 bps full, half-spread 5 bps): 14_800 × 0.0005 = 7.40

    Final total asset = 10_000 − 14.80 − 73.66 − 7.40 = 9_904.14, and
    initial − final must equal fees + spread + impact to 1e-6.
    """
    t, n = 3, 2
    close = np.full((t, n), 100.0)
    config = {
        "date_list": np.array(pd.date_range("2024-01-01", periods=t, freq="B")),
        "price_array": close,
        "open_array": close.copy(),
        "tech_array": np.zeros((t, n), dtype=np.float64),
        "volatility_array": np.full((t, n), 0.02),
        "volume_array": np.full((t, n), 1_000_000),
        "tic_list": np.array([f"S{i}" for i in range(n)]),
    }
    env = PortfolioOptimizationImpactEnv(
        config,
        initial_amount=10_000.0,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=10),
        initial_portfolio_state={"cash": 5000.0, "holdings": np.array([50.0, 0.0])},
        friction_pipeline=FrictionPipeline(
            [PseFeeSchedule(), SpreadHaircut(spread_array=np.full((t, n), 0.001))]
        ),
        cost_buffer_pct=0.02,
    )
    env.reset()
    initial_total = float(env._cash_balance + (env.holdings * 100.0).sum())
    assert initial_total == 10_000.0

    _state, _reward, _terminated, _truncated, info = env.step(
        np.array([-20.0, -20.0, 20.0])
    )

    # Trades realized as sized above; no cash clamp fired.
    np.testing.assert_array_equal(env.holdings, [0.0, 98.0])
    assert "cash_shortfall" not in info["friction"]

    impact_cost = info["cost"]
    fees_explicit = info["friction"]["fees_explicit"]
    spread_cost = info["friction"]["spread_cost"]
    assert impact_cost == pytest.approx(14.80)
    assert fees_explicit == pytest.approx(73.66)
    assert spread_cost == pytest.approx(7.40)

    # Conservation: the entire NAV drop is exactly the explicit costs.
    final_total = float(env.total_asset)
    assert final_total == pytest.approx(9_904.14)
    assert initial_total - final_total == pytest.approx(
        fees_explicit + spread_cost + impact_cost, abs=1e-6
    )

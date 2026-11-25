"""Prove the env's cash-shortfall clamp alarm CAN fire (tests-review ENV-3).

The committed-stack smoke evidence shows ``cash_shortfall`` is never
recorded in normal F0–F3 runs.  That is only meaningful if the alarm is
demonstrably live, so here we inject a hostile friction layer whose
``after_trade`` deducts more cash than the env holds and assert the
clamp block (env step 5c) records the overdraft:

* ``cash_shortfall`` equals the overdraft magnitude,
* ``cash_shortfall_over_tol`` is set iff the shortfall exceeds
  0.1% of ``initial_amount`` (tol = 0.001 * 1000 = 1.0 here),
* env cash is clamped to exactly 0.0.

Setup hand-computation: initial_amount = 1000, one stock at 100 with
zero impact (0 bps) and zero cost buffer; action [-20, 20] softmaxes to
~[0, 1], so the env buys 10 shares for exactly 1000.0 leaving cash 0.0.
The drain layer then subtracts ``drain`` from ctx.cash, so the
pre-clamp balance is exactly ``-drain`` and the shortfall equals
``drain``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
    PortfolioOptimizationImpactEnv,
)
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.frictions.pipeline import FrictionLayer, FrictionPipeline


class _CashDrainLayer(FrictionLayer):
    """Hostile layer: after trades execute, deduct a fixed cash amount.

    ``estimate_after_trade`` is left as the pass-through default on
    purpose, so the env's cash scaler cannot anticipate the drain and
    the balance is forced below zero.
    """

    name = "cash_drain_probe"

    def __init__(self, drain: float) -> None:
        self.drain = float(drain)

    def after_trade(self, ctx):
        ctx.cash -= self.drain
        return ctx


def _env(drain, initial_amount=1000.0):
    t, n = 3, 1
    close = np.full((t, n), 100.0)
    config = {
        "date_list": np.array(pd.date_range("2024-01-01", periods=t, freq="B")),
        "price_array": close,
        "open_array": close.copy(),
        "tech_array": np.zeros((t, n), dtype=np.float64),
        "volatility_array": np.full((t, n), 0.02),
        "volume_array": np.full((t, n), 1_000_000),
        "tic_list": np.array(["S0"]),
    }
    return PortfolioOptimizationImpactEnv(
        config,
        initial_amount=initial_amount,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=0),
        friction_pipeline=FrictionPipeline([_CashDrainLayer(drain)]),
        cost_buffer_pct=0.0,
    )


def test_overdraft_beyond_tolerance_fires_alarm_and_clamps_cash_to_zero():
    """Drain 50.0 against a 0.0 balance: shortfall = 50.0, tol = 1.0,
    so the over-tolerance alarm must be set and cash clamped to 0."""
    env = _env(drain=50.0)
    env.reset()

    _state, _reward, _terminated, _truncated, info = env.step(np.array([-20.0, 20.0]))

    assert env.holdings[0] == 10  # bought 10 x 100 with the full 1000
    assert info["friction"]["cash_shortfall"] == pytest.approx(50.0, abs=1e-9)
    assert info["friction"]["cash_shortfall_over_tol"] is True
    assert env.cash == 0.0
    assert env._cash_balance == 0.0


def test_overdraft_within_tolerance_records_shortfall_without_alarm():
    """Drain 0.5: shortfall = 0.5 <= tol = 1.0.  The shortfall is still
    recorded and cash still clamped, but the over-tolerance flag is not
    set (the clamp block only writes it when shortfall > tol)."""
    env = _env(drain=0.5)
    env.reset()

    _state, _reward, _terminated, _truncated, info = env.step(np.array([-20.0, 20.0]))

    assert info["friction"]["cash_shortfall"] == pytest.approx(0.5, abs=1e-9)
    assert "cash_shortfall_over_tol" not in info["friction"]
    assert env.cash == 0.0


def test_no_overdraft_records_no_shortfall():
    """Control: an empty pipeline with zero-cost impact leaves cash at
    exactly 0 after the all-in buy — no shortfall key, no clamp event.
    This is the normal-run direction the smoke evidence already shows."""
    t, n = 3, 1
    close = np.full((t, n), 100.0)
    config = {
        "date_list": np.array(pd.date_range("2024-01-01", periods=t, freq="B")),
        "price_array": close,
        "open_array": close.copy(),
        "tech_array": np.zeros((t, n), dtype=np.float64),
        "volatility_array": np.full((t, n), 0.02),
        "volume_array": np.full((t, n), 1_000_000),
        "tic_list": np.array(["S0"]),
    }
    env = PortfolioOptimizationImpactEnv(
        config,
        initial_amount=1000.0,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=0),
        friction_pipeline=FrictionPipeline([]),
        cost_buffer_pct=0.0,
    )
    env.reset()

    _state, _reward, _terminated, _truncated, info = env.step(np.array([-20.0, 20.0]))

    assert "cash_shortfall" not in info["friction"]
    assert "cash_shortfall_over_tol" not in info["friction"]
    assert env.cash >= 0.0

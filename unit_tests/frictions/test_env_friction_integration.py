"""Integration checks for frictions inside PortfolioOptimizationImpactEnv."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.env_market_impact.envs.env_portfolio_optimization_impact import PortfolioOptimizationImpactEnv
from meta.env_market_impact.envs.impact_models import BaselineImpactModel, SqrtImpactModel
from meta.frictions.execution_timing import NextDayExecution
from meta.frictions.fees import NyseFeeSchedule
from meta.frictions.board_lots import BoardLotRounding
from meta.frictions.pipeline import FrictionLayer, FrictionPipeline
from meta.frictions.settlement import SettlementLag
from meta.policies.classical import UniformBuyAndHold


def _config(close_prices, open_prices=None, quote_open_prices=None):
    close = np.asarray(close_prices, dtype=np.float64)
    if close.ndim == 1:
        close = close[:, None]
    open_ = np.asarray(open_prices, dtype=np.float64) if open_prices is not None else close.copy()
    if open_.ndim == 1:
        open_ = open_[:, None]
    t, n = close.shape
    cfg = {
        "date_list": np.array(pd.date_range("2024-01-01", periods=t, freq="B")),
        "price_array": close,
        "open_array": open_,
        "tech_array": np.zeros((t, n), dtype=np.float64),
        "volatility_array": np.full((t, n), 0.02),
        "volume_array": np.full((t, n), 1_000_000),
        "tic_list": np.array([f"S{i}" for i in range(n)]),
    }
    if quote_open_prices is not None:
        quote_open = np.asarray(quote_open_prices, dtype=np.float64)
        if quote_open.ndim == 1:
            quote_open = quote_open[:, None]
        cfg["quote_open_array"] = quote_open
        cfg["quote_price_array"] = close.copy()
    return cfg


def _env(close_prices, open_prices=None, pipeline=None, initial_amount=1000.0, cost_buffer_pct=0.0):
    return PortfolioOptimizationImpactEnv(
        _config(close_prices, open_prices),
        initial_amount=initial_amount,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=0),
        friction_pipeline=pipeline,
        cost_buffer_pct=cost_buffer_pct,
    )


def test_next_day_execution_fills_at_open_not_close():
    env = _env(
        close_prices=[100.0, 100.0, 100.0],
        open_prices=[100.0, 50.0, 100.0],
        pipeline=FrictionPipeline([NextDayExecution()]),
    )
    env.reset()
    _state, _reward, _terminated, _truncated, info = env.step(np.array([-20.0, 20.0]))
    assert env.holdings[0] == 20
    assert info["trades"][0]["notional"] == 1000.0
    assert env.total_asset == 2000.0
    assert info["friction"]["execution_price"] == "next_open"


def test_next_day_execution_preserves_permanent_impact_in_fill_price():
    env = PortfolioOptimizationImpactEnv(
        _config([100.0, 100.0, 100.0, 100.0], open_prices=[100.0, 100.0, 100.0, 100.0]),
        initial_amount=1000.0,
        max_trade_volume_pct=1.0,
        impact_model=SqrtImpactModel(Y=10.0, perm_fraction=1.0, perm_half_life_days=999999),
        friction_pipeline=FrictionPipeline([NextDayExecution()]),
    )
    env.reset()
    env.step(np.array([-20.0, 20.0]))
    perm_shift = env.impact_model.get_perm_state("S0")

    _state, _reward, _terminated, _truncated, info = env.step(np.array([20.0, -20.0]))

    assert info["trades"][0]["side"] == "sell"
    assert info["trades"][0]["price"] > 100.0
    np.testing.assert_allclose(info["trades"][0]["price"], 100.0 + perm_shift, rtol=1e-5)


def test_env_board_lot_uses_quoted_next_open_not_adjusted_execution_price():
    env = PortfolioOptimizationImpactEnv(
        _config(
            [42.820547, 42.820547, 42.820547],
            open_prices=[42.820547, 42.820547, 42.820547],
            quote_open_prices=[52.25, 52.25, 52.25],
        ),
        initial_amount=2500.0,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=0),
        friction_pipeline=FrictionPipeline([NextDayExecution(), BoardLotRounding()]),
    )
    env.reset()
    _state, _reward, _terminated, _truncated, info = env.step(np.array([-20.0, 20.0]))
    assert env.holdings[0] == 50
    np.testing.assert_allclose(info["quoted_execution_prices"], [52.25])


def test_env_preserves_inclusive_quoted_lot_boundary_precision():
    env = PortfolioOptimizationImpactEnv(
        _config(
            [42.0, 42.0, 42.0],
            open_prices=[42.0, 42.0, 42.0],
            quote_open_prices=[49.95, 49.95, 49.95],
        ),
        initial_amount=5000.0,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=0),
        friction_pipeline=FrictionPipeline([NextDayExecution(), BoardLotRounding()]),
    )
    env.reset()
    env.step(np.array([-20.0, 20.0]))
    assert env.holdings[0] == 100


def test_cash_scaler_prevents_free_transaction_cost_clamp():
    env = PortfolioOptimizationImpactEnv(
        _config([100.0, 100.0, 100.0]),
        initial_amount=1000.0,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=10),
        friction_pipeline=FrictionPipeline([]),
        cost_buffer_pct=0.0,
    )
    env.reset()

    _state, _reward, _terminated, _truncated, info = env.step(np.array([-20.0, 20.0]))

    assert env.cash >= 0.0
    assert "cash_shortfall" not in info["friction"]
    assert env.holdings[0] == 9
    assert env.total_asset < 1000.0


def test_cash_scaler_does_not_skip_large_impact_shortfall():
    cfg = _config([100.0, 100.0, 100.0])
    cfg["volatility_array"] = np.full((3, 1), 1.0)
    cfg["volume_array"] = np.full((3, 1), 1_000.0)
    env = PortfolioOptimizationImpactEnv(
        cfg,
        initial_amount=1000.0,
        max_trade_volume_pct=1.0,
        impact_model=SqrtImpactModel(Y=10.0, perm_fraction=0.25),
        cost_buffer_pct=0.10,
    )
    env.reset()

    _state, _reward, _terminated, _truncated, info = env.step(np.array([-20.0, 20.0]))

    assert env.cash >= 0.0
    assert "cash_shortfall" not in info["friction"]
    assert env.holdings[0] < 9


class _ResetProbe(FrictionLayer):
    name = "reset_probe"

    def __init__(self):
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_env_reset_resets_friction_pipeline_layers():
    layer = _ResetProbe()
    env = _env([100.0, 100.0, 100.0], pipeline=FrictionPipeline([layer]))
    env.reset()
    env.step(np.array([-20.0, 20.0]))
    env.reset()
    assert layer.reset_count == 2


def test_env_exposes_friction_state_roundtrip():
    layer = SettlementLag(cycle_days=2)
    env = _env(
        [100.0, 100.0, 100.0, 100.0],
        pipeline=FrictionPipeline([NyseFeeSchedule(), layer]),
        initial_amount=1000.0,
    )
    env.reset()
    env.holdings[0] = 10
    env.cash = 0.0
    env._cash_balance = 0.0
    env.step(np.array([20.0, -20.0]))
    state = env.get_friction_state()
    assert "settlement" in state
    assert state["settlement"]["pending"]

    restored = _env(
        [100.0, 100.0, 100.0, 100.0],
        pipeline=FrictionPipeline([NyseFeeSchedule(), SettlementLag(cycle_days=2)]),
        initial_amount=1000.0,
    )
    restored.reset()
    restored.set_friction_state(state)
    assert restored.get_friction_state() == state


def test_ubah_does_not_rebalance_after_initial_purchase_with_sizing_buffer():
    prices = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
        ]
    )
    env = _env(prices, initial_amount=1000.0, cost_buffer_pct=0.10)
    policy = UniformBuyAndHold()
    env.reset()

    _state, _reward, _terminated, _truncated, info = policy.step(env)
    assert len(info["trades"]) == 2
    first_holdings = env.holdings.copy()
    assert env.cash == 100.0

    for _ in range(3):
        _state, _reward, _terminated, _truncated, info = policy.step(env)
        assert info["trades"] == []
        assert info["reallocation_requested"] is False
        assert info["target_weights"] is None
        np.testing.assert_array_equal(env.holdings, first_holdings)
        assert env.cash == 100.0

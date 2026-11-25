"""F0-versus-F1 execution chronology (readiness finding 5.2, 2026-07-25).

The friction ladder declares two distinct chronologies:

* **F0** — no execution-timing layer, the optimistic ablation anchor: an action
  formed from day *t* information fills at that **same day's close**.
* **F1-F3** — ``NextDayExecution`` active, the realistic case: the same action
  is delayed and fills at the **day *t+1* open**.

Valuation always advances to the day *t+1* close in both branches, so F1 must
execute strictly *later* in wall-clock time than F0.

Before the fix the environment incremented ``time`` before resolving the
execution price, so F0 filled at the day *t+1* **close** — later than F1's
day *t+1* open. That inverted the ladder and made the Stage 3 execution-timing
sensitivity a next-open-versus-next-close comparison rather than the documented
next-open-versus-same-close.

These tests pin the dates and prices directly, using deliberately different
closes and opens so a wrong index cannot coincidentally produce a right answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
    PortfolioOptimizationImpactEnv,
)
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.frictions.execution_timing import NextDayExecution
from meta.frictions.pipeline import FrictionPipeline

# Deliberately distinct per-bar closes and opens: every candidate index maps to
# a unique price, so an off-by-one is always observable.
CLOSES = [100.0, 200.0, 400.0, 800.0]
OPENS = [110.0, 210.0, 410.0, 810.0]


def _config():
    close = np.asarray(CLOSES, dtype=np.float64)[:, None]
    open_ = np.asarray(OPENS, dtype=np.float64)[:, None]
    t, n = close.shape
    return {
        "date_list": np.array(pd.date_range("2024-01-01", periods=t, freq="B")),
        "price_array": close,
        "open_array": open_,
        "tech_array": np.zeros((t, n), dtype=np.float64),
        "volatility_array": np.full((t, n), 0.02),
        "volume_array": np.full((t, n), 1_000_000),
        "tic_list": np.array(["S0"]),
        "quote_price_array": close.copy(),
        "quote_open_array": open_.copy(),
    }


def _env(pipeline):
    return PortfolioOptimizationImpactEnv(
        _config(),
        initial_amount=10_000.0,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=0),
        friction_pipeline=pipeline,
        cost_buffer_pct=0.0,
    )


def _buy_everything():
    # Softmax over [cash, S0] that puts ~everything into the single stock.
    return np.array([-50.0, 50.0])


def test_f0_fills_at_the_same_day_close():
    """F0: decide on day 0, fill at day 0's close (100.0), value at day 1."""
    env = _env(pipeline=None)
    env.reset()
    _s, _r, _term, _trunc, info = env.step(_buy_everything())

    assert info["execution_lag_days"] == 0
    assert info["decision_date"] == info["execution_date"]
    assert info["base_execution_prices"][0] == pytest.approx(CLOSES[0])
    assert info["trades"][0]["price"] == pytest.approx(CLOSES[0])
    # Valuation still advances one bar.
    assert info["valuation_date"] == env.date_list[1]
    assert info["base_close_prices"][0] == pytest.approx(CLOSES[1])


def test_f1_fills_at_the_next_day_open():
    """F1: decide on day 0, fill at day 1's open (210.0), value at day 1."""
    env = _env(pipeline=FrictionPipeline([NextDayExecution()]))
    env.reset()
    _s, _r, _term, _trunc, info = env.step(_buy_everything())

    assert info["execution_lag_days"] == 1
    assert info["execution_date"] == env.date_list[1]
    assert info["base_execution_prices"][0] == pytest.approx(OPENS[1])
    assert info["trades"][0]["price"] == pytest.approx(OPENS[1])
    assert info["valuation_date"] == env.date_list[1]
    assert info["friction"]["execution_price"] == "next_open"


def test_f1_executes_later_than_f0_not_earlier():
    """The ladder's core invariant: adding the timing layer delays execution."""
    f0 = _env(pipeline=None)
    f0.reset()
    _s, _r, _t, _tr, f0_info = f0.step(_buy_everything())

    f1 = _env(pipeline=FrictionPipeline([NextDayExecution()]))
    f1.reset()
    _s, _r, _t, _tr, f1_info = f1.step(_buy_everything())

    assert f1_info["execution_date"] > f0_info["execution_date"]
    assert f1_info["execution_lag_days"] > f0_info["execution_lag_days"]
    # Both mark the result on the same valuation bar.
    assert f1_info["valuation_date"] == f0_info["valuation_date"]


def test_f0_does_not_execute_at_the_next_close():
    """Direct regression on the defect: F0 must not fill at day 1's close."""
    env = _env(pipeline=None)
    env.reset()
    _s, _r, _term, _trunc, info = env.step(_buy_everything())
    assert info["base_execution_prices"][0] != pytest.approx(CLOSES[1])


def test_execution_index_helper_matches_the_declared_ladder():
    assert _env(pipeline=None).execution_index_for_decision(2) == 2
    timed = _env(pipeline=FrictionPipeline([NextDayExecution()]))
    assert timed.execution_index_for_decision(2) == 3


def test_chronology_holds_across_a_whole_episode():
    """Every step of both branches keeps the declared date relationships."""
    for pipeline, expected_lag in ((None, 0), (FrictionPipeline([NextDayExecution()]), 1)):
        env = _env(pipeline=pipeline)
        env.reset()
        terminated = False
        steps = 0
        while not terminated:
            _s, _r, terminated, _tr, info = env.step(_buy_everything())
            steps += 1
            assert info["execution_lag_days"] == expected_lag
            # Execution never precedes the decision, never follows valuation.
            assert info["decision_date"] <= info["execution_date"]
            assert info["execution_date"] <= info["valuation_date"]
        assert steps > 1

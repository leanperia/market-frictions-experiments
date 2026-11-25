"""F0 pipeline-equivalence invariant (tests-review FRIC-2).

The ``meta/frictions/pipeline.py`` module docstring promises that an env
constructed with ``FrictionPipeline([])`` (empty layer list) produces the
same trajectory as one constructed with ``friction_pipeline=None``. This
is the F0 rung of the friction ladder: every higher friction level is
measured against F0, so an empty pipeline that silently perturbed the
trajectory would bias the whole ladder. Until now no test asserted it.

Both envs here use a nonzero ``BaselineImpactModel`` (10 bps) so the
comparison is not trivially zero-cost: impact costs are deducted from
cash on every trade in BOTH envs, and the equivalence must hold anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
    PortfolioOptimizationImpactEnv,
)
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.frictions.pipeline import FrictionPipeline

T, N = 12, 3  # 12 business days, 3 stocks -> 11 rebalancing steps
SEED = 7
INITIAL = 10_000.0

# Keys the impact model itself writes into the step diagnostics (added by
# the env unconditionally, pipeline or not). Anything else appearing in
# info["friction"] would mean the empty pipeline injected costs.
_IMPACT_ONLY_KEYS = {"impact_sell_side_cost", "sell_side_costs"}


def _close_panel() -> np.ndarray:
    """Seeded geometric random walk, ~1% daily vol around 100."""
    rng = np.random.default_rng(SEED)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(T, N)), axis=0)


def _config(close: np.ndarray) -> dict:
    t, n = close.shape
    return {
        "date_list": np.array(pd.date_range("2024-01-01", periods=t, freq="B")),
        "price_array": close,
        "open_array": close.copy(),
        "tech_array": np.zeros((t, n), dtype=np.float64),
        "volatility_array": np.full((t, n), 0.02),
        "volume_array": np.full((t, n), 1_000_000.0),
        "tic_list": np.array([f"S{i}" for i in range(n)]),
    }


def _env(pipeline) -> PortfolioOptimizationImpactEnv:
    # Each env gets its own impact-model instance so no state is shared
    # between the two trajectories being compared.
    return PortfolioOptimizationImpactEnv(
        _config(_close_panel()),
        initial_amount=INITIAL,
        max_trade_volume_pct=1.0,
        impact_model=BaselineImpactModel(basis_points=10),
        friction_pipeline=pipeline,
        cost_buffer_pct=0.01,
    )


def _action_sequence() -> np.ndarray:
    """Fixed seeded logits (cash + N stocks) forcing rebalancing each step."""
    rng = np.random.default_rng(SEED + 1)
    return rng.normal(0.0, 1.0, size=(T - 1, 1 + N))


def test_empty_pipeline_trajectory_identical_to_no_pipeline():
    env_none = _env(pipeline=None)
    env_empty = _env(pipeline=FrictionPipeline([]))
    env_none.reset()
    env_empty.reset()

    total_impact_cost = 0.0
    total_trades = 0
    for actions in _action_sequence():
        state_n, reward_n, term_n, trunc_n, info_n = env_none.step(actions)
        state_e, reward_e, term_e, trunc_e, info_e = env_empty.step(actions)

        # Step-by-step identical portfolio trajectory (bit-exact: the empty
        # pipeline branch performs only no-op transforms on float32 data).
        np.testing.assert_array_equal(env_empty.holdings, env_none.holdings)
        assert env_empty.cash == env_none.cash
        assert env_empty.total_asset == env_none.total_asset
        assert reward_e == reward_n
        assert (term_e, trunc_e) == (term_n, trunc_n)
        np.testing.assert_array_equal(state_e, state_n)

        # Identical executed trades (same fills, prices, impact costs).
        assert info_e["trades"] == info_n["trades"]
        assert info_e["cost"] == info_n["cost"]
        assert info_e["turnover"] == info_n["turnover"]
        np.testing.assert_array_equal(
            info_e["execution_prices"], info_n["execution_prices"]
        )

        total_impact_cost += info_n["cost"]
        total_trades += len(info_n["trades"])

    # Both envs must have terminated after T-1 steps.
    assert term_n and term_e

    # Guard against a vacuous pass: the shared trajectory actually traded
    # and paid nonzero baseline impact (10 bps of traded notional).
    assert total_trades > 0
    assert total_impact_cost > 0.0
    assert env_none.total_asset < INITIAL  # costs were really deducted


def test_empty_pipeline_friction_diagnostics_match_and_carry_no_layer_costs():
    """info['friction'] must be equivalent: same dict, impact-only keys.

    The empty pipeline records no layer diagnostics, so the per-step
    friction dict may contain only the impact model's own sell-side cost
    scalars (written by the env regardless of pipeline) — never fee /
    spread / settlement / lot-rounding entries.
    """
    env_none = _env(pipeline=None)
    env_empty = _env(pipeline=FrictionPipeline([]))
    env_none.reset()
    env_empty.reset()

    saw_sell_side_cost = False
    for actions in _action_sequence():
        *_, info_n = env_none.step(actions)
        *_, info_e = env_empty.step(actions)

        friction_n = info_n["friction"]
        friction_e = info_e["friction"]
        assert friction_e == friction_n
        assert set(friction_e) <= _IMPACT_ONLY_KEYS

        if "impact_sell_side_cost" in friction_e:
            saw_sell_side_cost = True
            # 10 bps of this step's sell notional, identically in both envs.
            expected = 0.001 * info_e["total_sell_value"]
            assert friction_e["impact_sell_side_cost"] == pytest.approx(expected, rel=1e-6)
            assert friction_e["sell_side_costs"] == pytest.approx(expected, rel=1e-6)

    # The seeded action sequence includes sell steps, so the impact-model
    # diagnostics path is genuinely exercised (comparison is not no-cost).
    assert saw_sell_side_cost

"""Option-1 income crediting and legal-share board lots — ADR-0003 §A1, handoff §7.3-4."""

from __future__ import annotations

import numpy as np
import pytest

from meta.frictions.board_lots import BoardLotRounding
from meta.frictions.pipeline import TradeContext


def _ctx(trades, prices, share_factors=None, holdings=None):
    n = len(trades)
    return TradeContext(
        time=0, date="2020-01-02",
        prices=np.asarray(prices, dtype=float),
        volume=np.full(n, 1e9), volatility=np.full(n, 0.01),
        holdings=np.zeros(n) if holdings is None else np.asarray(holdings, dtype=float),
        cash=1e9,
        target_weights=np.zeros(n + 1),
        trades=np.asarray(trades, dtype=np.int64),
        quoted_prices=np.asarray(prices, dtype=float),
        share_factors=None if share_factors is None else np.asarray(share_factors, dtype=float),
    )


class TestBoardLotsInLegalShareSpace:
    def test_unit_share_factor_reproduces_the_old_behaviour(self):
        """Pre-Option-1 panels must be bit-identical: panel units are legal shares."""
        layer = BoardLotRounding()
        # price 30 -> lot 100
        with_none = layer.before_trade(_ctx([1234, -1234], [30.0, 30.0]))
        with_ones = layer.before_trade(_ctx([1234, -1234], [30.0, 30.0], [1.0, 1.0]))
        np.testing.assert_array_equal(with_none.trades, [1200, -1200])
        np.testing.assert_array_equal(with_ones.trades, with_none.trades)

    def test_lot_is_enforced_in_legal_shares_not_panel_units(self):
        """A x12 stock dividend makes one legal share twelve panel shares, so a
        100-share lot occupies 1200 panel shares."""
        layer = BoardLotRounding()
        # 2500 panel shares = 208.33 legal shares -> 200 legal -> 2400 panel
        out = layer.before_trade(_ctx([2500], [30.0], [12.0]))
        assert out.trades[0] == 2400
        assert out.diagnostics["blocked_by_lot"][0] == 100

    def test_rounding_never_increases_a_trade(self):
        """The clamp that makes a fractional share factor safe."""
        layer = BoardLotRounding()
        rng = np.random.default_rng(0)
        for _ in range(300):
            qty = int(rng.integers(1, 100_000))
            factor = float(rng.uniform(0.2, 13.0))
            price = float(rng.choice([0.3, 3.0, 30.0, 300.0, 3000.0]))
            out = layer.before_trade(_ctx([qty], [price], [factor]))
            assert 0 <= out.trades[0] <= qty
            assert out.diagnostics["blocked_by_lot"][0] == qty - out.trades[0]

    def test_sign_is_preserved_for_sells(self):
        layer = BoardLotRounding()
        out = layer.before_trade(_ctx([-2500], [30.0], [12.0]))
        assert out.trades[0] == -2400

    def test_degenerate_share_factors_fall_back_to_unity(self):
        layer = BoardLotRounding()
        for bad in (0.0, -1.0, np.nan, np.inf):
            out = layer.before_trade(_ctx([1234], [30.0], [bad]))
            assert out.trades[0] == 1200


class TestDividendCrediting:
    def _env(self, dividend_array=None, share_factor_array=None):
        from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
            PortfolioOptimizationImpactEnv,
        )
        n_days, n_stocks = 6, 2
        price = np.full((n_days, n_stocks), 100.0)
        cfg = {
            "price_array": price.copy(),
            "open_array": price.copy(),
            "high_array": price.copy(),
            "low_array": price.copy(),
            "tech_array": np.zeros((n_days, n_stocks)),
            "volatility_array": np.full((n_days, n_stocks), 0.01),
            "volume_array": np.full((n_days, n_stocks), 1e9),
            "adv20_array": np.full((n_days, n_stocks), 1e9),
            "date_list": [f"2020-01-{d + 1:02d}" for d in range(n_days)],
            "tic_list": ["A", "B"],
        }
        if dividend_array is not None:
            cfg["dividend_array"] = dividend_array
        if share_factor_array is not None:
            cfg["share_factor_array"] = share_factor_array
        return PortfolioOptimizationImpactEnv(cfg, initial_amount=1_000_000.0)

    def test_no_dividend_array_leaves_cash_untouched(self):
        """Pre-Option-1 panels must behave exactly as before."""
        env = self._env()
        env.reset()
        env.step(np.array([0.0, 5.0, 5.0], dtype=np.float32))
        assert getattr(env, "_last_dividend_income", 0.0) == 0.0

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("dividend_array", np.nan, "non-finite"),
            ("dividend_array", -1.0, "non-negative"),
            ("share_factor_array", 0.0, "strictly positive"),
            ("share_factor_array", np.inf, "non-finite"),
        ],
    )
    def test_invalid_accounting_channels_are_rejected(self, field, value, match):
        array = np.ones((6, 2))
        if field == "dividend_array":
            array[:] = 0.0
        array[2, 0] = value
        with pytest.raises(ValueError, match=match):
            self._env(**{field: array})

    def test_income_is_credited_on_pre_trade_holdings(self):
        """The holder of record is whoever held *into* the ex-date."""
        n_days, n_stocks = 6, 2
        divs = np.zeros((n_days, n_stocks))
        divs[2, 0] = 1.50  # A goes ex on bar 2
        env = self._env(dividend_array=divs)
        env.reset()
        # build a position first
        env.step(np.array([0.0, 5.0, 5.0], dtype=np.float32))
        held = env.holdings.copy()
        cash_before = env._cash_balance
        env.step(np.array([0.0, 5.0, 5.0], dtype=np.float32))
        assert env._last_dividend_income == pytest.approx(held[0] * 1.50)
        assert env._last_dividend_income > 0.0
        # the credit is real cash, not a price effect
        assert env._cash_balance > cash_before - env._portfolio_value

    def test_income_never_moves_the_price_scale(self):
        n_days, n_stocks = 6, 2
        divs = np.zeros((n_days, n_stocks))
        divs[2, 0] = 25.0
        env_with = self._env(dividend_array=divs)
        env_without = self._env()
        env_with.reset(); env_without.reset()
        for _ in range(3):
            env_with.step(np.array([0.0, 5.0, 5.0], dtype=np.float32))
            env_without.step(np.array([0.0, 5.0, 5.0], dtype=np.float32))
        np.testing.assert_allclose(env_with.price_array, env_without.price_array)
        # ...but the dividend-paying book is worth strictly more
        assert env_with._portfolio_value > env_without._portfolio_value

    def test_ubah_reinvests_ex_date_income_pro_rata_and_leaves_no_cash(self):
        """UBAH compounds cash dividends without restoring equal weights.

        With flat prices and a zero-cost fill, equal initial holdings receive
        P5,000 each at the ex-date.  The policy re-invests the P10,000 cash in
        the existing 50:50 mix, taking both holdings from 5,000 to 5,050 and
        pinning the intended total-return convention at terminal cash ~= 0.
        """
        from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
            PortfolioOptimizationImpactEnv,
        )
        from meta.env_market_impact.envs.impact_models import BaselineImpactModel
        from meta.policies.classical import UniformBuyAndHold

        n_days, n_stocks = 4, 2
        prices = np.full((n_days, n_stocks), 100.0)
        dividends = np.zeros((n_days, n_stocks))
        dividends[2] = [1.0, 1.0]
        env = PortfolioOptimizationImpactEnv(
            {
                "price_array": prices,
                "open_array": prices,
                "high_array": prices,
                "low_array": prices,
                "tech_array": np.zeros((n_days, n_stocks)),
                "volatility_array": np.full((n_days, n_stocks), 0.01),
                "volume_array": np.full((n_days, n_stocks), 1e9),
                "adv20_array": np.full((n_days, n_stocks), 1e9),
                "dividend_array": dividends,
                "date_list": [f"2020-01-0{d + 1}" for d in range(n_days)],
                "tic_list": ["A", "B"],
            },
            initial_amount=1_000_000.0,
            max_trade_volume_pct=1.0,
            impact_model=BaselineImpactModel(basis_points=0),
            friction_pipeline=None,
            cost_buffer_pct=0.0,
        )
        env.reset()
        policy = UniformBuyAndHold()
        policy.step(env)  # the one initial allocation
        policy.step(env)  # F0 ex-date income is credited after the prior close
        policy.step(env)  # the tagged income is reinvested at this close

        np.testing.assert_allclose(env.holdings, [5050.0, 5050.0])
        assert env._last_dividend_income == pytest.approx(0.0)
        assert env._cash_balance == pytest.approx(0.0, abs=1e-6)
        assert env._dividend_reinvestment_cash == pytest.approx(0.0, abs=1e-6)

    def test_f0_prior_close_purchase_earns_next_ex_date_dividend(self):
        dividends = np.zeros((6, 2))
        dividends[1, 0] = 10.0
        env = self._env(dividend_array=dividends)
        env.reset()
        _state, _reward, _terminated, _truncated, info = env.step(
            np.array([-20.0, 20.0, 20.0], dtype=np.float32)
        )
        assert info["execution_date"] == "2020-01-01"
        assert info["valuation_date"] == "2020-01-02"
        assert env.holdings[0] > 0
        expected_income = float(env.holdings[0] * 10.0)
        assert info["dividend_income"] == pytest.approx(expected_income)
        assert env._cash_balance >= expected_income

    def test_f0_prior_close_sale_does_not_earn_next_ex_date_dividend(self):
        dividends = np.zeros((6, 2))
        dividends[1, 0] = 10.0
        env = self._env(dividend_array=dividends)
        env.initial_portfolio_state = {
            "cash": 0.0,
            "holdings": np.array([5_000.0, 0.0]),
        }
        env.reset()
        _state, _reward, _terminated, _truncated, info = env.step(
            np.array([20.0, -20.0, -20.0], dtype=np.float32)
        )
        assert env.holdings[0] == 0
        assert info["dividend_income"] == pytest.approx(0.0)

    def test_ubah_small_dividend_never_sells_to_rebuild_cost_buffer(self):
        from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
            PortfolioOptimizationImpactEnv,
        )
        from meta.env_market_impact.envs.impact_models import BaselineImpactModel
        from meta.policies.classical import UniformBuyAndHold

        prices = np.array(
            [[100.0, 100.0], [200.0, 200.0], [200.0, 200.0], [200.0, 200.0]]
        )
        dividends = np.zeros_like(prices)
        dividends[2] = [0.01, 0.01]
        cfg = {
            "price_array": prices,
            "open_array": prices,
            "tech_array": np.zeros_like(prices),
            "volatility_array": np.full_like(prices, 0.01),
            "volume_array": np.full_like(prices, 1e9),
            "dividend_array": dividends,
            "date_list": [f"2020-01-0{day}" for day in range(1, 5)],
            "tic_list": ["A", "B"],
        }
        env = PortfolioOptimizationImpactEnv(
            cfg,
            initial_amount=1_000_000.0,
            max_trade_volume_pct=1.0,
            impact_model=BaselineImpactModel(basis_points=0),
            cost_buffer_pct=0.007,
        )
        env.reset()
        policy = UniformBuyAndHold()
        policy.step(env)
        _state, _reward, _terminated, _truncated, ex_info = policy.step(env)
        _state, _reward, _terminated, _truncated, reinvest_info = policy.step(env)
        assert ex_info["trades"] == []
        assert reinvest_info["trades"] == []
        assert reinvest_info["dividend_reinvestment_residual"] > 0
        assert (env.holdings >= 0).all()

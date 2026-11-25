"""
Portfolio Optimization Environment with Market Impact

This environment adapts the POE (Portfolio Optimization Environment) from
Costa & Costa (2023) to the FinRL-Meta framework, incorporating a realistic
market impact model while preserving the original reward function (log return)
and the weight-based action paradigm from the paper.

The impact model integration (permanent impact tracking, ``end_day()`` decay,
trade-level logging) and the runner interface (normalizer stubs, ``total_asset``,
``cash``, etc.) are added on top of the original formulation to enable
consistent backtesting and comparison against the stock-trading environment.

Accounting model
~~~~~~~~~~~~~~~~
The portfolio is tracked via explicit **cash** and **holdings** (share
counts).  Portfolio value is ``cash + sum(holdings * prices)`` using market
prices.  After each rebalancing step, *actual* weights are derived from
holdings so that the next step's price-variation correctly reflects the
real portfolio composition (not just the softmax target).

Reference:
    Costa, C., & Costa, A. (2023). POE: A General Portfolio Optimization
    Environment for FinRL. In Anais do II Brazilian Workshop on Artificial
    Intelligence in Finance (pp. 132-143). SBC.
    https://doi.org/10.5753/bwaif.2023.231144
"""

from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple

import gymnasium as gym
import numpy as np

from .impact_models import ImpactModel
from .impact_models import SqrtImpactModel
from meta.frictions import (
    FrictionPipeline,
    FrictionStepDiagnostics,
    TradeContext,
    estimate_impact_cost,
)

EPS = 1e-8
EPS_PRICE = 1e-4   # price floor when applying permanent impact (avoid ≤0)
ACTION_LOGIT_LIMIT = 20.0

# Share counts are integers, and every trade is floored to a whole share, but they
# are stored as floats because holdings are multiplied by prices on every valuation
# and restored from a float-valued ``initial_portfolio_state``.
#
# float64, not float32 (PR27-5 / reset review N2). float32 represents integers
# exactly only up to 2**24 = 16,777,216, and the capital sensitivity suite
# deliberately scales PSE capital to ~USD 1M (PHP 55,000,000): a 14.3% NAV
# allocation to the panel's cheapest name (FLI, minimum close PHP 0.4699) is
# 16.7M shares, so precision degraded *inside the arm under study*. float64 is
# exact to 2**53, which no reachable position approaches.
#
# An integer dtype was considered and rejected: numpy would then silently truncate
# any fractional value written into ``holdings`` (a corporate-action or
# friction-layer bug would become invisible rather than loud), and every
# ``holdings * prices`` product would change promotion rules. float64 keeps the
# existing arithmetic and lets the property test assert exact integrality
# (unit_tests/properties/test_env_large_capital_precision.py).
HOLDINGS_DTYPE = np.float64


class PortfolioOptimizationImpactEnv(gym.Env):
    """
    A portfolio optimization environment that incorporates market impact.

    Preserves the original Costa & Costa (2023) reward (log return) and the
    softmax weight-based action space, while adding:

    * Config-dict input (compatible with ``MarketDataPreparator``).
    * Pluggable ``ImpactModel`` with ``end_day()`` decay and permanent
      impact tracked via the model.
    * Holdings-based accounting: cash and share counts are tracked
      explicitly so that portfolio value and weights always reflect
      the actual post-trade composition.
    * Trade-level logging (POV, turnover percentile) for post-hoc analysis.
    * Normalizer-state stubs so the runner and Optuna scripts work unchanged.
    * ``total_asset`` / ``cash`` / ``time`` / ``max_step`` attributes
      expected by the backtesting pipeline.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Dict,
        initial_amount: float = 1e6,
        reward_scaling: float = 1.0,
        max_trade_volume_pct: float = 0.1,
        impact_model: Optional[ImpactModel] = None,
        initial_portfolio_state: Optional[Dict] = None,
        friction_pipeline: Optional[FrictionPipeline] = None,
        cost_buffer_pct: float = 0.0,
    ) -> None:
        self.date_list = config["date_list"]
        price_array = config["price_array"]
        tech_array = config["tech_array"]
        self.volatility_array = config.get(
            "volatility_array", np.ones_like(price_array) * 0.02
        )
        self.volume_array = config.get("volume_array", np.ones_like(price_array) * 1e6)

        self.price_array = price_array.astype(np.float32)
        self.open_array = config.get("open_array", price_array).astype(np.float32)
        self.policy_price_history_array = config.get(
            "policy_price_history_array",
            np.empty((0, price_array.shape[1]), dtype=np.float64),
        ).astype(np.float64)
        # Exchange-rule thresholds are price-band boundaries. Preserve quote
        # precision rather than downcasting through float32 at exact bands
        # such as PSE PHP 49.95.
        self.quote_price_array = config.get("quote_price_array", price_array).astype(np.float64)
        self.quote_open_array = config.get(
            "quote_open_array", config.get("open_array", price_array)
        ).astype(np.float64)
        # Option-1 accounting channels (ADR-0003 §A1). Defaults preserve
        # pre-Option-1 behaviour exactly: a share factor of 1.0 means panel
        # units already are legal shares, and a zero dividend stream means the
        # panel carries no income (the superseded total-return scale folded it
        # into prices instead).
        self.share_factor_array = np.asarray(
            config.get("share_factor_array", np.ones_like(price_array)),
            dtype=np.float64,
        )
        self.dividend_array = np.asarray(
            config.get("dividend_array", np.zeros_like(price_array)),
            dtype=np.float64,
        )
        for name, array in (
            ("share_factor_array", self.share_factor_array),
            ("dividend_array", self.dividend_array),
        ):
            if array.shape != price_array.shape:
                raise ValueError(
                    f"{name} shape {array.shape} does not match price_array "
                    f"shape {price_array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if (self.share_factor_array <= 0).any():
            raise ValueError("share_factor_array must be strictly positive")
        if (self.dividend_array < 0).any():
            raise ValueError("dividend_array must be non-negative")
        # ADR-0009 causal-context channels. The mask distinguishes builder-
        # fabricated (forward-filled) rows from observed sessions; policies
        # that rank on history must know the difference, so the env records
        # whether the panel actually provided it rather than defaulting to
        # "all clean".
        halt_array = config.get("synthetic_halt_array")
        self.halt_mask_provided = halt_array is not None
        if halt_array is None:
            halt_array = np.zeros(price_array.shape, dtype=bool)
        self.synthetic_halt_array = np.asarray(halt_array, dtype=bool)
        if self.synthetic_halt_array.shape != price_array.shape:
            raise ValueError(
                f"synthetic_halt_array shape {self.synthetic_halt_array.shape} "
                f"does not match price_array shape {price_array.shape}"
            )
        n_stocks = price_array.shape[1]
        history_len = self.policy_price_history_array.shape[0]
        self.policy_date_history_array = np.asarray(
            config.get(
                "policy_date_history_array", np.empty(0, dtype="datetime64[ns]")
            ),
            dtype="datetime64[ns]",
        )
        self.policy_dividend_history_array = np.asarray(
            config.get(
                "policy_dividend_history_array",
                np.zeros((0, n_stocks), dtype=np.float64),
            ),
            dtype=np.float64,
        )
        self.policy_synthetic_halt_history_array = np.asarray(
            config.get(
                "policy_synthetic_halt_history_array",
                np.zeros((0, n_stocks), dtype=bool),
            ),
            dtype=bool,
        )
        if (
            "policy_date_history_array" in config
            and self.policy_date_history_array.shape[0] != history_len
        ):
            raise ValueError(
                f"policy_date_history_array length "
                f"{self.policy_date_history_array.shape[0]} does not match "
                f"policy_price_history_array length {history_len}"
            )
        if (
            "policy_dividend_history_array" in config
            and self.policy_dividend_history_array.shape
            != self.policy_price_history_array.shape
        ):
            raise ValueError(
                f"policy_dividend_history_array shape "
                f"{self.policy_dividend_history_array.shape} does not match "
                f"policy_price_history_array shape "
                f"{self.policy_price_history_array.shape}"
            )
        if (
            "policy_synthetic_halt_history_array" in config
            and self.policy_synthetic_halt_history_array.shape
            != self.policy_price_history_array.shape
        ):
            raise ValueError(
                f"policy_synthetic_halt_history_array shape "
                f"{self.policy_synthetic_halt_history_array.shape} does not "
                f"match policy_price_history_array shape "
                f"{self.policy_price_history_array.shape}"
            )
        if (
            self.halt_mask_provided
            and history_len
            and "policy_synthetic_halt_history_array" not in config
        ):
            # A masked panel always produces a masked history through the
            # adapter; silence here would treat unknown history as all-clean.
            raise ValueError(
                "config provides synthetic_halt_array and a non-empty "
                "policy_price_history_array but no "
                "policy_synthetic_halt_history_array"
            )
        self.tech_array = tech_array.astype(np.float32) * 2**-7

        self.stock_dim = self.price_array.shape[1]
        self.stock_symbols = config["tic_list"]
        self._initial_amount = initial_amount
        self.initial_capital = initial_amount  # runner compatibility
        self._reward_scaling = reward_scaling
        self._max_trade_volume_pct = max_trade_volume_pct
        self.impact_model = (
            impact_model if impact_model is not None else SqrtImpactModel()
        )
        self.initial_portfolio_state = initial_portfolio_state
        self.friction_pipeline = friction_pipeline
        self._last_friction_diagnostics = FrictionStepDiagnostics()
        # Cost-buffer reserves a small fraction of equity for fees / spread /
        # impact when sizing buys, preventing cash from drifting negative
        # after costs are deducted.  0.5% is enough to absorb PSE F2 costs
        # (~30-70bps fees + ~5-20bps spread + ~5-15bps impact).
        self._cost_buffer_pct = float(cost_buffer_pct)

        # Action: raw logits for (cash + N stocks) -> softmax
        action_space_dim = 1 + self.stock_dim

        # State: [price_variation(N), current_weights(1+N),
        #         last_action(1+N), perm_impact(N), tech(T)]
        self.state_dim = (
            self.stock_dim  # price variation
            + action_space_dim  # current weights (cash + stocks)
            + action_space_dim  # last action
            + self.stock_dim  # permanent impact
            + self.tech_array.shape[1]  # tech indicators
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-ACTION_LOGIT_LIMIT,
            high=ACTION_LOGIT_LIMIT,
            shape=(action_space_dim,),
            dtype=np.float32,
        )
        self.max_step = self.price_array.shape[0] - 1

    # ── Permanent impact helpers ──────────────────────────────────────

    def _get_perm_impact(self) -> np.ndarray:
        return self.impact_model.get_perm_state_array(self.stock_symbols)

    def _effective_prices(
        self, base_prices: np.ndarray, perm_shift: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply cumulative permanent impact to base (historical) prices.

        The impact model accumulates a per-symbol price-units shift in
        ``_perm_state`` whenever the agent trades, and decays it on
        ``end_day()``.  Without this helper the env values and executes
        trades at historical prices only — so the agent's own past trades
        have zero persistent effect on subsequent valuations or fills.

        ``perm_shift`` defaults to the model's current state.  Callers
        that need to snapshot before mutation (e.g. before the trade
        loop) pass an explicit copy.
        """
        if perm_shift is None:
            perm_shift = self._get_perm_impact()
        return np.maximum(np.asarray(base_prices) + np.asarray(perm_shift), EPS_PRICE).astype(base_prices.dtype)

    # ── Normalizer stubs (for runner compatibility) ───────────────────

    def get_normalizer_state(self) -> Optional[Dict]:
        return None

    def set_normalizer_state(self, state: Dict, freeze: bool = True) -> None:
        pass

    # ── Portfolio state transfer (train → OOS continuation) ────────────

    def get_portfolio_state(self) -> Dict:
        """Return current portfolio state for env-to-env transfer.

        Pass the returned dict as ``initial_portfolio_state`` when
        constructing the OOS environment so that the test run continues
        from the exact end-of-training cash and holdings.
        """
        return {
            "cash": self._cash_balance,
            "holdings": self.holdings.copy(),
            "dividend_reinvestment_cash": self._dividend_reinvestment_cash,
        }

    def get_friction_state(self) -> Dict:
        if self.friction_pipeline is None:
            return {}
        return self.friction_pipeline.get_state()

    def set_friction_state(self, state: Dict) -> None:
        if self.friction_pipeline is not None:
            self.friction_pipeline.set_state(state)

    # ── Turnover percentile helper ────────────────────────────────────

    def _calculate_turnover_percentiles(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        gross_notional = prices * volumes
        n = len(gross_notional)
        if n <= 1:
            return np.full(n, 50.0)
        sorted_indices = np.argsort(gross_notional)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(1, n + 1)
        return (ranks - 1) / (n - 1) * 100

    @property
    def _active_exec_index(self) -> int:
        """Execution bar for the step in flight (finding 5.2).

        ``step()`` publishes the resolved execution index here so the helper
        paths that re-enter the friction pipeline — affordability estimates and
        the post-cash-scaling quantizers — price, quote, and date their contexts
        on the bar the trade actually fills, not on the valuation bar. Outside a
        step it degrades to ``self.time``.
        """
        index = getattr(self, "_step_exec_index", None)
        return self.time if index is None else index

    def _uses_next_day_open_execution(self) -> bool:
        """True when an execution-timing layer defers fills to the next open."""
        return (
            self.friction_pipeline is not None
            and self.friction_pipeline.execution_price_source == "open"
        )

    def execution_index_for_decision(self, decision_index: int) -> int:
        """Bar index where an action decided on ``decision_index`` actually fills.

        The friction ladder declares two distinct chronologies, and they must not
        collapse into one another (2026-07-25 readiness finding 5.2):

        * **F0 (no execution-timing layer)** — the optimistic anchor: an action
          formed from day *t* information fills at that same day's close.
        * **F1-F3 (``NextDayExecution`` active)** — the realistic case: the same
          action is delayed and fills at the day *t+1* open.

        Valuation always advances to *t+1* regardless, so F1-F3 genuinely execute
        **later** than F0 rather than earlier.
        """
        return (
            decision_index + 1
            if self._uses_next_day_open_execution()
            else decision_index
        )

    def execution_prices_for_time(self, time: int) -> np.ndarray:
        """Return the price vector used to fill trades at ``time``."""
        if self.friction_pipeline is not None and self.friction_pipeline.execution_price_source == "open":
            return self.open_array[time]
        return self.price_array[time]

    def quoted_execution_prices_for_time(self, time: int) -> np.ndarray:
        """Return exchange-quoted prices for price-banded legal constraints."""
        if self.friction_pipeline is not None and self.friction_pipeline.execution_price_source == "open":
            return self.quote_open_array[time]
        return self.quote_price_array[time]

    def _estimate_impact_cost(
        self,
        trades: np.ndarray,
        prices: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> float:
        """Estimate impact costs without mutating the live impact model."""
        return estimate_impact_cost(
            self.impact_model,
            trades,
            prices,
            volatility,
            volume,
            self.stock_symbols,
        )

    def _apply_stateless_after_trade_costs(
        self,
        cash: float,
        trades: np.ndarray,
        prices: np.ndarray,
        volume: np.ndarray,
        volatility: np.ndarray,
        target_weights: Optional[np.ndarray],
        diagnostics: Optional[Dict],
    ) -> float:
        """Estimate explicit fee/spread cash deductions without mutating layers.

        Fees and spread are stateless after-trade cash costs. Settlement also
        has an after-trade hook, but it queues proceeds and must not be invoked
        during affordability checks because it is stateful and does not change
        immediate cash anyway.
        """
        if self.friction_pipeline is None:
            return float(cash)

        exec_index = self._active_exec_index
        ctx = TradeContext(
            time=exec_index,
            date=self.date_list[exec_index],
            prices=prices,
            volume=volume,
            volatility=volatility,
            holdings=self.holdings + trades,
            cash=float(cash),
            target_weights=(
                target_weights
                if target_weights is not None
                else np.zeros(1 + self.stock_dim, dtype=np.float32)
            ),
            trades=trades,
            quoted_prices=self.quoted_execution_prices_for_time(exec_index),
            close_prices=self.price_array[exec_index],
            open_prices=self.open_array[exec_index],
            share_factors=self.share_factor_array[exec_index],
            diagnostics=FrictionStepDiagnostics.coerce(diagnostics).to_dict(),
        )
        ctx = self.friction_pipeline.estimate_after_trade(ctx)
        return float(ctx.cash)

    def _cash_after_trades_estimate(
        self,
        trades: np.ndarray,
        prices: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
        *,
        include_impact: bool = True,
        include_pipeline_costs: bool = False,
        target_weights: Optional[np.ndarray] = None,
        diagnostics: Optional[Dict] = None,
    ) -> float:
        trades_f = trades.astype(np.float64)
        prices_f = prices.astype(np.float64)
        buys = np.where(trades_f > 0, trades_f * prices_f, 0.0).sum()
        sells = np.where(trades_f < 0, -trades_f * prices_f, 0.0).sum()
        cash = float(self._cash_balance + sells - buys)
        if include_impact:
            cash -= self._estimate_impact_cost(trades, prices, volatility, volume)
        if include_pipeline_costs:
            cash = self._apply_stateless_after_trade_costs(
                cash,
                trades,
                prices,
                volume,
                volatility,
                target_weights,
                diagnostics,
            )
        return cash

    def _scale_buys_to_cash(
        self,
        trades: np.ndarray,
        prices: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
        diagnostics: Dict,
        *,
        include_pipeline_costs: bool = False,
        target_weights: Optional[np.ndarray] = None,
        minimum_cash: float = 0.0,
    ) -> np.ndarray:
        """Scale buy trades down if estimated trade+impact costs exceed cash.

        This is a guardrail under the normal friction-level cost buffers. It
        runs before the friction pipeline, so later board-lot rounding can
        still restore legal PSE lot sizes after any scaling.
        """
        trades = trades.copy()
        buy_mask = trades > 0
        if not buy_mask.any():
            return trades

        if self._cash_after_trades_estimate(
            trades,
            prices,
            volatility,
            volume,
            include_impact=True,
            include_pipeline_costs=include_pipeline_costs,
            target_weights=target_weights,
            diagnostics=diagnostics,
        ) >= float(minimum_cash) - EPS:
            return trades

        original = trades.copy()
        lo, hi = 0.0, 1.0
        best = trades.copy()
        best[buy_mask] = 0
        for _ in range(24):
            mid = (lo + hi) / 2.0
            candidate = original.copy()
            candidate[buy_mask] = np.floor(original[buy_mask].astype(np.float64) * mid).astype(original.dtype)
            if self._cash_after_trades_estimate(
                candidate,
                prices,
                volatility,
                volume,
                include_impact=True,
                include_pipeline_costs=include_pipeline_costs,
                target_weights=target_weights,
                diagnostics=diagnostics,
            ) >= float(minimum_cash) - EPS:
                best = candidate
                lo = mid
            else:
                hi = mid

        diagnostics = FrictionStepDiagnostics.coerce(diagnostics)
        blocked = np.zeros_like(trades)
        blocked[buy_mask] = original[buy_mask] - best[buy_mask]
        diagnostics.add_array("blocked_by_cash", blocked)
        diagnostics["cash_scale"] = float(lo)
        return best

    def _reapply_post_cash_quantizers(
        self,
        trades: np.ndarray,
        prices: np.ndarray,
        volume: np.ndarray,
        volatility: np.ndarray,
        target_weights: np.ndarray,
        diagnostics: Dict,
    ) -> tuple[np.ndarray, FrictionStepDiagnostics]:
        """Re-run stateless trade quantizers after cash scaling.

        Cash scaling can produce non-board-lot share counts.  Reapplying the
        board-lot layer after scaling only reduces trade sizes, so it cannot
        violate ADV/outage/settlement caps already applied earlier.
        """
        if self.friction_pipeline is None:
            return trades, diagnostics
        exec_index = self._active_exec_index
        ctx = TradeContext(
            time=exec_index,
            date=self.date_list[exec_index],
            prices=prices,
            volume=volume,
            volatility=volatility,
            holdings=self.holdings,
            cash=self._cash_balance,
            target_weights=target_weights,
            trades=trades,
            quoted_prices=self.quoted_execution_prices_for_time(exec_index),
            close_prices=self.price_array[exec_index],
            open_prices=self.open_array[exec_index],
            share_factors=self.share_factor_array[exec_index],
            diagnostics=diagnostics,
        )
        ctx = self.friction_pipeline.after_cash_scaling(ctx)
        return np.asarray(ctx.trades, dtype=trades.dtype), ctx.diagnostics.clone()

    # ── Softmax normalisation (original paper, numerically stable) ────

    def _softmax_normalization(self, actions: np.ndarray) -> np.ndarray:
        """Normalizes the action vector using the softmax function."""
        actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        expected = 1 + self.stock_dim
        if actions.size != expected:
            raise ValueError(f"Expected action length {expected}, got {actions.size}.")
        # Classical policies use log(target_weight + 1e-8), which fits inside
        # this range while still letting zero-weight assets softmax to a
        # practically zero allocation for the 30-stock universes in scope.
        actions = np.nan_to_num(
            actions,
            nan=0.0,
            posinf=ACTION_LOGIT_LIMIT,
            neginf=-ACTION_LOGIT_LIMIT,
        )
        actions = np.clip(actions, -ACTION_LOGIT_LIMIT, ACTION_LOGIT_LIMIT)
        exp_actions = np.exp(actions - np.max(actions))
        return (exp_actions / np.sum(exp_actions)).astype(np.float32)

    # ── Compute actual weights from cash + holdings ───────────────────

    def _compute_weights(self, prices: np.ndarray) -> None:
        """Recompute ``_current_weights`` from explicit cash and holdings."""
        stock_values = self.holdings * prices
        self._portfolio_value = self._cash_balance + stock_values.sum()
        self.total_asset = self._portfolio_value

        if self._portfolio_value > EPS:
            self._current_weights[0] = self._cash_balance / self._portfolio_value
            self._current_weights[1:] = stock_values / self._portfolio_value
        else:
            self._current_weights[0] = 1.0
            self._current_weights[1:] = 0.0

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        self._step_exec_index = None

        if self.initial_portfolio_state is not None:
            # Continued-from-training: restore cash and holdings
            ps = self.initial_portfolio_state
            self._cash_balance = ps["cash"]
            self.holdings = ps["holdings"].copy().astype(HOLDINGS_DTYPE)
            self._dividend_reinvestment_cash = float(
                ps.get("dividend_reinvestment_cash", 0.0)
            )

            # Derive portfolio value from actual state at day-0 prices
            day0_prices = self.price_array[self.time]
            stock_values = self.holdings * day0_prices
            self._portfolio_value = self._cash_balance + stock_values.sum()
        else:
            # Fresh start: 100% cash
            self._cash_balance = self._initial_amount
            self._portfolio_value = self._initial_amount
            self.holdings = np.zeros(self.stock_dim, dtype=HOLDINGS_DTYPE)
            self._dividend_reinvestment_cash = 0.0

        self.total_asset = self._portfolio_value
        self.cash = self._cash_balance

        self._current_weights = np.zeros(1 + self.stock_dim, dtype=np.float32)
        if self.initial_portfolio_state is not None:
            self._compute_weights(self.price_array[self.time])
        else:
            self._current_weights[0] = 1.0  # 100% cash

        self._actions_memory = [self._current_weights.copy()]
        self._final_weights = [self._current_weights.copy()]
        self._asset_memory = {
            "initial": [self._portfolio_value],
            "final": [self._portfolio_value],
        }
        self._portfolio_return_memory = [0.0]
        self._date_memory = [self.date_list[self.time]]

        self.episode_return = 0.0
        self._last_dividend_income = 0.0

        if options is None or options.get("reset_impact_model", True):
            self.impact_model.reset()
        if self.friction_pipeline is not None and (
            options is None or options.get("reset_friction_pipeline", True)
        ):
            self.friction_pipeline.reset()

        return self._build_state(), {}

    # ── Step ──────────────────────────────────────────────────────────

    def step(
        self,
        actions: np.ndarray,
        *,
        rebalance: bool = True,
        reinvest_dividends: bool = False,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if rebalance and reinvest_dividends:
            raise ValueError(
                "target-weight rebalancing and buy-only dividend reinvestment "
                "are mutually exclusive"
            )
        # Decision / execution / valuation are three distinct dates (finding
        # 5.2). The action was formed from the observation at ``decision_index``;
        # it fills at ``exec_index`` (same close under F0, next open under
        # F1-F3); the resulting holdings are always marked at the next close.
        decision_index = self.time
        self.time += 1
        terminated = self.time >= self.max_step
        valuation_index = self.time
        exec_index = self.execution_index_for_decision(decision_index)
        # Published for the helper paths that re-enter the friction pipeline.
        self._step_exec_index = exec_index

        close_prices = self.price_array[valuation_index]
        exec_close_prices = self.price_array[exec_index]
        exec_prices = self.execution_prices_for_time(exec_index)
        quoted_exec_prices = self.quoted_execution_prices_for_time(exec_index)
        # Liquidity/risk constraints bind on the day the trade actually hits the
        # market, not on the valuation day.
        volatility = self.volatility_array[exec_index]
        volume = self.volume_array[exec_index]

        # Snapshot perm-state at step start so valuation/sizing/execution
        # use the cumulative price shift inherited from prior steps (after
        # the previous end_day decay).  New trades during this step add
        # further shift to _perm_state, but that compounding is left as a
        # within-step second-order effect.
        perm_shift_pre = self._get_perm_impact().copy()
        effective_exec_prices = self._effective_prices(exec_prices, perm_shift_pre)

        # ── 0. Cash-dividend chronology ───────────────────────────────
        # F1-F3 fill at the t+1 open, so the holdings carried into that open
        # receive the t+1 ex-date income before the fill. F0 fills at the t
        # close; its t+1 income belongs to the post-fill holdings and is
        # credited only after execution below. This prevents tomorrow's cash
        # from funding a prior-close trade.
        dividend_rates = self.dividend_array[valuation_index]
        dividend_before_trade = exec_index == valuation_index
        dividend_income = 0.0
        if dividend_before_trade:
            dividend_income = float(np.dot(self.holdings, dividend_rates))
            if dividend_income:
                self._cash_balance += dividend_income
                self._dividend_reinvestment_cash += dividend_income
        self._last_dividend_income = dividend_income

        # ── 1. Mark-to-market before rebalancing ──────────────────────
        # Implementation-shortfall valuation (ADR-0001): value holdings at the
        # unaffected (base) price; the agent's own permanent impact must not
        # inflate its NAV. Execution below still fills at effective prices.
        stock_values_pre = self.holdings * exec_prices
        value_pre = self._cash_balance + stock_values_pre.sum()
        self._asset_memory["initial"].append(value_pre)

        # ── 2. Target weights from softmax, or observed hold weights ──
        if rebalance:
            weights = self._softmax_normalization(actions)
        else:
            weights = np.zeros(1 + self.stock_dim, dtype=np.float32)
            if value_pre > EPS:
                weights[0] = self._cash_balance / value_pre
                weights[1:] = stock_values_pre / value_pre
            else:
                weights[0] = 1.0
        self._actions_memory.append(weights)

        # ── 3. Compute rebalancing trades (in shares) ─────────────────
        # Reserve a small cash buffer (default 0.5%) for fees/spread/impact
        # so the buys don't drive cash into negative territory after costs.
        # Friction-level configs can raise this when fees are heavy.
        reinvestment_budget = 0.0
        reinvestment_cash_floor = 0.0
        if rebalance:
            sizing_value = value_pre * (1.0 - self._cost_buffer_pct)
            target_stock_values = sizing_value * weights[1:]
            trades_in_value = target_stock_values - stock_values_pre
            trades_in_shares = np.divide(
                trades_in_value,
                effective_exec_prices,
                out=np.zeros_like(trades_in_value),
                where=effective_exec_prices > 0,
            ).astype(int)
        elif reinvest_dividends:
            # UBAH income compounding is a distinct accounting intent, not a
            # target-weight rebalance. Only tagged dividend cash can fund it;
            # arbitrary initial/residual cash is never swept into stocks.
            reinvestment_budget = min(
                max(0.0, float(self._dividend_reinvestment_cash)),
                max(0.0, float(self._cash_balance)),
            )
            reinvestment_cash_floor = float(self._cash_balance) - reinvestment_budget
            held_values = np.maximum(stock_values_pre, 0.0)
            held_total = float(held_values.sum())
            if reinvestment_budget > EPS and held_total > EPS:
                desired_buy_values = reinvestment_budget * held_values / held_total
                trades_in_shares = np.floor(
                    np.divide(
                        desired_buy_values,
                        effective_exec_prices,
                        out=np.zeros_like(desired_buy_values),
                        where=effective_exec_prices > 0,
                    )
                ).astype(int)
            else:
                trades_in_shares = np.zeros(self.stock_dim, dtype=int)
        else:
            trades_in_shares = np.zeros(self.stock_dim, dtype=int)

        # Volume constraint
        volume_limit = (volume * self._max_trade_volume_pct).astype(int)
        trades_in_shares = np.clip(trades_in_shares, -volume_limit, volume_limit)

        # Can't sell more shares than currently held (no short selling)
        for i in range(self.stock_dim):
            if trades_in_shares[i] < 0:
                trades_in_shares[i] = max(trades_in_shares[i], -int(self.holdings[i]))

        # Turnover percentiles for logging (uses base prices for liquidity
        # ranking — the agent's impact doesn't change the cross-sectional
        # liquidity order)
        turnover_percentiles = self._calculate_turnover_percentiles(exec_prices, volume)

        # Guard cash before the friction pipeline so downstream layers that
        # quantise trades (PSE board lots) can still enforce their constraints.
        pre_pipeline_diagnostics = FrictionStepDiagnostics()
        trades_in_shares = self._scale_buys_to_cash(
            trades_in_shares,
            effective_exec_prices,
            volatility,
            volume,
            pre_pipeline_diagnostics,
            minimum_cash=(
                reinvestment_cash_floor if reinvest_dividends else 0.0
            ),
        )
        reinvestment_cash_before_execution = float(self._cash_balance)

        # ── 3b. Friction pipeline (before trade) ─────────────────────
        # Layers can clip trades, route through next-day fills, round to
        # board lots, etc. An empty pipeline is a no-op — verified by
        # Prototype #1.  ctx.prices is the EFFECTIVE price (base + perm)
        # so fees, spread, settlement, etc. all charge on the post-impact
        # mid that the agent actually faces.
        if self.friction_pipeline is not None:
            ctx = TradeContext(
                time=exec_index,
                date=self.date_list[exec_index],
                prices=effective_exec_prices,
                volume=volume,
                volatility=volatility,
                holdings=self.holdings,
                cash=self._cash_balance,
                target_weights=weights,
                trades=trades_in_shares,
                quoted_prices=quoted_exec_prices,
                close_prices=exec_close_prices,
                open_prices=self.open_array[exec_index],
                share_factors=self.share_factor_array[exec_index],
                diagnostics=pre_pipeline_diagnostics,
            )
            ctx = self.friction_pipeline.before_trade(ctx)
            trades_in_shares = np.asarray(ctx.trades, dtype=trades_in_shares.dtype)
            effective_exec_prices = np.asarray(ctx.prices, dtype=np.float32)
            self._cash_balance = ctx.cash
            self._last_friction_diagnostics = ctx.diagnostics.clone()
        else:
            self._last_friction_diagnostics = FrictionStepDiagnostics()

        trades_in_shares = self._scale_buys_to_cash(
            trades_in_shares,
            effective_exec_prices,
            volatility,
            volume,
            self._last_friction_diagnostics,
            include_pipeline_costs=True,
            target_weights=weights,
            minimum_cash=(
                reinvestment_cash_floor if reinvest_dividends else 0.0
            ),
        )
        trades_in_shares, self._last_friction_diagnostics = self._reapply_post_cash_quantizers(
            trades_in_shares,
            effective_exec_prices,
            volume,
            volatility,
            weights,
            self._last_friction_diagnostics,
        )

        # ── 4. Execute trades through impact model ────────────────────
        total_impact_cost = 0.0
        total_traded_value = 0.0
        total_buy_value = 0.0
        total_sell_value = 0.0
        total_sell_impact_cost = 0.0
        trades = []

        for i in range(self.stock_dim):
            ts = trades_in_shares[i]
            if ts == 0:
                continue

            # Trade at the EFFECTIVE price (base + accumulated perm shift
            # from prior steps and prior trades within this step).
            trade_price = float(effective_exec_prices[i])
            impact = self.impact_model.apply_trade(
                ts,
                trade_price,
                volatility[i],
                volume[i],
                self.stock_symbols[i],
            )
            total_impact_cost += impact.cost
            notional = abs(ts) * trade_price
            pov = abs(ts) / volume[i] if volume[i] > 0 else 0.0
            side = "buy" if ts > 0 else "sell"

            if ts > 0:
                total_buy_value += notional
            else:
                total_sell_value += notional
                total_sell_impact_cost += float(impact.cost)
            total_traded_value += notional

            trades.append(
                {
                    "stock_idx": i,
                    "symbol": str(self.stock_symbols[i]),
                    "side": side,
                    "signed_shares": int(ts),
                    "shares": abs(ts),
                    "price": trade_price,
                    "notional": notional,
                    "pov": pov,
                    "turnover_percentile": turnover_percentiles[i],
                    "impact_cost": float(impact.cost),
                }
            )

            # Update cash: buy costs cash, sell generates cash
            self._cash_balance -= ts * trade_price
            self._cash_balance -= impact.cost
            self.holdings[i] += ts

        # ── 5. Decay permanent impact ─────────────────────────────────
        # Decay is anchored to the date the trade printed.
        self.impact_model.end_day(self.date_list[exec_index])

        # ── 5b. Friction pipeline (after trade) ──────────────────────
        if total_sell_impact_cost > 0:
            self._last_friction_diagnostics.add_scalar("impact_sell_side_cost", total_sell_impact_cost)
            self._last_friction_diagnostics.add_scalar("sell_side_costs", total_sell_impact_cost)

        if self.friction_pipeline is not None:
            ctx = TradeContext(
                time=exec_index,
                date=self.date_list[exec_index],
                prices=effective_exec_prices,
                volume=volume,
                volatility=volatility,
                holdings=self.holdings,
                cash=self._cash_balance,
                target_weights=weights,
                trades=trades_in_shares,
                quoted_prices=quoted_exec_prices,
                close_prices=exec_close_prices,
                open_prices=self.open_array[exec_index],
                share_factors=self.share_factor_array[exec_index],
                diagnostics=self._last_friction_diagnostics,
            )
            ctx = self.friction_pipeline.after_trade(ctx)
            self._cash_balance = ctx.cash
            self._last_friction_diagnostics = ctx.diagnostics.clone()

        # ── 5c. Clamp cash invariant: if costs nibbled cash slightly
        # below zero despite the cost buffer, clamp to 0 and record the
        # shortfall as a diagnostic.  Anything beyond the tolerance is a
        # real bug worth surfacing — log it via the diagnostics dict.
        if self._cash_balance < 0:
            shortfall = float(-self._cash_balance)
            tol = 0.001 * self._initial_amount
            self._last_friction_diagnostics["cash_shortfall"] = shortfall
            if shortfall > tol:
                self._last_friction_diagnostics["cash_shortfall_over_tol"] = True
            self._cash_balance = 0.0

        if reinvest_dividends:
            reinvestment_spent = max(
                0.0,
                reinvestment_cash_before_execution - float(self._cash_balance),
            )
            self._dividend_reinvestment_cash = max(
                0.0,
                float(self._dividend_reinvestment_cash) - reinvestment_spent,
            )
        else:
            reinvestment_spent = 0.0

        # Under F0, execution was yesterday's close relative to the valuation
        # bar. Credit the current ex-date only now, using the holdings that
        # actually crossed that close.
        if not dividend_before_trade:
            dividend_income = float(np.dot(self.holdings, dividend_rates))
            if dividend_income:
                self._cash_balance += dividend_income
                self._dividend_reinvestment_cash += dividend_income
            self._last_dividend_income = dividend_income

        # ── 6. Post-trade portfolio value and actual weights ──────────
        # Implementation-shortfall valuation (ADR-0001): mark holdings at the
        # unaffected (base) close so the agent's own permanent impact never
        # inflates NAV/returns. effective_close_prices is retained only for the
        # diagnostics/observation, not for valuation.
        effective_close_prices = self._effective_prices(close_prices)
        self._compute_weights(close_prices)
        self.cash = self._cash_balance

        self._final_weights.append(self._current_weights.copy())
        self._asset_memory["final"].append(self._portfolio_value)

        # ── 7. Reward: log return (original paper) ────────────────────
        last_value = self._asset_memory["final"][-2]
        rate_of_return = self._portfolio_value / last_value if last_value > EPS else 1.0
        portfolio_return = rate_of_return - 1
        log_return = np.log(max(rate_of_return, EPS))

        self._portfolio_return_memory.append(portfolio_return)
        self._date_memory.append(self.date_list[valuation_index])
        reward = log_return * self._reward_scaling

        if terminated:
            self.episode_return = (
                self.total_asset / self._initial_amount
                if self._initial_amount > 0
                else 0.0
            )

        turnover = total_traded_value / self.total_asset if self.total_asset > 0 else 0
        info = {
            "reallocation_requested": bool(rebalance),
            "dividend_reinvestment_requested": bool(reinvest_dividends),
            "dividend_income": float(dividend_income),
            "dividend_reinvestment_budget": float(reinvestment_budget),
            "dividend_reinvestment_spent": float(reinvestment_spent),
            "dividend_reinvestment_residual": float(
                self._dividend_reinvestment_cash
            ),
            # Auditable chronology (finding 5.2): which bar the action was formed
            # on, which bar it filled on, and which bar the result is marked at.
            "decision_date": self.date_list[decision_index],
            "execution_date": self.date_list[exec_index],
            "valuation_date": self.date_list[valuation_index],
            "execution_lag_days": int(exec_index - decision_index),
            "turnover": turnover,
            "cost": total_impact_cost,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "cash": self.cash,
            "trades": trades,
            "execution_prices": effective_exec_prices.copy(),
            "base_execution_prices": exec_prices.copy(),
            "quoted_execution_prices": quoted_exec_prices.copy(),
            "close_prices": effective_close_prices.copy(),
            "base_close_prices": close_prices.copy(),
            "volume": volume.copy(),
            "target_weights": weights.copy() if rebalance else None,
            "realized_weights": self._current_weights.copy(),
            "friction": self._last_friction_diagnostics.to_dict(),
        }

        return self._build_state(), reward, terminated, False, info

    # ── State construction ────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """Build the flat observation vector.

        Components (original POE spirit, adapted for config-dict input):
        * price_variation(N): ratio curr_price / prev_price  (1.0 at t=0)
        * current_weights(1+N): actual cash weight + stock weights
        * last_action(1+N): the previous softmax weights (target)
        * permanent_impact(N): cumulative impact from the model
        * tech(T): technical indicators
        """
        perm_impact = self._get_perm_impact()

        if self.time == 0:
            price_var = np.ones(self.stock_dim, dtype=np.float32)
        else:
            # Price variation uses EFFECTIVE prices (base + perm shift)
            # so the agent's observation reflects its own lingering
            # market impact, not just historical data.
            prev_base = self.price_array[self.time - 1]
            curr_base = self.price_array[self.time]
            prev_eff = self._effective_prices(prev_base, perm_impact)
            curr_eff = self._effective_prices(curr_base, perm_impact)
            price_var = np.where(prev_eff > 0, curr_eff / prev_eff, 1.0).astype(np.float32)

        state_components = [
            price_var,
            self._current_weights,
            self._actions_memory[-1],
            perm_impact,
            self.tech_array[min(self.time, self.max_step)],
        ]
        return np.hstack(state_components).astype(np.float32)

    def render(self, mode="human"):
        return self.total_asset


def build_portfolio_env(
    env_config: Dict,
    settings: Mapping,
    *,
    seed: int,
    reward_scaling: Optional[float] = None,
) -> PortfolioOptimizationImpactEnv:
    """Construct and seed the supported friction-aware portfolio environment.

    ``env_config`` contains already aligned market arrays. Only ``market``
    (with the legacy ``name`` fallback), ``environment`` and ``friction`` are
    consumed from resolved settings. Acquisition, causal feature preparation,
    configuration layering and campaign admission remain with the caller.

    Calibration may override reward scaling without mutating its settings.
    Each call owns fresh pipeline/impact state and returns an already reset
    environment. This is the same construction used by primary evaluation.
    """
    from meta.frictions import build_impact_model, build_pipeline_from_config

    environment = settings.get("environment", {})
    friction = settings.get("friction", {})
    pipeline = build_pipeline_from_config(
        friction,
        env_config,
        market=settings.get("market", settings.get("name", "unknown")),
    )
    env = PortfolioOptimizationImpactEnv(
        config=env_config,
        initial_amount=float(environment.get("initial_amount", 1e6)),
        reward_scaling=(
            float(environment.get("reward_scaling", 1.0))
            if reward_scaling is None
            else reward_scaling
        ),
        max_trade_volume_pct=float(friction.get("max_trade_volume_pct", 0.1)),
        impact_model=build_impact_model(friction),
        friction_pipeline=pipeline,
        cost_buffer_pct=float(friction.get("cost_buffer_pct", 0.0)),
    )
    env.reset(seed=seed)
    return env

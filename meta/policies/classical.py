"""Non-RL portfolio policies.

These produce target portfolio weights at each step and feed them into
``PortfolioOptimizationImpactEnv.step`` through a softmax-inverse trick
(``logits = log(target_weights + ε)`` ⇒ ``softmax(logits) ≈ target``).
The env's trade computation, friction stack, and diagnostics layer
fire identically to RL — that's the whole point of the Stage 1
deterministic grid / Stage 2 non-RL comparator path.

Each policy needs read access to the env's ``price_array`` and current
``time`` index. We don't try to be SB3-compatible; the deterministic
non-RL runner loop calls ``policy.target_weights(env)`` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize


EPS = 1e-8


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    """softmax(log(w + ε)) ≈ w for w ≥ 0 summing to 1."""
    return np.log(np.asarray(weights, dtype=np.float64) + EPS).astype(np.float32)


def _observed_prices(env) -> np.ndarray:
    """Prices available when the policy chooses the next executed allocation."""
    current = np.asarray(env.price_array[: env.time + 1], dtype=np.float64)
    history = np.asarray(
        getattr(env, "policy_price_history_array", np.empty((0, env.stock_dim))),
        dtype=np.float64,
    )
    return np.vstack([history, current]) if history.size else current


def _observed_context(env) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dates, closes, dividends, and halt flags observable at the decision index.

    The sibling of ``_observed_prices`` for policies that rank on causal
    total-return history (ADR-0009). Every channel ends at ``env.time``
    inclusive; pre-window history channels are prepended when present. A stub
    or config that carries price history without aligned dates cannot place
    that history on the calendar, so it is rejected rather than guessed at.
    """
    t = env.time
    dates = np.asarray(env.date_list[: t + 1], dtype="datetime64[ns]")
    closes = np.asarray(env.price_array[: t + 1], dtype=np.float64)
    dividends = np.asarray(env.dividend_array[: t + 1], dtype=np.float64)
    halts = np.asarray(env.synthetic_halt_array[: t + 1], dtype=bool)
    history = np.asarray(
        getattr(env, "policy_price_history_array", np.empty((0, env.stock_dim))),
        dtype=np.float64,
    )
    if history.size:
        hist_dates = np.asarray(
            env.policy_date_history_array, dtype="datetime64[ns]"
        )
        if hist_dates.shape[0] != history.shape[0]:
            raise ValueError(
                "policy_date_history_array does not align with "
                "policy_price_history_array; cannot place history on the "
                "calendar"
            )
        hist_dividends = np.asarray(
            env.policy_dividend_history_array, dtype=np.float64
        )
        hist_halts = np.asarray(
            env.policy_synthetic_halt_history_array, dtype=bool
        )
        for name, channel in (
            ("policy_dividend_history_array", hist_dividends),
            ("policy_synthetic_halt_history_array", hist_halts),
        ):
            if channel.shape != history.shape:
                raise ValueError(
                    f"{name} shape {channel.shape} does not align with "
                    f"policy_price_history_array shape {history.shape}"
                )
        dates = np.concatenate([hist_dates, dates])
        closes = np.vstack([history, closes])
        dividends = np.vstack([hist_dividends, dividends])
        halts = np.vstack([hist_halts, halts])
    return dates, closes, dividends, halts


class NonRLPolicy(ABC):
    name: str = "non_rl"

    @abstractmethod
    def target_weights(self, env) -> np.ndarray:
        """Return target weights of shape (1 + N,): [cash, stock_1, ..., stock_N]."""

    def act(self, env) -> np.ndarray:
        w = np.asarray(self.target_weights(env), dtype=np.float64)
        # Fail loudly on non-finite target weights rather than letting NaN/inf
        # propagate into the env action (a silent-fallback hazard: the
        # ``sum <= 0`` guard below does not catch NaN because ``nan <= 0`` is
        # False). A deterministic allocation rule that cannot produce finite
        # weights is a bug, not an all-cash signal.
        if not np.isfinite(w).all():
            raise ValueError(
                f"{type(self).__name__}.target_weights returned non-finite weights: {w!r}"
            )
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / w.size
        else:
            w = w / s
        return weights_to_logits(w)

    def rebalance_requested(self, env) -> bool:
        """Whether this step should translate target weights into trades."""
        return True

    def step(self, env):
        """Advance the shared environment according to this rule."""
        return env.step(self.act(env), rebalance=self.rebalance_requested(env))


# ── Uniform Buy-and-Hold ──────────────────────────────────────────────


class UniformBuyAndHold(NonRLPolicy):
    """Hold a uniform initial allocation and compound cash dividends pro rata.

    After the one initial allocation, the policy never submits another target
    weight. The environment receives an explicit buy-only income-reinvestment
    instruction instead, so it can use the contemporaneous execution price
    without exposing a future open to the policy. Dividend cash is compounded
    in the realised (drifted) mix; existing shares are never sold.
    """

    name = "ubah"

    def __init__(self) -> None:
        self._initial_allocation_submitted = False

    def target_weights(self, env) -> np.ndarray:
        n_stocks = env.stock_dim
        w = np.zeros(1 + n_stocks)
        w[1:] = 1.0 / n_stocks
        return w

    def step(self, env):
        initial_allocation = self.rebalance_requested(env)
        return env.step(
            self.act(env),
            rebalance=initial_allocation,
            reinvest_dividends=not initial_allocation,
        )

    def rebalance_requested(self, env) -> bool:
        if not self._initial_allocation_submitted:
            self._initial_allocation_submitted = True
            existing_holdings = np.asarray(
                getattr(env, "holdings", []), dtype=np.float64
            )
            if not (existing_holdings.size and np.abs(existing_holdings).sum() > 0):
                return True
        return False


# ── Uniform Constant Rebalanced Portfolio ─────────────────────────────


class UniformConstantRebalancedPortfolio(NonRLPolicy):
    """Daily-rebalanced uniform weights across the N stocks (0 cash)."""

    name = "ucrp"

    def target_weights(self, env) -> np.ndarray:
        n_stocks = env.stock_dim
        w = np.zeros(1 + n_stocks)
        w[1:] = 1.0 / n_stocks
        return w


# ── Markowitz max-Sharpe (long-only) ──────────────────────────────────


class MarkowitzMaxSharpe(NonRLPolicy):
    """Rolling-window long-only maximum-Sharpe portfolio.

    On each step, estimate ``mu`` and ``Sigma`` from trailing daily returns,
    using Ledoit-Wolf covariance shrinkage, and solve

        max  mu^T w / sqrt(w^T Sigma w)
        s.t. w >= 0,  1^T w = 1.
    """

    name = "markowitz"

    def __init__(self, lookback: int = 252) -> None:
        self.lookback = int(lookback)

    def target_weights(self, env) -> np.ndarray:
        prices = _observed_prices(env)
        n_stocks = env.stock_dim

        window = prices[-(self.lookback + 1) :]
        if window.shape[0] < 3:
            w = np.zeros(1 + n_stocks)
            w[1:] = 1.0 / n_stocks
            return w

        rets = np.diff(window, axis=0) / np.maximum(window[:-1], EPS)
        mu = rets.mean(axis=0)
        if np.max(mu) <= 0:
            w = np.zeros(1 + n_stocks)
            w[0] = 1.0
            return w
        cov = LedoitWolf().fit(rets).covariance_
        weights = _solve_long_only_max_sharpe(mu, cov)
        w = np.zeros(1 + n_stocks)
        w[1:] = weights
        return w


def _solve_long_only_max_sharpe(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Solve max-Sharpe through the standard convex variable transformation."""
    n = mu.size
    positive = np.maximum(mu, 0.0)
    start_w = positive / positive.sum()
    scale = 1.0 / max(float(mu @ start_w), EPS)
    x0 = np.concatenate([start_w * scale, [scale]])

    result = minimize(
        lambda x: float(x[:-1] @ cov @ x[:-1]),
        x0,
        method="SLSQP",
        bounds=[(0.0, None)] * n + [(EPS, None)],
        constraints=[
            {"type": "eq", "fun": lambda x: float(mu @ x[:-1] - 1.0)},
            {"type": "eq", "fun": lambda x: float(x[:-1].sum() - x[-1])},
        ],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or result.x[-1] <= EPS:
        raise RuntimeError(f"Markowitz max-Sharpe optimisation failed: {result.message}")
    weights = np.maximum(result.x[:-1] / result.x[-1], 0.0)
    return weights / weights.sum()


# ── Z-score mean reversion ────────────────────────────────────────────


class ZScoreMeanReversion(NonRLPolicy):
    """Equally weight stocks whose rolling price z-score is below a threshold.

    The long-only implementation holds cash when no asset has a sufficiently
    negative z-score.
    """

    name = "zscore"

    def __init__(self, lookback: int = 60, threshold: float = 1.0) -> None:
        self.lookback = int(lookback)
        self.threshold = float(threshold)

    def target_weights(self, env) -> np.ndarray:
        prices = _observed_prices(env)
        n_stocks = env.stock_dim
        window = prices[-self.lookback :]
        if window.shape[0] < self.lookback:
            w = np.zeros(1 + n_stocks)
            w[0] = 1.0
            return w
        mu = window.mean(axis=0)
        sigma = window.std(axis=0, ddof=1) + EPS
        z = (window[-1] - mu) / sigma
        signal = z < -self.threshold
        w = np.zeros(1 + n_stocks)
        if signal.any():
            w[1:][signal] = 1.0 / int(signal.sum())
        else:
            w[0] = 1.0
        return w


# ── MACD trend following ──────────────────────────────────────────────


class MacdTrend(NonRLPolicy):
    """Long stocks with a positive MACD histogram (MACD above signal).

    Cash bucket = fraction of stocks with non-positive histogram.
    """

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)

    def _ema(self, series: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(series)
        out[0] = series[0]
        for i in range(1, series.shape[0]):
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _macd_histogram(self, prices: np.ndarray) -> np.ndarray:
        fast = self._ema(prices, self.fast)
        slow = self._ema(prices, self.slow)
        macd_line = fast - slow
        signal_line = self._ema(macd_line, self.signal)
        return macd_line - signal_line

    def target_weights(self, env) -> np.ndarray:
        prices = _observed_prices(env)
        n_stocks = env.stock_dim
        window = prices
        n = 1 + n_stocks
        if window.shape[0] < self.slow + self.signal:
            w = np.zeros(n)
            w[1:] = 1.0 / n_stocks
            return w
        hist = self._macd_histogram(window)[-1]
        in_trend = hist > 0
        n_in = int(in_trend.sum())
        w = np.zeros(n)
        if n_in == 0:
            w[0] = 1.0
        else:
            w[1:][in_trend] = 1.0 / n_in
        return w


# ── Hierarchical Risk Parity (López de Prado 2016) ────────────────────


class HierarchicalRiskParity(NonRLPolicy):
    """Hierarchical Risk Parity via recursive bisection.

    López de Prado, M. (2016). "Building Diversified Portfolios that
    Outperform Out of Sample." *Journal of Portfolio Management*
    42(4):59–69.

    Uses single-linkage clustering on correlation distance
    ``d_ij = sqrt(0.5 * (1 - rho_ij))``, then allocates by recursive
    bisection with inverse-variance weighting at each split.  No
    matrix inversion required.
    """

    name = "hrp"

    def __init__(self, lookback: int = 252) -> None:
        self.lookback = int(lookback)

    def target_weights(self, env) -> np.ndarray:
        prices = _observed_prices(env)
        n_stocks = env.stock_dim
        window = prices[-(self.lookback + 1) :]

        if window.shape[0] < 5:
            w = np.zeros(1 + n_stocks)
            w[1:] = 1.0 / n_stocks
            return w

        rets = np.diff(window, axis=0) / np.maximum(window[:-1], EPS)
        cov = np.cov(rets, rowvar=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(rets, rowvar=False)

        # Numerical safety: ensure valid correlation matrix
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        dist = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)
        dist = (dist + dist.T) / 2.0  # enforce symmetry

        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = list(leaves_list(link))

        weights = _hrp_recursive_bisection(cov, sort_ix)
        w = np.zeros(1 + n_stocks)
        w[1:] = weights
        return w


def _hrp_recursive_bisection(cov: np.ndarray, sort_ix: list) -> np.ndarray:
    """Allocate by recursive bisection with inverse-variance weighting."""
    n = cov.shape[0]
    weights = np.ones(n)
    items = [sort_ix]

    while items:
        next_items = []
        for cluster in items:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            left_var = _cluster_var(cov, left)
            right_var = _cluster_var(cov, right)
            total_var = left_var + right_var

            if total_var < EPS:
                alpha = 0.5
            else:
                alpha = 1.0 - left_var / total_var  # inverse-variance: smaller var gets more

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)

            if len(left) > 1:
                next_items.append(left)
            if len(right) > 1:
                next_items.append(right)
        items = next_items

    s = weights.sum()
    if s > EPS:
        weights /= s
    else:
        weights[:] = 1.0 / n
    return weights


def _cluster_var(cov: np.ndarray, indices: list) -> float:
    """Variance of an inverse-variance portfolio within a cluster."""
    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.diag(sub_cov)
    diag = np.maximum(diag, EPS)
    ivp = 1.0 / diag
    s = ivp.sum()
    if s < EPS:
        ivp = np.ones(len(indices)) / len(indices)
    else:
        ivp /= s
    return float(ivp @ sub_cov @ ivp)


# ── Equal Risk Contribution / Risk Parity ─────────────────────────────


class EqualRiskContribution(NonRLPolicy):
    """Equal Risk Contribution (Risk Parity) allocation.

    Maillard, S., Roncalli, T., Teiletche, J. (2010). "The Properties
    of Equally Weighted Risk Contribution Portfolios." *Journal of
    Portfolio Management* 36(4):60–70.

    Solves a Spinu log-barrier risk-budgeting objective on a covariance matrix
    regularized as ``Sigma + ridge * I``. Here ``RC_i = w_i * (Sigma w)_i``
    is the risk contribution of asset i.

    Falls back to inverse-variance weights if the optimiser fails.
    """

    name = "erc"

    def __init__(self, lookback: int = 252, ridge: float = 1e-4) -> None:
        self.lookback = int(lookback)
        self.ridge = float(ridge)

    def target_weights(self, env) -> np.ndarray:
        prices = _observed_prices(env)
        n_stocks = env.stock_dim
        window = prices[-(self.lookback + 1) :]

        if window.shape[0] < 5:
            w = np.zeros(1 + n_stocks)
            w[1:] = 1.0 / n_stocks
            return w

        rets = np.diff(window, axis=0) / np.maximum(window[:-1], EPS)
        cov = np.cov(rets, rowvar=False) + self.ridge * np.eye(n_stocks)

        weights = _solve_erc(cov)
        w = np.zeros(1 + n_stocks)
        w[1:] = weights
        return w


def _solve_erc(cov: np.ndarray) -> np.ndarray:
    """Solve for equal risk contribution weights.

    Uses the Spinu (2013) log-barrier formulation:
    ``min  0.5 * y^T Σ y  -  Σ_i log(y_i)``  over y > 0,
    then normalise w = y / Σy.  This is convex and well-conditioned.
    The log-barrier ensures y > 0 automatically.
    """
    n = cov.shape[0]

    # Solve in log-space: y = exp(x) to ensure positivity
    x0 = np.zeros(n)

    def objective(x):
        y = np.exp(x)
        port_var = 0.5 * float(y @ cov @ y)
        log_barrier = -float(x.sum())  # -Σ log(y_i) = -Σ x_i
        return port_var + log_barrier

    def gradient(x):
        y = np.exp(x)
        # d/dx_i of 0.5 * y^T Σ y = y_i * (Σy)_i  (chain rule via exp)
        grad_var = y * (cov @ y)
        grad_barrier = -np.ones(n)
        return grad_var + grad_barrier

    try:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=gradient,
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(result.message)
        y = np.exp(result.x)
        weights = y / y.sum()
        weights = np.maximum(weights, 0.0)
        s = weights.sum()
        if s > EPS:
            return weights / s
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("ERC optimiser failed: %s; falling back to inverse-variance", exc)

    # Fallback: inverse-variance
    diag = np.diag(cov)
    diag = np.maximum(diag, EPS)
    ivp = 1.0 / diag
    return ivp / ivp.sum()


# ── Month-end cadence (ADR-0009) ──────────────────────────────────────


class _MonthEndRebalancedPolicy(NonRLPolicy):
    """Shared cadence for policies that trade only at month-end formations.

    Formation happens at the final trading day of each calendar month on the
    fixed master calendar: the decision at index ``t`` is a formation iff the
    next trading date belongs to a different month. Reading ``date_list`` one
    date ahead is exchange-calendar knowledge, not price look-ahead
    (ADR-0009 Decision 6); prices, dividends, and halt flags beyond ``t``
    remain unobservable. On non-formation days ``rebalance_requested`` is
    False and the environment holds the drifting position, which is the
    one-month holding rule.

    The first decision of a fresh run enters the standing portfolio of the
    most recent computable formation (ADR-0009 Decision 5); a restored run
    (non-zero holdings) waits for the next formation instead.
    """

    def __init__(self) -> None:
        self._initial_entry_pending = True

    @staticmethod
    def _is_formation_index(env) -> bool:
        t = env.time
        if t + 1 >= len(env.date_list):
            return False
        current_month = np.datetime64(env.date_list[t], "M")
        next_month = np.datetime64(env.date_list[t + 1], "M")
        return current_month != next_month

    def _wants_initial_entry(self, env) -> bool:
        """True on the first decision of a fresh (non-restored) run."""
        if not self._initial_entry_pending or env.time != 0:
            return False
        holdings = np.asarray(getattr(env, "holdings", []), dtype=np.float64)
        return not (holdings.size and np.abs(holdings).sum() > 0)

    def rebalance_requested(self, env) -> bool:
        return self._is_formation_index(env) or self._wants_initial_entry(env)

    def step(self, env):
        result = super().step(env)
        self._initial_entry_pending = False
        return result


class MonthlyUniformConstantRebalancedPortfolio(_MonthEndRebalancedPolicy):
    """Equal weights over the full universe, rebalanced at month-ends only.

    The predeclared secondary cadence control for CSM-LO (ADR-0009 Decision
    7): identical universe and month-end/next-open schedule, no momentum rank
    and no eligibility screen, so the pair isolates the cross-sectional
    signal from the turnover reduction of monthly rebalancing. Never an RL
    method and never a primary Holm-family comparator.
    """

    name = "ucrp_monthly"

    def target_weights(self, env) -> np.ndarray:
        n_stocks = env.stock_dim
        w = np.zeros(1 + n_stocks)
        w[1:] = 1.0 / n_stocks
        return w


class CrossSectionalMomentumLongOnly(_MonthEndRebalancedPolicy):
    """Long-only prior-(2-12) cross-sectional momentum (ADR-0009).

    At each month-end formation the policy compounds dividend-inclusive daily
    total returns (panel-share units) over the eleven months ending one month
    before formation, ranks the eligible assets, and equal-weights the top
    ``top_n``; fewer than ``min_eligible`` eligible names targets 100% cash.

    Eligibility is the frozen staleness rule: the anchor month-end close
    (thirteen months before the holding month, i.e. the month-end preceding
    the first scored month) must exist and be non-synthetic, and no synthetic
    forward-filled row may occur from the first scored month's first trading
    day through the formation close. Observed zero-volume rows stay eligible;
    ``volume == 0`` is never consulted. Exact ties break by lexical ticker
    order. Formation diagnostics accumulate in ``self.formation_log``.
    """

    name = "csm_lo"

    #: Prior-(2-12): skip the most recent month, score the 11 before it.
    SCORE_MONTHS = 11
    SKIP_MONTHS = 1

    def __init__(self, top_n: int = 10, min_eligible: int = 10) -> None:
        super().__init__()
        self.top_n = int(top_n)
        self.min_eligible = int(min_eligible)
        self.formation_log: list[dict] = []
        self._current_target: Optional[np.ndarray] = None

    def _require_context(self, env) -> None:
        if not getattr(env, "halt_mask_provided", False):
            raise ValueError(
                "csm_lo requires the is_synthetic_halt mask (ADR-0009); the "
                "panel or env config did not provide synthetic_halt_array"
            )

    def target_weights(self, env) -> np.ndarray:
        self._require_context(env)
        n_stocks = env.stock_dim
        if self._is_formation_index(env):
            dates, closes, dividends, halts = _observed_context(env)
            self._current_target = self._form(
                env, dates, closes, dividends, halts, formation_idx=len(dates) - 1
            )
        elif self._wants_initial_entry(env):
            dates, closes, dividends, halts = _observed_context(env)
            history_len = len(dates) - (env.time + 1)
            formation_idx = self._latest_formation_index(dates, history_len)
            if formation_idx is not None:
                self._current_target = self._form(
                    env, dates, closes, dividends, halts, formation_idx
                )
        if self._current_target is None:
            w = np.zeros(1 + n_stocks)
            w[0] = 1.0
            return w
        return self._current_target

    def rebalance_requested(self, env) -> bool:
        if self._is_formation_index(env):
            return True
        if self._wants_initial_entry(env):
            # Entering costs a trade; without a computable standing formation
            # the fresh run is already all-cash and there is nothing to enter.
            dates, _, _, _ = _observed_context(env)
            history_len = len(dates) - (env.time + 1)
            return self._latest_formation_index(dates, history_len) is not None
        return False

    @staticmethod
    def _latest_formation_index(dates: np.ndarray, history_len: int) -> Optional[int]:
        """Most recent month-end index strictly inside the pre-window history."""
        if history_len < 1:
            return None
        months = dates.astype("datetime64[M]")
        # dates[history_len] is the first in-window date, so every history
        # index has a successor to compare against.
        boundaries = np.nonzero(months[1 : history_len + 1] != months[:history_len])[0]
        if boundaries.size == 0:
            return None
        return int(boundaries[-1])

    def _form(
        self,
        env,
        dates: np.ndarray,
        closes: np.ndarray,
        dividends: np.ndarray,
        halts: np.ndarray,
        formation_idx: int,
    ) -> np.ndarray:
        n_stocks = env.stock_dim
        tickers = np.asarray(
            getattr(env, "stock_symbols", np.arange(n_stocks).astype(str))
        ).astype(str)
        months = dates[: formation_idx + 1].astype("datetime64[M]")
        formation_month = months[formation_idx]
        anchor_month = formation_month - np.timedelta64(
            self.SCORE_MONTHS + self.SKIP_MONTHS, "M"
        )
        score_end_month = formation_month - np.timedelta64(self.SKIP_MONTHS, "M")

        record: dict = {
            "formation_date": str(dates[formation_idx].astype("datetime64[D]")),
            "top_n": self.top_n,
            "min_eligible": self.min_eligible,
        }

        anchor_candidates = np.nonzero(months == anchor_month)[0]
        score_end_candidates = np.nonzero(months == score_end_month)[0]
        if anchor_candidates.size == 0 or score_end_candidates.size == 0:
            record.update(
                {
                    "eligible": [],
                    "selected": [],
                    "excluded": {tic: "insufficient_history" for tic in tickers},
                    "all_cash": True,
                }
            )
            self.formation_log.append(record)
            return self._cash_target(n_stocks)

        anchor_idx = int(anchor_candidates[-1])
        score_end_idx = int(score_end_candidates[-1])

        prev = closes[anchor_idx:score_end_idx]
        curr = closes[anchor_idx + 1 : score_end_idx + 1]
        divs = dividends[anchor_idx + 1 : score_end_idx + 1]
        scores = ((curr + divs) / prev).prod(axis=0) - 1.0

        synthetic_anchor = halts[anchor_idx]
        synthetic_window = halts[anchor_idx + 1 : formation_idx + 1].any(axis=0)
        eligible = ~synthetic_anchor & ~synthetic_window

        excluded = {}
        for i in np.nonzero(~eligible)[0]:
            excluded[str(tickers[i])] = (
                "synthetic_anchor" if synthetic_anchor[i] else "synthetic_in_window"
            )

        record["scores"] = {
            str(tickers[i]): float(scores[i]) for i in range(n_stocks)
        }
        record["eligible"] = [str(t) for t in tickers[eligible]]
        record["excluded"] = excluded

        eligible_idx = np.nonzero(eligible)[0]
        if eligible_idx.size < self.min_eligible:
            record.update({"selected": [], "all_cash": True})
            self.formation_log.append(record)
            return self._cash_target(n_stocks)

        # Highest score first; exact ties resolved by lexical ticker order.
        order = eligible_idx[
            np.lexsort((tickers[eligible_idx], -scores[eligible_idx]))
        ]
        selected = order[: self.top_n]
        record.update(
            {"selected": [str(t) for t in tickers[selected]], "all_cash": False}
        )
        self.formation_log.append(record)

        w = np.zeros(1 + n_stocks)
        w[1 + selected] = 1.0 / selected.size
        return w

    @staticmethod
    def _cash_target(n_stocks: int) -> np.ndarray:
        w = np.zeros(1 + n_stocks)
        w[0] = 1.0
        return w


# ── Factory ───────────────────────────────────────────────────────────


# Legacy class imports and registry identifiers remain readable for old
# configuration layers and invalidated output provenance.
BuyAndHold = UniformBuyAndHold
EqualWeight = UniformConstantRebalancedPortfolio


NON_RL_REGISTRY: Dict[str, type] = {
    "ubah": UniformBuyAndHold,
    "ucrp": UniformConstantRebalancedPortfolio,
    "bh": UniformBuyAndHold,
    "ew": UniformConstantRebalancedPortfolio,
    "markowitz": MarkowitzMaxSharpe,
    "zscore": ZScoreMeanReversion,
    "macd": MacdTrend,
    "hrp": HierarchicalRiskParity,
    "erc": EqualRiskContribution,
    "csm_lo": CrossSectionalMomentumLongOnly,
    "ucrp_monthly": MonthlyUniformConstantRebalancedPortfolio,
}


def build_non_rl_policy(name: str, params: Optional[Dict[str, Any]] = None) -> NonRLPolicy:
    if name not in NON_RL_REGISTRY:
        raise ValueError(f"Unknown non-RL policy: {name!r}")
    return NON_RL_REGISTRY[name](**(params or {}))


# Legacy public names remain available for old callers and result provenance.
ClassicalPolicy = NonRLPolicy
CLASSICAL_REGISTRY = NON_RL_REGISTRY
build_classical_policy = build_non_rl_policy

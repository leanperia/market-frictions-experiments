#!/usr/bin/env python3
"""Headline backtest metrics for experiment cells.

Extracted from ``tools/run_experiment_cell.py:run_deterministic`` so the metric
math is importable and unit-testable without constructing an env (tests-review
SUITE-1). The extraction is numerically behaviour-preserving — same metric keys,
values, ddof and NaN paths — with ONE deliberate hardening: the rf-adjusted path
now aligns via ``tools.rf_series.align_rf`` (coverage_threshold=0.9), which
*raises* on an out-of-window rf series instead of silently zero-filling to the
raw Sharpe (STAT-1). At full rf coverage the numbers are identical to the legacy
``reindex().ffill().bfill().fillna(0.0)`` idiom.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from tools.rf_series import align_rf


def _align_benchmark_returns(benchmark_series: pd.Series, dates: pd.Series) -> np.ndarray:
    """Align benchmark returns to portfolio sessions.

    A portfolio calendar may include a synthesised cancelled session so an
    outage constraint can execute. The market benchmark has no return on such
    a session; it must be zero rather than the preceding session's return.
    """
    return benchmark_series.reindex(pd.to_datetime(dates)).fillna(0.0).values


def compute_backtest_metrics(
    df: pd.DataFrame,
    rf_series: pd.Series | None = None,
    benchmark_series: pd.Series | None = None,
) -> Dict[str, float]:
    """Compute the headline backtest metric dict from a per-day backtest frame.

    ``df`` must contain ``portfolio_value``; the first row is the initial
    state (not an observed return period). ``date`` plus the optional
    friction-diagnostic columns (``turnover``, ``impact_cost``, ...) are
    consumed when present, exactly as the inline block in
    ``run_deterministic`` did.
    """
    eq = df["portfolio_value"].astype(float)
    daily_ret = eq.pct_change().dropna()
    n = len(df)
    # First row is the initial state, not an observed return period.
    n_periods = max(len(daily_ret), 1)
    annual_vol = float(daily_ret.std(ddof=1) * np.sqrt(252)) if len(daily_ret) > 1 else float("nan")
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    drawdown = eq / eq.cummax() - 1.0
    max_dd = float(drawdown.min())

    # ── Sortino ratio (downside deviation with zero threshold) ────────
    downside = daily_ret.copy()
    downside[downside > 0] = 0.0
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))
    sortino = float((daily_ret.mean() * 252) / downside_dev) if downside_dev > 1e-12 else float("nan")

    # ── Calmar ratio ──────────────────────────────────────────────────
    ann_ret = float((1.0 + total_return) ** (252 / n_periods) - 1.0)
    calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-12 else float("nan")

    # ── CVaR (Expected Shortfall) at 5% and 1% ───────────────────────
    sorted_rets = np.sort(daily_ret.values)
    n_obs = len(sorted_rets)
    cvar_5_idx = max(int(np.floor(0.05 * n_obs)), 1)
    cvar_1_idx = max(int(np.floor(0.01 * n_obs)), 1)
    cvar_5 = float(sorted_rets[:cvar_5_idx].mean()) if cvar_5_idx > 0 else float("nan")
    cvar_1 = float(sorted_rets[:cvar_1_idx].mean()) if cvar_1_idx > 0 else float("nan")

    # ── Higher moments ────────────────────────────────────────────────
    if n_obs > 3:
        ret_arr = daily_ret.values
        mu = ret_arr.mean()
        sigma = ret_arr.std(ddof=1)
        if sigma > 1e-12:
            skewness = float(np.mean(((ret_arr - mu) / sigma) ** 3))
            excess_kurtosis = float(np.mean(((ret_arr - mu) / sigma) ** 4) - 3.0)
        else:
            skewness = float("nan")
            excess_kurtosis = float("nan")
    else:
        skewness = float("nan")
        excess_kurtosis = float("nan")

    # ── Rf-adjusted metrics ─────────────────────────────────────────────
    sharpe_rf = float("nan")
    sortino_rf = float("nan")
    if rf_series is not None and "date" in df.columns and not daily_ret.empty:
        dates = pd.to_datetime(df["date"]).iloc[1:]
        # align_rf raises if the rf series covers < 90% of the return dates,
        # instead of silently zero-filling to the raw Sharpe (tests-review
        # STAT-1). Full-coverage output is identical to the legacy
        # reindex().ffill().bfill().fillna(0.0).values idiom.
        aligned_rf = align_rf(rf_series, dates)
        excess_ret = daily_ret.values - aligned_rf
        excess_mean = float(np.mean(excess_ret))
        excess_std = float(np.std(excess_ret, ddof=1))
        if excess_std > 1e-12:
            sharpe_rf = float(excess_mean * 252 / (excess_std * np.sqrt(252)))
        # Sortino rf: downside deviation of excess returns
        excess_down = np.where(excess_ret < 0, excess_ret, 0.0)
        excess_down_dev = float(np.sqrt(np.mean(excess_down ** 2)) * np.sqrt(252))
        if excess_down_dev > 1e-12:
            sortino_rf = float(excess_mean * 252 / excess_down_dev)

    # ── Benchmark-relative metrics ────────────────────────────────────
    information_ratio = float("nan")
    jensen_alpha = float("nan")
    beta = float("nan")
    treynor_ratio = float("nan")
    if benchmark_series is not None and "date" in df.columns and not daily_ret.empty:
        dates = pd.to_datetime(df["date"]).iloc[1:]
        bm_aligned = _align_benchmark_returns(benchmark_series, dates)
        # Use rf if available for excess returns, otherwise raw
        if rf_series is not None:
            aligned_rf = align_rf(rf_series, dates)
        else:
            aligned_rf = np.zeros(len(dates))
        port_excess = daily_ret.values - aligned_rf
        bm_excess = bm_aligned - aligned_rf
        active_ret = daily_ret.values - bm_aligned

        # Information Ratio
        active_std = float(np.std(active_ret, ddof=1))
        if active_std > 1e-12:
            information_ratio = float(np.mean(active_ret) * 252 / (active_std * np.sqrt(252)))

        # Jensen alpha and beta via OLS: R_p - rf = alpha + beta*(R_b - rf) + e
        bm_var = float(np.var(bm_excess, ddof=1))
        if bm_var > 1e-16 and len(port_excess) > 2:
            beta = float(np.cov(port_excess, bm_excess, ddof=1)[0, 1] / bm_var)
            jensen_alpha = float((np.mean(port_excess) - beta * np.mean(bm_excess)) * 252)
            if abs(beta) > 1e-12:
                treynor_ratio = float(np.mean(port_excess) * 252 / beta)

    metrics = {
        "n_days": int(n),
        "total_return": total_return,
        "annual_return": ann_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": float((daily_ret.mean() * 252) / annual_vol) if annual_vol and annual_vol > 0 else float("nan"),
        "sharpe_ratio_rf": sharpe_rf,
        "sortino_ratio": sortino,
        "sortino_ratio_rf": sortino_rf,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "cvar_5pct": cvar_5,
        "cvar_1pct": cvar_1,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "information_ratio": information_ratio,
        "jensen_alpha": jensen_alpha,
        "beta": beta,
        "treynor_ratio": treynor_ratio,
        "final_equity": float(eq.iloc[-1]),
        "total_turnover": float(df.get("turnover", pd.Series(dtype=float)).fillna(0.0).sum()),
        "total_impact_cost": float(df.get("impact_cost", pd.Series(dtype=float)).fillna(0.0).sum()),
        "total_fees_explicit": float(df.get("fees_explicit", pd.Series(dtype=float)).fillna(0.0).sum()),
        "total_spread_cost": float(df.get("spread_cost", pd.Series(dtype=float)).fillna(0.0).sum()),
        "blocked_by_adv": float(df.get("blocked_by_adv", pd.Series(dtype=float)).fillna(0.0).sum()),
        "blocked_by_lot": float(df.get("blocked_by_lot", pd.Series(dtype=float)).fillna(0.0).sum()),
        "blocked_by_cash": float(df.get("blocked_by_cash", pd.Series(dtype=float)).fillna(0.0).sum()),
        "blocked_by_outage": float(df.get("blocked_by_outage", pd.Series(dtype=float)).fillna(0.0).sum()),
        "cash_shortfall_total": float(df.get("cash_shortfall", pd.Series(dtype=float)).fillna(0.0).sum()),
        "cash_shortfall_steps": int((df.get("cash_shortfall", pd.Series(dtype=float)).fillna(0.0) > 0).sum()),
    }
    return metrics

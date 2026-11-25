"""Allocation-rule sensitivity using dividends already in the frozen panel.

Only the rolling return observations change. The original policy implementations
and their optimisation helpers remain untouched. Dividends here inform targets;
the shared environment remains solely responsible for crediting portfolio cash.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from .classical import (
    EPS, EqualRiskContribution, HierarchicalRiskParity, MarkowitzMaxSharpe,
    _hrp_recursive_bisection, _observed_context, _solve_erc,
    _solve_long_only_max_sharpe,
)


def observed_total_returns(env, lookback: int) -> np.ndarray:
    """Return at most ``lookback`` causal, aligned dividend-inclusive returns.

    Price and cash distributions are in the panel's common adjusted share unit.
    The algebraic form preserves the original price-return calculation exactly
    when distributions are zero, including its denominator floor. No halt rows
    or dates are removed: the original calendar and rolling windows are retained.
    """
    dates, closes, dividends, _halts = _observed_context(env)
    if closes.shape != dividends.shape or dates.shape != (len(closes),):
        raise ValueError("Observed price/dividend/date channels are not aligned")
    if len(dates) > 1 and not np.all(dates[1:] > dates[:-1]):
        raise ValueError("Observed history must have strictly increasing dates")
    window = closes[-(lookback + 1):]
    income = dividends[-(lookback + 1):]
    if not np.isfinite(window).all() or not np.isfinite(income).all():
        raise ValueError("Observed prices and dividends must be finite")
    return (np.diff(window, axis=0) + income[1:]) / np.maximum(window[:-1], EPS)


class DividendInclusiveMarkowitz(MarkowitzMaxSharpe):
    name = "markowitz_tr"

    def target_weights(self, env) -> np.ndarray:
        rets = observed_total_returns(env, self.lookback)
        n_stocks = env.stock_dim
        w = np.zeros(1 + n_stocks)
        if rets.shape[0] < 2:
            w[1:] = 1.0 / n_stocks
            return w
        mu = rets.mean(axis=0)
        if np.max(mu) <= 0:
            w[0] = 1.0
            return w
        cov = LedoitWolf().fit(rets).covariance_
        w[1:] = _solve_long_only_max_sharpe(mu, cov)
        return w


class DividendInclusiveHRP(HierarchicalRiskParity):
    name = "hrp_tr"

    def target_weights(self, env) -> np.ndarray:
        rets = observed_total_returns(env, self.lookback)
        n_stocks = env.stock_dim
        w = np.zeros(1 + n_stocks)
        if rets.shape[0] < 4:
            w[1:] = 1.0 / n_stocks
            return w
        cov = np.cov(rets, rowvar=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(rets, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        dist = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        link = linkage(squareform(dist, checks=False), method="single")
        w[1:] = _hrp_recursive_bisection(cov, list(leaves_list(link)))
        return w


class DividendInclusiveERC(EqualRiskContribution):
    name = "erc_tr"

    def target_weights(self, env) -> np.ndarray:
        rets = observed_total_returns(env, self.lookback)
        n_stocks = env.stock_dim
        w = np.zeros(1 + n_stocks)
        if rets.shape[0] < 4:
            w[1:] = 1.0 / n_stocks
            return w
        cov = np.cov(rets, rowvar=False) + self.ridge * np.eye(n_stocks)
        w[1:] = _solve_erc(cov)
        return w


DIVIDEND_INCLUSIVE_POLICIES = {
    "markowitz": DividendInclusiveMarkowitz,
    "hrp": DividendInclusiveHRP,
    "erc": DividendInclusiveERC,
}

"""Bid-ask spread haircut.

Estimator: Corwin & Schultz (2012) two-day high-low estimator.

    S_hat = 2 (e^α - 1) / (1 + e^α)

with

    α = (sqrt(2β) - sqrt(β)) / (3 - 2 sqrt(2)) - sqrt(γ / (3 - 2 sqrt(2)))
    β = sum over two days of (ln(H_d / L_d))^2
    γ = (ln(H_{t,t-1} / L_{t,t-1}))^2

Where ``H_{t,t-1}`` is the max(High_t, High_{t-1}) and ``L_{t,t-1}``
is the min(Low_t, Low_{t-1}). The published estimator can be negative
on noisy days; we clip at zero and apply a per-market floor for
realism.

The execution charge is half-spread × notional per trade, applied as a
cash deduction (mathematically equivalent to filling at the bid/ask
side of the mid). This layer expects a precomputed
``spread_array`` of shape ``(time, asset)`` (fraction-of-price units).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .pipeline import FrictionLayer, TradeContext


def corwin_schultz_spread(
    high: np.ndarray,
    low: np.ndarray,
    spread_floor: float = 0.0,
) -> np.ndarray:
    """Per-day CS spread estimate. ``high``/``low`` shape: (time, asset).

    Returns an array of the same shape; the first row is filled with
    the spread_floor (no previous day to pair with).
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if high.shape != low.shape:
        raise ValueError("high and low must share shape (time, asset)")
    T, N = high.shape
    out = np.full((T, N), spread_floor)

    if T < 2:
        return out

    h_t = high[1:]
    l_t = low[1:]
    h_prev = high[:-1]
    l_prev = low[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h_t / l_t) ** 2 + np.log(h_prev / l_prev) ** 2
        h_pair = np.maximum(h_t, h_prev)
        l_pair = np.minimum(l_t, l_prev)
        gamma = np.log(h_pair / l_pair) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(np.clip(gamma / denom, 0.0, None))
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = np.where(np.isnan(spread), spread_floor, spread)
    spread = np.maximum(spread, spread_floor)
    out[1:] = spread
    return out


class SpreadHaircut(FrictionLayer):
    name = "spread"

    def __init__(self, spread_array: np.ndarray, spread_floor: float = 0.0) -> None:
        self.spread_array = np.asarray(spread_array, dtype=np.float64)
        if self.spread_array.ndim != 2:
            raise ValueError("spread_array must be (time, asset)")
        self.spread_floor = float(spread_floor)

    def _apply(self, ctx: TradeContext) -> TradeContext:
        if ctx.time >= self.spread_array.shape[0]:
            return ctx
        half_spread = np.maximum(self.spread_array[ctx.time], self.spread_floor) / 2.0
        prices = np.asarray(ctx.prices, dtype=np.float64)
        trades = ctx.trades.astype(np.float64)
        notional = np.abs(trades) * prices
        per_trade_cost = notional * half_spread
        cost = float(per_trade_cost.sum())
        sell_side = float(np.where(trades < 0, per_trade_cost, 0.0).sum())

        ctx.cash -= cost
        ctx.diagnostics.add_scalar("spread_cost", cost)
        ctx.diagnostics["half_spread_avg"] = float(half_spread.mean())
        ctx.diagnostics.add_scalar("sell_side_costs", sell_side)
        return ctx

    def after_trade(self, ctx: TradeContext) -> TradeContext:
        return self._apply(ctx)

    def estimate_after_trade(self, ctx: TradeContext) -> TradeContext:
        return self._apply(ctx)

"""Cash settlement lag (NYSE T+2 → T+1 since 2024; PSE T+2 → T+2).

We don't change how the env tracks cash (single ``_cash_balance``).
Instead the layer maintains a side-account of in-transit sell proceeds
and only lets *buys* draw on ``settled_cash = cash − in_transit_total``.
Buys that would exceed settled cash are scaled down proportionally;
the deficit is reported as ``blocked_by_cash``.

NYSE's settlement cycle compressed from T+2 to T+1 on 2024-05-28; that
boundary is a constructor argument (``t1_effective_date``) so the
NYSE config can flip behaviour mid-sample without code changes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence

import numpy as np
import pandas as pd

from .pipeline import FrictionLayer, TradeContext


@dataclass
class _Pending:
    settle_date: pd.Timestamp
    amount: float


class SettlementLag(FrictionLayer):
    name = "settlement"

    def __init__(
        self,
        cycle_days: int = 2,
        t1_effective_date: Optional[str] = None,
        trading_dates: Optional[Sequence] = None,
    ) -> None:
        self.default_cycle = int(cycle_days)
        self._t1_effective = pd.Timestamp(t1_effective_date) if t1_effective_date else None
        self._queue: Deque[_Pending] = deque()
        self._trading_dates = None
        if trading_dates is not None:
            dates = pd.to_datetime(list(trading_dates)).normalize()
            self._trading_dates = pd.DatetimeIndex(sorted(pd.unique(dates)))

    def reset(self) -> None:
        self._queue.clear()

    def get_state(self) -> dict:
        return {
            "pending": [
                {"settle_date": p.settle_date.isoformat(), "amount": p.amount}
                for p in self._queue
            ]
        }

    def set_state(self, state: dict) -> None:
        self._queue.clear()
        for item in state.get("pending", []):
            self._queue.append(
                _Pending(
                    settle_date=pd.Timestamp(item["settle_date"]).normalize(),
                    amount=float(item["amount"]),
                )
            )

    def _cycle_for(self, date: pd.Timestamp) -> int:
        if self._t1_effective is not None and date >= self._t1_effective:
            return 1
        return self.default_cycle

    def _mature(self, today: pd.Timestamp) -> float:
        matured = 0.0
        while self._queue and self._queue[0].settle_date <= today:
            matured += self._queue.popleft().amount
        return matured

    def _in_transit_total(self) -> float:
        return float(sum(p.amount for p in self._queue))

    def _settle_date(self, today: pd.Timestamp) -> pd.Timestamp:
        cycle = self._cycle_for(today)
        if self._trading_dates is not None and len(self._trading_dates) > 0:
            pos = self._trading_dates.searchsorted(today)
            if pos < len(self._trading_dates) and self._trading_dates[pos] == today:
                target = pos + cycle
                if target < len(self._trading_dates):
                    return self._trading_dates[target]
                # Tail of calendar: not enough trading dates remain to express the
                # full settlement cycle.  Fall back to a business-day offset so
                # proceeds are never made available earlier than T+N requires.
                return today + pd.offsets.BDay(cycle)
        return today + pd.offsets.BDay(cycle)

    def before_trade(self, ctx: TradeContext) -> TradeContext:
        today = pd.Timestamp(ctx.date).normalize()
        self._mature(today)
        settled = ctx.cash - self._in_transit_total()

        # Compute notional value of *buys only*. If they exceed settled cash, scale down.
        prices = np.asarray(ctx.prices, dtype=np.float64)
        trades = np.asarray(ctx.trades, dtype=np.int64)
        buy_mask = trades > 0
        buy_notional = float((trades[buy_mask] * prices[buy_mask]).sum())

        scale = 1.0
        blocked = np.zeros_like(trades)
        if buy_notional > settled and buy_notional > 0:
            scale = max(0.0, settled / buy_notional)
            scaled_buys = (trades[buy_mask].astype(np.float64) * scale).astype(np.int64)
            blocked[buy_mask] = trades[buy_mask] - scaled_buys
            new_trades = trades.copy()
            new_trades[buy_mask] = scaled_buys
            ctx.trades = new_trades.astype(ctx.trades.dtype, copy=False)

        ctx.diagnostics["settled_cash"] = settled
        ctx.diagnostics["in_transit_cash"] = self._in_transit_total()
        ctx.diagnostics.add_array("blocked_by_cash", blocked)
        ctx.diagnostics["settlement_scale"] = scale
        return ctx

    def after_trade(self, ctx: TradeContext) -> TradeContext:
        # Queue NET sell proceeds for delayed settlement.
        #
        # The amount that will actually mature into spendable cash is
        # gross sell proceeds minus the fees and spread costs the
        # broker/clearer takes out of those sells.  The fee/spread
        # layers (which must run earlier in the pipeline order) accumulate
        # their sell-side share into ctx.diagnostics["sell_side_costs"].
        # We subtract that here so subsequent steps see an in-transit
        # balance the agent can actually count on.
        today = pd.Timestamp(ctx.date).normalize()
        prices = np.asarray(ctx.prices, dtype=np.float64)
        trades = np.asarray(ctx.trades, dtype=np.int64)
        gross = float(np.where(trades < 0, -trades * prices, 0.0).sum())
        sell_side_costs = float(ctx.diagnostics.get("sell_side_costs", 0.0))
        net = max(0.0, gross - sell_side_costs)
        if net > 0:
            settle_date = self._settle_date(today)
            self._queue.append(_Pending(settle_date=settle_date, amount=net))
        # Update the in_transit_cash diagnostic to reflect the new queue
        ctx.diagnostics["in_transit_cash"] = self._in_transit_total()
        return ctx

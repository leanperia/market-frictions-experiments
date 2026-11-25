"""Board-lot rounding (PSE).

The Philippine Stock Exchange uses a price-banded board-lot table: the
lot size depends on the stock's quoted last-traded price. Buys round *down*
toward zero (you can't trade fractions of a lot); sells round *down*
in absolute value as well — but only what would have been traded gets
recorded as ``blocked_by_lot``.

The default schedule is the PSE Academy table effective since 2014.
Override via the ``schedule`` constructor argument if PSE updates it.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .pipeline import FrictionLayer, TradeContext

# Each tuple is (upper_price_inclusive, lot_size). The last entry uses
# float("inf") to catch everything from PHP 5,000 upward.
DEFAULT_PSE_LOT_TABLE: Tuple[Tuple[float, int], ...] = (
    (0.0099, 1_000_000),
    (0.0490, 100_000),
    (0.2490, 10_000),
    (0.4950, 10_000),
    (4.99, 1_000),
    (9.99, 100),
    (19.98, 100),
    (49.95, 100),
    (99.95, 10),
    (199.9, 10),
    (499.8, 10),
    (999.5, 10),
    (1_999.0, 5),
    (4_998.0, 5),
    (float("inf"), 5),
)


def _lot_for_price(price: float, schedule: Sequence[Tuple[float, int]]) -> int:
    for upper, lot in schedule:
        if price <= upper:
            return lot
    return schedule[-1][1]


class BoardLotRounding(FrictionLayer):
    name = "board_lots"

    def __init__(self, schedule: Sequence[Tuple[float, int]] = DEFAULT_PSE_LOT_TABLE) -> None:
        self.schedule: List[Tuple[float, int]] = list(schedule)

    def _round_one(
        self, shares: int, price: float, share_factor: float = 1.0
    ) -> Tuple[int, int]:
        """Round one ticker's trade down to a whole number of board lots.

        The lot is defined in the shares that **legally existed** on the trade
        date, but the panel counts shares on the current basis, so with
        ``share_factor = f`` one legal share occupies ``f`` panel shares. The
        constraint is therefore enforced in legal-share space and converted
        back (handoff §7.4).

        PSE stock dividends make ``f`` fractional, so integral legal shares and
        integral panel shares cannot both hold. The conversion back floors, and
        the result is clamped to the original magnitude, so a rounding artifact
        can only ever *reduce* a trade — it can never let one exceed what the
        lot permits, and it can never manufacture shares.
        """
        if shares == 0 or price <= 0:
            return shares, 0
        lot = _lot_for_price(price, self.schedule)
        sign = 1 if shares > 0 else -1
        magnitude = abs(shares)

        if not np.isfinite(share_factor) or share_factor <= 0:
            share_factor = 1.0
        if lot <= 1 and share_factor == 1.0:
            return shares, 0

        legal = magnitude / share_factor
        kept_legal = np.floor(legal / lot) * lot
        kept = int(np.floor(kept_legal * share_factor))
        kept = max(0, min(kept, magnitude))
        blocked = magnitude - kept
        return sign * kept, blocked

    def before_trade(self, ctx: TradeContext) -> TradeContext:
        new_trades = ctx.trades.copy()
        blocked = np.zeros_like(ctx.trades, dtype=np.int64)
        prices = ctx.quoted_prices if ctx.quoted_prices is not None else ctx.prices
        factors = ctx.share_factors
        for i, shares in enumerate(ctx.trades):
            factor = 1.0 if factors is None else float(factors[i])
            new_trades[i], blocked[i] = self._round_one(
                int(shares), float(prices[i]), factor
            )
        ctx.trades = new_trades
        ctx.diagnostics.add_array("blocked_by_lot", blocked)
        return ctx

    def after_cash_scaling(self, ctx: TradeContext) -> TradeContext:
        return self.before_trade(ctx)

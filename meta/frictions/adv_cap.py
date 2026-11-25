"""Participation cap against rolling-20-day ADV.

The MACE env already clips against *same-day* volume via
``max_trade_volume_pct``. That is operationally a bit forgiving: on
crisis days, observed volume spikes and the cap relaxes. The plan's
F2 cap is a fraction of the *prior 20-day ADV* — a tighter, more
conservative constraint that doesn't peek at today's volume.

This layer expects an ``adv20`` array passed at construction time
(time × asset). It indexes by ``ctx.time``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pipeline import FrictionLayer, TradeContext


class AdvCap(FrictionLayer):
    name = "adv_cap"

    def __init__(self, adv20_array: np.ndarray, max_pov: float = 0.10) -> None:
        self.adv20_array = np.asarray(adv20_array)
        if self.adv20_array.ndim != 2:
            raise ValueError("adv20_array must be (time, asset)")
        self.max_pov = float(max_pov)

    def before_trade(self, ctx: TradeContext) -> TradeContext:
        if ctx.time >= self.adv20_array.shape[0]:
            return ctx
        adv_today = self.adv20_array[ctx.time]
        cap = (adv_today * self.max_pov).astype(np.int64)
        signs = np.sign(ctx.trades)
        magnitude = np.abs(ctx.trades).astype(np.int64)
        capped_magnitude = np.minimum(magnitude, cap)
        new_trades = (signs.astype(np.int64) * capped_magnitude).astype(ctx.trades.dtype)
        blocked = magnitude - capped_magnitude

        ctx.trades = new_trades
        ctx.diagnostics.add_array("blocked_by_adv", blocked)
        with np.errstate(divide="ignore", invalid="ignore"):
            pov = np.where(adv_today > 0, capped_magnitude / adv_today, 0.0)
        ctx.diagnostics["participation_pct"] = pov
        return ctx

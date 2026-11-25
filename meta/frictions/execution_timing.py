"""Next-day open execution.

The portfolio env follows standard Gym timing: the action produced from
the observation at day ``t`` is consumed by ``step()``, which advances the
environment to day ``t+1``. This layer therefore does not buffer actions
again.

The layer advertises ``uses_open_prices`` so the env selects the day
``t+1`` open before constructing the trade context. It does not overwrite
``ctx.prices`` itself, because by that point the env has already applied
any permanent-impact shift to the selected open price.
"""

from __future__ import annotations

from .pipeline import FrictionLayer, TradeContext


class NextDayExecution(FrictionLayer):
    name = "execution_timing"
    execution_price_source = "open"

    def before_trade(self, ctx: TradeContext) -> TradeContext:
        ctx.diagnostics["execution_lag_days"] = 1
        ctx.diagnostics["execution_price"] = "next_open"
        return ctx

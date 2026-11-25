"""Outage filter — zero trades on disrupted trading days.

PSE has half a dozen documented outages between 2016 and 2025, seeded
in ``meta/calendars/pse_outages.yaml``. Severity (full-day vs partial)
is interpreted here, not in the calendar:

* ``full_day_cancellation`` — *all* trades blocked.
* ``mid_session_halt`` / ``opening_delay`` / ``pre_open_delay`` —
  trades scaled by the fraction of the session that survived
  (``(session_minutes - duration) / session_minutes``).

NYSE has no equivalent calendar of single-day cancellations in our
sample, so this layer is PSE-only by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import yaml

from .pipeline import FrictionLayer, TradeContext

DEFAULT_SESSION_MINUTES = 330


class OutageFilter(FrictionLayer):
    name = "outages"

    def __init__(
        self,
        outages: Iterable[Dict],
        session_minutes: int = DEFAULT_SESSION_MINUTES,
        block_full_day_only: bool = False,
    ) -> None:
        self.session_minutes = int(session_minutes)
        self.block_full_day_only = bool(block_full_day_only)
        self._by_date: Dict[pd.Timestamp, Dict] = {}
        for entry in outages:
            ts = pd.Timestamp(entry["date"])
            self._by_date[ts.normalize()] = dict(entry)

    @classmethod
    def from_yaml(cls, path: Union[str, Path], **kwargs) -> "OutageFilter":
        data = yaml.safe_load(Path(path).read_text())
        return cls(outages=data.get("outages", []), **kwargs)

    def _scale_for(self, entry: Dict) -> float:
        nature = entry.get("nature", "")
        if nature == "full_day_cancellation":
            return 0.0
        if self.block_full_day_only:
            return 1.0
        dur = float(entry.get("duration_minutes", 0))
        surviving = max(0.0, self.session_minutes - dur)
        return surviving / self.session_minutes

    def before_trade(self, ctx: TradeContext) -> TradeContext:
        date = pd.Timestamp(ctx.date).normalize()
        entry = self._by_date.get(date)
        if entry is None:
            return ctx
        scale = self._scale_for(entry)
        # The env has already selected the execution price for this date
        # (next open when the execution-timing layer is active). Outages are
        # therefore modeled as a share-quantity constraint: full-day synthetic
        # rows carry forward prices and scale to zero; partial halts keep the
        # real daily prices but reduce executable size.
        original = ctx.trades.copy()
        scaled = (ctx.trades.astype(np.float64) * scale).astype(ctx.trades.dtype)
        blocked = original - scaled
        ctx.trades = scaled
        ctx.diagnostics.add_array("blocked_by_outage", np.abs(blocked))
        ctx.diagnostics["outage_scale"] = scale
        ctx.diagnostics["outage_nature"] = entry.get("nature")
        return ctx

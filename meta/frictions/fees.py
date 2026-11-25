"""Per-market explicit fee schedules.

The constructor takes the components that matter for the markets in
scope and an optional CMEPA effective date that flips the PSE stock
transaction tax (STT) between its pre- and post-CMEPA rates.

Defaults reflect publicly-documented rates at the time of writing
(2026-05) and are intentionally low-fidelity — the goal is for the
*directional* magnitude of the fee burden to be right (NYSE ~1-3 bps,
PSE ~30-70 bps) so the F1 ablation can show explicit fees moving
Sharpe in the expected direction. Tighten the numbers later when we
have signed-off broker schedules.

References
----------
* NYSE — SEC Section 31 fee on sells, FINRA TAF per share.
* PSE — PSE / SCCP / regulator fees, broker commission, 12% VAT on
  commission, stock transaction tax (STT) of 0.6% on sells pre-CMEPA
  and 0.1% post-CMEPA (Republic Act No. 12214, "CMEPA", effective
  2025-07-01).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .pipeline import FrictionLayer, TradeContext


# ──────────────────────────────────────────────────────────────────────
# NYSE
# ──────────────────────────────────────────────────────────────────────


@dataclass
class NyseFeeSchedule(FrictionLayer):
    # Retail US-equity fee schedule (ADR-0001). Broker commission is 0 in the
    # post-2019 zero-commission era; only the SEC Section 31 fee and the FINRA
    # Trading Activity Fee (TAF) apply, both on the sell side.
    #
    # Date-accurate SEC Section 31 rate per $1M of sells (SEC fee rate
    # advisories, FY2020-FY2026) and FINRA TAF per share (FINRA schedule).
    # Each tuple is (effective_date, rate); rates before the first entry use
    # the first entry's rate.
    commission_bps: float = 0.0          # retail zero-commission (post-2019)
    # FINRA TAF per-trade cap (USD); raised with the 2026 rate adjustment
    # (FINRA fee adjustment schedule, SR-FINRA-2024-019).
    finra_taf_cap_schedule: tuple = (
        ("2020-01-01", 8.30),            # through 2025-12-31
        ("2026-01-01", 9.79),            # 2026 onward
    )
    finra_taf_schedule: tuple = (
        ("2020-01-01", 0.000119),        # through 2021-12-31
        ("2022-01-01", 0.000145),        # 2022
        ("2023-01-01", 0.000166),        # 2023-2025
        ("2026-01-01", 0.000195),        # 2026 onward (SR-FINRA-2024-019)
    )
    sec_fee_schedule: tuple = (
        ("2020-01-01", 2.21e-5),         # $22.10/M (FY2020, window start)
        ("2021-02-25", 5.10e-6),         # $5.10/M
        ("2022-05-14", 2.29e-5),         # $22.90/M
        ("2023-02-27", 8.00e-6),         # $8.00/M
        ("2024-05-22", 2.78e-5),         # $27.80/M
        ("2025-05-14", 0.0),             # $0.00/M (fee suspended)
        ("2026-04-04", 2.06e-5),         # $20.60/M
    )
    name: str = field(default="fees_nyse", init=False)

    def _apply(self, ctx: TradeContext) -> TradeContext:
        date = pd.Timestamp(ctx.date).normalize()
        sec_fee_rate = self._sec_fee_rate_for(date)
        prices = np.asarray(ctx.prices, dtype=np.float64)
        trades = np.asarray(ctx.trades, dtype=np.float64)
        notional = np.abs(trades) * prices
        is_sell = trades < 0
        commission = notional * (self.commission_bps / 10_000.0)
        sec = np.where(is_sell, notional * sec_fee_rate, 0.0)
        taf_rate = self._taf_rate_for(date)
        taf_cap = self._taf_cap_for(date)
        taf = np.where(is_sell, np.minimum(np.abs(trades) * taf_rate, taf_cap), 0.0)
        total = commission + sec + taf
        sell_side = (
            np.where(is_sell, commission, 0.0).sum() + sec.sum() + taf.sum()
        )

        ctx.cash -= float(total.sum())
        ctx.diagnostics.add_scalar("fees_explicit", float(total.sum()))
        ctx.diagnostics["fees_by_type"] = {
            "commission": float(commission.sum()),
            "sec": float(sec.sum()),
            "finra_taf": float(taf.sum()),
        }
        ctx.diagnostics.add_scalar("sell_side_costs", float(sell_side))
        return ctx

    def _sec_fee_rate_for(self, date: pd.Timestamp) -> float:
        rate = self.sec_fee_schedule[0][1]
        for eff_date_str, eff_rate in self.sec_fee_schedule:
            if date >= pd.Timestamp(eff_date_str):
                rate = eff_rate
            else:
                break
        return rate

    def _taf_rate_for(self, date: pd.Timestamp) -> float:
        rate = self.finra_taf_schedule[0][1]
        for eff_date_str, eff_rate in self.finra_taf_schedule:
            if date >= pd.Timestamp(eff_date_str):
                rate = eff_rate
            else:
                break
        return rate

    def _taf_cap_for(self, date: pd.Timestamp) -> float:
        cap = self.finra_taf_cap_schedule[0][1]
        for eff_date_str, eff_cap in self.finra_taf_cap_schedule:
            if date >= pd.Timestamp(eff_date_str):
                cap = eff_cap
            else:
                break
        return cap

    def after_trade(self, ctx: TradeContext) -> TradeContext:
        return self._apply(ctx)

    def estimate_after_trade(self, ctx: TradeContext) -> TradeContext:
        return self._apply(ctx)


# ──────────────────────────────────────────────────────────────────────
# PSE (with CMEPA switch on STT)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PseFeeSchedule(FrictionLayer):
    commission_bps: float = 25.0          # broker commission, 25 bps typical retail
    vat_on_commission: float = 0.12       # 12% VAT on commission
    sccp_fee_rate: float = 0.0001         # 0.01% SCCP clearing
    pse_transaction_fee_rate: float = 0.00005   # 0.005%
    stt_pre_cmepa: float = 0.006          # 0.6% stock transaction tax on sells
    stt_post_cmepa: float = 0.001         # 0.1% STT after CMEPA
    cmepa_effective_date: str = "2025-07-01"
    name: str = field(default="fees_pse", init=False)

    def _apply(self, ctx: TradeContext) -> TradeContext:
        date = pd.Timestamp(ctx.date).normalize()
        prices = np.asarray(ctx.prices, dtype=np.float64)
        trades = np.asarray(ctx.trades, dtype=np.float64)
        notional = np.abs(trades) * prices
        is_sell = trades < 0
        stt_rate = self._stt_for(date)

        commission = notional * (self.commission_bps / 10_000.0)
        vat = commission * self.vat_on_commission
        sccp = notional * self.sccp_fee_rate
        pse_tx = notional * self.pse_transaction_fee_rate
        stt = np.where(is_sell, notional * stt_rate, 0.0)
        total = commission + vat + sccp + pse_tx + stt
        side_sell_share = np.where(is_sell, 1.0, 0.0)
        sell_side = float(
            (commission * side_sell_share).sum()
            + (vat * side_sell_share).sum()
            + (sccp * side_sell_share).sum()
            + (pse_tx * side_sell_share).sum()
            + stt.sum()
        )

        ctx.cash -= float(total.sum())
        ctx.diagnostics.add_scalar("fees_explicit", float(total.sum()))
        ctx.diagnostics["fees_by_type"] = {
            "commission": float(commission.sum()),
            "vat_on_commission": float(vat.sum()),
            "sccp": float(sccp.sum()),
            "pse_transaction_fee": float(pse_tx.sum()),
            "stock_transaction_tax": float(stt.sum()),
        }
        ctx.diagnostics["stt_rate_today"] = stt_rate
        ctx.diagnostics.add_scalar("sell_side_costs", sell_side)
        return ctx

    def _stt_for(self, date: pd.Timestamp) -> float:
        cutoff = pd.Timestamp(self.cmepa_effective_date)
        return self.stt_post_cmepa if date >= cutoff else self.stt_pre_cmepa

    def after_trade(self, ctx: TradeContext) -> TradeContext:
        return self._apply(ctx)

    def estimate_after_trade(self, ctx: TradeContext) -> TradeContext:
        return self._apply(ctx)

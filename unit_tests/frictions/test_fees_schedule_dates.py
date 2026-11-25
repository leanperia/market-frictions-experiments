"""Date-accuracy checks for the NYSE SEC Section 31 / FINRA TAF schedules.

Pins the exact rate picked for a sample of dates spanning every effective
window in ``NyseFeeSchedule.sec_fee_schedule`` and ``finra_taf_schedule``
(tests-review FRIC-3), including the 2025-05-14 SEC fee suspension and the
2026-04-04 reinstatement, plus the 8.30 USD per-trade TAF cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta.frictions.fees import NyseFeeSchedule
from meta.frictions.pipeline import TradeContext


def _ctx(date, trades, prices=None):
    n = len(trades)
    return TradeContext(
        time=0,
        date=pd.Timestamp(date),
        prices=np.array(prices or [100.0] * n, dtype=np.float64),
        volume=np.array([1e7] * n),
        volatility=np.array([0.02] * n),
        holdings=np.zeros(n),
        cash=1_000_000.0,
        target_weights=np.zeros(n + 1),
        trades=np.array(trades, dtype=np.int64),
    )


# (date, expected SEC fee rate per dollar of sells, expected TAF rate per share)
# Each date falls strictly inside one effective window of the source schedules.
SCHEDULE_CASES = [
    ("2020-06-01", 2.21e-5, 0.000119),  # FY2020 window start rates
    ("2021-03-01", 5.10e-6, 0.000119),  # SEC cut effective 2021-02-25
    ("2022-06-01", 2.29e-5, 0.000145),  # SEC raise 2022-05-14; TAF 2022 step
    ("2023-03-01", 8.00e-6, 0.000166),  # SEC cut 2023-02-27; TAF 2023 step
    ("2024-06-01", 2.78e-5, 0.000166),  # SEC raise 2024-05-22
    ("2025-06-01", 0.0, 0.000166),      # SEC fee suspended 2025-05-14
    ("2026-04-05", 2.06e-5, 0.000195),  # SEC fee reinstated 2026-04-04; TAF 2026 step (SR-FINRA-2024-019)
]


@pytest.mark.parametrize("date,sec_rate,taf_rate", SCHEDULE_CASES)
def test_sell_pays_exact_sec_and_taf_rate_for_date(date, sec_rate, taf_rate):
    # Sell 1000 shares at $100 -> notional = 100,000.
    # SEC fee = 100,000 * sec_rate; TAF = min(1000 * taf_rate, 8.30) — the cap
    # never binds here (max 1000 * 0.000166 = 0.166). Default schedule has
    # commission_bps = 0, so fees_explicit = sec + taf exactly.
    layer = NyseFeeSchedule()
    ctx = _ctx(date, [-1000])
    layer.after_trade(ctx)
    expected_sec = 100_000.0 * sec_rate
    expected_taf = 1000 * taf_rate
    assert ctx.diagnostics["fees_by_type"]["sec"] == pytest.approx(expected_sec, abs=1e-12)
    assert ctx.diagnostics["fees_by_type"]["finra_taf"] == pytest.approx(expected_taf, abs=1e-12)
    assert ctx.diagnostics["fees_by_type"]["commission"] == 0.0
    assert ctx.diagnostics["fees_explicit"] == pytest.approx(expected_sec + expected_taf, abs=1e-12)


def test_sec_fee_is_zero_during_2025_suspension():
    # 2025-05-14 advisory suspends the Section 31 fee: a 2025-06-01 sell of
    # 1000 shares at $100 pays SEC = 0 exactly; only TAF (1000 * 0.000166
    # = 0.166) remains.
    layer = NyseFeeSchedule()
    ctx = _ctx("2025-06-01", [-1000])
    layer.after_trade(ctx)
    assert ctx.diagnostics["fees_by_type"]["sec"] == 0.0
    assert ctx.diagnostics["fees_explicit"] == pytest.approx(0.166, abs=1e-12)


def test_sec_fee_reinstated_2026_04_04():
    # 2026-04-04 advisory reinstates the fee at $20.60/M: a 2026-04-05 sell
    # with notional 100,000 pays SEC = 100,000 * 2.06e-5 = 2.06.
    layer = NyseFeeSchedule()
    ctx = _ctx("2026-04-05", [-1000])
    layer.after_trade(ctx)
    assert ctx.diagnostics["fees_by_type"]["sec"] == pytest.approx(2.06, abs=1e-12)


def test_taf_rate_and_cap_step_up_in_2026():
    # SR-FINRA-2024-019: from 2026-01-01 the TAF is $0.000195/share, cap $9.79.
    # A 1000-share sell on 2026-04-05 pays TAF = 1000 * 0.000195 = 0.195.
    layer = NyseFeeSchedule()
    ctx = _ctx("2026-04-05", [-1000])
    layer.after_trade(ctx)
    assert ctx.diagnostics["fees_by_type"]["finra_taf"] == pytest.approx(0.195, abs=1e-12)


def test_taf_cap_binds_at_9_79_in_2026():
    # Sell 100,000 shares on 2026-04-05: uncapped TAF = 100,000 * 0.000195
    # = 19.50 > 9.79, so the 2026 per-trade cap binds and TAF charged is 9.79.
    layer = NyseFeeSchedule()
    ctx = _ctx("2026-04-05", [-100_000])
    layer.after_trade(ctx)
    assert 100_000 * 0.000195 > 9.79  # cap genuinely binds
    assert ctx.diagnostics["fees_by_type"]["finra_taf"] == pytest.approx(9.79, abs=1e-12)


def test_taf_capped_at_8_30_per_trade_for_large_sell():
    # Sell 100,000 shares on 2024-06-01: uncapped TAF = 100,000 * 0.000166
    # = 16.60 > 8.30, so the per-trade cap binds and TAF charged is exactly
    # 8.30. SEC fee is uncapped: 100,000 sh * $100 * 2.78e-5 = 278.00.
    layer = NyseFeeSchedule()
    ctx = _ctx("2024-06-01", [-100_000])
    layer.after_trade(ctx)
    assert 100_000 * 0.000166 > 8.30  # cap genuinely binds in this scenario
    assert ctx.diagnostics["fees_by_type"]["finra_taf"] == pytest.approx(8.30, abs=1e-12)
    assert ctx.diagnostics["fees_by_type"]["sec"] == pytest.approx(278.0, abs=1e-12)
    assert ctx.diagnostics["fees_explicit"] == pytest.approx(286.30, abs=1e-12)


@pytest.mark.parametrize("date", [d for d, _, _ in SCHEDULE_CASES])
def test_buys_pay_neither_sec_nor_taf_on_any_date(date):
    # Both SEC Section 31 and FINRA TAF are sell-side only; with the default
    # zero commission a buy incurs no explicit fees on any schedule date.
    layer = NyseFeeSchedule()
    ctx = _ctx(date, [1000])
    layer.after_trade(ctx)
    assert ctx.diagnostics["fees_by_type"]["sec"] == 0.0
    assert ctx.diagnostics["fees_by_type"]["finra_taf"] == 0.0
    assert ctx.diagnostics["fees_explicit"] == 0.0

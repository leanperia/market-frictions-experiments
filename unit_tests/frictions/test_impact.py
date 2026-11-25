"""Regression checks for shared impact helpers."""

from __future__ import annotations

import numpy as np
import pytest

from meta.env_market_impact.envs.impact_models import OWImpactModel
from meta.frictions.impact import estimate_impact_cost


def test_estimate_impact_cost_does_not_mutate_ow_transient_state():
    model = OWImpactModel()

    estimate_impact_cost(
        model,
        trades=[1_000],
        prices=[100.0],
        volatility=[0.02],
        volume=[1_000_000],
        symbols=["A"],
    )

    assert dict(model._transient_states) == {}


# ---------------------------------------------------------------------------
# Regression: OW transient decay must be owned by end_day() ALONE.
#
# Two coupled bugs were fixed here:
#   * (round 1) end_day() inherited from ImpactModel never decayed
#     _transient_states, so an idle symbol's residual was frozen.
#   * (codex 2026-06-15) the round-1 fix left apply_trade() ALSO decaying the
#     stored state, so a symbol traded on consecutive days was decayed twice
#     between days (once in apply_trade, once in end_day).
#
# Correct contract: end_day() applies exactly ONE exp(-kappa) decay per trading
# day to every symbol; apply_trade() consumes the stored residual AS-IS. The env
# calls apply_trade() during a step and end_day() at the end of the same step, so
# between consecutive trading days exactly one decay must elapse.
#
# These tests isolate the REAL transient charge from apply_trade() via total-cost
# differencing: with identical inputs the perm+temp components are identical
# across trades, so trans_cost(trade_k) = total(trade_k) - total(first_trade),
# where the first trade (empty residual) has trans_cost == 0.
# ---------------------------------------------------------------------------

_TRADE_KW = dict(
    trade_size=1_000,
    price=100.0,
    volatility=0.02,
    volume=1_000_000,
    symbol="A",
)


def _instant_impact_frac(model: OWImpactModel) -> float:
    part = abs(_TRADE_KW["trade_size"]) / _TRADE_KW["volume"]
    return model.Y * _TRADE_KW["volatility"] * np.sqrt(part)


def _real_trans_cost_after_end_days(n_end_days: int, *, half_life_days: float) -> float:
    """Transient component charged by a REAL apply_trade() on the trade that
    follows ``n_end_days`` end_day() calls after an initial day-0 trade.

    Isolated as total(second trade) - total(first trade): identical inputs make
    the perm+temp parts cancel, and the first trade's residual is empty so its
    trans_cost is 0.
    """
    model = OWImpactModel(Y=0.6, perm_fraction=0.25, half_life_days=half_life_days)
    base = model.apply_trade(**_TRADE_KW).cost          # trans_cost == 0 (empty residual)
    for d in range(n_end_days):
        model.end_day(f"2024-01-{d + 1:02d}")
    nxt = model.apply_trade(**_TRADE_KW).cost           # perm+temp (same) + trans
    return nxt - base


def test_ow_consecutive_day_trade_is_decayed_exactly_once():
    """codex 2026-06-15 regression: with half_life_days=1.0 (exp(-kappa)=0.5), a
    trade on the day AFTER a prior trade (one end_day between them) must be
    charged transient cost = 0.5 * I1 * |x| * P — NOT 0.25 (which the old
    double-decay produced)."""
    model = OWImpactModel(Y=0.6, perm_fraction=0.25, half_life_days=1.0)
    i1 = _instant_impact_frac(model)
    expected_one_decay = 0.5 * i1 * _TRADE_KW["trade_size"] * _TRADE_KW["price"]
    double_decay = 0.25 * i1 * _TRADE_KW["trade_size"] * _TRADE_KW["price"]

    trans = _real_trans_cost_after_end_days(1, half_life_days=1.0)
    assert trans == pytest.approx(expected_one_decay, rel=1e-9)
    assert trans != pytest.approx(double_decay, rel=1e-6)


@pytest.mark.parametrize("n_end_days", [1, 2, 3])
def test_ow_transient_residual_decays_once_per_end_day(n_end_days):
    """The real apply_trade transient charge after N end_day() calls equals
    I1 * exp(-kappa * N) * |x| * P — one decay per elapsed day, owned solely by
    end_day(); idle and consecutive-traded days behave identically."""
    model = OWImpactModel(Y=0.6, perm_fraction=0.25, half_life_days=0.08)
    i1 = _instant_impact_frac(model)
    expected = i1 * np.exp(-model.kappa * n_end_days) * _TRADE_KW["trade_size"] * _TRADE_KW["price"]

    trans = _real_trans_cost_after_end_days(n_end_days, half_life_days=0.08)
    assert trans == pytest.approx(expected, rel=1e-9)

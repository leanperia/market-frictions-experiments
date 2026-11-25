"""Shared impact-model family helpers for experiment runners and envs."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from meta.env_market_impact.envs.impact_models import (
    ACImpactModel,
    BaselineImpactModel,
    ImpactModel,
    OWImpactModel,
    SqrtImpactModel,
)

IMPACT_MODEL_FAMILIES = {
    "baseline": BaselineImpactModel,
    "sqrt": SqrtImpactModel,
    "ac": ACImpactModel,
    "ow": OWImpactModel,
}


def build_impact_model(friction_cfg: Mapping[str, Any]) -> ImpactModel:
    spec = friction_cfg.get("impact_model", {"name": "baseline"})
    name = spec.get("name", "baseline")
    if name not in IMPACT_MODEL_FAMILIES:
        raise ValueError(f"Unknown impact model family: {name}")
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return IMPACT_MODEL_FAMILIES[name](**kwargs)


def estimate_impact_cost(
    impact_model: ImpactModel,
    trades,
    prices,
    volatility,
    volume,
    symbols: Sequence[str],
) -> float:
    """Estimate impact costs without mutating the live impact model."""

    model = copy.copy(impact_model)
    for attr in ("_perm_state", "_transient_states"):
        if hasattr(impact_model, attr):
            setattr(model, attr, copy.deepcopy(getattr(impact_model, attr)))
    if hasattr(model, "_impact_records"):
        model._impact_records = []

    total = 0.0
    for idx, shares in enumerate(trades):
        if shares == 0:
            continue
        impact = model.apply_trade(
            shares,
            float(prices[idx]),
            float(volatility[idx]),
            float(volume[idx]),
            str(symbols[idx]),
        )
        total += float(impact.cost)
    return total

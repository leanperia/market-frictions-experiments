"""Execution diagnostics helpers for the friction stack.

The friction stack records per-step execution diagnostics that are consumed by
the canonical environment and the experiment runner. ``FrictionStepDiagnostics``
keeps that Interface stable while still behaving like a mutable mapping so the
friction layers can update it incrementally.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np


@dataclass
class FrictionStepDiagnostics(MutableMapping[str, Any]):
    """Mutable mapping with small helpers for scalar and vector diagnostics."""

    _data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> "FrictionStepDiagnostics":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            return cls(dict(value))
        raise TypeError(f"Unsupported diagnostics payload: {type(value)!r}")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def clone(self) -> "FrictionStepDiagnostics":
        return type(self)(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            elif isinstance(value, dict):
                out[key] = dict(value)
            else:
                out[key] = value
        return out

    def add_scalar(self, key: str, value: float) -> None:
        self._data[key] = float(self._data.get(key, 0.0)) + float(value)

    def add_array(self, key: str, value: Any) -> None:
        arr = np.asarray(value)
        prev = self._data.get(key)
        self._data[key] = arr.copy() if prev is None else np.asarray(prev) + arr


def _sum_abs(value: Any) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64)
    return float(np.abs(arr).sum())


def _mean(value: Any) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _max(value: Any) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64)
    return float(arr.max()) if arr.size else 0.0


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _array_at(value: Any, idx: int, default: float = 0.0) -> float:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _float(arr.item())
    if idx >= arr.size:
        return default
    return _float(arr.reshape(-1)[idx])


def flatten_step_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    friction = FrictionStepDiagnostics.coerce(info.get("friction")).to_dict()
    trades = info.get("trades", []) or []
    fees_by_type = friction.get("fees_by_type", {}) or {}
    reallocation_requested = bool(info.get("reallocation_requested", True))
    target = info.get("target_weights") if reallocation_requested else None
    realized = info.get("realized_weights")
    divergence = 0.0
    if target is not None and realized is not None:
        divergence = float(np.abs(np.asarray(target) - np.asarray(realized)).sum())

    row = {
        "reallocation_requested": reallocation_requested,
        "turnover": _float(info.get("turnover")),
        "impact_cost": _float(info.get("cost")),
        "total_buy_value": _float(info.get("total_buy_value")),
        "total_sell_value": _float(info.get("total_sell_value")),
        "cash": _float(info.get("cash")),
        "n_trades": len(trades),
        "traded_shares": float(sum(_float(t.get("shares")) for t in trades)),
        "traded_notional": float(sum(_float(t.get("notional")) for t in trades)),
        "trade_pov_mean": float(np.mean([_float(t.get("pov")) for t in trades])) if trades else 0.0,
        "trade_pov_max": float(np.max([_float(t.get("pov")) for t in trades])) if trades else 0.0,
        "realized_target_l1": divergence,
        "fees_explicit": _float(friction.get("fees_explicit")),
        "spread_cost": _float(friction.get("spread_cost")),
        "half_spread_avg": _float(friction.get("half_spread_avg")),
        "participation_pct_mean": _mean(friction.get("participation_pct")),
        "participation_pct_max": _max(friction.get("participation_pct")),
        "blocked_by_adv": _sum_abs(friction.get("blocked_by_adv")),
        "blocked_by_lot": _sum_abs(friction.get("blocked_by_lot")),
        "blocked_by_cash": _sum_abs(friction.get("blocked_by_cash")),
        "blocked_by_outage": _sum_abs(friction.get("blocked_by_outage")),
        "settled_cash": _float(friction.get("settled_cash")),
        "in_transit_cash": _float(friction.get("in_transit_cash")),
        "settlement_scale": _float(friction.get("settlement_scale", 1.0)),
        "execution_lag_days": _float(friction.get("execution_lag_days")),
        "execution_price": friction.get("execution_price", ""),
        "outage_scale": _float(friction.get("outage_scale", 1.0)),
        "outage_nature": friction.get("outage_nature", ""),
        "cash_shortfall": _float(friction.get("cash_shortfall")),
        "cash_shortfall_over_tol": bool(friction.get("cash_shortfall_over_tol", False)),
        "sell_side_costs": _float(friction.get("sell_side_costs")),
        "impact_sell_side_cost": _float(friction.get("impact_sell_side_cost")),
    }
    for key, value in fees_by_type.items():
        row[f"fees_{key}"] = _float(value)
    return row


def expand_asset_step_info(
    date: Any,
    stock_symbols: Sequence[str],
    info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    friction = FrictionStepDiagnostics.coerce(info.get("friction")).to_dict()
    stock_dim = len(stock_symbols)
    reallocation_requested = bool(info.get("reallocation_requested", True))
    target_payload = info.get("target_weights") if reallocation_requested else None
    target = np.asarray(
        target_payload if target_payload is not None else np.full(1 + stock_dim, np.nan),
        dtype=np.float64,
    )
    realized = np.asarray(info.get("realized_weights", np.zeros(1 + stock_dim)), dtype=np.float64)
    exec_prices = info.get("execution_prices")
    base_exec_prices = info.get("base_execution_prices")
    quoted_exec_prices = info.get("quoted_execution_prices")
    close_prices = info.get("close_prices")
    base_close_prices = info.get("base_close_prices")
    volume = info.get("volume")
    trade_by_idx = {int(t["stock_idx"]): t for t in info.get("trades", []) or []}

    rows = []
    for idx, symbol in enumerate(stock_symbols):
        trade = trade_by_idx.get(idx, {})
        rows.append(
            {
                "date": date,
                "reallocation_requested": reallocation_requested,
                "stock_idx": idx,
                "symbol": str(symbol),
                "target_weight": _array_at(target, idx + 1),
                "realized_weight": _array_at(realized, idx + 1),
                "execution_price": _array_at(exec_prices, idx),
                "base_execution_price": _array_at(base_exec_prices, idx),
                "quoted_execution_price": _array_at(quoted_exec_prices, idx),
                "close_price": _array_at(close_prices, idx),
                "base_close_price": _array_at(base_close_prices, idx),
                "volume": _array_at(volume, idx),
                "side": trade.get("side", ""),
                "executed_shares": _float(trade.get("signed_shares")),
                "abs_executed_shares": _float(trade.get("shares")),
                "executed_notional": _float(trade.get("notional")),
                "trade_pov": _float(trade.get("pov")),
                "turnover_percentile": _float(trade.get("turnover_percentile")),
                "impact_cost": _float(trade.get("impact_cost")),
                "participation_pct": _array_at(friction.get("participation_pct"), idx),
                "blocked_by_adv": _array_at(friction.get("blocked_by_adv"), idx),
                "blocked_by_lot": _array_at(friction.get("blocked_by_lot"), idx),
                "blocked_by_cash": _array_at(friction.get("blocked_by_cash"), idx),
                "blocked_by_outage": _array_at(friction.get("blocked_by_outage"), idx),
                "outage_scale": _float(friction.get("outage_scale", 1.0)),
                "outage_nature": friction.get("outage_nature", ""),
                "execution_mode": friction.get("execution_price", ""),
            }
        )
    return rows

"""Friction-pipeline scaffolding.

A ``FrictionPipeline`` is an ordered list of ``FrictionLayer`` instances
that transform a per-step ``TradeContext``. The env calls
``pipeline.before_trade(ctx)`` after computing target trades but before
executing them, and ``pipeline.after_trade(ctx)`` once trades have been
applied and impact has decayed. Each layer can mutate ``ctx.trades`` and
record numbers under ``ctx.diagnostics``.

An env constructed with ``FrictionPipeline()`` (empty layer list) must
produce the same trajectory as one constructed with ``friction_pipeline
= None``. Stateful layers should implement ``reset()`` so training and
backtest episodes do not leak state across resets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .diagnostics import FrictionStepDiagnostics


@dataclass
class TradeContext:
    """Mutable per-step state passed through the pipeline.

    Layers are free to overwrite ``trades`` (integer share deltas, sign
    = side) and ``cash``, and to write into ``diagnostics``. Anything
    else should be treated as read-only — the env owns those fields.
    """

    time: int
    date: Any
    prices: np.ndarray
    volume: np.ndarray
    volatility: np.ndarray
    holdings: np.ndarray
    cash: float
    target_weights: np.ndarray
    trades: np.ndarray
    quoted_prices: Optional[np.ndarray] = None
    close_prices: Optional[np.ndarray] = None
    open_prices: Optional[np.ndarray] = None
    # Current shares per share held on this bar (CRSP CFACSHR analogue). Board
    # lots are defined in the shares that legally existed on the trade date, so
    # a layer enforcing them must convert into that space and back. ``None``
    # means the panel predates Option-1 accounting and panel units *are* legal
    # shares.
    share_factors: Optional[np.ndarray] = None
    diagnostics: FrictionStepDiagnostics = field(default_factory=FrictionStepDiagnostics)

    def __post_init__(self) -> None:
        self.diagnostics = FrictionStepDiagnostics.coerce(self.diagnostics)


class FrictionLayer:
    """Base class for a single friction layer.

    Subclasses override either ``before_trade`` or ``after_trade`` (or
    both). The default implementations are pass-through so subclasses
    only implement the side they care about.
    """

    name: str = "friction_layer"
    layer_id: Optional[str] = None
    execution_price_source: str = "close"

    def reset(self) -> None:
        """Reset any episode-local state held by the layer."""
        return None

    def before_trade(self, ctx: TradeContext) -> TradeContext:
        return ctx

    def after_trade(self, ctx: TradeContext) -> TradeContext:
        return ctx

    def estimate_after_trade(self, ctx: TradeContext) -> TradeContext:
        return ctx

    def after_cash_scaling(self, ctx: TradeContext) -> TradeContext:
        return ctx

    @property
    def uses_open_prices(self) -> bool:
        return self.execution_price_source == "open"


class FrictionPipeline:
    """Ordered composition of friction layers.

    An empty pipeline is a documented no-op — used by Prototype #1 to
    verify the plumbing.
    """

    def __init__(self, layers: Optional[List[FrictionLayer]] = None) -> None:
        self.layers: List[FrictionLayer] = list(layers) if layers else []
        self._validate_layer_ordering()

    def _validate_layer_ordering(self) -> None:
        """Check that settlement appears after all sell-side cost layers.

        Settlement reads ``sell_side_costs`` from diagnostics, which is
        written by ``fees*`` and ``spread`` layers.  If settlement comes
        before any configured cost producer, it silently omits those costs
        and computes incorrect net proceeds.
        """
        names = [getattr(layer, "name", type(layer).__name__) for layer in self.layers]
        if "settlement" in names:
            settlement_idx = names.index("settlement")
            preceding = names[:settlement_idx]
            following = names[settlement_idx + 1:]

            def is_sell_side_cost_producer(name: str) -> bool:
                return name.startswith("fees") or name.startswith("spread")

            preceding_cost_producers = [n for n in preceding if is_sell_side_cost_producer(n)]
            following_cost_producers = [n for n in following if is_sell_side_cost_producer(n)]
            if not preceding_cost_producers or following_cost_producers:
                raise ValueError(
                    f"FrictionPipeline: 'settlement' at position {settlement_idx} "
                    f"requires all fees/spread layers before it (they write "
                    f"'sell_side_costs' to diagnostics). Layers after settlement: "
                    f"{following_cost_producers}. "
                    f"Layer order: {names}"
                )

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        names = [getattr(layer, "name", type(layer).__name__) for layer in self.layers]
        return f"FrictionPipeline(layers={names})"

    def before_trade(self, ctx: TradeContext) -> TradeContext:
        for layer in self.layers:
            ctx = layer.before_trade(ctx)
        return ctx

    def after_trade(self, ctx: TradeContext) -> TradeContext:
        for layer in self.layers:
            ctx = layer.after_trade(ctx)
        return ctx

    def estimate_after_trade(self, ctx: TradeContext) -> TradeContext:
        for layer in self.layers:
            ctx = layer.estimate_after_trade(ctx)
        return ctx

    def after_cash_scaling(self, ctx: TradeContext) -> TradeContext:
        for layer in self.layers:
            ctx = layer.after_cash_scaling(ctx)
        return ctx

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for key, layer in self._stateful_layers():
            get_state = getattr(layer, "get_state", None)
            if callable(get_state):
                state[key] = get_state()
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        for key, layer in self._stateful_layers():
            if key in state:
                set_state = getattr(layer, "set_state", None)
                if callable(set_state):
                    set_state(state[key])

    def _stateful_layers(self):
        seen: Dict[str, int] = {}
        for layer in self.layers:
            base_key = getattr(layer, "layer_id", None) or getattr(layer, "name", type(layer).__name__)
            seen[base_key] = seen.get(base_key, 0) + 1
            key = base_key if seen[base_key] == 1 else f"{base_key}#{seen[base_key]}"
            yield key, layer

    @property
    def execution_price_source(self) -> str:
        sources = {
            getattr(layer, "execution_price_source", "close")
            for layer in self.layers
            if getattr(layer, "execution_price_source", "close") != "close"
        }
        if len(sources) > 1:
            raise ValueError(f"Conflicting execution price sources in pipeline: {sorted(sources)}")
        return next(iter(sources), "close")

    @property
    def uses_open_prices(self) -> bool:
        return self.execution_price_source == "open"

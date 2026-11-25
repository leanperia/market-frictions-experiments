"""Layer registry and pipeline construction for the friction stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from .markets import get_market_layer_builders
from .pipeline import FrictionLayer, FrictionPipeline

LayerFactory = Callable[[Dict[str, Any], Mapping[str, Any]], FrictionLayer]


@dataclass(frozen=True)
class LayerBuilder:
    factory: LayerFactory
    required_env_keys: tuple[str, ...] = ()

    def build(self, name: str, params: Dict[str, Any], env_config: Mapping[str, Any]) -> FrictionLayer:
        missing = [key for key in self.required_env_keys if key not in env_config]
        if missing:
            raise ValueError(f"Layer {name!r} requires env inputs: {missing}")
        return self.factory(dict(params or {}), env_config)


def _build_execution_timing(params: Dict[str, Any], _env_config: Mapping[str, Any]) -> FrictionLayer:
    from .execution_timing import NextDayExecution

    return NextDayExecution(**params)


def _build_adv_cap(params: Dict[str, Any], env_config: Mapping[str, Any]) -> FrictionLayer:
    from .adv_cap import AdvCap

    return AdvCap(adv20_array=env_config["adv20_array"], **params)


def _build_spread(params: Dict[str, Any], env_config: Mapping[str, Any]) -> FrictionLayer:
    import numpy as np

    from .spread import SpreadHaircut, corwin_schultz_spread

    floor = params.pop("spread_floor", 0.0005)
    flat_spread = params.pop("flat_spread", None)
    high = np.asarray(env_config["high_array"], dtype=np.float64)
    if flat_spread is not None:
        # Flat per-market effective spread (ADR-0001): daily Corwin-Schultz is
        # unreliable for liquid large-caps (fails ~45% of days, volatility-
        # inflated otherwise). Use a constant literature-calibrated spread and
        # set the floor to that value so it is applied exactly.
        spread = np.full(high.shape, float(flat_spread), dtype=np.float64)
        return SpreadHaircut(spread, spread_floor=float(flat_spread))
    spread = corwin_schultz_spread(high, env_config["low_array"], spread_floor=floor)
    return SpreadHaircut(spread, spread_floor=floor)


def _build_settlement(params: Dict[str, Any], env_config: Mapping[str, Any]) -> FrictionLayer:
    from .settlement import SettlementLag

    return SettlementLag(trading_dates=env_config.get("date_list"), **params)


CORE_LAYER_BUILDERS = {
    "execution_timing": LayerBuilder(factory=_build_execution_timing),
    "adv_cap": LayerBuilder(factory=_build_adv_cap, required_env_keys=("adv20_array",)),
    "spread": LayerBuilder(factory=_build_spread, required_env_keys=("high_array", "low_array")),
    "settlement": LayerBuilder(factory=_build_settlement),
}


def _market_builder(name: str, market: str) -> LayerBuilder | None:
    market_builders = get_market_layer_builders(market)
    spec = market_builders.get(name)
    if spec is None:
        return None
    return LayerBuilder(
        factory=spec["factory"],
        required_env_keys=tuple(spec.get("required_env_keys", ())),
    )


def build_layer(
    name: str,
    params: Dict[str, Any],
    env_config: Mapping[str, Any],
    *,
    market: str,
) -> FrictionLayer:
    builder = CORE_LAYER_BUILDERS.get(name) or _market_builder(name, market)
    if builder is None:
        raise ValueError(f"Unknown or unsupported friction layer {name!r} for market {market!r}")
    return builder.build(name, params, env_config)


def build_pipeline_from_config(
    friction_config: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    env_config: Mapping[str, Any],
    *,
    market: str,
) -> FrictionPipeline:
    layer_specs = (
        list(friction_config)
        if isinstance(friction_config, Sequence) and not isinstance(friction_config, Mapping)
        else list((friction_config or {}).get("layers", []))
    )
    seen_ids: Dict[str, int] = {}
    layers = []
    for spec in layer_specs:
        name = spec["name"]
        layer = build_layer(name, spec.get("params", {}), env_config, market=market)
        base_id = spec.get("id") or name
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        layer.layer_id = base_id if seen_ids[base_id] == 1 else f"{base_id}#{seen_ids[base_id]}"
        layers.append(layer)
    return FrictionPipeline(layers)

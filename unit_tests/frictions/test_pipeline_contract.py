"""Contract checks for the friction pipeline seam."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.diagnostics import FrictionStepDiagnostics
from meta.frictions.pipeline import FrictionLayer, FrictionPipeline, TradeContext


def _ctx(trades):
    n = len(trades)
    return TradeContext(
        time=0,
        date=pd.Timestamp("2024-01-15"),
        prices=np.array([100.0] * n, dtype=np.float64),
        volume=np.array([1e6] * n),
        volatility=np.array([0.02] * n),
        holdings=np.zeros(n),
        cash=1_000.0,
        target_weights=np.zeros(n + 1),
        trades=np.array(trades, dtype=np.int64),
        diagnostics=FrictionStepDiagnostics(),
    )


class _EstimateProbe(FrictionLayer):
    name = "estimate_probe"

    def estimate_after_trade(self, ctx: TradeContext) -> TradeContext:
        ctx.cash -= 12.5
        ctx.diagnostics["estimate_probe"] = True
        return ctx


class _QuantizerProbe(FrictionLayer):
    name = "quantizer_probe"

    def after_cash_scaling(self, ctx: TradeContext) -> TradeContext:
        ctx.trades = (ctx.trades // 2).astype(ctx.trades.dtype)
        ctx.diagnostics["quantizer_probe"] = True
        return ctx


class _StateProbe(FrictionLayer):
    name = "state_probe"

    def __init__(self) -> None:
        self.value = 7

    def get_state(self):
        return {"value": self.value}

    def set_state(self, state):
        self.value = state["value"]


def test_estimate_after_trade_uses_layer_hook():
    pipeline = FrictionPipeline([_EstimateProbe()])
    out = pipeline.estimate_after_trade(_ctx([10]))
    assert out.cash == 987.5
    assert out.diagnostics["estimate_probe"] is True


def test_after_cash_scaling_uses_layer_hook():
    pipeline = FrictionPipeline([_QuantizerProbe()])
    out = pipeline.after_cash_scaling(_ctx([9, -7]))
    np.testing.assert_array_equal(out.trades, np.array([4, -4]))
    assert out.diagnostics["quantizer_probe"] is True


def test_state_keys_use_stable_layer_ids_not_positions():
    probe = _StateProbe()
    probe.layer_id = "settlement_cash"
    pipeline = FrictionPipeline([probe])
    assert pipeline.get_state() == {"settlement_cash": {"value": 7}}


# ── Layer ordering validation ────────────────────────────────────────


class _FakeLayer(FrictionLayer):
    """Minimal layer stub for ordering tests."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_settlement_after_fees_is_valid():
    """fees -> settlement should not raise."""
    pipeline = FrictionPipeline([_FakeLayer("fees"), _FakeLayer("settlement")])
    assert len(pipeline) == 2


def test_settlement_after_spread_is_valid():
    """spread -> settlement should not raise."""
    pipeline = FrictionPipeline([_FakeLayer("spread"), _FakeLayer("settlement")])
    assert len(pipeline) == 2


def test_settlement_after_both_fees_and_spread_is_valid():
    """fees -> spread -> settlement should not raise."""
    pipeline = FrictionPipeline([
        _FakeLayer("fees"),
        _FakeLayer("spread"),
        _FakeLayer("settlement"),
    ])
    assert len(pipeline) == 3


def test_settlement_before_fees_raises():
    """settlement -> fees must raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="settlement.*requires.*fees/spread"):
        FrictionPipeline([_FakeLayer("settlement"), _FakeLayer("fees")])


def test_settlement_alone_raises():
    """settlement without any cost producer must raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="settlement.*requires.*fees/spread"):
        FrictionPipeline([_FakeLayer("settlement")])


def test_settlement_between_fees_and_spread_raises():
    """fees -> settlement -> spread must raise because spread costs would be omitted."""
    import pytest
    with pytest.raises(ValueError, match="Layers after settlement.*spread"):
        FrictionPipeline([
            _FakeLayer("fees"),
            _FakeLayer("settlement"),
            _FakeLayer("spread"),
        ])


def test_settlement_between_spread_and_fees_raises():
    """spread -> settlement -> fees must raise because fee costs would be omitted."""
    import pytest
    with pytest.raises(ValueError, match="Layers after settlement.*fees"):
        FrictionPipeline([
            _FakeLayer("spread"),
            _FakeLayer("settlement"),
            _FakeLayer("fees"),
        ])


def test_settlement_after_market_specific_fees_is_valid():
    """fees_nyse -> settlement should pass (prefix matching)."""
    pipeline = FrictionPipeline([_FakeLayer("fees_nyse"), _FakeLayer("settlement")])
    assert len(pipeline) == 2


def test_settlement_after_market_specific_fees_pse_is_valid():
    """fees_pse -> settlement should pass (prefix matching)."""
    pipeline = FrictionPipeline([_FakeLayer("fees_pse"), _FakeLayer("settlement")])
    assert len(pipeline) == 2


def test_pipeline_without_settlement_is_always_valid():
    """No settlement layer means no ordering constraint."""
    pipeline = FrictionPipeline([_FakeLayer("adv_cap"), _FakeLayer("fees")])
    assert len(pipeline) == 2


def test_empty_pipeline_is_valid():
    """Empty pipeline should not raise."""
    pipeline = FrictionPipeline([])
    assert len(pipeline) == 0

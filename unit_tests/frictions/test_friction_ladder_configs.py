"""Instantiation checks for the seven active F0-F3 friction ladder configs.

Each YAML under ``configs/friction/`` is loaded for real through
``build_pipeline_from_config`` / ``build_impact_model`` and asserted to
produce exactly the intended layer stack and impact model.  This catches
silent drift between the experiment ladder and the registry (renamed
layers, dropped params, reordered stacks) that no other test exercises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from meta.frictions import build_impact_model, build_pipeline_from_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "friction"

T, N = 5, 2


def _env_config():
    """Synthetic T=5, N=2 env inputs covering every registry-required key."""
    rng = np.random.default_rng(42)
    prices = 100.0 + rng.uniform(0.0, 1.0, size=(T, N))
    return {
        "date_list": np.array(pd.date_range("2024-01-01", periods=T, freq="B")),
        "price_array": prices,
        "open_array": prices * 0.999,
        "high_array": prices * 1.01,
        "low_array": prices * 0.99,
        "adv20_array": np.full((T, N), 1_000_000.0, dtype=np.float64),
        "volume_array": np.full((T, N), 2_000_000.0, dtype=np.float64),
        "volatility_array": np.full((T, N), 0.02, dtype=np.float64),
        "tic_list": ["AAA", "BBB"],
    }


def _load(filename):
    with open(CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)["friction"]


def _build(filename, market):
    friction_cfg = _load(filename)
    pipeline = build_pipeline_from_config(friction_cfg, _env_config(), market=market)
    return friction_cfg, pipeline


def _class_names(pipeline):
    return [type(layer).__name__ for layer in pipeline.layers]


def _layer(pipeline, cls_name):
    for layer in pipeline.layers:
        if type(layer).__name__ == cls_name:
            return layer
    raise AssertionError(f"no layer of class {cls_name!r} in pipeline")


# ── ladder inventory ─────────────────────────────────────────────────


def test_ladder_inventory_is_exactly_the_seven_active_configs():
    """The experiment ladder is F0-F3 only.

    F4 (operational frictions: outages/halts) was evaluated and rejected,
    so an accidental F4 config — or a renamed level — must fail this pin.
    """
    assert sorted(p.name for p in CONFIG_DIR.glob("*.yaml")) == [
        "F0_bare.yaml",
        "F1_basic_nyse.yaml",
        "F1_basic_pse.yaml",
        "F2_liquidity_nyse.yaml",
        "F2_liquidity_pse.yaml",
        "F3_structural_nyse.yaml",
        "F3_structural_pse.yaml",
    ]


# ── F0 ───────────────────────────────────────────────────────────────


def test_f0_bare_is_empty_pipeline_with_10bps_baseline():
    friction_cfg, pipeline = _build("F0_bare.yaml", market="nyse")

    assert friction_cfg["level"] == "F0"
    assert friction_cfg["cost_buffer_pct"] == pytest.approx(0.002)
    assert friction_cfg["max_trade_volume_pct"] == pytest.approx(1.0)
    assert pipeline.layers == []

    impact = build_impact_model(friction_cfg)
    assert type(impact).__name__ == "BaselineImpactModel"
    assert impact.basis_points == 10
    assert impact.fee_rate == pytest.approx(0.0010)


def test_f0_bare_is_market_agnostic():
    """F0 has no layers, so it must build identically under either market."""
    friction_cfg = _load("F0_bare.yaml")
    for market in ("nyse", "pse"):
        pipeline = build_pipeline_from_config(friction_cfg, _env_config(), market=market)
        assert pipeline.layers == []


# ── F1 ───────────────────────────────────────────────────────────────


def test_f1_nyse_layers_and_params():
    friction_cfg, pipeline = _build("F1_basic_nyse.yaml", market="nyse")

    assert friction_cfg["level"] == "F1"
    assert friction_cfg["cost_buffer_pct"] == pytest.approx(0.002)
    assert _class_names(pipeline) == ["NextDayExecution", "NyseFeeSchedule", "SpreadHaircut"]

    fees = _layer(pipeline, "NyseFeeSchedule")
    assert fees.commission_bps == pytest.approx(0.0)

    # flat_spread=0.0004 in the YAML: the registry builds a constant spread
    # array and sets the floor to the flat value (overriding spread_floor's
    # 0.0005 — see _build_spread in meta/frictions/registry.py, ADR-0001).
    spread = _layer(pipeline, "SpreadHaircut")
    assert spread.spread_floor == pytest.approx(0.0004)
    np.testing.assert_allclose(spread.spread_array, np.full((T, N), 0.0004))

    impact = build_impact_model(friction_cfg)
    assert type(impact).__name__ == "BaselineImpactModel"
    assert impact.basis_points == 0
    assert impact.fee_rate == pytest.approx(0.0)


def test_f1_pse_layers_and_params():
    friction_cfg, pipeline = _build("F1_basic_pse.yaml", market="pse")

    assert friction_cfg["level"] == "F1"
    assert friction_cfg["cost_buffer_pct"] == pytest.approx(0.005)
    assert _class_names(pipeline) == ["NextDayExecution", "PseFeeSchedule", "SpreadHaircut"]

    fees = _layer(pipeline, "PseFeeSchedule")
    assert fees.commission_bps == pytest.approx(25.0)
    assert fees.cmepa_effective_date == "2025-07-01"

    # flat_spread=0.0012 overrides spread_floor=0.0010 (see NYSE note above).
    spread = _layer(pipeline, "SpreadHaircut")
    assert spread.spread_floor == pytest.approx(0.0012)
    np.testing.assert_allclose(spread.spread_array, np.full((T, N), 0.0012))

    impact = build_impact_model(friction_cfg)
    assert type(impact).__name__ == "BaselineImpactModel"
    assert impact.basis_points == 0


# ── F2 ───────────────────────────────────────────────────────────────


def test_f2_nyse_layers_and_params():
    friction_cfg, pipeline = _build("F2_liquidity_nyse.yaml", market="nyse")

    assert friction_cfg["level"] == "F2"
    assert friction_cfg["cost_buffer_pct"] == pytest.approx(0.003)
    assert _class_names(pipeline) == [
        "NextDayExecution",
        "AdvCap",
        "NyseFeeSchedule",
        "SpreadHaircut",
    ]

    assert _layer(pipeline, "AdvCap").max_pov == pytest.approx(0.10)
    assert _layer(pipeline, "NyseFeeSchedule").commission_bps == pytest.approx(0.0)
    assert _layer(pipeline, "SpreadHaircut").spread_floor == pytest.approx(0.0004)

    impact = build_impact_model(friction_cfg)
    assert type(impact).__name__ == "SqrtImpactModel"
    assert impact.Y == pytest.approx(0.6)
    assert impact.perm_fraction == pytest.approx(0.25)


def test_f2_pse_layers_and_params():
    friction_cfg, pipeline = _build("F2_liquidity_pse.yaml", market="pse")

    assert friction_cfg["level"] == "F2"
    assert friction_cfg["cost_buffer_pct"] == pytest.approx(0.005)
    assert _class_names(pipeline) == [
        "NextDayExecution",
        "AdvCap",
        "PseFeeSchedule",
        "SpreadHaircut",
    ]

    assert _layer(pipeline, "AdvCap").max_pov == pytest.approx(0.10)
    fees = _layer(pipeline, "PseFeeSchedule")
    assert fees.commission_bps == pytest.approx(25.0)
    assert fees.cmepa_effective_date == "2025-07-01"
    assert _layer(pipeline, "SpreadHaircut").spread_floor == pytest.approx(0.0012)

    impact = build_impact_model(friction_cfg)
    assert type(impact).__name__ == "SqrtImpactModel"
    assert impact.Y == pytest.approx(0.6)
    assert impact.perm_fraction == pytest.approx(0.25)


# ── F3 ───────────────────────────────────────────────────────────────


def test_f3_nyse_layers_and_params():
    friction_cfg, pipeline = _build("F3_structural_nyse.yaml", market="nyse")

    assert friction_cfg["level"] == "F3"
    assert friction_cfg["cost_buffer_pct"] == pytest.approx(0.003)
    assert _class_names(pipeline) == [
        "NextDayExecution",
        "AdvCap",
        "NyseFeeSchedule",
        "SpreadHaircut",
        "SettlementLag",
    ]

    settlement = _layer(pipeline, "SettlementLag")
    assert settlement.default_cycle == 2
    # T+2 -> T+1 switch on the SEC effective date (ADR-0001).
    assert settlement._t1_effective == pd.Timestamp("2024-05-28")

    impact = build_impact_model(friction_cfg)
    assert type(impact).__name__ == "SqrtImpactModel"
    assert impact.Y == pytest.approx(0.6)
    assert impact.perm_fraction == pytest.approx(0.25)


def test_f3_pse_layers_and_params():
    friction_cfg, pipeline = _build("F3_structural_pse.yaml", market="pse")

    assert friction_cfg["level"] == "F3"
    assert friction_cfg["cost_buffer_pct"] == pytest.approx(0.007)
    assert _class_names(pipeline) == [
        "NextDayExecution",
        "AdvCap",
        "PseFeeSchedule",
        "SpreadHaircut",
        "SettlementLag",
        "BoardLotRounding",
    ]

    settlement = _layer(pipeline, "SettlementLag")
    assert settlement.default_cycle == 2
    # PSE stays on T+2 throughout the window: no T+1 switch configured.
    assert settlement._t1_effective is None

    impact = build_impact_model(friction_cfg)
    assert type(impact).__name__ == "SqrtImpactModel"
    assert impact.Y == pytest.approx(0.6)
    assert impact.perm_fraction == pytest.approx(0.25)


# ── load-bearing layer ORDER constraints (F3) ────────────────────────


@pytest.mark.parametrize(
    "filename,market",
    [("F3_structural_nyse.yaml", "nyse"), ("F3_structural_pse.yaml", "pse")],
)
def test_f3_fees_and_spread_run_before_settlement(filename, market):
    """settlement.after_trade queues NET sell proceeds (gross minus the
    sell-side fees+spread already recorded in diagnostics), so the fee and
    spread layers must run strictly EARLIER in the layer list.  A reorder
    must fail here loudly, not silently overstate settled cash.
    """
    _, pipeline = _build(filename, market=market)
    names = _class_names(pipeline)

    settlement_idx = names.index("SettlementLag")
    fees_idx = names.index("NyseFeeSchedule" if market == "nyse" else "PseFeeSchedule")
    spread_idx = names.index("SpreadHaircut")

    assert fees_idx < settlement_idx
    assert spread_idx < settlement_idx


def test_f3_pse_board_lots_is_last_layer():
    """board_lots.before_trade rounds to legal increments, so it must run
    AFTER settlement.before_trade (which scales buys to settled cash and can
    produce fractional share counts) — i.e. board_lots is the LAST layer.
    """
    _, pipeline = _build("F3_structural_pse.yaml", market="pse")
    names = _class_names(pipeline)

    assert names.index("BoardLotRounding") > names.index("SettlementLag")
    assert names[-1] == "BoardLotRounding"


# ── layer_id stability ───────────────────────────────────────────────


F1_TO_F3 = [
    ("F1_basic_nyse.yaml", "nyse"),
    ("F1_basic_pse.yaml", "pse"),
    ("F2_liquidity_nyse.yaml", "nyse"),
    ("F2_liquidity_pse.yaml", "pse"),
    ("F3_structural_nyse.yaml", "nyse"),
    ("F3_structural_pse.yaml", "pse"),
]


@pytest.mark.parametrize("filename,market", F1_TO_F3)
def test_layer_ids_are_unique_and_match_yaml_names(filename, market):
    """The registry assigns ``layer.layer_id`` from the YAML layer name; the
    ids key checkpoint state, so they must be unique and stable per config.
    """
    friction_cfg, pipeline = _build(filename, market=market)

    yaml_names = [spec["name"] for spec in friction_cfg["layers"]]
    layer_ids = [layer.layer_id for layer in pipeline.layers]

    assert layer_ids == yaml_names
    assert len(set(layer_ids)) == len(layer_ids)

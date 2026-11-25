"""Observable contracts for shared friction-aware environment construction.

The complete reset/step trace oracle was recorded through the original public
cell builder at fac3f54c, before either runner adopted the shared constructor.
Its market inputs are invented, not copied from licensed panels. Hashes bind
observations, chronology, fills, costs and diagnostics at every step exactly.
Only rewards permit two float64 ULPs for NumPy's CPU-specific log dispatch;
their reference values independently reproduce all original hashes at fac3f54c.
"""

import copy
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from meta.data_adapters.local_panel import load_env_config
from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
    build_portfolio_env,
)

FIXTURES = Path(__file__).parent / "fixtures"
PANEL = FIXTURES / "construction_panel.csv"
ORACLE = json.loads((FIXTURES / "construction_oracle.json").read_text())
ROOT = Path(__file__).resolve().parents[2]


def _settings(market, level):
    # The portable-construction checks must not import the research runner or
    # training libraries just to discover the seven existing friction files.
    path = next(
        path
        for path in sorted((ROOT / "configs/friction").glob(f"{level}_*.yaml"))
        if level == "F0" or path.stem.endswith("_" + market)
    )
    friction = yaml.safe_load(path.read_text())["friction"]
    return {
        "market": market,
        "environment": {
            "initial_amount": ORACLE["initial_amount"],
            "reward_scaling": ORACLE["reward_scaling"],
        },
        "data": {"accounting_model": "option1_crsp"},
        "friction": friction,
    }


def _normal(value):
    if isinstance(value, (np.datetime64, date, datetime)):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_normal(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _normal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    return value


def _step_record(env, action):
    observation, reward, terminated, truncated, info = env.step(np.array(action))
    return {
        "observation": observation,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "holdings": env.holdings.copy(),
        "wealth": env.total_asset,
        "info": info,
    }


def _trace(env):
    observation, info = env.reset(seed=ORACLE["seed"])
    trace = [{"observation": observation, "info": info}]
    for action in ORACLE["actions"]:
        trace.append(_step_record(env, action))
    return _normal(trace)


def assert_frozen_trace(trace, expected):
    """Keep original hashes, bounding only independently recorded rewards."""
    np.testing.assert_array_max_ulp(
        np.array([row["reward"] for row in trace[1:]], dtype=np.float64),
        np.array(expected["rewards"], dtype=np.float64),
        maxulp=2,
    )
    canonical = copy.deepcopy(trace)
    for row, reward in zip(canonical[1:], expected["rewards"]):
        row["reward"] = reward
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected["trace_sha256"]


@pytest.mark.parametrize("market", ["nyse", "pse"])
@pytest.mark.parametrize("level", ["F0", "F1", "F2", "F3"])
@pytest.mark.parametrize(
    "caller", ["construction", "primary_evaluation", "calibration"]
)
def test_constructed_environment_preserves_the_frozen_synthetic_trace(
    market, level, caller
):
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest() == ORACLE["panel_sha256"]
    settings = _settings(market, level)
    if caller == "construction":
        env = build_portfolio_env(
            load_env_config(PANEL, require_option1_accounting=True),
            settings,
            seed=ORACLE["seed"],
        )
    elif caller == "primary_evaluation":
        from tools.run_experiment_cell import build_env

        env = build_env(settings, PANEL, seed=ORACLE["seed"])
    else:
        from tools.run_hparam_tuning import build_env

        env = build_env(
            load_env_config(PANEL, require_option1_accounting=True),
            settings,
            reward_scaling=ORACLE["reward_scaling"],
            seed=ORACLE["seed"],
        )
    trace = _trace(env)
    expected = ORACLE["cases"][f"{market}__{level}"]
    assert_frozen_trace(trace, expected)
    assert trace[-1]["terminated"] and not trace[-1]["truncated"]
    assert env.total_asset == expected["final_equity"]
    assert sum(len(row["info"]["trades"]) for row in trace[1:]) == expected["trades"]
    assert (
        sum(row["info"]["dividend_income"] for row in trace[1:])
        == expected["dividend_income"]
    )
    env.close()


def test_calibration_reward_override_does_not_change_accounting_or_settings():
    from tools.run_hparam_tuning import build_env

    settings = _settings("pse", "F3")
    before = copy.deepcopy(settings)
    inputs = load_env_config(PANEL, require_option1_accounting=True)
    configured = build_portfolio_env(inputs, settings, seed=7)
    calibrated = build_env(inputs, settings, seed=7, reward_scaling=7.5)

    for action in ORACLE["actions"]:
        state_a, reward_a, terminated_a, truncated_a, info_a = configured.step(
            np.array(action)
        )
        state_b, reward_b, terminated_b, truncated_b, info_b = calibrated.step(
            np.array(action)
        )
        np.testing.assert_array_equal(state_a, state_b)
        assert reward_b == pytest.approx(3 * reward_a, abs=1e-14)
        assert (terminated_a, truncated_a) == (terminated_b, truncated_b)
        assert _normal(info_a) == _normal(info_b)
    assert settings == before
    configured.close()
    calibrated.close()


def test_construction_and_reset_keep_impact_and_settlement_state_independent():
    settings = _settings("pse", "F3")
    inputs = load_env_config(PANEL, require_option1_accounting=True)
    reference = build_portfolio_env(inputs, settings, seed=7)
    expected = _trace(reference)
    reference.close()
    first = build_portfolio_env(inputs, settings, seed=7)
    second = build_portfolio_env(inputs, settings, seed=7)

    # Two environments trade concurrently. Reset only first while second has
    # live impact/settlement state; second must follow its uninterrupted path.
    for index, action in enumerate(ORACLE["actions"]):
        if index == 3:
            first.reset(seed=7)
        _step_record(first, action)
        assert _normal(_step_record(second, action)) == expected[index + 1]
    assert _trace(first) == expected
    first.close()
    second.close()


def test_constructed_environment_applies_the_nonzero_f0_cost():
    inputs = {
        "date_list": np.array(["2024-05-23", "2024-05-24"], dtype="datetime64[ns]"),
        "tic_list": np.array(["SYNTHETIC"]),
        "price_array": np.full((2, 1), 100.0),
        "tech_array": np.zeros((2, 1)),
        "volume_array": np.full((2, 1), 1_000_000.0),
    }
    settings = {
        "market": "nyse",
        "environment": {"initial_amount": 1000.0},
        "friction": {
            "level": "F0",
            "max_trade_volume_pct": 1.0,
            "impact_model": {"name": "baseline", "basis_points": 10},
            "layers": [],
        },
    }
    env = build_portfolio_env(inputs, settings, seed=7)

    _, _, terminated, truncated, info = env.step(np.array([-20.0, 20.0]))

    # A 1,000 cash account cannot afford ten 100 shares plus the 10 bps
    # charge: nine shares cost 900 + 0.90 and leave 99.10 cash.
    np.testing.assert_array_equal(env.holdings, [9.0])
    assert env.cash == 99.1
    assert env.total_asset == 999.1
    assert info["cost"] == pytest.approx(0.9, abs=1e-12)
    assert terminated and not truncated


def test_frozen_construction_contract_without_avx512():
    """Exercise the hosted-runner CPU dispatch through real public environments."""
    process_env = os.environ.copy()
    process_env["NPY_DISABLE_CPU_FEATURES"] = (
        "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from unit_tests.frictions.test_environment_construction import "
            "test_constructed_environment_preserves_the_frozen_synthetic_trace as check; "
            "[check(market, level, 'construction') "
            "for market in ('nyse', 'pse') for level in ('F0', 'F1', 'F2', 'F3')]",
        ],
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "changed", ["reward", "wealth", "holdings", "cost", "observation"]
)
def test_trace_contract_rejects_changes_outside_reward_roundoff(changed):
    env = build_portfolio_env(
        load_env_config(PANEL, require_option1_accounting=True),
        _settings("pse", "F0"),
        seed=ORACLE["seed"],
    )
    trace = _trace(env)
    env.close()
    expected = ORACLE["cases"]["pse__F0"]
    assert_frozen_trace(trace, expected)

    if changed == "reward":
        # Three ULPs must fail: only the measured one/two-ULP dispatch delta
        # is admissible. No broad decimal rounding or relative tolerance.
        for _ in range(3):
            trace[1]["reward"] = float(np.nextafter(trace[1]["reward"], np.inf))
    elif changed == "wealth":
        trace[1]["wealth"] = float(np.nextafter(trace[1]["wealth"], np.inf))
    elif changed == "holdings":
        trace[1]["holdings"][0] += 1
    elif changed == "cost":
        trace[1]["info"]["cost"] = float(np.nextafter(trace[1]["info"]["cost"], np.inf))
    else:
        trace[1]["observation"][0] = float(
            np.nextafter(trace[1]["observation"][0], np.inf)
        )
    with pytest.raises(AssertionError):
        assert_frozen_trace(trace, expected)

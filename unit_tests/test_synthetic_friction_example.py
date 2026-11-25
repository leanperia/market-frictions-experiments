"""The supported non-RL example must work without private inputs or RL packages."""

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("market", ["nyse", "pse"])
def test_synthetic_friction_example_runs_without_training_dependencies(market):
    # Simulate absent optional third-party packages in a fresh interpreter.
    # Production accounting/environment code remains real and unmocked.
    script = """
import importlib.abc
import runpy
import sys
class UnavailableTrainingPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'stable_baselines3', 'optuna'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, UnavailableTrainingPackages())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tools/synthetic_friction_example.py"),
            "--market",
            market,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["market"] == market
    assert output["campaign_evidence"] is False
    assert output["weight_order"] == ["cash", "SYN-A", "SYN-B"]
    assert math.isfinite(output["final_equity"]) and output["final_equity"] > 0
    steps = output["steps"]
    assert len(steps) == 7
    assert all(row["decision_date"] < row["execution_date"] for row in steps)
    assert sum(len(row["trades"]) for row in steps) > 0
    assert sum(row["costs"]["fees"] for row in steps) > 0
    assert sum(row["costs"]["impact"] for row in steps) > 0
    assert any(row["in_transit_cash"] > 0 for row in steps)
    assert all(row["cash"] >= 0 for row in steps)
    for row in steps:
        for name in ("intended_weights", "realized_weights"):
            assert len(row[name]) == 3
            assert all(math.isfinite(value) and value >= 0 for value in row[name])
            assert sum(row[name]) == pytest.approx(1.0, abs=1e-6)
    assert any(row["intended_weights"] != row["realized_weights"] for row in steps)
    assert sum(row["blocked"]["adv"] for row in steps) > 0
    if market == "pse":
        assert sum(row["blocked"]["lots"] for row in steps) > 0

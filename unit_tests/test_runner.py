"""Contract tests for the experiment runner (synthetic data only, no PyTorch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tools import run_cell as rc
from tools.check_parity import compare_metrics, config_differences
from tools.manifest import Cell, cell_layers, cell_params, expand, find_cell, load_manifest
from tools.run_grid import main as grid_main
from tools.run_grid import select_cells

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_manifest(ROOT / "manifests" / "cifer2027.yaml")

# Metric keys of summary.json["backtest"], as in the result files of the paper.
BACKTEST_KEYS = {
    "n_days", "total_return", "annual_return", "annual_volatility", "sharpe_ratio",
    "sharpe_ratio_rf", "sortino_ratio", "sortino_ratio_rf", "calmar_ratio",
    "max_drawdown", "cvar_5pct", "cvar_1pct", "skewness", "excess_kurtosis",
    "information_ratio", "jensen_alpha", "beta", "treynor_ratio", "final_equity",
    "total_turnover", "total_impact_cost", "total_fees_explicit", "total_spread_cost",
    "blocked_by_adv", "blocked_by_lot", "blocked_by_cash", "blocked_by_outage",
    "cash_shortfall_total", "cash_shortfall_steps",
}


# ── Manifest ─────────────────────────────────────────────────────────


def test_paper_manifest_has_the_full_grid():
    cells = expand(MANIFEST)
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))
    by_group = {}
    for c in cells:
        by_group[c.group] = by_group.get(c.group, 0) + 1
    assert by_group == {
        "non_rl": 2 * 8 * 4,
        "cadence_control": 2 * 4,
        "signal_tuned": 2 * 2,
        "rl": 2 * 4 * 4 * 21,
    }
    assert len(cells) == 748
    # The order is part of the interface (cluster array indices).
    assert cells[0].cell_id == "nyse__ubah__F0__seed1"
    assert cells[-1].cell_id == "pse__td3__F3__seed21"


def test_cell_layers_and_params():
    ppo = find_cell(MANIFEST, "pse__ppo_mlp__F2__seed3")
    assert cell_layers(MANIFEST, ppo) == [
        "configs/base/pse_adjusted_pseedge.yaml",
        "configs/friction/F2_liquidity_pse.yaml",
        "configs/algo/ppo_mlp.yaml",
        "configs/tuned_stage1/ppo_mlp__pse.yaml",
    ]
    tuned = find_cell(MANIFEST, "nyse__zscore_tuned__F3__seed1")
    assert cell_params(MANIFEST, tuned) == {"lookback": 180, "threshold": 1.0}
    assert cell_params(MANIFEST, find_cell(MANIFEST, "nyse__zscore__F3__seed1")) is None
    for layer in {l for c in expand(MANIFEST) for l in cell_layers(MANIFEST, c)}:
        assert (ROOT / layer).exists(), layer


@pytest.mark.parametrize(
    "cell_id, reference_hash",
    [
        # config_hash values recorded in the result files of the paper.
        ("pse__ubah__F3__seed1", "8ffd14c07736d28c472c943c61e8c2f13cc54d6a4028c3f4bbc9445471c5afb1"),
        ("nyse__erc__F0__seed1", "e592e1b11b81f1470bdbf9fe19f361c696052c14951056139d2efa49715f0011"),
        ("pse__csm_lo__F2__seed1", "23c320f2c7840fc178e1c8bbf2f9d8afe784cf9b82003e21113badcd6fb593ce"),
        ("nyse__macd_tuned__F3__seed1", "f88ee232a6d989ebce86e788879501e07363214dfe7400f04bc9ad159aaca424"),
        ("pse__ppo_mlp__F3__seed1", "1e174eafd987739afbd92d6949598706a95a6cbe4eb3ffac99c3903ba45fa814"),
        ("nyse__td3__F1__seed7", "172021cf6791977eadf7542b29bf58dcfe76024c9b833ea7c57a3a0132888dd0"),
        ("nyse__ppo_eiie__F0__seed21", "66ea4be626126f1ccc756da92a9f77a1611c11407ec4b4940995f9af81abbd5f"),
    ],
)
def test_effective_configuration_equals_the_paper(cell_id, reference_hash):
    cfg = rc.build_cell_config(MANIFEST, find_cell(MANIFEST, cell_id))
    assert rc.config_sha256(cfg) == reference_hash


def test_config_hash_ignores_bookkeeping_keys():
    cfg = rc.build_cell_config(MANIFEST, find_cell(MANIFEST, "pse__ubah__F0__seed1"))
    h = rc.config_sha256(cfg)
    cfg["__layers__"] = ["something else"]
    assert rc.config_sha256(cfg) == h
    cfg["runtime"]["seed"] = 2
    assert rc.config_sha256(cfg) != h


def test_deep_merge_later_layers_win():
    merged = rc.deep_merge({"a": {"b": 1, "c": 2}, "d": [1]}, {"a": {"b": 5}, "d": [2]})
    assert merged == {"a": {"b": 5, "c": 2}, "d": [2]}


def test_select_cells_filters():
    cells = expand(MANIFEST)
    picked = select_cells(cells, groups=["rl"], strategies=["sac"], seeds=[1, 2], markets=["pse"])
    assert [c.cell_id for _, c in picked] == [
        f"pse__sac__{f}__seed{s}" for f in ("F0", "F1", "F2", "F3") for s in (1, 2)
    ]
    for index, cell in picked:
        assert cells[index] == cell


def test_grid_list(capsys):
    assert grid_main(["--list", "--group", "signal_tuned"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split("\t")[1] for line in lines] == [
        "nyse__zscore_tuned__F3__seed1",
        "nyse__macd_tuned__F3__seed1",
        "pse__zscore_tuned__F3__seed1",
        "pse__macd_tuned__F3__seed1",
    ]


# ── Data paths ───────────────────────────────────────────────────────


def test_data_paths_resolve_under_the_data_root(tmp_path):
    assert rc.resolve_data_path("data/a/train.csv", tmp_path) == tmp_path / "a" / "train.csv"
    assert rc.resolve_data_path("a/train.csv", tmp_path) == tmp_path / "a" / "train.csv"
    assert rc.resolve_data_path("/abs/x.csv", tmp_path) == Path("/abs/x.csv")
    assert rc.resolve_data_path(None, tmp_path) is None


def test_history_splits():
    cfg = {"data": {"train_csv": "data/p/train.csv", "valid_csv": "data/p/valid.csv",
                    "test_csv": "data/p/test.csv"}}
    root = Path("/r")
    assert rc.history_csvs(cfg, "test", root) == [Path("/r/p/train.csv"), Path("/r/p/valid.csv")]
    assert rc.history_csvs(cfg, "valid", root) == [Path("/r/p/train.csv")]
    assert rc.history_csvs(cfg, "train", root) == []


# ── End to end on synthetic data ─────────────────────────────────────

TICKERS = ["SYN-A", "SYN-B", "SYN-C", "SYN-D"]
FEATURES = ["zopen", "zhigh", "zlow", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25", "zd_30"]


def _write_synthetic_data(root: Path) -> None:
    """Invented prices for three splits, a risk-free series and a benchmark."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=150)
    rows = []
    prices = np.array([20.0, 50.0, 80.0, 120.0])
    for d in dates:
        prices = prices * np.exp(rng.normal(0.0005, 0.015, size=len(TICKERS)))
        for tic, close in zip(TICKERS, prices):
            row = {
                "date": d.strftime("%Y-%m-%d"), "tic": tic,
                "open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close,
                "volume": 2_000_000.0,
                "quote_open": close * 0.999, "quote_high": close * 1.01,
                "quote_low": close * 0.99, "quote_close": close,
                "cum_share_factor": 1.0, "dividend_per_panel_share": 0.0, "is_synthetic_halt": 0,
            }
            row.update({f: 0.0 for f in FEATURES})
            rows.append(row)
    panel = pd.DataFrame(rows)
    (root / "panel").mkdir(parents=True)
    splits = {"train": dates[:90], "valid": dates[90:115], "test": dates[115:]}
    for name, split_dates in splits.items():
        keep = panel["date"].isin(split_dates.strftime("%Y-%m-%d"))
        panel[keep].to_csv(root / "panel" / f"{name}.csv", index=False)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "rate": 2.0}).to_csv(root / "rf.csv", index=False)
    bench = panel.groupby("date")["close"].mean().reset_index()
    bench.to_csv(root / "bench.csv", index=False)


@pytest.fixture()
def synthetic(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_synthetic_data(data_root)
    base = {
        "name": "synthetic_pse",
        "market": "pse",
        "data": {
            "accounting_model": "option1_crsp",
            "train_csv": "data/panel/train.csv",
            "valid_csv": "data/panel/valid.csv",
            "test_csv": "data/panel/test.csv",
            "rf_csv": "data/rf.csv",
            "rf_type": "daily",
            "benchmark_csv": "data/bench.csv",
        },
        "environment": {"initial_amount": 1_000_000, "reward_scaling": 1.0},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    manifest = yaml.safe_load((ROOT / "manifests" / "cifer2027.yaml").read_text())
    manifest["markets"] = {
        "pse": {"base": str(base_path), "friction": manifest["markets"]["pse"]["friction"]}
    }
    for group in manifest["groups"]:
        group["markets"] = ["pse"]
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    return {"data_root": data_root, "out": tmp_path / "out", "manifest_path": manifest_path,
            "manifest": load_manifest(manifest_path)}


@pytest.mark.parametrize("strategy, friction", [("ubah", "F0"), ("ucrp", "F3"), ("zscore_tuned", "F3")])
def test_non_rl_cell_writes_one_result_directory(synthetic, strategy, friction):
    cell = Cell("pse", strategy, friction, 1)
    out = rc.run_cell(cell, data_root=synthetic["data_root"], output_root=synthetic["out"],
                      manifest=synthetic["manifest"])
    assert out["status"] == "ok"
    result = Path(out["result_dir"])
    assert result.name == f"pse__{strategy}__{friction}__seed1"
    summary = json.loads((result / "summary.json").read_text())
    assert set(summary["backtest"]) == BACKTEST_KEYS
    assert summary["algo"] == strategy
    assert summary["friction_level"] == friction
    assert summary["policy_kind"] == "non_rl"
    assert summary["training_performed"] is False
    assert summary["backtest"]["n_days"] == 35
    if strategy == "zscore_tuned":
        # A 252-day look-back has too little history here: the strategy stays in
        # cash, its Sharpe ratio is undefined, and the summary says so.
        assert summary["backtest"]["sharpe_ratio_rf"] is None
        assert {"metric": "backtest.sharpe_ratio_rf", "reason": "non_finite"} in summary[
            "degenerate_metrics"
        ]
    else:
        assert np.isfinite(summary["backtest"]["sharpe_ratio_rf"])
    values = pd.read_csv(result / "backtest" / "portfolio_values.csv")
    assert len(values) == 35
    assert values["date"].iloc[0] == "2023-06-12"
    assert values["portfolio_value"].iloc[0] == pytest.approx(1_000_000)
    assert summary["backtest"]["final_equity"] == pytest.approx(values["portfolio_value"].iloc[-1])
    cfg = yaml.safe_load((result / "merged_config.yaml").read_text())
    assert rc.config_sha256(cfg) == summary["config_sha256"]
    if strategy == "zscore_tuned":
        assert cfg["algo"]["params"] == {"lookback": 252, "threshold": 2.0}


def test_cell_is_deterministic_and_skipped_when_done(synthetic):
    cell = Cell("pse", "markowitz", "F3", 1)
    kwargs = dict(data_root=synthetic["data_root"], manifest=synthetic["manifest"])
    first = rc.run_cell(cell, output_root=synthetic["out"] / "a", **kwargs)
    again = rc.run_cell(cell, output_root=synthetic["out"] / "a", **kwargs)
    assert again["status"] == "skipped"
    other = rc.run_cell(cell, output_root=synthetic["out"] / "b", **kwargs)
    a = json.loads((Path(first["result_dir"]) / "summary.json").read_text())["backtest"]
    b = json.loads((Path(other["result_dir"]) / "summary.json").read_text())["backtest"]
    assert a == b


def test_changed_configuration_is_refused(synthetic):
    cell = Cell("pse", "ubah", "F0", 1)
    kwargs = dict(data_root=synthetic["data_root"], output_root=synthetic["out"])
    rc.run_cell(cell, manifest=synthetic["manifest"], **kwargs)
    changed = json.loads(json.dumps(synthetic["manifest"]))
    changed["strategies"]["ubah"] = {"algo": "configs/algo/ucrp.yaml"}
    with pytest.raises(RuntimeError, match="different configuration"):
        rc.run_cell(cell, manifest=changed, **kwargs)
    assert rc.run_cell(cell, manifest=changed, force=True, **kwargs)["status"] == "ok"


def test_validation_split(synthetic):
    out = rc.run_cell(Cell("pse", "ucrp", "F1", 1), data_root=synthetic["data_root"],
                      output_root=synthetic["out"], manifest=synthetic["manifest"], split="valid")
    summary = json.loads((Path(out["result_dir"]) / "summary.json").read_text())
    assert summary["cell_id"] == "pse__ucrp__F1__seed1__valid"
    assert summary["backtest"]["n_days"] == 25


def test_model_option_is_rl_only(synthetic):
    with pytest.raises(ValueError, match="RL cells only"):
        rc.run_cell(Cell("pse", "ubah", "F0", 1), data_root=synthetic["data_root"],
                    output_root=synthetic["out"], manifest=synthetic["manifest"],
                    model_path=Path("model.zip"))


def test_grid_runs_selected_cells_and_skips_done(synthetic, capsys):
    argv = ["--manifest", str(synthetic["manifest_path"]), "--data-root", str(synthetic["data_root"]),
            "--output-root", str(synthetic["out"]), "--group", "non_rl", "--strategy", "ubah", "ucrp",
            "--friction", "F2", "--workers", "2"]
    assert grid_main(argv) == 0
    assert (synthetic["out"] / "pse__ubah__F2__seed1" / "summary.json").exists()
    assert (synthetic["out"] / "pse__ucrp__F2__seed1" / "summary.json").exists()
    assert (synthetic["out"] / "logs" / "pse__ubah__F2__seed1.log").exists()
    capsys.readouterr()
    assert grid_main(argv) == 0
    assert "2 already done, 0 to run" in capsys.readouterr().out


def test_rl_cell_smoke(synthetic):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    out = rc.run_cell(Cell("pse", "ppo_mlp", "F3", 1), data_root=synthetic["data_root"],
                      output_root=synthetic["out"], manifest=synthetic["manifest"],
                      total_timesteps=128, threads=1)
    result = Path(out["result_dir"])
    summary = json.loads((result / "summary.json").read_text())
    assert summary["training_performed"] is True
    assert summary["total_timesteps"] == 128
    assert (result / "checkpoint" / "model.zip").exists()
    reloaded = rc.run_cell(Cell("pse", "ppo_mlp", "F3", 1), data_root=synthetic["data_root"],
                           output_root=synthetic["out"] / "reload", manifest=synthetic["manifest"],
                           total_timesteps=128, threads=1,
                           model_path=result / "checkpoint" / "model.zip")
    again = json.loads((Path(reloaded["result_dir"]) / "summary.json").read_text())
    assert again["training_performed"] is False
    assert again["backtest"] == summary["backtest"]


# ── Parity helpers ───────────────────────────────────────────────────


def test_compare_metrics_classifies_differences():
    result = compare_metrics({"a": 1.0, "b": 2.0, "c": 3.0, "d": None},
                             {"a": 1.0, "b": 2.0 + 1e-15, "c": 3.1, "d": None}, 1e-12)
    assert result["exact"] == 2
    assert [d["metric"] for d in result["within_tolerance"]] == ["b"]
    assert [d["metric"] for d in result["mismatch"]] == ["c"]


def test_config_differences_lists_paths():
    diffs = config_differences({"a": {"b": 1, "c": 2}}, {"a": {"b": 1, "c": 3, "d": 4}})
    assert diffs == ["a.c: 2 != 3", "a.d: missing in runner"]

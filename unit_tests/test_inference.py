"""Tests of the statistical inference (synthetic data only, no PyTorch).

The reference values in ``test_*_matches_study_implementation`` were computed
with the study's original implementation on the same synthetic inputs. They
pin the random-number scheme: a change in the order of the random draws
changes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tools import inference as inf

ROOT = Path(__file__).resolve().parents[1]


def synthetic_returns():
    rng = np.random.default_rng(2027)
    x = rng.normal(0.0004, 0.01, 120)
    r = rng.normal(0.0002, 0.012, (5, 120)) + 0.3 * x
    return x, r


# ── statistical functions ───────────────────────────────────────────────────


def test_hierarchical_bootstrap_matches_study_implementation():
    x, r = synthetic_returns()
    shared = np.random.default_rng(11).integers(0, 5, size=(300, 5))
    out = inf.hierarchical_paired_bootstrap_delta(
        x, r, 252, n_boot=300, block_length=7, seed=3, shared_seed_indices=shared)
    assert out["non_rl_sharpe"] == 0.22020849297663894
    assert out["rl_median_sharpe"] == 1.657342254075424
    assert out["delta"] == -1.4371337610987849
    assert out["ci_low"] == -5.349776253722465
    assert out["ci_high"] == 2.5410581479197014
    assert out["prob_delta_positive"] == 0.29
    assert out["pvalue_uncorrected"] == 0.4717607973421927
    assert out["n_boot_effective"] == 300
    # Without shared seed draws the seed resample comes from the same generator.
    out = inf.hierarchical_paired_bootstrap_delta(x, r, 252, n_boot=300, block_length=7, seed=3)
    assert out["pvalue_uncorrected"] == 0.44518272425249167
    assert (out["ci_low"], out["ci_high"]) == (-4.914394981692322, 2.0331861862456004)


def test_ledoit_wolf_test_matches_study_implementation():
    x, r = synthetic_returns()
    out = inf.lw_sharpe_diff_test(x, r[0], 252, n_boot=200, block_length=7, seed=5)
    assert out["diff"] == -2.638089544131124
    assert out["pvalue"] == 0.1890547263681592
    assert (out["ci_low"], out["ci_high"]) == (-6.773189563467953, 1.3480753979196045)
    assert not out["significant_0.05"]


def test_median_interval_wilson_and_indices_match_study_implementation():
    out = inf.bootstrap_median_ci(np.array([0.1, -0.3, 0.5, 0.2, 0.9]), n_boot=500, seed=1001)
    assert out == {"median": 0.2, "ci_low": -0.3, "ci_high": 0.9}
    assert inf.wilson_interval(8, 21) == (0.20751011368564493, 0.5912134622488425)
    idx = inf.stationary_bootstrap_indices(8, 3, 2, np.random.default_rng(0))
    assert idx.tolist() == [[6, 6, 1, 0, 1, 2, 3, 4], [5, 6, 7, 0, 3, 4, 0, 1]]


def test_stationary_bootstrap_blocks():
    rng = np.random.default_rng(1)
    idx = inf.stationary_bootstrap_indices(50, 10**9, 20, rng)
    # Practically no restarts: each row is one block that wraps around.
    assert np.all((np.diff(idx, axis=1) % 50) == 1)
    idx = inf.stationary_bootstrap_indices(200, 4, 500, np.random.default_rng(2))
    assert idx.min() >= 0 and idx.max() < 200
    continued = (np.diff(idx, axis=1) % 200) == 1
    assert 0.70 < continued.mean() < 0.80  # restart probability 1/4 (plus chance continuations)


def test_bootstrap_p_value_definition():
    x, r = synthetic_returns()
    n_boot = 199
    out = inf.hierarchical_paired_bootstrap_delta(x, r, 252, n_boot=n_boot, block_length=5, seed=0)
    k = round(out["pvalue_uncorrected"] * (n_boot + 1)) - 1
    assert out["pvalue_uncorrected"] == (1 + k) / (n_boot + 1)
    assert 0 < out["pvalue_uncorrected"] <= 1
    # Same seeds, same result.
    again = inf.hierarchical_paired_bootstrap_delta(x, r, 252, n_boot=n_boot, block_length=5, seed=0)
    assert again == out


def test_degenerate_contrasts():
    x, r = synthetic_returns()
    assert np.isnan(inf.hierarchical_paired_bootstrap_delta(x[:9], r[:, :9], 252)["pvalue_uncorrected"])
    flat = np.zeros_like(x)
    out = inf.hierarchical_paired_bootstrap_delta(flat, r, 252, n_boot=50)
    assert np.isnan(out["pvalue_uncorrected"]) and np.isnan(out["non_rl_sharpe"])
    with pytest.raises(ValueError):
        inf.hierarchical_paired_bootstrap_delta(x, r[:, :-1], 252)
    with pytest.raises(ValueError):
        inf.hierarchical_paired_bootstrap_delta(x, r, 252, n_boot=10, shared_seed_indices=np.zeros((9, 5), int))


def test_holm_bonferroni():
    adjusted, reject = inf.holm_bonferroni([0.01, 0.04, 0.03, 0.005, np.nan])
    assert np.allclose(adjusted[:4], [0.03, 0.06, 0.06, 0.02], rtol=0, atol=1e-15)
    assert np.isnan(adjusted[4])
    assert reject.tolist() == [True, False, False, True, False]
    adjusted, _ = inf.holm_bonferroni([0.9, 0.8])
    assert adjusted.tolist() == [1.0, 1.0]
    assert inf.is_borderline([0.2, 0.05]) and not inf.is_borderline([0.02, 0.08, np.nan])


def test_family_definitions_match_the_paper():
    assert inf.PRIMARY.expected_contrasts_per_market == 32
    assert inf.SIGNAL_TUNED.expected_contrasts_per_market == 40
    assert inf.DIVIDEND.expected_contrasts_per_market == 52
    # The larger families append to the smaller ones, so shared contrasts keep their seeds.
    assert inf.SIGNAL_TUNED.non_rl_algorithms[:8] == inf.PRIMARY.non_rl_algorithms
    assert inf.DIVIDEND.non_rl_algorithms[:10] == inf.SIGNAL_TUNED.non_rl_algorithms
    assert inf.PRIMARY.family_label == "holm_within_market_32"
    order = sorted(inf.PAIRWISE_ALGORITHMS)
    assert len(order) == 13
    assert order.index("csm_lo") * 13 + order.index("ucrp_monthly") == 11


# ── synthetic cell results ──────────────────────────────────────────────────


def write_cell(root: Path, market: str, algo: str, seed: int, values: np.ndarray, dates) -> None:
    d = root / f"{market}__{algo}__F3__seed{seed}"
    (d / "backtest").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": [x.strftime("%Y-%m-%d") for x in dates], "portfolio_value": values}).to_csv(
        d / "backtest" / "portfolio_values.csv", index=False)
    summary = {"seed": seed, "backtest": {"sharpe_ratio_rf": 0.0, "total_return": float(values[-1] / values[0] - 1),
                                          "total_turnover": float(seed)}}
    (d / "summary.json").write_text(json.dumps(summary))


def synthetic_world(tmp_path: Path, n_days: int = 60, drop=()):
    """Data root (rates and test calendars) and cell results for every paper strategy."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    data_root = tmp_path / "data"
    manifest = yaml.safe_load((ROOT / "manifests" / "cifer2027.yaml").read_text())
    for market in inf.MARKETS:
        cfg = yaml.safe_load((ROOT / manifest["markets"][market]["base"]).read_text())
        test_csv = inf.resolve_data_path(cfg["data"]["test_csv"], data_root)
        test_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": np.repeat([x.strftime("%Y-%m-%d") for x in dates], 2), "tic": ["A", "B"] * n_days}
                     ).to_csv(test_csv, index=False)
        rf_csv = inf.resolve_data_path(cfg["data"]["rf_csv"], data_root)
        rf_csv.parent.mkdir(parents=True, exist_ok=True)
        rf_dates = pd.bdate_range("2020-12-01", periods=n_days + 40)
        pd.DataFrame({"date": rf_dates.strftime("%Y-%m-%d"), "rate": 1.0 + 0.01 * np.arange(len(rf_dates))}
                     ).to_csv(rf_csv, index=False)
    cells = tmp_path / "cells"
    algos = inf.required_algorithms(["primary", "signal_tuned", "dividend"])
    for market in inf.MARKETS:
        common = rng.normal(0.0003, 0.01, n_days - 1)
        for algo in algos:
            seeds = inf.EVALUATION_SEEDS if algo in inf.RL_ALGORITHMS else (1,)
            for seed in seeds:
                if (market, algo, seed) in drop:
                    continue
                rets = 0.5 * common + rng.normal(0.0002, 0.01, n_days - 1)
                values = 100000.0 * np.concatenate([[1.0], np.cumprod(1 + rets)])
                write_cell(cells, market, algo, seed, values, dates)
    return data_root, cells, dates


def test_build_family_seeds_holm_and_missing_cells(tmp_path):
    data_root, cells, dates = synthetic_world(tmp_path, drop={("pse", "macd", 1)})
    manifest = yaml.safe_load((ROOT / "manifests" / "cifer2027.yaml").read_text())
    rf, cal = inf.load_market_inputs(manifest, data_root, ROOT)
    assert len(cal["nyse"]) == len(dates) - 1
    inputs = inf.collect_inputs(cells, inf.required_algorithms(["primary"]))
    assert len(inputs) == 2 * (9 + 4 * 21) - 1
    table, seeds = inf.build_family(inputs, rf, cal, inf.PRIMARY, n_bootstrap=40, block_length=5,
                                    log=lambda _: None)
    assert len(table) == 64 and len(seeds) == 8
    row = table[(table.market == "pse") & (table.non_rl_algo == "zscore") & (table.rl_algo == "sac")].iloc[0]
    assert row.seed_resample_rng_seed == 502000 and row.date_resample_rng_seed == 10205
    row = table[(table.market == "nyse") & (table.non_rl_algo == "csm_lo") & (table.rl_algo == "td3")].iloc[0]
    assert row.seed_resample_rng_seed == 403000 and row.date_resample_rng_seed == 307
    # The missing PSE MACD cell: its 4 contrasts enter at p = 1; the family keeps 32 members.
    missing = table[(table.market == "pse") & (table.non_rl_algo == "macd")]
    assert missing.degenerate_contrast.all() and (missing.pvalue_primary_uncorrected == 1.0).all()
    for market in inf.MARKETS:
        fam = table[table.market == market]
        assert (fam.family_size == 32).all()
        adjusted, _ = inf.holm_bonferroni(fam.pvalue_primary_uncorrected.to_numpy())
        assert np.array_equal(adjusted, fam.p_holm.to_numpy())
    assert not table[table.market == "nyse"].degenerate_contrast.any()
    assert seeds.rl_seed_sharpe_by_seed.map(lambda s: len(json.loads(s))).tolist() == [21] * 8
    # A contrast does not depend on the family it is computed in.
    big, _ = inf.build_family(inputs.assign(), rf, cal, inf.SIGNAL_TUNED, n_bootstrap=40, block_length=5,
                              log=lambda _: None)
    cols = ["market", "non_rl_algo", "rl_algo", "pvalue_primary_uncorrected", "delta_ci_low_95"]
    shared = big[big.non_rl_algo.isin(inf.PRIMARY.non_rl_algorithms)][cols].reset_index(drop=True)
    assert shared.equals(table[cols].reset_index(drop=True))


def test_parallel_workers_give_identical_results(tmp_path):
    data_root, cells, _ = synthetic_world(tmp_path)
    manifest = yaml.safe_load((ROOT / "manifests" / "cifer2027.yaml").read_text())
    rf, cal = inf.load_market_inputs(manifest, data_root, ROOT)
    inputs = inf.collect_inputs(cells, inf.required_algorithms(["primary"]))
    one, _ = inf.build_family(inputs, rf, cal, inf.PRIMARY, n_bootstrap=30, block_length=5,
                              cache=inf.ContrastCache(1), log=lambda _: None)
    two, _ = inf.build_family(inputs, rf, cal, inf.PRIMARY, n_bootstrap=30, block_length=5,
                              cache=inf.ContrastCache(2), log=lambda _: None)
    pd.testing.assert_frame_equal(one, two)


# Columns of the study's result files (the public output contract).
HEADLINE_COLUMNS = [
    "market", "friction", "non_rl_algo", "rl_algo", "n_rl_seeds", "non_rl_sharpe_rf",
    "rl_median_sharpe_rf", "delta_non_rl_minus_typical_rl", "delta_ci_low_95", "delta_ci_high_95",
    "prob_delta_positive", "pvalue_primary_uncorrected", "degenerate_contrast", "degenerate_reason",
    "n_boot_effective", "seed_resample_rng_seed", "date_resample_rng_seed", "rl_seeds_exceeding_non_rl",
    "rl_seed_exceeding_proportion", "rl_seed_exceeding_ci_low_95", "rl_seed_exceeding_ci_high_95",
    "common_date_start", "common_date_end", "common_date_count", "date_support_complete",
    "expected_date_start", "expected_date_end", "expected_date_count", "p_holm",
    "significant_family_adjusted", "correction_family", "family_size", "block_length", "n_bootstrap",
]
SEED_COLUMNS = [
    "market", "friction", "rl_algo", "n_seeds", "n_seeds_declared", "rl_seed_sharpe_by_seed",
    "rl_median_sharpe_rf_rl_window", "rl_median_ci_low_95", "rl_median_ci_high_95", "rl_iqm_sharpe_rf",
    "rl_mean_sharpe_rf", "total_return_median_sharpe_seed", "turnover_median_sharpe_seed",
    "median_ci_rng_seed", "degenerate_cell", "degenerate_reason", "common_date_start", "common_date_end",
    "common_date_count", "date_support_complete", "expected_date_start", "expected_date_end",
    "expected_date_count",
]
CADENCE_COLUMNS = [
    "analysis_scope", "market", "friction", "strategy", "cadence_control",
    "sharpe_diff_strategy_minus_control", "ci_low_95", "ci_high_95", "pvalue_uncorrected",
    "degenerate_comparison", "degenerate_reason", "p_holm", "significant_family_adjusted",
    "correction_family", "family_size",
]


def test_run_inference_end_to_end(tmp_path):
    data_root, cells, _ = synthetic_world(tmp_path)
    out = inf.run_inference(cells, data_root, tmp_path / "out", families=["signal_tuned", "dividend"],
                            n_bootstrap=20, workers=1, log=lambda _: None)
    assert set(out) == {"primary", "signal_tuned", "dividend"}
    primary = pd.read_csv(out["primary"]["headline"])
    assert list(primary.columns) == HEADLINE_COLUMNS + ["block_length_sensitive"]
    assert len(primary) == 64 and not primary.degenerate_contrast.any()
    assert list(pd.read_csv(out["primary"]["seed_statistics"]).columns) == SEED_COLUMNS
    sens = pd.read_csv(out["primary"]["sensitivity"])
    assert list(sens.columns) == ["market", "non_rl_algo", "rl_algo", "block_length", "p_holm",
                                  "significant_family_adjusted", "flips_vs_primary"]
    assert sorted(sens.block_length.unique()) == [10, 42] and len(sens) == 128
    cadence = pd.read_csv(out["primary"]["cadence"])
    assert list(cadence.columns) == CADENCE_COLUMNS and len(cadence) == 2
    assert (cadence.p_holm == cadence.pvalue_uncorrected).all()  # family of one per market

    tuned = pd.read_csv(out["signal_tuned"]["headline"])
    assert list(tuned.columns) == HEADLINE_COLUMNS + ["inference_role", "original_p_holm_32",
                                                      "block_length_sensitive"]
    assert len(tuned) == 80 and (tuned.family_size == 40).all()
    assert tuned.original_p_holm_32.isna().sum() == 2 * 2 * 4  # the tuned rules were not in the 32
    dividend = pd.read_csv(out["dividend"]["headline"])
    assert list(dividend.columns) == HEADLINE_COLUMNS + ["inference_role"]
    assert len(dividend) == 104 and (dividend.family_size == 52).all()
    dsens = pd.read_csv(out["dividend"]["sensitivity"])
    assert list(dsens.columns) == HEADLINE_COLUMNS + ["inference_role", "significant_family_adjusted_primary",
                                                      "flips_vs_primary"]
    # Raw p-values of shared contrasts are identical across the three families.
    keys = ["market", "non_rl_algo", "rl_algo"]
    merged = primary.merge(dividend, on=keys, suffixes=("", "_52"))
    assert len(merged) == 64
    assert np.allclose(merged.pvalue_primary_uncorrected, merged.pvalue_primary_uncorrected_52, rtol=0, atol=5e-9)


def test_post_hoc_family_refuses_missing_cells(tmp_path):
    data_root, cells, _ = synthetic_world(tmp_path, drop={("nyse", "zscore_tuned", 1)})
    with pytest.raises(ValueError, match="incomplete or degenerate"):
        inf.run_inference(cells, data_root, tmp_path / "out", families=["signal_tuned"],
                          n_bootstrap=10, log=lambda _: None)


def test_parity_comparison_marks_incomplete_rows(tmp_path):
    from tools import check_inference_parity as cip

    rows = [{"market": m, "non_rl_algo": a, "rl_algo": "sac", "pvalue_primary_uncorrected": "0.5",
             "p_holm": "1.0"} for m in ("nyse", "pse") for a in ("ubah", "macd")]
    ref = tmp_path / "ref.csv"
    pd.DataFrame(rows).to_csv(ref, index=False)
    ours = tmp_path / "ours.csv"
    rows[0]["pvalue_primary_uncorrected"] = "0.4"   # a real difference (U.S., complete inputs)
    rows[3]["pvalue_primary_uncorrected"] = "0.9"   # PSE MACD: input missing, not checkable
    pd.DataFrame(rows).to_csv(ours, index=False)
    missing = {"pse__sac__F3__seed4"}
    out = cip.compare_file("primary", ours, ref, missing)
    assert not out["bytes_identical"]
    assert (out["rows_identical"], out["rows_different"], out["rows_unchecked"]) == (1, 1, 2)
    assert out["p_values"] == {"identical": 3, "different": 1, "unchecked": 4}
    assert out["differences"][0]["column"] == "pvalue_primary_uncorrected"
    cells = cip.needed_cells(["primary"])
    assert len(cells) == 2 * (9 + 4 * 21)

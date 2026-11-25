#!/usr/bin/env python3
"""Check that the runner reproduces the frozen results of the paper.

This is not a unit test: it needs the licensed data and the result directories
of the original study, which are not in this repository.

For every deterministic (non-RL) cell of the manifest (groups ``non_rl``,
``cadence_control`` and ``signal_tuned``) the script

1. runs the cell with ``tools/run_grid.py`` (cells with a result are skipped),
2. finds the reference result directory of the same cell,
3. compares the configuration (hash of the effective configuration and, where
   the reference has it, every key of ``merged_config.yaml``),
4. compares every metric in ``summary.json["backtest"]`` (exact, or
   ``|diff| <= --tolerance``), and the daily portfolio values where the
   reference has them.

With ``--rl-checkpoint-seeds`` it also evaluates the saved reference RL models
(no training) for those seeds and compares their test metrics. This needs the
RL requirements.

    python tools/check_parity.py --data-root DATA --reference-root REFS \\
        --output-root runs/parity --workers 4

Exit status 0 means: no mismatch above the tolerance and no missing reference.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.manifest import DEFAULT_MANIFEST, Cell, expand, load_manifest  # noqa: E402
from tools.run_grid import cell_environment, run_one, select_cells  # noqa: E402

DETERMINISTIC_GROUPS = ("non_rl", "cadence_control", "signal_tuned")

# Reference result directories of the paper, relative to --reference-root.
REFERENCE_DIRS: Dict[str, str] = {
    **{
        name: "stage1_deterministic_grid/2026-08-cifer-eval"
        for name in ("ubah", "ucrp", "markowitz", "hrp", "erc", "zscore", "macd")
    },
    "csm_lo": "stage1_deterministic_csm_lo/2026-08-td3-csm-lo",
    "ucrp_monthly": "stage1_deterministic_ucrp_monthly/2026-08-td3-csm-lo",
    "zscore_tuned": "signal_rule_selected_test/2026-09-12-cifer-signal-tuning/test",
    "macd_tuned": "signal_rule_selected_test/2026-09-12-cifer-signal-tuning/test",
    "ppo_mlp": "stage2_rl_evaluation/2026-08-cifer-eval",
    "ppo_eiie": "stage2_rl_evaluation/2026-08-cifer-eval",
    "sac": "stage2_rl_evaluation/2026-08-cifer-eval",
    "td3": "stage2_rl_evaluation/2026-08-td3-csm-lo",
}


def reference_dir(reference_root: Path, cell: Cell) -> Path:
    base = reference_root / REFERENCE_DIRS[cell.strategy]
    if cell.strategy.endswith("_tuned"):
        rule = cell.strategy[: -len("_tuned")]
        pattern = f"{cell.market}__{rule}__{cell.friction}__validtune-test__*"
    else:
        pattern = f"{cell.cell_id}__*"
    matches = sorted(p for p in base.glob(pattern) if (p / "summary.json").exists())
    if len(matches) != 1:
        raise LookupError(f"{cell.cell_id}: {len(matches)} reference directories match {base / pattern}")
    return matches[0]


def _strip(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if not str(k).startswith("__")}


def config_differences(ours: Any, ref: Any, path: str = "") -> List[str]:
    if isinstance(ours, dict) and isinstance(ref, dict):
        out: List[str] = []
        for key in sorted(set(ours) | set(ref), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in ours:
                out.append(f"{sub}: missing in runner")
            elif key not in ref:
                out.append(f"{sub}: missing in reference")
            else:
                out.extend(config_differences(ours[key], ref[key], sub))
        return out
    return [] if ours == ref else [f"{path}: {ours!r} != {ref!r}"]


def compare_metrics(ours: Dict[str, Any], ref: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    exact, within, mismatch = [], [], []
    max_abs = 0.0
    for key in sorted(set(ours) | set(ref)):
        a, b = ours.get(key, "<missing>"), ref.get(key, "<missing>")
        if a == b:
            exact.append(key)
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diff = abs(float(a) - float(b))
            if math.isfinite(diff):
                max_abs = max(max_abs, diff)
            if diff <= tolerance:
                within.append({"metric": key, "abs_diff": diff})
                continue
            mismatch.append({"metric": key, "runner": a, "reference": b, "abs_diff": diff})
        else:
            mismatch.append({"metric": key, "runner": a, "reference": b})
    return {
        "n_metrics": len(exact) + len(within) + len(mismatch),
        "exact": len(exact),
        "within_tolerance": within,
        "mismatch": mismatch,
        "max_abs_diff": max_abs,
    }


def compare_paths(ours: Path, ref: Path) -> Optional[Dict[str, Any]]:
    """Compare daily portfolio values when the reference kept them."""
    ref_csv = ref / "backtest" / "portfolio_values.csv"
    our_csv = ours / "backtest" / "portfolio_values.csv"
    if not ref_csv.exists() or not our_csv.exists():
        return None
    a = pd.read_csv(our_csv, float_precision="round_trip")
    b = pd.read_csv(ref_csv, float_precision="round_trip")
    if len(a) != len(b) or list(a["date"]) != list(b["date"]):
        return {"identical": False, "reason": "different dates"}
    differing = []
    for col in sorted(set(a.columns) & set(b.columns)):
        if col == "date":
            continue
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x = a[col].to_numpy(dtype=float)
            y = b[col].to_numpy(dtype=float)
            same = (x == y) | (np.isnan(x) & np.isnan(y))  # empty columns read as NaN
            if not same.all():
                diff = float(np.nanmax(np.abs(x[~same] - y[~same])))
                differing.append({"column": col, "max_abs_diff": diff})
        elif not a[col].fillna("").astype(str).equals(b[col].fillna("").astype(str)):
            differing.append({"column": col})
    return {
        "identical": not differing,
        "rows": len(a),
        "columns_compared": len(set(a.columns) & set(b.columns)) - 1,
        "differing_columns": differing,
    }


def compare_cell(cell: Cell, ours: Path, ref: Path, tolerance: float) -> Dict[str, Any]:
    our_summary = json.loads((ours / "summary.json").read_text())
    ref_summary = json.loads((ref / "summary.json").read_text())
    result: Dict[str, Any] = {
        "cell_id": cell.cell_id,
        "group": cell.group,
        "reference": str(ref),
        "config_hash_equal": our_summary["config_sha256"] == ref_summary.get("config_hash"),
        "metrics": compare_metrics(our_summary["backtest"], ref_summary["backtest"], tolerance),
    }
    ref_cfg_path = ref / "merged_config.yaml"
    if ref_cfg_path.exists():
        our_cfg = _strip(yaml.safe_load((ours / "merged_config.yaml").read_text()))
        ref_cfg = _strip(yaml.safe_load(ref_cfg_path.read_text()))
        result["config_differences"] = config_differences(our_cfg, ref_cfg)
    path = compare_paths(ours, ref)
    if path is not None:
        result["daily_values"] = path
    return result


def check_rl_configurations(args: argparse.Namespace, cells: List[tuple[int, Cell]]) -> Dict[str, Any]:
    """Compare the effective configuration of every RL cell with its reference.

    This does not train anything: it checks the layer merge (base, friction,
    algorithm, tuned values, seed) against the reference ``merged_config.yaml``
    and the reference configuration hash.
    """
    from tools.run_cell import build_cell_config, config_sha256

    manifest = load_manifest(args.manifest)
    checked, problems = 0, []
    for _, cell in cells:
        try:
            ref = reference_dir(args.reference_root, cell)
        except LookupError as exc:
            problems.append({"cell_id": cell.cell_id, "problem": str(exc)})
            continue
        cfg = build_cell_config(manifest, cell)
        ref_summary = json.loads((ref / "summary.json").read_text())
        ref_cfg = _strip(yaml.safe_load((ref / "merged_config.yaml").read_text()))
        diffs = config_differences(_strip(cfg), ref_cfg)
        checked += 1
        if config_sha256(cfg) != ref_summary.get("config_hash") or diffs:
            problems.append({"cell_id": cell.cell_id, "config_differences": diffs[:5]})
    return {"cells": len(cells), "checked": checked, "equal": checked - len(
        [p for p in problems if "config_differences" in p]), "problems": problems}


def run_grid_subprocess(args: argparse.Namespace, groups: Sequence[str]) -> int:
    command = [
        sys.executable, str(PROJECT_ROOT / "tools" / "run_grid.py"),
        "--manifest", str(args.manifest),
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root / "cells"),
        "--group", *groups,
        "--workers", str(args.workers),
        "--threads-per-cell", "1",
    ]
    return subprocess.call(command)


def evaluate_reference_models(args: argparse.Namespace, cells: List[tuple[int, Cell]]) -> List[Dict[str, Any]]:
    """Evaluate the saved reference RL models with the runner (no training)."""
    out_root = args.output_root / "rl_reference_models"
    grid_args = argparse.Namespace(
        output_root=out_root, manifest=args.manifest, data_root=args.data_root,
        split="test", device="cpu", threads_per_cell=args.rl_threads,
        total_timesteps=None, force=False,
    )
    env = cell_environment(args.rl_threads)
    jobs = []
    for index, cell in cells:
        ref = reference_dir(args.reference_root, cell)
        model = ref / "checkpoint" / "model.zip"
        if not model.exists():
            print(f"[parity] no saved model for {cell.cell_id}; skipped")
            continue
        jobs.append((index, cell, ref, model))

    def work(job):
        index, cell, ref, model = job
        if not (out_root / cell.cell_id / "summary.json").exists():
            result = run_one(index, cell, args=grid_args, extra_args=["--model", str(model)], env=env)
            if result["returncode"] != 0:
                return {"cell_id": cell.cell_id, "error": f"run failed, see {result['log']}"}
        return compare_cell(cell, out_root / cell.cell_id, ref, args.tolerance)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        return list(pool.map(work, jobs))


def cell_status(result: Dict[str, Any]) -> str:
    if "error" in result:
        return "error"
    m = result["metrics"]
    bad_config = not result["config_hash_equal"] or result.get("config_differences")
    if m["mismatch"] or bad_config:
        return "mismatch"
    if m["within_tolerance"]:
        return "within_tolerance"
    return "exact"


def print_table(title: str, results: List[Dict[str, Any]]) -> None:
    print(f"\n## {title}\n")
    print("| Group | Cells | Exact | Within tolerance | Mismatch | Config equal | Daily values identical |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    groups = sorted({r.get("group", "") for r in results})
    for group in groups + ["all"]:
        rows = [r for r in results if group == "all" or r.get("group") == group]
        status = [cell_status(r) for r in rows]
        config_ok = sum(1 for r in rows if r.get("config_hash_equal") and not r.get("config_differences"))
        paths = [r["daily_values"]["identical"] for r in rows if "daily_values" in r]
        print(
            f"| {group} | {len(rows)} | {status.count('exact')} | {status.count('within_tolerance')} | "
            f"{status.count('mismatch') + status.count('error')} | {config_ok}/{len(rows)} | "
            f"{(str(sum(paths)) + '/' + str(len(paths))) if paths else 'n/a'} |"
        )
    for r in results:
        s = cell_status(r)
        if s == "exact":
            continue
        if s == "error":
            print(f"- {r['cell_id']}: {r['error']}")
            continue
        m = r["metrics"]
        detail = ", ".join(
            f"{d['metric']} {d.get('abs_diff', float('nan')):.3g}"
            for d in (m["mismatch"] + m["within_tolerance"])
        )
        extra = ""
        if not r["config_hash_equal"]:
            extra += " [config hash differs]"
        if r.get("config_differences"):
            extra += f" [config: {'; '.join(r['config_differences'][:3])}]"
        print(f"- {r['cell_id']} ({s}): {detail}{extra}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="Directory that holds the reference result directories.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "parity")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--compare-only", action="store_true", help="Do not run missing cells.")
    parser.add_argument("--rl-checkpoint-seeds", type=int, nargs="*", default=[],
                        help="Also evaluate the saved reference RL models of these seeds.")
    parser.add_argument("--rl-threads", type=int, default=2)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    cells = expand(manifest)
    deterministic = select_cells(cells, groups=DETERMINISTIC_GROUPS)

    if not args.compare_only:
        code = run_grid_subprocess(args, DETERMINISTIC_GROUPS)
        if code != 0:
            print("[parity] some cells failed; they are reported as errors")

    results = []
    for _, cell in deterministic:
        ours = args.output_root / "cells" / cell.cell_id
        try:
            ref = reference_dir(args.reference_root, cell)
        except LookupError as exc:
            results.append({"cell_id": cell.cell_id, "group": cell.group, "error": str(exc)})
            continue
        if not (ours / "summary.json").exists():
            results.append({"cell_id": cell.cell_id, "group": cell.group, "error": "no result"})
            continue
        results.append(compare_cell(cell, ours, ref, args.tolerance))
    print_table("Deterministic cells", results)

    rl_config = check_rl_configurations(args, select_cells(cells, groups=["rl"]))
    print("\n## RL cells: effective configuration\n")
    print(f"{rl_config['equal']}/{rl_config['cells']} RL cells have the reference configuration "
          "(hash and every key of merged_config.yaml).")
    for problem in rl_config["problems"][:20]:
        print(f"- {problem}")

    rl_results: List[Dict[str, Any]] = []
    if args.rl_checkpoint_seeds:
        rl_cells = select_cells(cells, groups=["rl"], seeds=args.rl_checkpoint_seeds)
        rl_results = evaluate_reference_models(args, rl_cells)
        for r in rl_results:
            r.setdefault("group", "rl_reference_models")
        print_table("RL: saved reference models evaluated by the runner", rl_results)

    report = {
        "tolerance": args.tolerance,
        "deterministic": results,
        "rl_configurations": rl_config,
        "rl_reference_models": rl_results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[parity] report: {report_path}")
    failed = [r for r in results + rl_results if cell_status(r) in ("mismatch", "error")]
    return 1 if failed or rl_config["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())

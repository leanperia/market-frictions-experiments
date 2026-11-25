#!/usr/bin/env python3
"""Check that tools/inference.py reproduces the inference result files of the paper.

This is not a unit test: it needs the licensed data and the result
directories of the original study, which are not in this repository.

1. Inputs. For every F3 cell that the selected families need, the script takes
   the runner's result from ``--cell-root`` if it exists, else the study's
   result directory (with its daily ``backtest/portfolio_values.csv``) from
   ``--reference-daily-root``. A runner result is compared with the study's
   ``summary.json`` (every metric, exact) and, where the study's daily file
   is available, byte for byte with it. The dividend-inclusive cells
   (``*_tr``) always come from the study's follow-up directory, because the
   runner does not produce them yet. The selected inputs are linked under
   ``<output-root>/cells``.
2. Inference. ``tools.inference.run_inference`` on those inputs, output under
   ``<output-root>/inference``.
3. Comparison with the study's CSV files, cell by cell as text (the same
   number formats are written, so equal text means equal values to the
   printed precision; the post hoc files print every float in full, so equal
   text there means bit-identical doubles). Rows whose inputs were not all
   available are reported as not checkable: a contrast needs its non-RL cell
   and all 21 RL seeds; a Holm-adjusted value needs the complete market family.

    python tools/check_inference_parity.py --data-root DATA \\
        --reference-root REFERENCE_RUNS --reference-daily-root REFERENCE_RUNS \\
        --cell-root runs/cells --output-root runs/inference-parity --workers 4

``--reference-root`` holds the study's result directories (``summary.json``
files and the inference CSVs). ``--reference-daily-root`` holds result
directories with the daily files, in the same layout; it may be the same
directory. Without it, only cells in ``--cell-root`` are used.

Exit status 0: every checkable value is identical, no runner input differs from
the study, and no selected family is incomplete.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from tools.check_parity import reference_dir  # noqa: E402
from tools.inference import (  # noqa: E402
    CADENCE_CONTROLS,
    FAMILIES,
    HEADLINE_FRICTION,
    MARKETS,
    required_algorithms,
    run_inference,
)
from tools.manifest import DEFAULT_MANIFEST, Cell, load_manifest  # noqa: E402

# The study's inference result directories, relative to --reference-root.
REFERENCE_OUTPUTS = {
    "primary": "stage3_statistical_inference/2026-08-td3-csm-lo",
    "signal_tuned": "stage3_signal_tuned_inference/2026-09-12-cifer-signal-tuning",
    "dividend": "cifer_followups/2026-09-12-cifer-followups/item3_inference",
}
DIVIDEND_VARIANTS = "cifer_followups/2026-09-12-cifer-followups/item3_variants"
POST_HOC_FILES = [
    "contrasts_block10_B10000.csv", "contrasts_block21_B10000.csv", "contrasts_block42_B10000.csv",
    "seed_statistics_block10_B10000.csv", "seed_statistics_block21_B10000.csv",
    "seed_statistics_block42_B10000.csv", "headline_non_rl_vs_typical_rl.csv",
    "headline_block_length_sensitivity.csv",
]
OUTPUT_FILES = {
    "primary": ["headline_non_rl_vs_typical_rl.csv", "headline_rl_seed_statistics.csv",
                "headline_block_length_sensitivity.csv", "secondary_cadence_control.csv"],
    "signal_tuned": POST_HOC_FILES,
    "dividend": POST_HOC_FILES,
}
P_COLUMNS = ("pvalue_primary_uncorrected", "p_holm", "pvalue_uncorrected")


# ── inputs ──────────────────────────────────────────────────────────────────


def needed_cells(families: Sequence[str]) -> List[Cell]:
    cells = []
    rl = set(FAMILIES["primary"].rl_algorithms)
    for market in MARKETS:
        for algo in required_algorithms(families):
            seeds = FAMILIES["primary"].evaluation_seeds if algo in rl else (1,)
            for seed in seeds:
                cells.append(Cell(market, algo, HEADLINE_FRICTION[market], seed))
    return cells


def study_dir(root: Path, cell: Cell) -> Optional[Path]:
    """The study's result directory of a cell under ``root`` (None if absent)."""
    if cell.strategy.endswith("_tr"):
        path = root / DIVIDEND_VARIANTS / f"{cell.market}__{cell.strategy}"
        return path if (path / "summary.json").exists() else None
    try:
        return reference_dir(root, cell)
    except (LookupError, KeyError):
        return None


def assemble_inputs(args: argparse.Namespace, cells: List[Cell]) -> Dict[str, Any]:
    """Choose the source of every cell and link it under <output-root>/cells."""
    target = args.output_root / "cells"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for cell in cells:
        record: Dict[str, Any] = {"cell_id": cell.cell_id}
        ref_summary_dir = study_dir(args.reference_root, cell)
        ref_daily_dir = None
        for root in (args.reference_daily_root, args.reference_root):
            if root is None:
                continue
            d = study_dir(root, cell)
            if d is not None and (d / "backtest" / "portfolio_values.csv").exists():
                ref_daily_dir = d
                break
        ours = args.cell_root / cell.cell_id if args.cell_root else None
        source = None
        if cell.strategy.endswith("_tr"):
            source, record["source"] = ref_daily_dir, "study (dividend variant)"
        elif ours is not None and (ours / "summary.json").exists():
            source, record["source"] = ours, "runner"
            if ref_summary_dir is not None:
                a = json.loads((ours / "summary.json").read_text())["backtest"]
                b = json.loads((ref_summary_dir / "summary.json").read_text())["backtest"]
                record["summary_identical"] = a == b
            if ref_daily_dir is not None:
                record["daily_identical"] = filecmp.cmp(
                    ours / "backtest" / "portfolio_values.csv",
                    ref_daily_dir / "backtest" / "portfolio_values.csv", shallow=False)
        elif ref_daily_dir is not None:
            source, record["source"] = ref_daily_dir, "study daily file"
        else:
            record["source"] = "missing"
        link = target / cell.cell_id
        if link.is_symlink() or link.exists():
            link.unlink()
        if source is not None:
            link.symlink_to(source.resolve(), target_is_directory=True)
            record["path"] = str(source)
        records.append(record)
    missing = {r["cell_id"] for r in records if r["source"] == "missing"}
    return {
        "cells": records,
        "missing": sorted(missing),
        "by_source": {s: sum(1 for r in records if r["source"] == s) for s in sorted({r["source"] for r in records})},
        "runner_summary_differs": [r["cell_id"] for r in records if r.get("summary_identical") is False],
        "runner_daily_differs": [r["cell_id"] for r in records if r.get("daily_identical") is False],
        "runner_daily_identical": sum(1 for r in records if r.get("daily_identical") is True),
        "runner_summary_identical": sum(1 for r in records if r.get("summary_identical") is True),
    }


# ── which rows can be checked ───────────────────────────────────────────────


def contrast_complete(market: str, non_rl: str, rl: str, missing: Set[str]) -> bool:
    friction = HEADLINE_FRICTION[market]
    if f"{market}__{non_rl}__{friction}__seed1" in missing:
        return False
    return all(f"{market}__{rl}__{friction}__seed{s}" not in missing
               for s in FAMILIES["primary"].evaluation_seeds)


def rl_complete(market: str, rl: str, missing: Set[str]) -> bool:
    friction = HEADLINE_FRICTION[market]
    return all(f"{market}__{rl}__{friction}__seed{s}" not in missing
               for s in FAMILIES["primary"].evaluation_seeds)


def row_checkable(family: str, row: pd.Series, column: str, missing: Set[str]) -> bool:
    market = row.get("market")
    if "non_rl_algo" in row.index:
        if column in ("p_holm", "significant_family_adjusted", "flips_vs_primary",
                      "block_length_sensitive", "significant_family_adjusted_primary",
                      "original_p_holm_32"):
            design = FAMILIES[family]
            return all(contrast_complete(market, a, b, missing)
                       for a in design.non_rl_algorithms for b in design.rl_algorithms)
        return contrast_complete(market, row["non_rl_algo"], row["rl_algo"], missing)
    if "rl_algo" in row.index:  # seed statistics
        return rl_complete(market, row["rl_algo"], missing)
    if "cadence_control" in row.index:
        friction = HEADLINE_FRICTION[market]
        return all(f"{market}__{a}__{friction}__seed1" not in missing for a in CADENCE_CONTROLS[0])
    return True


# ── comparison ──────────────────────────────────────────────────────────────


def compare_file(family: str, ours: Path, ref: Path, missing: Set[str]) -> Dict[str, Any]:
    if not ours.exists():
        return {"file": ours.name, "error": "not produced"}
    if not ref.exists():
        return {"file": ours.name, "error": f"no reference file {ref}"}
    a = pd.read_csv(ours, dtype=str, keep_default_na=False)
    b = pd.read_csv(ref, dtype=str, keep_default_na=False)
    result: Dict[str, Any] = {"file": ours.name, "rows": len(b), "columns": len(b.columns),
                              "bytes_identical": filecmp.cmp(ours, ref, shallow=False)}
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        result["error"] = (f"shape or columns differ: {len(a)}x{list(a.columns)} vs "
                           f"{len(b)}x{list(b.columns)}")
        return result
    cells_identical = cells_different = cells_unchecked = 0
    rows_identical = rows_different = rows_unchecked = 0
    p = {"identical": 0, "different": 0, "unchecked": 0}
    differences = []
    for i in range(len(b)):
        row_b = b.iloc[i]
        row_state = "identical"
        for col in b.columns:
            checkable = row_checkable(family, row_b, col, missing)
            same = a.iat[i, a.columns.get_loc(col)] == row_b[col]
            if not checkable:
                cells_unchecked += 1
                state = "unchecked"
                if row_state == "identical":
                    row_state = "unchecked"
            elif same:
                cells_identical += 1
                state = "identical"
            else:
                cells_different += 1
                state = "different"
                row_state = "different"
                if len(differences) < 20:
                    differences.append({"row": i, "column": col, "ours": a.iat[i, a.columns.get_loc(col)],
                                        "reference": row_b[col]})
            if col in P_COLUMNS:
                p[state] += 1
        rows_identical += row_state == "identical"
        rows_different += row_state == "different"
        rows_unchecked += row_state == "unchecked"
    result.update({
        "cells_identical": cells_identical, "cells_different": cells_different,
        "cells_unchecked": cells_unchecked, "rows_identical": rows_identical,
        "rows_different": rows_different, "rows_unchecked": rows_unchecked,
        "p_values": p, "differences": differences,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="The study's result directories (summary.json files and inference CSVs).")
    parser.add_argument("--reference-daily-root", type=Path, default=None,
                        help="Result directories of the study that contain the daily portfolio values.")
    parser.add_argument("--cell-root", type=Path, default=PROJECT_ROOT / "runs" / "cells",
                        help="The runner's cell results (preferred over the study's daily files).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "inference-parity")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--family", nargs="+", default=["primary", "signal_tuned", "dividend"],
                        choices=sorted(FAMILIES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compare-only", action="store_true",
                        help="Compare existing outputs; do not run the inference.")
    return parser


def print_report(report: Dict[str, Any]) -> None:
    inputs = report["inputs"]
    print("\n## Inputs\n")
    print(f"cells needed: {len(inputs['cells'])}; by source: {inputs['by_source']}")
    print(f"runner cells with summary.json identical to the study: {inputs['runner_summary_identical']}; "
          f"different: {len(inputs['runner_summary_differs'])}")
    print(f"runner cells with daily file byte-identical to the study: {inputs['runner_daily_identical']}; "
          f"different: {len(inputs['runner_daily_differs'])}")
    if inputs["missing"]:
        print(f"missing cells: {len(inputs['missing'])} (e.g. {inputs['missing'][:5]})")
    print("\n## Inference files\n")
    print("| Family | File | Rows | Rows identical | Rows different | Rows not checkable "
          "| p-values identical | p-values different | p-values not checkable | File bytes identical |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for family, results in report["families"].items():
        if isinstance(results, dict) and "error" in results:
            print(f"| {family} | (not run: {results['error']}) | | | | | | | | |")
            continue
        for r in results:
            if "error" in r:
                print(f"| {family} | {r['file']} | error: {r['error']} | | | | | | | |")
                continue
            p = r["p_values"]
            print(f"| {family} | {r['file']} | {r['rows']} | {r['rows_identical']} | {r['rows_different']} | "
                  f"{r['rows_unchecked']} | {p['identical']} | {p['different']} | {p['unchecked']} | "
                  f"{'yes' if r['bytes_identical'] else 'no'} |")
            for d in r["differences"][:5]:
                print(f"  - row {d['row']} {d['column']}: ours {d['ours']!r}, study {d['reference']!r}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    families = list(dict.fromkeys(["primary"] + args.family if "signal_tuned" in args.family else args.family))
    args.output_root.mkdir(parents=True, exist_ok=True)
    cells = needed_cells(families)
    inputs = assemble_inputs(args, cells)
    missing = set(inputs["missing"])
    report: Dict[str, Any] = {"inputs": inputs, "families": {f: None for f in families}}
    t0 = time.time()
    runnable = []
    for family in families:
        algos = FAMILIES[family].non_rl_algorithms + FAMILIES[family].rl_algorithms
        family_missing = [c for c in missing if c.split("__")[1] in algos]
        if FAMILIES[family].post_hoc and family_missing:
            report["families"][family] = {"error": f"{len(family_missing)} input cells missing"}
        else:
            runnable.append(family)
    if not args.compare_only and runnable:
        run_inference(args.output_root / "cells", args.data_root, args.output_root / "inference",
                      families=runnable, manifest=load_manifest(args.manifest), workers=args.workers)
    report["inference_seconds"] = time.time() - t0
    for family in runnable:
        ref_dir = args.reference_root / REFERENCE_OUTPUTS[family]
        report["families"][family] = [
            compare_file(family, args.output_root / "inference" / family / name, ref_dir / name, missing)
            for name in OUTPUT_FILES[family]
        ]
    print_report(report)
    path = args.output_root / "inference_parity_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[parity] report: {path}")
    bad = bool(inputs["runner_summary_differs"] or inputs["runner_daily_differs"])
    for results in report["families"].values():
        if isinstance(results, dict):
            bad = True
            continue
        for r in results:
            if "error" in r or r["cells_different"]:
                bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the cells of a manifest on one computer.

Each cell runs as a separate ``tools/run_cell.py`` process. ``--workers N``
runs N cells at the same time. A cell whose result already exists (same
configuration) is skipped, so an interrupted grid can be started again.

    python tools/run_grid.py --data-root /path/to/data --list
    python tools/run_grid.py --data-root /path/to/data --group non_rl --workers 4
    python tools/run_grid.py --data-root /path/to/data --strategy ppo_mlp --seed 1 2 3

Each cell writes its log to ``<output-root>/logs/<cell_id>.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.manifest import DEFAULT_MANIFEST, Cell, expand, load_manifest  # noqa: E402

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def select_cells(
    cells: Sequence[Cell],
    *,
    groups: Optional[Sequence[str]] = None,
    markets: Optional[Sequence[str]] = None,
    strategies: Optional[Sequence[str]] = None,
    frictions: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[tuple[int, Cell]]:
    """``(manifest index, cell)`` pairs that pass every given filter."""
    out = []
    for index, cell in enumerate(cells):
        if groups and cell.group not in groups:
            continue
        if markets and cell.market not in markets:
            continue
        if strategies and cell.strategy not in strategies:
            continue
        if frictions and cell.friction not in frictions:
            continue
        if seeds and cell.seed not in seeds:
            continue
        out.append((index, cell))
    return out


def cell_environment(threads_per_cell: int, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    for var in THREAD_ENV_VARS:
        env[var] = str(threads_per_cell)
    env.update(extra or {})
    return env


def run_one(
    index: int,
    cell: Cell,
    *,
    args: argparse.Namespace,
    extra_args: Sequence[str] = (),
    env: Dict[str, str],
) -> Dict[str, object]:
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "test" else f"__{args.split}"
    log_path = log_dir / f"{cell.cell_id}{suffix}.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_cell.py"),
        "--manifest", str(args.manifest),
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--index", str(index),
        "--split", args.split,
        "--device", args.device,
        "--threads", str(args.threads_per_cell),
        *extra_args,
    ]
    if args.total_timesteps is not None:
        command += ["--total-timesteps", str(args.total_timesteps)]
    if args.force:
        command.append("--force")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    return {
        "cell_id": cell.cell_id,
        "returncode": code,
        "seconds": time.time() - t0,
        "log": str(log_path),
    }


def is_done(cell: Cell, args: argparse.Namespace) -> bool:
    suffix = "" if args.split == "test" else f"__{args.split}"
    return (args.output_root / f"{cell.cell_id}{suffix}" / "summary.json").exists()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "cells")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--group", nargs="+", default=None)
    parser.add_argument("--market", nargs="+", default=None)
    parser.add_argument("--strategy", nargs="+", default=None)
    parser.add_argument("--friction", nargs="+", default=None)
    parser.add_argument("--seed", nargs="+", type=int, default=None)
    parser.add_argument("--split", choices=["test", "valid"], default="test")
    parser.add_argument("--workers", type=int, default=1, help="Cells run at the same time.")
    parser.add_argument("--threads-per-cell", type=int, default=1,
                        help="BLAS and PyTorch threads per cell (the paper: 1 non-RL, 2 RL).")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print the selected cells and stop.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    selected = select_cells(
        expand(manifest),
        groups=args.group,
        markets=args.market,
        strategies=args.strategy,
        frictions=args.friction,
        seeds=args.seed,
    )
    if args.list:
        for index, cell in selected:
            print(f"{index}\t{cell.cell_id}\t{cell.group}")
        return 0
    if args.data_root is None:
        raise SystemExit("--data-root is required")

    todo = [(i, c) for i, c in selected if args.force or not is_done(c, args)]
    print(f"[grid] {len(selected)} cells selected, {len(selected) - len(todo)} already done, "
          f"{len(todo)} to run with {args.workers} worker(s)", flush=True)
    env = cell_environment(args.threads_per_cell)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, i, c, args=args, env=env) for i, c in todo]
        for n, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = "ok" if result["returncode"] == 0 else f"FAILED ({result['returncode']})"
            print(f"[grid] {n}/{len(todo)} {result['cell_id']} {status} "
                  f"{result['seconds']:.0f}s", flush=True)
            if result["returncode"] != 0:
                failures.append(result)
    for result in failures:
        print(f"[grid] failed: {result['cell_id']} (log: {result['log']})")
    print(json.dumps({"selected": len(selected), "run": len(todo), "failed": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the statistical inference of the paper on the runner's cell results.

Input: the F3 cell results written by ``tools/run_cell.py`` / ``tools/run_grid.py``
(``<cell-root>/<market>__<strategy>__F3__seed<n>/summary.json`` and
``backtest/portfolio_values.csv``), and the data directory (for the risk-free
rates and the test calendar named in ``configs/base/``).

    python tools/run_inference.py --data-root /path/to/data --cell-root runs/cells \\
        --output-root runs/inference --workers 4

Families (``--family``, default ``primary signal_tuned``):

    primary       8 non-RL x 4 RL = 32 contrasts per market, block-length
                  sensitivity (10, 42), secondary cadence control
                  (CSM-LO vs monthly UCRP). Needs the non_rl and
                  cadence_control F3 cells and the RL F3 cells, seeds 1-21.
    signal_tuned  post hoc, 40 per market: adds zscore_tuned and macd_tuned
                  (the signal_tuned cells). Also runs ``primary``.
    dividend      post hoc, 52 per market: adds markowitz_tr, hrp_tr, erc_tr.
                  The runner does not yet produce these dividend-inclusive
                  cells; put their result directories (same layout, seed 1)
                  under --cell-root to run this family.

Outputs, under ``<output-root>/<family>/``, have the file names, columns and
number formats of the study's result files (docs/inference.md). A contrast
whose cells are missing enters its Holm family at p = 1 in ``primary``; the
post hoc families refuse to run with a missing cell.

With B = 10,000 one contrast takes about 5 seconds; ``--workers`` runs
contrasts in parallel processes (results do not depend on it). The complete
primary + signal_tuned run computes 240 contrasts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One BLAS thread per process (set before NumPy is imported); the bootstrap is
# element-wise NumPy code.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from tools.inference import FAMILIES, N_BOOTSTRAP, run_inference  # noqa: E402
from tools.manifest import DEFAULT_MANIFEST, load_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory with the panels and risk-free rates (as for run_cell.py).")
    parser.add_argument("--cell-root", type=Path, default=PROJECT_ROOT / "runs" / "cells",
                        help="Directory with the cell result directories (default runs/cells).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "inference")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--family", nargs="+", default=["primary", "signal_tuned"],
                        choices=sorted(FAMILIES), help="Families to compute.")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Bootstrap draws (the paper: 10,000). Other values do not reproduce the paper.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes for the contrasts.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    files = run_inference(
        args.cell_root, args.data_root, args.output_root,
        families=args.family, manifest=load_manifest(args.manifest),
        n_bootstrap=args.n_bootstrap, workers=args.workers,
    )
    for family, paths in files.items():
        for path in paths.values():
            print(f"[inference] {family}: {path}")
    print(f"[inference] done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

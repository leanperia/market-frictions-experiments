#!/usr/bin/env python3
"""Run one experiment cell and write one result directory.

A cell is (market, strategy, friction level, seed). The cell's configuration
layers come from a manifest (default ``manifests/cifer2027.yaml``)::

    python tools/run_cell.py --data-root /path/to/data --output-root runs/cells \\
        --market pse --strategy ubah --friction F3 --seed 1

    python tools/run_cell.py --data-root /path/to/data --output-root runs/cells \\
        --index 17            # the 18th cell of the manifest (0-based)

Result directory ``<output-root>/<market>__<strategy>__<friction>__seed<n>/``:

    merged_config.yaml            effective configuration (all layers merged)
    summary.json                  metrics (key ``backtest``) and run facts
    backtest/portfolio_values.csv daily portfolio value and friction diagnostics
    backtest/trades.csv           per-asset rows with executed trades
    backtest/formation_diagnostics.jsonl   (formation-based strategies only)
    backtest/asset_diagnostics.csv         (only with --asset-diagnostics)
    checkpoint/model.zip          (RL cells only)

Paths in the configuration files are written as ``data/<path>``. The runner
reads them as ``<data-root>/<path>``. Non-RL strategies are evaluated on the test
split with the earlier splits as price history. RL strategies are trained on
the training split and evaluated deterministically on the test split.

The semantics are those of the research runner that produced the paper
(``tools/run_experiment_cell.py`` in the private development repository).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta.data_adapters.local_panel import load_env_config  # noqa: E402
from meta.env_market_impact.envs.env_portfolio_optimization_impact import (  # noqa: E402
    build_portfolio_env,
)
from meta.frictions.diagnostics import expand_asset_step_info, flatten_step_info  # noqa: E402
from meta.policies import build_non_rl_policy  # noqa: E402
from tools.manifest import (  # noqa: E402
    DEFAULT_MANIFEST,
    Cell,
    cell_layers,
    cell_params,
    expand,
    find_cell,
    load_manifest,
)
from tools.portfolio_metrics import compute_backtest_metrics  # noqa: E402
from tools.rf_series import load_rf_series  # noqa: E402

NON_RL_KINDS = {"non_rl", "classical"}
DEFAULT_RL_TIMESTEPS = 300_000
SPLIT_KEYS = {"train": "train_csv", "valid": "valid_csv", "test": "test_csv"}


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


# Copied from tools/run_experiment_cell.py (deep_merge).
def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def merge_layers(paths: Iterable[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    return merged


def config_sha256(cfg: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the configuration.

    Keys that start with ``__`` (bookkeeping) are excluded. For the same
    effective configuration this equals the ``config_hash`` recorded by the
    research runner of the paper.
    """
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("__")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_cell_config(
    manifest: Dict[str, Any],
    cell: Cell,
    *,
    total_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge the layers of a cell and set the seed (and the RL budget, if given)."""
    layers = cell_layers(manifest, cell)
    cfg = merge_layers(PROJECT_ROOT / layer for layer in layers)
    params = cell_params(manifest, cell)
    if params is not None:
        cfg.setdefault("algo", {})["params"] = params
    cfg.setdefault("runtime", {})["seed"] = int(cell.seed)
    if total_timesteps is not None:
        cfg.setdefault("training", {})["total_timesteps"] = int(total_timesteps)
    cfg["__layers__"] = list(layers)
    return cfg


def is_rl(cfg: Mapping[str, Any]) -> bool:
    return cfg["algo"].get("kind", "rl") not in NON_RL_KINDS


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────


def resolve_data_path(value: str | Path | None, data_root: Path) -> Optional[Path]:
    """``data/<path>`` (or ``<path>``) in a configuration -> ``<data_root>/<path>``."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        path = Path(*parts[1:])
    return Path(data_root) / path


def split_csv(cfg: Mapping[str, Any], split: str, data_root: Path) -> Path:
    key = SPLIT_KEYS[split]
    path = resolve_data_path(cfg.get("data", {}).get(key), data_root)
    if path is None:
        raise ValueError(f"configuration has no data.{key}")
    if not path.exists():
        raise FileNotFoundError(f"data.{key} not found: {path}")
    return path


def history_csvs(cfg: Mapping[str, Any], split: str, data_root: Path) -> List[Path]:
    """Earlier splits that give pre-window price history to the policies.

    Validation sees the training split; test sees training and validation.
    Same rule as ``_history_csvs_for_split`` in the research runner.
    """
    data_cfg = cfg.get("data", {})
    train = resolve_data_path(data_cfg.get("train_csv"), data_root)
    valid = resolve_data_path(data_cfg.get("valid_csv"), data_root)
    if split == "valid":
        return [train] if train is not None else []
    if split == "test":
        return [p for p in (train, valid) if p is not None]
    return []


def outage_calendar(cfg: Mapping[str, Any], data_root: Path) -> Optional[Path]:
    for spec in cfg.get("friction", {}).get("layers", []) or []:
        if spec.get("name") == "outages":
            calendar = (spec.get("params") or {}).get("calendar")
            return resolve_data_path(calendar, data_root)
    return None


def build_env(cfg: Dict[str, Any], split: str, data_root: Path, seed: int):
    """Construct the friction-aware environment for one split (already reset)."""
    env_config = load_env_config(
        split_csv(cfg, split, data_root),
        outage_calendar=outage_calendar(cfg, data_root),
        history_csv=history_csvs(cfg, split, data_root),
        require_option1_accounting=(
            cfg.get("data", {}).get("accounting_model") == "option1_crsp"
        ),
    )
    return build_portfolio_env(env_config, cfg, seed=seed)


def load_rf(cfg: Dict[str, Any], data_root: Path) -> Optional[pd.Series]:
    rf = load_rf_series(cfg, resolve_fn=lambda value: resolve_data_path(value, data_root))
    if rf is None and cfg.get("data", {}).get("rf_csv") is not None:
        print(f"[warn] risk-free series not loaded: {cfg['data']['rf_csv']}", file=sys.stderr)
    return rf


# Same as _load_benchmark_series in tools/run_experiment_cell.py.
def load_benchmark(cfg: Dict[str, Any], data_root: Path) -> Optional[pd.Series]:
    bm_csv = cfg.get("data", {}).get("benchmark_csv")
    if bm_csv is None:
        return None
    bm_path = resolve_data_path(bm_csv, data_root)
    if bm_path is None or not bm_path.exists():
        print(f"[warn] benchmark_csv not found: {bm_csv}", file=sys.stderr)
        return None
    df = pd.read_csv(bm_path)
    df["date"] = pd.to_datetime(df["date"])
    close_col = "close" if "close" in df.columns else "Close"
    if close_col not in df.columns:
        print(f"[warn] benchmark_csv has no close column: {list(df.columns)}", file=sys.stderr)
        return None
    df = df.set_index("date").sort_index()
    bm_ret = df[close_col].pct_change().dropna()
    bm_ret.index.name = "date"
    return bm_ret


# ──────────────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────────────


# Copied from tools/run_experiment_cell.py (_RLActor).
class RLActor:
    """Give an SB3 model the same ``step(env)`` surface as ``NonRLPolicy``."""

    def __init__(self, model, initial_state):
        self._model = model
        self._state = initial_state

    def step(self, env):
        action, _ = self._model.predict(self._state, deterministic=True)
        self._state, reward, terminated, truncated, info = env.step(action)
        return self._state, reward, terminated, truncated, info


# Copied from tools/run_experiment_cell.py (run_deterministic). ``policy`` is a
# NonRLPolicy (``rl=False``) or a Stable-Baselines3 model (``rl=True``).
def run_deterministic(
    policy,
    env,
    *,
    rl: bool,
    rf_series: pd.Series | None = None,
    benchmark_series: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    state, _ = env.reset()
    rows = [{
        "date": env.date_list[env.time],
        "portfolio_value": env.total_asset,
        "reward": 0.0,
        "cash": float(getattr(env, "cash", env.total_asset)),
        "n_trades": 0,
    }]
    asset_rows: List[Dict[str, Any]] = []
    actor = RLActor(policy, state) if rl else policy
    for _ in range(env.max_step):
        _state, reward, terminated, _truncated, info = actor.step(env)
        date = env.date_list[env.time]
        rows.append(
            {
                "date": date,
                "portfolio_value": env.total_asset,
                "reward": float(reward),
                **flatten_step_info(info),
            }
        )
        asset_rows.extend(expand_asset_step_info(date, env.stock_symbols, info))
        if terminated:
            break
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype == object:
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(0.0)
    metrics = compute_backtest_metrics(df, rf_series, benchmark_series)
    asset_df = pd.DataFrame(asset_rows)
    if asset_df.empty:
        trades_df = asset_df.copy()
    else:
        trade_like = asset_df["abs_executed_shares"].astype(float).abs() > 0
        trades_df = asset_df[trade_like].reset_index(drop=True)
    return df, asset_df, trades_df, metrics


# Copied from tools/run_experiment_cell.py (_json_safe).
def _json_safe(value: Any, *, path: str, degenerate: List[Dict[str, str]]) -> Any:
    """Replace non-finite floats by None and record their dotted paths."""
    if isinstance(value, Mapping):
        return {
            key: _json_safe(
                item, path=f"{path}.{key}" if path else str(key), degenerate=degenerate
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]", degenerate=degenerate)
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        degenerate.append({"metric": path, "reason": "non_finite"})
        return None
    return value


def software_versions() -> Dict[str, str]:
    import gymnasium
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "gymnasium": gymnasium.__version__,
    }


# ──────────────────────────────────────────────────────────────────────
# Cell
# ──────────────────────────────────────────────────────────────────────


def run_cell(
    cell: Cell,
    *,
    data_root: Path,
    output_root: Path,
    manifest: Optional[Dict[str, Any]] = None,
    split: str = "test",
    total_timesteps: Optional[int] = None,
    device: str = "cpu",
    threads: Optional[int] = None,
    model_path: Optional[Path] = None,
    force: bool = False,
    asset_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Run one cell. Returns ``{"cell_id", "status", "result_dir"}``.

    A cell whose ``summary.json`` exists with the same configuration hash is
    skipped. If the configuration changed, the cell is refused unless
    ``force`` is true.
    """
    if manifest is None:
        manifest = load_manifest(DEFAULT_MANIFEST)
    if split not in SPLIT_KEYS or split == "train":
        raise ValueError("split must be 'test' or 'valid'")
    data_root = Path(data_root)
    cfg = build_cell_config(manifest, cell, total_timesteps=total_timesteps)
    rl = is_rl(cfg)
    if rl:
        cfg.setdefault("training", {}).setdefault("total_timesteps", DEFAULT_RL_TIMESTEPS)
    if model_path is not None and not rl:
        raise ValueError("--model applies to RL cells only")
    h = config_sha256(cfg)

    cell_id = cell.cell_id if split == "test" else f"{cell.cell_id}__{split}"
    result_dir = Path(output_root) / cell_id
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not force:
        recorded = json.loads(summary_path.read_text()).get("config_sha256")
        if recorded == h:
            print(f"[skip] {cell_id} (result exists)")
            return {"cell_id": cell_id, "status": "skipped", "result_dir": str(result_dir)}
        raise RuntimeError(
            f"{summary_path} exists with a different configuration "
            f"({recorded} != {h}); use --force to replace it"
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "merged_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    seed = int(cell.seed)
    rf_series = load_rf(cfg, data_root)
    benchmark_series = load_benchmark(cfg, data_root)
    versions = software_versions()
    runtime: Dict[str, Any] = {}
    train_time = 0.0
    trained = False
    print(f"[cell] {cell_id}")

    if not rl:
        random.seed(seed)
        np.random.seed(seed)
        algo_cfg = cfg["algo"]
        policy = build_non_rl_policy(algo_cfg["name"], algo_cfg.get("params", {}))
        env = build_env(cfg, split, data_root, seed)
        bt_df, asset_df, trades_df, metrics = run_deterministic(
            policy, env, rl=False, rf_series=rf_series, benchmark_series=benchmark_series
        )
        formation_log = getattr(policy, "formation_log", None)
    else:
        from tools import rl as rl_tools

        runtime.update(rl_tools.configure_torch_threads(threads))
        versions.update(rl_tools.software_versions())
        rl_tools.set_seed(seed)
        train_env = build_env(cfg, "train", data_root, seed)
        if model_path is not None:
            model = rl_tools.load_model(cfg, str(model_path), train_env, device=device)
            print(f"[cell] loaded model {model_path}")
        else:
            steps = int(cfg["training"]["total_timesteps"])
            print(f"[cell] training device={device} timesteps={steps}")
            t0 = time.time()
            model = rl_tools.train_model(
                cfg, train_env, seed=seed, device=device, total_timesteps=steps
            )
            train_time = time.time() - t0
            trained = True
            (result_dir / "checkpoint").mkdir(exist_ok=True)
            model.save(str(result_dir / "checkpoint" / "model"))
        env = build_env(cfg, split, data_root, seed)
        bt_df, asset_df, trades_df, metrics = run_deterministic(
            model, env, rl=True, rf_series=rf_series, benchmark_series=benchmark_series
        )
        formation_log = None

    backtest_dir = result_dir / "backtest"
    backtest_dir.mkdir(exist_ok=True)
    bt_df.to_csv(backtest_dir / "portfolio_values.csv", index=False)
    trades_df.to_csv(backtest_dir / "trades.csv", index=False)
    if asset_diagnostics:
        asset_df.to_csv(backtest_dir / "asset_diagnostics.csv", index=False)
    if formation_log:
        with (backtest_dir / "formation_diagnostics.jsonl").open("w", encoding="utf-8") as fh:
            for entry in formation_log:
                fh.write(json.dumps(entry, default=str) + "\n")

    summary: Dict[str, Any] = {
        "cell_id": cell_id,
        "market": cell.market,
        "algo": cell.strategy,
        "friction_level": cfg.get("friction", {}).get("level"),
        "seed": seed,
        "split": split,
        "policy_kind": "rl" if rl else "non_rl",
        "config_layers": cfg["__layers__"],
        "config_sha256": h,
        "training_performed": trained,
        "total_timesteps": int(cfg["training"]["total_timesteps"]) if trained else 0,
        "model_loaded_from": str(model_path) if model_path is not None else None,
        "device": device if rl else "cpu",
        "train_time_seconds": train_time,
        "software": versions,
        "runtime": runtime,
        "backtest": metrics,
    }
    degenerate: List[Dict[str, str]] = []
    summary = _json_safe(summary, path="", degenerate=degenerate)
    if degenerate:
        summary["degenerate_metrics"] = degenerate
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=str, allow_nan=False))
    tmp.replace(summary_path)  # summary.json appears only when the cell is complete
    print(
        f"[done] {cell_id} sharpe_rf={metrics.get('sharpe_ratio_rf', float('nan')):.4f} "
        f"total_return={metrics['total_return']:.4f}"
    )
    return {"cell_id": cell_id, "status": "ok", "result_dir": str(result_dir)}


# ──────────────────────────────────────────────────────────────────────
# Command line
# ──────────────────────────────────────────────────────────────────────


def select_cell(manifest: Dict[str, Any], args: argparse.Namespace) -> Cell:
    cells = expand(manifest)
    if args.index is not None:
        if not 0 <= args.index < len(cells):
            raise SystemExit(f"--index must be in [0, {len(cells) - 1}]")
        return cells[args.index]
    if args.cell is not None:
        return find_cell(manifest, args.cell)
    missing = [n for n in ("market", "strategy", "friction") if getattr(args, n) is None]
    if missing:
        raise SystemExit("give --index, --cell, or all of --market --strategy --friction")
    if args.strategy not in manifest["strategies"]:
        raise SystemExit(f"unknown strategy {args.strategy!r}")
    if args.market not in manifest["markets"]:
        raise SystemExit(f"unknown market {args.market!r}")
    if args.friction not in manifest["markets"][args.market]["friction"]:
        raise SystemExit(f"unknown friction level {args.friction!r}")
    return Cell(args.market, args.strategy, args.friction, int(args.seed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory that holds the panels, rates and benchmarks.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "cells")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int, default=None,
                        help="0-based position of the cell in the expanded manifest.")
    parser.add_argument("--cell", default=None, help="Cell id, e.g. pse__ubah__F3__seed1.")
    parser.add_argument("--market", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--friction", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", choices=["test", "valid"], default="test",
                        help="Evaluation split (default test).")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="RL training budget (default from config: 300,000).")
    parser.add_argument("--device", default="cpu", help="PyTorch device for RL (default cpu).")
    parser.add_argument("--threads", type=int, default=None,
                        help="PyTorch threads for RL (the paper used 2).")
    parser.add_argument("--model", type=Path, default=None,
                        help="Evaluate this saved RL model instead of training.")
    parser.add_argument("--asset-diagnostics", action="store_true",
                        help="Also write per-asset, per-day diagnostics.")
    parser.add_argument("--force", action="store_true", help="Replace an existing result.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    cell = select_cell(manifest, args)
    run_cell(
        cell,
        data_root=args.data_root,
        output_root=args.output_root,
        manifest=manifest,
        split=args.split,
        total_timesteps=args.total_timesteps,
        device=args.device,
        threads=args.threads,
        model_path=args.model,
        force=args.force,
        asset_diagnostics=args.asset_diagnostics,
    )


if __name__ == "__main__":
    main()

"""Experiment manifests: the list of cells in a grid and the layers of each cell.

A cell is one (market, strategy, friction level, seed). A manifest (for example
``manifests/cifer2027.yaml``) declares the configuration layers of each market
and strategy, and the groups of cells to run. ``expand`` returns all cells in a
fixed order, so that an integer index identifies a cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "cifer2027.yaml"


@dataclass(frozen=True)
class Cell:
    market: str
    strategy: str
    friction: str
    seed: int
    group: str = ""

    @property
    def cell_id(self) -> str:
        return f"{self.market}__{self.strategy}__{self.friction}__seed{self.seed}"


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be a mapping")
    for key in ("markets", "strategies", "groups"):
        if key not in manifest:
            raise ValueError(f"{path}: manifest has no '{key}' section")
    for group in manifest["groups"]:
        for strategy in group["strategies"]:
            if strategy not in manifest["strategies"]:
                raise ValueError(f"group {group['name']}: unknown strategy {strategy!r}")
        for market in group["markets"]:
            if market not in manifest["markets"]:
                raise ValueError(f"group {group['name']}: unknown market {market!r}")
            for friction in group["frictions"]:
                if friction not in manifest["markets"][market]["friction"]:
                    raise ValueError(
                        f"group {group['name']}: no friction {friction!r} for {market}"
                    )
    return manifest


def expand(manifest: Dict[str, Any]) -> List[Cell]:
    """All cells of the manifest, in the fixed manifest order, without duplicates."""
    cells: List[Cell] = []
    seen = set()
    for group in manifest["groups"]:
        for market in group["markets"]:
            for strategy in group["strategies"]:
                for friction in group["frictions"]:
                    for seed in group["seeds"]:
                        cell = Cell(market, strategy, friction, int(seed), group["name"])
                        if cell.cell_id in seen:
                            continue
                        seen.add(cell.cell_id)
                        cells.append(cell)
    return cells


def find_cell(manifest: Dict[str, Any], cell_id: str) -> Cell:
    for cell in expand(manifest):
        if cell.cell_id == cell_id:
            return cell
    raise KeyError(f"cell {cell_id!r} is not in the manifest")


def cell_layers(manifest: Dict[str, Any], cell: Cell) -> List[str]:
    """Configuration files of a cell, relative to the repository root, in merge order."""
    market = manifest["markets"][cell.market]
    strategy = manifest["strategies"][cell.strategy]
    layers = [market["base"], market["friction"][cell.friction], strategy["algo"]]
    tuned = strategy.get("tuned")
    if tuned:
        layers.append(tuned.format(market=cell.market))
    return layers


def cell_params(manifest: Dict[str, Any], cell: Cell) -> Optional[Dict[str, Any]]:
    """Strategy parameters that replace ``algo.params`` for this market, if any."""
    params = manifest["strategies"][cell.strategy].get("params")
    if params is None:
        return None
    if cell.market not in params:
        raise ValueError(f"strategy {cell.strategy}: no params for market {cell.market}")
    return dict(params[cell.market])

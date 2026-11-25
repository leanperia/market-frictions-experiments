"""PSE-specific friction-layer builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from ..board_lots import BoardLotRounding
from ..fees import PseFeeSchedule
from ..outages import OutageFilter


def _build_fees(params: Dict[str, Any], _env_config: Mapping[str, Any]) -> PseFeeSchedule:
    return PseFeeSchedule(**dict(params or {}))


def _build_board_lots(params: Dict[str, Any], _env_config: Mapping[str, Any]) -> BoardLotRounding:
    return BoardLotRounding(**dict(params or {}))


def _build_outages(params: Dict[str, Any], _env_config: Mapping[str, Any]) -> OutageFilter:
    params = dict(params or {})
    calendar = params.pop("calendar", None)
    if calendar is None:
        return OutageFilter(outages=params.pop("outages", []), **params)
    calendar_path = Path(calendar)
    if not calendar_path.is_absolute() and not calendar_path.exists():
        calendar_path = Path(__file__).resolve().parents[3] / calendar_path
    return OutageFilter.from_yaml(calendar_path, **params)


LAYER_BUILDERS = {
    "fees": {
        "factory": _build_fees,
        "required_env_keys": (),
    },
    "board_lots": {
        "factory": _build_board_lots,
        "required_env_keys": (),
    },
    "outages": {
        "factory": _build_outages,
        "required_env_keys": (),
    },
}

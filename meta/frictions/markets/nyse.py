"""NYSE-specific friction-layer builders."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ..fees import NyseFeeSchedule


def _build_fees(params: Dict[str, Any], _env_config: Mapping[str, Any]) -> NyseFeeSchedule:
    return NyseFeeSchedule(**dict(params or {}))


LAYER_BUILDERS = {
    "fees": {
        "factory": _build_fees,
        "required_env_keys": (),
    },
}

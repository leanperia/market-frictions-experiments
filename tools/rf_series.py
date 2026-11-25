"""tools/rf_series.py — Deterministic risk-free rate loader and alignment helper.

This module owns the full rf lifecycle:

  1. ``load_rf_series(cfg)``  —  load an rf CSV (date-column autodetect),
     convert annual-percent to daily decimal (÷25200 convention), handle
     rf_type ``'daily'`` vs ``'step_function'``.  For ``step_function``, the
     business-day forward-fill grid is bounded by ``index.max()`` of the data
     itself — ``pd.Timestamp.today()`` is **never** consulted, making the
     output deterministic.

  2. ``align_rf(rf_series, date_index, *, coverage_threshold)``  —  reindex
     rf onto a supplied DatetimeIndex and raise a clear ``ValueError`` when
     coverage is below *coverage_threshold* (default 0.9) instead of silently
     bfill-filling to zero.  For full-coverage inputs the result is numerically
     identical to the legacy
     ``rf.reindex(dates).ffill().bfill().fillna(0.0).values`` idiom.

References:
    STAT-1 review finding: today() non-determinism in step-function loaders.
    ADR-0001: /25200 annual-to-daily decimal convention is settled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Settled convention (ADR-0001): annual percent ÷ 25200 = daily decimal.
# 25200 = 100 (percent-to-fraction) × 252 (trading days).
_ANNUAL_TO_DAILY = 25200.0

_DATE_COLUMN_CANDIDATES = ("observation_date", "date", "effective_date")


def _detect_date_column(columns: list[str]) -> str | None:
    for cand in _DATE_COLUMN_CANDIDATES:
        if cand in columns:
            return cand
    return None


def load_rf_series(
    cfg: Dict[str, Any],
    *,
    resolve_fn=None,
) -> "pd.Series | None":
    """Load a daily risk-free-rate series aligned to calendar dates.

    Config keys (under ``data``):
      rf_csv        : path to CSV with a date column and a rate column.
      rf_column     : name of the rate column (default: auto-detect).
      rf_type       : ``'daily'`` (default) or ``'step_function'``.

    Returns a pandas Series indexed by ``pd.Timestamp`` with daily rf as a
    decimal fraction (not percent), or ``None`` when rf_csv is not configured
    or the file cannot be found.

    The returned series name is ``'rf_daily'``.

    Notes
    -----
    ``step_function`` rates are forward-filled onto a business-day grid.  The
    grid end is ``annual_rate.index.max()`` — the last date in the raw CSV.
    ``pd.Timestamp.today()`` is never used.
    """
    data_cfg = cfg.get("data", {})
    rf_csv = data_cfg.get("rf_csv")
    if rf_csv is None:
        return None

    # Resolve the CSV path, using the supplied resolver if given.
    if resolve_fn is not None:
        rf_path = resolve_fn(rf_csv)
    else:
        rf_path = Path(rf_csv)

    if rf_path is None or not rf_path.exists():
        return None

    df = pd.read_csv(rf_path)
    date_col = _detect_date_column(list(df.columns))
    if date_col is None:
        return None

    df[date_col] = pd.to_datetime(df[date_col])

    rf_col = data_cfg.get("rf_column")
    if rf_col is None:
        rate_cols = [c for c in df.columns if c != date_col]
        if len(rate_cols) != 1:
            return None
        rf_col = rate_cols[0]

    df[rf_col] = pd.to_numeric(df[rf_col], errors="coerce")
    df = df.dropna(subset=[rf_col]).set_index(date_col).sort_index()
    annual_rate: pd.Series = df[rf_col]

    rf_type = data_cfg.get("rf_type", "daily")
    if rf_type == "step_function":
        # Forward-fill onto a business-day grid.
        # The grid end is the last date present in the data — NEVER today().
        end = annual_rate.index.max()
        full_idx = pd.date_range(annual_rate.index.min(), end, freq="B")
        annual_rate = annual_rate.reindex(full_idx).ffill().bfill()

    daily_rf = annual_rate / _ANNUAL_TO_DAILY
    daily_rf.index.name = "date"
    daily_rf.name = "rf_daily"
    return daily_rf


def load_rf_series_from_path(
    rf_path: Path,
    rf_col: str | None = None,
    rf_type: str = "daily",
) -> pd.Series:
    """Low-level loader: load rf from an explicit path without a config dict.

    Parameters
    ----------
    rf_path:
        CSV file with a recognised date column and a rate column.
    rf_col:
        Rate column name.  Auto-detected (first non-date column) when ``None``.
    rf_type:
        ``'daily'`` or ``'step_function'``.

    Returns
    -------
    pd.Series
        Daily decimal rf indexed by pd.Timestamp.

    Raises
    ------
    ValueError
        When the CSV has no recognisable date column.
    """
    df = pd.read_csv(rf_path)
    date_col = _detect_date_column(list(df.columns))
    if date_col is None:
        raise ValueError(
            f"rf_csv has no recognised date column ({list(_DATE_COLUMN_CANDIDATES)}): "
            f"{rf_path}"
        )
    df[date_col] = pd.to_datetime(df[date_col])

    if rf_col is None:
        rate_cols = [c for c in df.columns if c != date_col]
        if len(rate_cols) != 1:
            raise ValueError(
                f"rf_csv has ambiguous rate columns (expected exactly one): {rate_cols}"
            )
        rf_col = rate_cols[0]

    df[rf_col] = pd.to_numeric(df[rf_col], errors="coerce")
    df = df.dropna(subset=[rf_col]).set_index(date_col).sort_index()
    annual_rate: pd.Series = df[rf_col]

    if rf_type == "step_function":
        end = annual_rate.index.max()
        full_idx = pd.date_range(annual_rate.index.min(), end, freq="B")
        annual_rate = annual_rate.reindex(full_idx).ffill().bfill()

    daily_rf = annual_rate / _ANNUAL_TO_DAILY
    daily_rf.index.name = "date"
    daily_rf.name = "rf_daily"
    return daily_rf


def align_rf(
    rf_series: pd.Series,
    date_index: pd.DatetimeIndex | pd.Index,
    *,
    coverage_threshold: float = 0.9,
) -> np.ndarray:
    """Reindex *rf_series* onto *date_index* and return a numpy array.

    For full-coverage inputs, the result is **numerically identical** to the
    legacy idiom::

        rf_series.reindex(date_index).ffill().bfill().fillna(0.0).values

    Parameters
    ----------
    rf_series:
        Daily rf series produced by ``load_rf_series``.
    date_index:
        Target dates (the portfolio return dates).
    coverage_threshold:
        Fraction [0, 1] of requested dates that must have a non-NaN rf value
        after reindexing (before ffill/bfill).  Defaults to 0.9 (90 %).
        Use 0.0 to skip the check.

    Returns
    -------
    np.ndarray of float
        Shape ``(len(date_index),)`` with daily decimal rf values.

    Raises
    ------
    ValueError
        When the fraction of dates covered by *rf_series* is below
        *coverage_threshold*.
    """
    if len(date_index) == 0:
        return np.zeros(0, dtype=float)

    date_index = pd.DatetimeIndex(date_index)
    aligned = rf_series.reindex(date_index)

    if coverage_threshold > 0.0:
        n_covered = int(aligned.notna().sum())
        n_total = len(date_index)
        coverage = n_covered / n_total if n_total > 0 else 1.0
        if coverage < coverage_threshold:
            raise ValueError(
                f"rf_series coverage {coverage:.1%} is below the required threshold "
                f"{coverage_threshold:.1%} ({n_covered}/{n_total} requested dates have "
                f"rf values).  Check that rf_csv spans the backtest period."
            )

    result = aligned.ffill().bfill().fillna(0.0).values
    return result.astype(float)

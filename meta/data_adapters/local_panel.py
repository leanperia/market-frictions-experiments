"""Build MACE-env config dicts from local long-panel CSVs.

Source CSVs follow the schema produced by the TradeMaster fork:
``date, tic, open, high, low, close, volume, zopen, zhigh, zlow,
zclose, zd_5..zd_30, ...``. Required columns for the impact env are
``date, tic, open, high, low, close, volume``; the zscored columns are
used as the tech_array features (no extra TA-Lib dependency). Corrected
adjusted panels may additionally contain ``quote_open/high/low/close``;
these preserve raw exchange-quoted prices for price-banded constraints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

DEFAULT_TECH_FEATURES: tuple[str, ...] = (
    "zopen",
    "zhigh",
    "zlow",
    "zclose",
    "zd_5",
    "zd_10",
    "zd_15",
    "zd_20",
    "zd_25",
    "zd_30",
)
QUOTE_PRICE_COLUMNS: tuple[str, ...] = (
    "quote_open",
    "quote_high",
    "quote_low",
    "quote_close",
)
OPTION1_ACCOUNTING_COLUMNS: tuple[str, ...] = (
    "cum_share_factor",
    "dividend_per_panel_share",
)
# ADR-0009: per-row mask distinguishing builder-fabricated (forward-filled)
# rows from observed sessions. Optional so pre-ADR-0009 panels stay loadable;
# consumers that need it must key off its absence, never assume all-clean.
SYNTHETIC_HALT_COLUMN = "is_synthetic_halt"


def _validate_panel(df: pd.DataFrame, required: Sequence[str], path: Path) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    duplicates = df.duplicated(["date", "tic"])
    if duplicates.any():
        sample = df.loc[duplicates, ["date", "tic"]].head().to_dict("records")
        raise ValueError(f"{path} contains duplicate date/ticker rows: {sample}")
    counts = df.groupby("date")["tic"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"{path} has an inconsistent ticker count per day.")
    ticker_sets = df.groupby("date")["tic"].agg(lambda s: tuple(sorted(pd.unique(s))))
    expected = ticker_sets.iloc[0] if not ticker_sets.empty else ()
    bad_dates = ticker_sets[ticker_sets != expected]
    if not bad_dates.empty:
        first_bad = bad_dates.index[0]
        raise ValueError(
            f"{path} has an inconsistent ticker set; first mismatch on {first_bad!s}."
        )
    numeric_required = [col for col in required if col not in ("date", "tic")]
    for col in numeric_required:
        values = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{path} contains non-finite values in {col}.")
    if (df["close"] <= 0).any():
        raise ValueError(f"{path} contains non-positive close prices.")
    for col in ("open", "high", "low"):
        if col in df.columns and (df[col] <= 0).any():
            raise ValueError(f"{path} contains non-positive {col} prices.")
    if "volume" in df.columns and (df["volume"] < 0).any():
        raise ValueError(f"{path} contains negative volume.")


def _load_outages(path: Optional[Path]) -> List[dict]:
    if path is None:
        return []
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        return []
    return list(data.get("outages", []))


def insert_full_day_outage_rows(df: pd.DataFrame, outages: Sequence[dict]) -> pd.DataFrame:
    """Insert flat zero-volume rows for missing full-day outage dates.

    Some vendors omit cancelled sessions entirely. An outage-stress study
    needs those dates in the panel so its layer can block trades and produce
    diagnostics; the corrected V2 campaign does not enable that layer.
    Synthetic rows carry forward the previous close as OHLC and set derived
    return-style features to zero.
    """
    if not outages:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    existing = set(pd.to_datetime(df["date"].unique()).normalize())
    min_date = df["date"].min().normalize()
    max_date = df["date"].max().normalize()
    additions = []

    for entry in outages:
        if entry.get("nature") != "full_day_cancellation":
            continue
        outage_date = pd.Timestamp(entry["date"]).normalize()
        if outage_date in existing or outage_date < min_date or outage_date > max_date:
            continue
        previous_dates = [d for d in existing if d < outage_date]
        if not previous_dates:
            continue
        previous_date = max(previous_dates)
        rows = df[df["date"].dt.normalize() == previous_date].copy()
        rows["date"] = outage_date
        if "close" in rows:
            previous_close = rows["close"].copy()
            for col in ("open", "high", "low", "close", "adjcp"):
                if col in rows:
                    rows[col] = previous_close
        if "quote_close" in rows:
            previous_quote_close = rows["quote_close"].copy()
            for col in QUOTE_PRICE_COLUMNS:
                if col in rows:
                    rows[col] = previous_quote_close
        if "volume" in rows:
            rows["volume"] = 0
        if SYNTHETIC_HALT_COLUMN in rows:
            # The copied template row may be an observed session; the inserted
            # row never is.
            rows[SYNTHETIC_HALT_COLUMN] = 1
        for col in rows.columns:
            if col.startswith("z") or col.startswith("zd_"):
                rows[col] = 0.0
        additions.append(rows)
        existing.add(outage_date)

    if not additions:
        return df
    return pd.concat([df, *additions], ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily_return, 20-day volatility, and adv20 per ticker.

    Both volatility and adv20 are *shifted by one day* so they reflect
    information available at the start of the trading day (no lookahead).
    Early rows (where no prior observation exists) use conservative fallback
    values rather than back-filling from future rows.
    """
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("tic")["close"].pct_change()
    df["volatility"] = df.groupby("tic")["daily_return"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=2).std()
    )
    df["adv20"] = df.groupby("tic")["volume"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=1).mean()
    )
    df["volatility"] = df.groupby("tic")["volatility"].ffill().fillna(0.02)
    df["adv20"] = df.groupby("tic")["adv20"].ffill().fillna(df["volume"])
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def _read_panel_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str)
    return df


def _normalise_history_paths(history_csv: Optional[Path | Sequence[Path]]) -> List[Path]:
    if history_csv is None:
        return []
    if isinstance(history_csv, (str, Path)):
        return [Path(history_csv)]
    return [Path(p) for p in history_csv]


def load_panel(
    csv_path: Path,
    tech_features: Optional[Sequence[str]] = None,
    outage_calendar: Optional[Path] = None,
    history_csv: Optional[Path | Sequence[Path]] = None,
    require_option1_accounting: bool = False,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    current = _read_panel_csv(csv_path)
    current_min = current["date"].min().normalize()
    current_max = current["date"].max().normalize()

    history_parts = []
    for path in _normalise_history_paths(history_csv):
        hist = _read_panel_csv(path)
        hist = hist[hist["date"].dt.normalize() < current_min]
        if not hist.empty:
            history_parts.append(hist)

    df = pd.concat([*history_parts, current], ignore_index=True) if history_parts else current
    df = insert_full_day_outage_rows(df, _load_outages(outage_calendar))

    tech = list(tech_features) if tech_features is not None else list(DEFAULT_TECH_FEATURES)
    required: List[str] = ["date", "tic", "open", "high", "low", "close", "volume", *tech]
    quote_present = [col in df.columns for col in QUOTE_PRICE_COLUMNS]
    if any(quote_present) and not all(quote_present):
        missing_quote = [col for col in QUOTE_PRICE_COLUMNS if col not in df.columns]
        raise ValueError(f"{csv_path} has an incomplete quoted-price schema: {missing_quote}")
    if all(quote_present):
        required.extend(QUOTE_PRICE_COLUMNS)
    accounting_present = [col in df.columns for col in OPTION1_ACCOUNTING_COLUMNS]
    if any(accounting_present) and not all(accounting_present):
        missing_accounting = [
            col for col in OPTION1_ACCOUNTING_COLUMNS if col not in df.columns
        ]
        raise ValueError(
            f"{csv_path} has an incomplete Option-1 accounting schema: "
            f"{missing_accounting}"
        )
    if require_option1_accounting and not all(accounting_present):
        raise ValueError(
            f"{csv_path} is configured for Option-1 accounting but is missing "
            f"{list(OPTION1_ACCOUNTING_COLUMNS)}"
        )
    if all(accounting_present):
        required.extend(OPTION1_ACCOUNTING_COLUMNS)
    halt_present = SYNTHETIC_HALT_COLUMN in df.columns
    if halt_present:
        # Mixing a masked split with an unmasked history leaves NaN in the
        # concatenated column; the finiteness check below rejects it loudly.
        required.append(SYNTHETIC_HALT_COLUMN)
    _validate_panel(df, required, csv_path)
    if halt_present and not df[SYNTHETIC_HALT_COLUMN].isin([0, 1]).all():
        raise ValueError(
            f"{csv_path} contains {SYNTHETIC_HALT_COLUMN} values outside {{0, 1}}."
        )
    if all(accounting_present):
        if (df["cum_share_factor"] <= 0).any():
            raise ValueError(f"{csv_path} contains non-positive cum_share_factor.")
        if (df["dividend_per_panel_share"] < 0).any():
            raise ValueError(
                f"{csv_path} contains negative dividend_per_panel_share."
            )
    df = _add_derived(df)
    policy_history = df[df["date"].dt.normalize() < current_min].copy()
    df = df[
        (df["date"].dt.normalize() >= current_min)
        & (df["date"].dt.normalize() <= current_max)
    ].copy()
    df.attrs["policy_history"] = policy_history
    _validate_panel(df, required, csv_path)
    return df


def build_env_config(
    df: pd.DataFrame,
    tech_features: Optional[Sequence[str]] = None,
) -> Dict:
    """Return the dict consumed by PortfolioOptimizationImpactEnv.

    Layout matches MarketDataPreparator.create_env_config so the env
    can stay agnostic to the data source.
    """
    tech = list(tech_features) if tech_features is not None else list(DEFAULT_TECH_FEATURES)
    policy_history = df.attrs.get("policy_history")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    date_list = df["date"].unique()
    tic_list = df["tic"].unique()

    price_rows: list = []
    open_rows: list = []
    high_rows: list = []
    low_rows: list = []
    quote_price_rows: list = []
    quote_open_rows: list = []
    quote_high_rows: list = []
    quote_low_rows: list = []
    share_factor_rows: list = []
    dividend_rows: list = []
    halt_rows: list = []
    tech_rows: list = []
    vol_rows: list = []
    volu_rows: list = []
    adv_rows: list = []
    halt_present = SYNTHETIC_HALT_COLUMN in df.columns
    for date in date_list:
        date_df = df[df["date"] == date]
        price_rows.append(date_df["close"].values)
        open_rows.append(date_df["open"].values)
        high_rows.append(date_df["high"].values)
        low_rows.append(date_df["low"].values)
        quote_price_rows.append(date_df.get("quote_close", date_df["close"]).values)
        quote_open_rows.append(date_df.get("quote_open", date_df["open"]).values)
        quote_high_rows.append(date_df.get("quote_high", date_df["high"]).values)
        quote_low_rows.append(date_df.get("quote_low", date_df["low"]).values)
        # Option-1 accounting channels (ADR-0003 §A1). Defaults keep pre-Option-1
        # panels loadable: a share factor of 1.0 means "panel units are legal
        # shares", and zero income means "this panel carries no dividend stream".
        share_factor_rows.append(
            date_df.get(
                "cum_share_factor", pd.Series(1.0, index=date_df.index)
            ).values
        )
        dividend_rows.append(
            date_df.get(
                "dividend_per_panel_share", pd.Series(0.0, index=date_df.index)
            ).values
        )
        if halt_present:
            halt_rows.append(date_df[SYNTHETIC_HALT_COLUMN].values)
        tech_rows.append(date_df[tech].values.flatten())
        vol_rows.append(date_df["volatility"].values)
        volu_rows.append(date_df["volume"].values)
        adv_rows.append(date_df["adv20"].values)

    config = {
        "date_list": date_list,
        "price_array": np.array(price_rows, dtype=np.float64),
        "open_array": np.array(open_rows, dtype=np.float64),
        "high_array": np.array(high_rows, dtype=np.float64),
        "low_array": np.array(low_rows, dtype=np.float64),
        "quote_price_array": np.array(quote_price_rows, dtype=np.float64),
        "quote_open_array": np.array(quote_open_rows, dtype=np.float64),
        "quote_high_array": np.array(quote_high_rows, dtype=np.float64),
        "quote_low_array": np.array(quote_low_rows, dtype=np.float64),
        "share_factor_array": np.array(share_factor_rows, dtype=np.float64),
        "dividend_array": np.array(dividend_rows, dtype=np.float64),
        "tech_array": np.array(tech_rows, dtype=np.float64),
        "volatility_array": np.array(vol_rows, dtype=np.float64),
        "volume_array": np.array(volu_rows, dtype=np.float64),
        "adv20_array": np.array(adv_rows, dtype=np.float64),
        "tic_list": tic_list,
    }
    if halt_present:
        config["synthetic_halt_array"] = np.array(halt_rows, dtype=bool)
    if isinstance(policy_history, pd.DataFrame) and not policy_history.empty:
        history = policy_history.sort_values(["date", "tic"]).reset_index(drop=True)
        history_dates = history["date"].unique()
        history_rows = []
        history_dividend_rows = []
        history_halt_rows = []
        for date in history_dates:
            date_df = history[history["date"] == date].set_index("tic")
            history_rows.append(date_df.loc[tic_list, "close"].values)
            history_dividend_rows.append(
                date_df.reindex(tic_list)
                .get("dividend_per_panel_share", pd.Series(0.0, index=tic_list))
                .values
            )
            if halt_present:
                history_halt_rows.append(
                    date_df.loc[tic_list, SYNTHETIC_HALT_COLUMN].values
                )
        config["policy_price_history_array"] = np.array(history_rows, dtype=np.float64)
        config["policy_date_history_array"] = np.array(
            history_dates, dtype="datetime64[ns]"
        )
        config["policy_dividend_history_array"] = np.array(
            history_dividend_rows, dtype=np.float64
        )
        if halt_present:
            config["policy_synthetic_halt_history_array"] = np.array(
                history_halt_rows, dtype=bool
            )
    return config


def load_env_config(
    csv_path: Path,
    tech_features: Optional[Sequence[str]] = None,
    outage_calendar: Optional[Path] = None,
    history_csv: Optional[Path | Sequence[Path]] = None,
    require_option1_accounting: bool = False,
) -> Dict:
    df = load_panel(
        csv_path,
        tech_features=tech_features,
        outage_calendar=outage_calendar,
        history_csv=history_csv,
        require_option1_accounting=require_option1_accounting,
    )
    return build_env_config(df, tech_features=tech_features)

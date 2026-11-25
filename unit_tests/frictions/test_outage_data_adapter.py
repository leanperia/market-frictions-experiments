"""Data-adapter checks for missing full-day outage sessions."""

from __future__ import annotations

import pandas as pd

from meta.data_adapters.local_panel import DEFAULT_TECH_FEATURES, load_env_config


def test_missing_full_day_outage_date_is_inserted(tmp_path):
    rows = []
    for date in ("2022-01-03", "2022-01-05"):
        for tic in ("A", "B"):
            row = {
                "date": date,
                "tic": tic,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
                "quote_open": 110.0,
                "quote_high": 111.0,
                "quote_low": 109.0,
                "quote_close": 110.0,
            }
            row.update({feature: 0.0 for feature in DEFAULT_TECH_FEATURES})
            rows.append(row)
    csv_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    outage_path = tmp_path / "outages.yaml"
    outage_path.write_text(
        "outages:\n"
        "  - date: 2022-01-04\n"
        "    nature: full_day_cancellation\n"
        "    duration_minutes: 330\n",
        encoding="utf-8",
    )

    cfg = load_env_config(csv_path, outage_calendar=outage_path)
    dates = [str(pd.Timestamp(d).date()) for d in cfg["date_list"]]
    assert dates == ["2022-01-03", "2022-01-04", "2022-01-05"]
    outage_idx = dates.index("2022-01-04")
    assert cfg["volume_array"][outage_idx].sum() == 0
    assert (cfg["quote_open_array"][outage_idx] == 110.0).all()
    assert (cfg["quote_price_array"][outage_idx] == 110.0).all()


def test_prior_split_prices_are_exposed_only_as_policy_history(tmp_path):
    def write_panel(path, dates, closes):
        rows = []
        for date, close in zip(dates, closes):
            for tic in ("A", "B"):
                row = {
                    "date": date,
                    "tic": tic,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000,
                }
                row.update({feature: 0.0 for feature in DEFAULT_TECH_FEATURES})
                rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    history_path = tmp_path / "history.csv"
    test_path = tmp_path / "test.csv"
    write_panel(history_path, ["2021-12-30", "2021-12-31"], [98.0, 99.0])
    write_panel(test_path, ["2022-01-03", "2022-01-04"], [100.0, 101.0])

    cfg = load_env_config(test_path, history_csv=history_path)
    assert cfg["price_array"].shape == (2, 2)
    assert cfg["policy_price_history_array"].shape == (2, 2)
    assert (cfg["price_array"][0] == 100.0).all()
    assert (cfg["policy_price_history_array"][-1] == 99.0).all()


def test_panel_rejects_same_count_but_different_ticker_set(tmp_path):
    rows = []
    for date, tickers in (("2024-01-02", ("A", "B")), ("2024-01-03", ("A", "C"))):
        for tic in tickers:
            row = {
                "date": date,
                "tic": tic,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
            }
            row.update({feature: 0.0 for feature in DEFAULT_TECH_FEATURES})
            rows.append(row)
    csv_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    try:
        load_env_config(csv_path)
    except ValueError as exc:
        assert "ticker set" in str(exc)
        return
    raise AssertionError("expected ticker-set validation failure")


def test_panel_rejects_non_finite_required_values(tmp_path):
    rows = []
    for date in ("2024-01-02", "2024-01-03"):
        for tic in ("A", "B"):
            row = {
                "date": date,
                "tic": tic,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
            }
            row.update({feature: 0.0 for feature in DEFAULT_TECH_FEATURES})
            rows.append(row)
    rows[0]["zopen"] = float("nan")
    csv_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    try:
        load_env_config(csv_path)
    except ValueError as exc:
        assert "non-finite" in str(exc)
        return
    raise AssertionError("expected non-finite validation failure")

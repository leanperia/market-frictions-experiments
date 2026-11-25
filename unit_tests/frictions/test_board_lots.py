"""Direction check for BoardLotRounding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta.frictions.board_lots import BoardLotRounding
from meta.frictions.pipeline import TradeContext


def _ctx(prices, trades, quoted_prices=None):
    return TradeContext(
        time=0,
        date=pd.Timestamp("2024-01-15"),
        prices=np.array(prices, dtype=np.float64),
        volume=np.array([1e7] * len(prices)),
        volatility=np.array([0.02] * len(prices)),
        holdings=np.zeros(len(prices)),
        cash=1e7,
        target_weights=np.zeros(len(prices) + 1),
        trades=np.array(trades, dtype=np.int64),
        quoted_prices=(
            np.array(quoted_prices, dtype=np.float64)
            if quoted_prices is not None
            else None
        ),
    )


def test_price_bands_round_to_expected_lot_sizes():
    layer = BoardLotRounding()

    cases = [
        (0.5, 1_057, 1_000, 57),
        (15.0, 157, 100, 57),
        (150.0, 257, 250, 7),
        (1500.0, 17, 15, 2),
    ]
    for price, trade, expected_kept, expected_blocked in cases:
        out = layer.before_trade(_ctx(prices=[price], trades=[trade]))
        assert out.trades[0] == expected_kept
        assert out.diagnostics["blocked_by_lot"][0] == expected_blocked


def test_rounds_buys_down_records_blocked():
    layer = BoardLotRounding()
    ctx = _ctx(prices=[150.0], trades=[157])  # lot=10 -> keep 150, block 7
    out = layer.before_trade(ctx)
    assert out.trades[0] == 150
    assert out.diagnostics["blocked_by_lot"][0] == 7


def test_rounds_sells_down_in_magnitude():
    layer = BoardLotRounding()
    ctx = _ctx(prices=[150.0], trades=[-257])
    out = layer.before_trade(ctx)
    assert out.trades[0] == -250
    assert out.diagnostics["blocked_by_lot"][0] == 7


def test_zero_trade_passthrough():
    layer = BoardLotRounding()
    ctx = _ctx(prices=[150.0, 1500.0], trades=[0, 0])
    out = layer.before_trade(ctx)
    np.testing.assert_array_equal(out.trades, [0, 0])
    np.testing.assert_array_equal(out.diagnostics["blocked_by_lot"], [0, 0])


def test_boundary_prices_follow_pse_academy_lot_table():
    layer = BoardLotRounding()

    cases = [
        (0.0099, 1_234_567, 1_000_000),
        (0.0100, 1_234_567, 1_200_000),
        (0.0500, 1_234_567, 1_230_000),
        (50.0000, 1_234_567, 1_234_560),
        (1000.0000, 1_234_567, 1_234_565),
    ]
    for price, trade, expected_kept in cases:
        out = layer.before_trade(_ctx(prices=[price], trades=[trade]))
        assert out.trades[0] == expected_kept


def test_quoted_price_selects_lot_band_for_adjusted_accounting_price():
    layer = BoardLotRounding()
    out = layer.before_trade(
        _ctx(prices=[42.820547], quoted_prices=[52.25], trades=[57])
    )
    assert out.trades[0] == 50
    assert out.diagnostics["blocked_by_lot"][0] == 7

"""Adapters that turn local CSV panels into env-config dicts.

The MACE env (and downstream friction pipeline) consumes a dict of
NumPy arrays (date_list, price_array, tech_array, volatility_array,
volume_array, adv20_array, tic_list). The legacy MACE pipeline builds
that dict from Yahoo Finance downloads via MarketDataPreparator;
this package builds the same shape from the long-panel CSVs that
already live under data/from_trademaster_fork/portfolio_management/.
"""

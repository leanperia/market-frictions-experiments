# Data

The market data used in the paper come from licensed sources and are **not**
included. The licences do not allow redistribution of the prices or of data
derived from them. This page describes what the code expects, so that you can
build equivalent panels from data that you are licensed to use.

## Panels

`meta/data_adapters/local_panel.py` reads one long-format CSV per split
(`train.csv`, `valid.csv`, `test.csv`), one row per trading date and ticker.
The base configurations in `configs/base/` give the expected paths under
`data/` (ignored by git).

| Column | Meaning |
|---|---|
| `date`, `tic` | Trading date and ticker |
| `open`, `high`, `low`, `close` | Split- and dividend-adjusted prices on a common share basis |
| `quote_open`, `quote_high`, `quote_low`, `quote_close` | Unadjusted exchange-quoted prices (used for price-banded rules such as PSE board lots) |
| `volume` | Traded volume |
| `zopen` … `zd_30` | Normalised price features used as the policy's technical features |
| `cum_share_factor`, `dividend_per_panel_share` | Corporate-action accounting channels |
| `is_synthetic_halt` | 1 for rows that the panel builder inserted (forward-filled), for example on exchange closures |

## Universes and splits used in the paper

| | U.S. proxy | PSE proxy |
|---|---|---|
| Stocks | 30 | 30 |
| Training | 19 Mar 2008 – 29 Dec 2017 (2,465 sessions) | 3 Mar 2008 – 29 Dec 2017 (2,397 sessions) |
| Validation | 2 Jan 2018 – 31 Dec 2019 (503) | 3 Jan 2018 – 27 Dec 2019 (486) |
| Test | 2 Jan 2020 – 26 Dec 2025 (1,505) | 2 Jan 2020 – 28 Nov 2025 (1,446) |
| Initial capital | USD 100,000 | PHP 5,500,000 |

U.S. tickers: AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DD, DIS, GS, HD,
HON, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, PG, TRV, UNH, V, VZ, WMT.

PSE tickers: AC, AEV, AGI, ALI, AP, BDO, BPI, DMC, FGEN, FLI, FPH, GLO, ICT,
JFC, JGS, MBT, MEG, MER, MWC, PCOR, PNB, RLC, SCC, SECB, SM, SMC, SMPH, TEL,
URC, VLL.

Both universes are selected with information up to the end of the sample, so
they have survivorship and look-ahead bias. The paper discusses this limitation.

## Risk-free rates

| Market | Series | Source |
|---|---|---|
| U.S. | 3-month Treasury bill, secondary market, discount basis (DTB3) | [FRED](https://fred.stlouisfed.org/series/DTB3) |
| PSE | Bangko Sentral ng Pilipinas target reverse repurchase (RRP) rate, as a step function of policy decisions | [BSP](https://www.bsp.gov.ph/) |

`tools/rf_series.py` converts annual percentages to daily decimals by dividing
by 25,200.

## Fingerprints

Each file in `configs/tuned_stage1/` records SHA-256 fingerprints of the panels
and rate files used for calibration. A rebuilt panel that matches these
fingerprints is byte-identical to the panel used in the paper.

# Notices and attribution

## Upstream code

This repository is derived from a private fork of
[FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)
(Copyright (c) 2024 AI4Finance Foundation Inc., MIT License). The `meta/`
package name and directory layout come from that project. The MIT notice in
`LICENSE` must be kept in all copies.

- `meta/env_market_impact/envs/env_portfolio_optimization_impact.py` extends the
  portfolio environment of Costa, C., & Costa, A. (2023). *POE: A General Portfolio
  Optimization Environment for FinRL.* https://doi.org/10.5753/bwaif.2023.231144
- `meta/env_market_impact/envs/impact_models.py` implements market-impact models
  from the literature:
  - Bouchaud, J.-P., Farmer, J. D., & Lillo, F. (2009). How markets slowly digest
    changes in supply and demand.
  - Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
  - Gatheral, J. (2010). No-dynamic-arbitrage and market impact.
  - Obizhaeva, A., & Wang, J. (2013). Optimal trading strategy and supply/demand
    dynamics.

The friction pipeline (`meta/frictions/`), the non-RL policies (`meta/policies/`),
the panel adapter (`meta/data_adapters/`), the metrics (`tools/`), the
configurations and the tests were written for this research.

## Market rules and assumptions

Fee schedules, settlement cycles, board lots and spreads in `meta/frictions/` and
`configs/friction/` are **modelling assumptions** with effective dates. They are
not legal or trading advice, and they are not a complete record of exchange rules.
The flat spreads (4 bps U.S., 12 bps PSE, full spread) and the impact coefficient
`Y = 0.6` are assumed values, not calibrated estimates.

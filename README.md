# market-frictions-experiments

Code, configurations and reproducibility materials for the paper:

> L. L. Peria, J. C. Crisostomo, and A. R. Valdez, "Evaluating Reinforcement
> Learning Portfolio Strategies Under Modeled Market Frictions: U.S. and
> Philippine Evidence," in *Proc. IEEE Symposium on Computational Intelligence
> for Financial Engineering and Economics (CIFEr), IEEE SSCI 2027*.

The study compares four reinforcement-learning (RL) configurations (PPO-MLP,
PPO-EIIE, SAC, TD3) with eight passive, risk-model, signal-based and momentum
strategies in a U.S. and a Philippine Stock Exchange (PSE) proxy, under a ladder
of four modelled execution scenarios (F0–F3).

> **Status: initial release (November 2026).** This repository contains the
> execution model, the portfolio environment, the non-RL strategies, the metrics,
> all experiment configurations and the complete RL hyperparameters. The
> experiment runner and the statistical-inference code will follow (see
> [Roadmap](#roadmap)). Licensed market data are not included (see
> [docs/data.md](docs/data.md)).

## Contents

| Path | Content |
|---|---|
| `meta/frictions/` | Friction pipeline: fees, spread, volume cap, market impact, settlement, board lots, execution timing, outages, diagnostics |
| `meta/frictions/markets/` | U.S. and PSE market rules with effective dates |
| `meta/env_market_impact/envs/` | Portfolio environment with market impact (Gymnasium) and impact models |
| `meta/policies/` | Non-RL strategies (UBAH, UCRP, Markowitz, HRP, ERC, Z-score, MACD, CSM-LO) and the EIIE feature extractor |
| `meta/data_adapters/` | Loader from long-format panel CSVs to the environment |
| `tools/portfolio_metrics.py`, `tools/rf_series.py` | Performance metrics and risk-free alignment |
| `tools/synthetic_friction_example.py` | Self-contained example on invented data |
| `configs/friction/` | The F0–F3 friction ladder for each market |
| `configs/algo/`, `configs/tuned_stage1/` | Base algorithm settings and the selected RL configurations with tuning provenance |
| `configs/base/` | Market data configurations (expected paths, initial capital, risk-free series) |
| `docs/hyperparameters.md` | Search spaces, all selected values, fixed settings and networks |
| `docs/data.md` | Data schema, universes, splits and sources |
| `unit_tests/` | Contract tests for the execution model and environment |

## Friction ladder

| Level | Execution assumptions |
|---|---|
| F0 | 10 bps proportional cost; same-day close execution |
| F1 | Market-specific fees; next-day-open execution; assumed full spreads of 4 bps (U.S.) and 12 bps (PSE), charged at half-spread per side |
| F2 | F1 plus a 10% cap on lagged 20-day average daily volume and square-root market impact (assumed coefficient `Y = 0.6`) |
| F3 | F2 plus cash-account settlement (PSE T+2; U.S. T+2, then T+1 from 28 May 2024) and PSE board-lot rounding |

The spreads and the impact coefficient are assumptions, not calibrated
estimates. See [NOTICE.md](NOTICE.md).

## Quick start

Python 3.10 is the tested version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # CPU only, no PyTorch
python -m pytest -q unit_tests -k 'not primary_evaluation and not calibration'
python tools/synthetic_friction_example.py
```

The synthetic example runs the friction-aware environment on invented prices
(`--market pse` or `--market nyse`) and prints, for each step, the intended and
realised weights, the costs, the blocked quantities and the final equity.

For the RL components (`meta/policies/eiie_extractor.py` and, later, the runner):

```bash
pip install -r requirements-rl.txt
```

## Roadmap

1. **Experiment runner.** One command runs one experiment cell (configuration,
   market, friction level, seed) and writes one result directory. A manifest
   lists the full grid. A local runner executes the grid on one computer. A
   cluster scheduler is an optional adapter only; no scheduler is needed.
2. **Statistical inference.** Hierarchical paired stationary bootstrap with Holm
   correction, the simulation study of family-wise error and detection rates,
   and the table and figure builders.
3. **Data preparation.** Instructions to build the panels from your own licensed
   data, checked against the fingerprints in `configs/tuned_stage1/`.
4. **`market-frictions` library.** The execution model will move to a separate,
   installable package with a stable API. This repository will then depend on a
   pinned version. The paper's state is kept at the tag `cifer-2027-camera-ready`.

## Notes on the code

- The `meta/` package name comes from the FinRL-Meta fork in which this code was
  developed. It will change when the library is extracted.
- Code comments that refer to `ADR-NNNN` refer to decision records in the private
  development repository.

## Licence and citation

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for upstream attribution
(FinRL-Meta, POE, and the market-impact literature). To cite, use
[CITATION.cff](CITATION.cff) or the reference above.

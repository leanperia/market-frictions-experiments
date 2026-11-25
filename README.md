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

> **Status (November 2026).** This repository contains the execution model, the
> portfolio environment, the non-RL strategies, the metrics, all experiment
> configurations, the complete RL hyperparameters, the experiment runner and the
> statistical inference. The simulation study and the table and figure builders
> will follow (see [Roadmap](#roadmap)). Licensed market data are not included
> (see [docs/data.md](docs/data.md)).

## Contents

| Path | Content |
|---|---|
| `meta/frictions/` | Friction pipeline: fees, spread, volume cap, market impact, settlement, board lots, execution timing, outages, diagnostics |
| `meta/frictions/markets/` | U.S. and PSE market rules with effective dates |
| `meta/env_market_impact/envs/` | Portfolio environment with market impact (Gymnasium) and impact models |
| `meta/policies/` | Non-RL strategies (UBAH, UCRP, Markowitz, HRP, ERC, Z-score, MACD, CSM-LO) and the EIIE feature extractor |
| `meta/data_adapters/` | Loader from long-format panel CSVs to the environment |
| `tools/portfolio_metrics.py`, `tools/rf_series.py` | Performance metrics and risk-free alignment |
| `tools/run_cell.py`, `tools/run_grid.py` | Experiment runner: one cell, or a grid on one computer ([docs/runner.md](docs/runner.md)) |
| `tools/check_parity.py` | Comparison of runner results with the original result files of the paper |
| `tools/inference.py`, `tools/run_inference.py` | Hierarchical paired bootstrap, Holm correction and the paper's contrast families ([docs/inference.md](docs/inference.md)) |
| `tools/check_inference_parity.py` | Comparison of the inference results with the original result files of the paper |
| `manifests/cifer2027.yaml` | The 748 cells of the paper |
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

For the RL components (`meta/policies/eiie_extractor.py` and the RL cells of the
runner):

```bash
pip install -r requirements-rl.txt
```

### Running the experiments

You need panels built from data that you are licensed to use
([docs/data.md](docs/data.md)). Put them in one directory with the layout of
`configs/base/*.yaml` (the `data/` prefix of those paths is replaced by
`--data-root`). Then:

```bash
# one cell = (market, strategy, friction level, seed) -> one result directory
python tools/run_cell.py --data-root /path/to/data --market pse --strategy ucrp --friction F3 --seed 1

# the deterministic part of the paper (76 cells), 4 at a time
python tools/run_grid.py --data-root /path/to/data --group non_rl cadence_control signal_tuned --workers 4

# the RL cells (672 cells; each trains a new policy for 300,000 steps)
python tools/run_grid.py --data-root /path/to/data --group rl --workers 4 --threads-per-cell 2
```

Results go to `runs/cells/` (ignored by git). Cells with a result are skipped.
See [docs/runner.md](docs/runner.md) for the options, the result files and how
to use a cluster scheduler (map the array index to `--index`).

The statistical inference uses the F3 cells (deterministic cells and RL seeds
1–21):

```bash
python tools/run_inference.py --data-root /path/to/data --cell-root runs/cells --workers 4
```

It writes the contrast tables of the paper (primary 32 contrasts per market,
the cadence control and the post hoc signal-tuning family) to
`runs/inference/` ([docs/inference.md](docs/inference.md)).

What is verified (with the original data, see [docs/runner.md](docs/runner.md)):
all 76 deterministic cells of the paper reproduce bit for bit, the configuration
of all 672 RL cells is identical, 27 saved RL models of the study give bit-identical
test results, and one complete RL training (PPO-MLP, PSE, F3, seed 1) reproduced
the study's model byte for byte. Bit-identical results need the same CPU
kernels and thread counts as the study; on a CPU with AVX-512, Markowitz and ERC
differ by about 1e-3 in Sharpe ratio unless you select the AVX2 kernels
(docs/runner.md). The other 671 RL trainings were not repeated. The inference
code reproduces all 20 inference result files of the paper byte for byte (all
p-values and Holm decisions of the primary, signal-tuning and dividend
families) from the study's daily portfolio values; of these inputs, the runner
regenerated the 22 deterministic F3 cells and 6 of the 168 F3 RL cells
(from saved models), all byte-identical ([docs/inference.md](docs/inference.md)).

## Roadmap

1. **Experiment runner.** Done: `tools/run_cell.py`, `tools/run_grid.py` and
   `manifests/cifer2027.yaml` ([docs/runner.md](docs/runner.md)).
2. **Statistical inference.** Done: hierarchical paired stationary bootstrap
   with Holm correction and the paper's contrast families
   (`tools/run_inference.py`, [docs/inference.md](docs/inference.md)). Still to
   come: the runner cells of the three dividend-inclusive risk models (the
   inference of that family is ready), the simulation study of family-wise
   error and detection rates, and the table and figure builders.
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

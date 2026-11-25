# Statistical inference

`tools/inference.py` contains the statistical inference of the paper and
`tools/run_inference.py` runs it on the cell results of the runner
([runner.md](runner.md)). No scheduler or cluster is necessary.

## What is computed

The estimand of a contrast is, at F3 and on the test window (2020–2025),

    Δ = Sharpe(non-RL strategy) − median over the 21 training seeds of Sharpe(RL seed)

with risk-free-adjusted daily returns (the rate of `configs/base/<market>.yaml`)
and annualisation by √252. A positive Δ favours the non-RL strategy.

Each bootstrap draw resamples the 21 RL seeds with replacement and applies one
stationary-bootstrap date sequence (mean block length 21 trading days) jointly
to the non-RL path and to every sampled RL path. With B = 10,000 draws the
two-sided, null-centred p-value is

    p = (1 + #{b : |Δ*_b − Δ| ≥ |Δ|}) / (B + 1)

The 95% interval in the output is the percentile interval of the draws
(descriptive). Holm's correction is applied separately within each market at
α = 0.05. A contrast whose inputs are missing or invalid enters its family at
p = 1, so that the family size does not change.

| Family | Contrasts per market | Role in the paper |
|---|---:|---|
| `primary` | 32: UBAH, UCRP, ERC, HRP, Markowitz, Z-score, MACD, CSM-LO × PPO-MLP, PPO-EIIE, SAC, TD3 | primary analysis (Section III) |
| `signal_tuned` | 40: `primary` + tuned Z-score and tuned MACD | post hoc signal tuning |
| `dividend` | 52: `signal_tuned` + dividend-inclusive Markowitz, HRP, ERC | post hoc dividend-inclusive estimation |
| cadence control | 1: CSM-LO vs monthly UCRP | secondary, separately declared |

Robustness checks, as in the study:

- Block length: every family is computed again with mean block lengths 10 and
  42. A decision that changes is flagged (`flips_vs_primary`,
  `block_length_sensitive`).
- Monte-Carlo borderline rule: if any Holm-adjusted p of the block-21 family
  is in [0.03, 0.07], the family is computed again with B = 100,000 and that
  result is reported. (It did not fire in the paper.) In `primary` the
  block-length check then also uses B = 100,000; in the post hoc families it
  uses B = 10,000, as in the respective study scripts.
- Cadence control: Ledoit–Wolf studentised stationary-bootstrap test of equal
  Sharpe ratios (HAC standard error, B = 1,000, block length 21).

### Random numbers

All seeds are fixed integers, so the results are exactly reproducible:

| Draws | Seed |
|---|---|
| RL-seed resample of RL configuration k (0 = PPO-MLP … 3 = TD3), shared by its contrasts | 400000 + 1000k (U.S.), 500000 + 1000k (PSE) |
| Date resample of the contrast of non-RL strategy j (position in the family) and RL configuration k | 100k + j (U.S.), 10000 + 100k + j (PSE) |
| Median-Sharpe interval of RL configuration k | 1000 + k (U.S.), 1100 + k (PSE) |
| Cadence control | 11 |

Because the larger families append strategies to the smaller ones, a contrast
has the same seeds and therefore the same raw p-value in every family; only the
Holm adjustment differs. `run_inference.py` computes each shared contrast once.
The cadence seed is `i·13 + j`, where i < j are the positions of CSM-LO and
monthly UCRP in the sorted list of the 13 F3 strategies (in the study this test
was one entry of an all-pairs table).

## Running it

```bash
# 1. the cells (see runner.md): non-RL, cadence control, tuned signal rules, RL at F3
python tools/run_grid.py --data-root DATA --group non_rl cadence_control signal_tuned --friction F3 --workers 4
python tools/run_grid.py --data-root DATA --group rl --friction F3 --workers 4 --threads-per-cell 2
# 2. the inference
python tools/run_inference.py --data-root DATA --cell-root runs/cells --output-root runs/inference --workers 4
```

| Argument | Meaning |
|---|---|
| `--data-root` | Data directory, as for `run_cell.py` (risk-free rates and test calendar) |
| `--cell-root` | Cell results (default `runs/cells`) |
| `--output-root` | Output directory (default `runs/inference`) |
| `--family` | `primary`, `signal_tuned`, `dividend` (default `primary signal_tuned`; `signal_tuned` also runs `primary`) |
| `--workers` | Parallel processes for the bootstrap; the results do not depend on it |
| `--n-bootstrap` | Number of draws (10,000 in the paper; other values do not reproduce the paper) |
| `--manifest` | Manifest that names the market base configurations |

The inference needs, per market, the F3 cells of the 8 non-RL strategies, of
`ucrp_monthly`, of the two tuned signal rules (for `signal_tuned`) and of the 4
RL configurations with seeds 1–21 (168 RL cells for both markets). The
`dividend` family also needs `<market>__{markowitz,hrp,erc}_tr__F3__seed1`; the
runner does not produce these dividend-inclusive cells yet, so this family
can only be run with result directories from elsewhere. With B = 10,000 one
contrast takes about 5 seconds on one core; `primary` + `signal_tuned` are 240
contrasts (about 4 minutes with 6 workers).

### Output files

Under `<output-root>/<family>/`, with the names, columns and number formats of
the study's result files:

| Family | Files |
|---|---|
| `primary` | `headline_non_rl_vs_typical_rl.csv` (one row per contrast), `headline_rl_seed_statistics.csv` (per RL configuration: seed Sharpe ratios, median and its interval, IQM, mean), `headline_block_length_sensitivity.csv`, `secondary_cadence_control.csv` |
| `signal_tuned`, `dividend` | `contrasts_block{10,21,42}_B10000.csv`, `seed_statistics_block{10,21,42}_B10000.csv`, `headline_non_rl_vs_typical_rl.csv`, `headline_block_length_sensitivity.csv` |

Main columns of a contrast row: `non_rl_sharpe_rf`, `rl_median_sharpe_rf`,
`delta_non_rl_minus_typical_rl`, `delta_ci_low_95`, `delta_ci_high_95`,
`pvalue_primary_uncorrected`, `p_holm`, `significant_family_adjusted`,
`degenerate_contrast` (and its reason), the two random seeds, the number of RL
seeds above the non-RL Sharpe with a Wilson interval, and the date support.
The `primary` files print floats with 8 decimals (6 in the sensitivity file);
the post hoc files print every float in full.

Not ported: the supplementary diagnostics of the study that the paper does not
use for its claims (per-cell ensemble statistics with BCa intervals,
probabilistic and deflated Sharpe ratios, Calmar ratios, the all-pairs
ensemble table other than the cadence control) and the simulation study of
family-wise error and detection rates.

## Parity with the paper

`tools/check_inference_parity.py` assembles the inputs, runs the inference and
compares every output file with the study's file. It needs the licensed data
and the study's result directories, which are not public.

```bash
python tools/check_inference_parity.py --data-root DATA \
    --reference-root REFERENCE_RUNS --reference-daily-root REFERENCE_DAILY_RUNS \
    --cell-root runs/cells --output-root runs/inference-parity --workers 6
```

For each cell it uses the runner's result in `--cell-root` if there is one
(and compares it with the study's `summary.json` and, byte for byte, with the
study's daily file), else the study's daily file from `--reference-daily-root`.
Rows whose inputs are not all available are reported as not checkable.

### Results (November 2026)

Computer and software as in [runner.md](runner.md) (AVX2 kernels selected).
Inputs: 196 F3 cells. 28 came from the runner — the 22 deterministic F3 cells
and 6 RL cells evaluated from the study's saved models (one seed each of
PPO-MLP U.S., PPO-EIIE PSE, SAC and TD3 in both markets) — and all 28 have a
`summary.json` and a daily file byte-identical to the study's. The other 162
RL cells and the 6 dividend-inclusive cells are the study's daily files.

| Family | Files | Rows | Rows identical | p-values identical | Files byte-identical |
|---|---:|---:|---:|---:|---:|
| `primary` (incl. sensitivity and cadence control) | 4 | 202 | 202 | 260/260 | 4/4 |
| `signal_tuned` | 8 | 504 | 504 | 960/960 | 8/8 |
| `dividend` | 8 | 648 | 648 | 1,248/1,248 | 8/8 |

All 20 output files are byte-identical to the study's files: every contrast,
raw and Holm-adjusted p-value, interval, seed statistic, block-length check and
the cadence control. The post hoc files print floats in full, so their values
are bit-identical. The inference took 2.5 minutes with 6 workers.

In addition, the study's original functions, run separately on the same
inputs, gave the same results as this code: all 312 distinct contrasts (52 per
market at each of the three block lengths) bit-identical, all 744 Holm-adjusted
p-values and decisions of the three families identical (the `primary` ones to
the printed precision), the 24 seed-statistics rows and the cadence control
identical.

What is not verified: the runner produced the daily paths of only 6 of the 168
F3 RL cells (all byte-identical to the study's). The other 162 F3 RL
evaluations are expected to be identical as well (all 27 seed-1 models checked
in [runner.md](runner.md) and these 6 were), but they were not repeated. A
complete repetition of the inference from runner outputs alone needs the 168
RL evaluations (about 3.5 minutes each on one core from the saved models, or a
full training of each policy).

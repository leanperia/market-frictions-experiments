# Experiment runner

The runner has three commands. None of them needs a cluster or a scheduler.

| Command | Purpose |
|---|---|
| `tools/run_cell.py` | Run one cell and write one result directory |
| `tools/run_grid.py` | Run many cells of a manifest on one computer |
| `tools/check_parity.py` | Compare results with the reference results of the paper (needs the original result files) |

The statistical inference on the cell results is described in
[inference.md](inference.md).

## Cells and the manifest

A cell is one combination of market, strategy, friction level and seed.
`manifests/cifer2027.yaml` lists the cells of the paper (748 cells):

| Group | Cells | Content |
|---|---:|---|
| `non_rl` | 64 | UBAH, UCRP, Markowitz, HRP, ERC, Z-score, MACD, CSM-LO × 2 markets × F0–F3, seed 1 |
| `cadence_control` | 8 | UCRP rebalanced monthly × 2 markets × F0–F3, seed 1 |
| `signal_tuned` | 4 | Z-score and MACD with validation-selected parameters × 2 markets, F3, seed 1 |
| `rl` | 672 | PPO-MLP, PPO-EIIE, SAC, TD3 × 2 markets × F0–F3 × seeds 1–21 |

The configuration of a cell is the merge, in this order, of the market base file
(`configs/base/`), the friction level (`configs/friction/`), the strategy
(`configs/algo/`) and, for RL, the selected hyperparameters
(`configs/tuned_stage1/<algo>__<market>.yaml`). Later files override earlier
ones key by key. For the tuned signal rules the manifest gives the parameters,
which replace `algo.params`. The RL hyperparameters were selected under F3 and
are used unchanged at every friction level; each RL cell trains a new policy.

## One cell

```bash
python tools/run_cell.py --data-root /path/to/data --output-root runs/cells \
    --market pse --strategy ucrp --friction F3 --seed 1
python tools/run_cell.py --data-root /path/to/data --cell nyse__sac__F2__seed4 --threads 2
python tools/run_cell.py --data-root /path/to/data --index 100     # position in the manifest
```

Paths in the configuration files are written as `data/<path>`; the runner reads
`<data-root>/<path>`. See [data.md](data.md) for the expected files.

The result directory `<output-root>/<market>__<strategy>__<friction>__seed<n>/`
contains `merged_config.yaml`, `summary.json` and `backtest/portfolio_values.csv`
(daily portfolio value and execution diagnostics), `backtest/trades.csv`, and
for RL cells `checkpoint/model.zip`. `--asset-diagnostics` adds the per-asset,
per-day file. `summary.json["backtest"]` has the same metric names and
definitions as the result files of the paper (`tools/portfolio_metrics.py`),
for example `sharpe_ratio_rf` (risk-free-adjusted Sharpe ratio, used in the
paper), `sharpe_ratio`, `annual_volatility`, `total_return`, `annual_return`,
`max_drawdown` and `total_turnover`. An undefined metric (for example the Sharpe
ratio of a strategy that stays in cash) is written as `null` and listed under
`degenerate_metrics`.

What a cell does:

- Non-RL: build the strategy, build the environment on the test split with the
  training and validation splits as price history, and step through the test
  split.
- RL: train a new model on the training split for `training.total_timesteps`
  (300,000) steps, save it, then step deterministically through the test split.
  `--total-timesteps` changes the budget (for a quick check); `--model` evaluates
  a saved model without training.

If `summary.json` exists with the same configuration hash, the cell is skipped.
If the configuration changed, the runner stops; `--force` replaces the result.
`--split valid` evaluates on the validation split instead (used to select the
signal-rule parameters).

## A grid on one computer

```bash
python tools/run_grid.py --list                                # all cells with their index
python tools/run_grid.py --data-root /path/to/data --group non_rl cadence_control signal_tuned --workers 4
python tools/run_grid.py --data-root /path/to/data --strategy ppo_mlp --market pse \
    --friction F3 --seed 1 2 3 --threads-per-cell 2
```

Each cell runs in its own process; `--workers` sets how many run at the same
time. `--threads-per-cell` sets the BLAS and PyTorch threads of each cell (the
paper used 1 for non-RL cells and 2 for RL cells). Memory use is about 1.5 GB per
cell. A non-RL cell takes about one minute; on the CPUs of the study an RL cell
took about 12–20 minutes (PPO) or 1.2–2 hours (SAC, TD3).

## On a cluster

No scheduler code is necessary. Map the array index of the scheduler to
`--index`. The order of the cells is fixed by the manifest
(`python tools/run_grid.py --list`). For example, with Slurm:

```bash
#SBATCH --array=76-747          # the RL cells
#SBATCH --cpus-per-task=2
python tools/run_cell.py --data-root "$DATA" --output-root "$OUT" \
    --index "$SLURM_ARRAY_TASK_ID" --threads 2
```

## Parity with the paper

`tools/check_parity.py` runs every deterministic cell (76) and compares
`summary.json["backtest"]` and the effective configuration with the original
result files of the study. It also checks the effective configuration of the 672
RL cells, and with `--rl-checkpoint-seeds` it evaluates the saved RL models of
the study with this runner. It needs the licensed data and the original result
directories, which are not public.

```bash
python tools/check_parity.py --data-root DATA --reference-root REFERENCE_RUNS \
    --output-root runs/parity --workers 4 [--rl-checkpoint-seeds 1]
```

### Results of the parity check

On 25 November 2026 the repository at the tag `cifer-2027-camera-ready` was
cloned fresh on a compute node of the original cluster (AMD EPYC 7513), with a
new environment installed from `requirements.txt` only and no environment
variables set. All 76 deterministic cells reproduced the study's `summary.json`
exactly (29 of 29 metrics each), and all 672 RL cells had the study's
effective configuration. The input panels matched the SHA-256 fingerprints in
`configs/tuned_stage1/`.

### Floating-point results depend on the CPU

The reference results were computed on AMD EPYC 7513 processors, which have
AVX2 but not AVX-512. On a CPU with AVX-512, NumPy, OpenBLAS and PyTorch select
different machine-code kernels. These kernels round and sum in a different
order, so some numbers differ in the last bits:

- Most non-RL strategies: the metrics differ by at most about 1e-14 (most often
  only `beta`, `jensen_alpha` and `treynor_ratio`, which use `numpy.cov`).
- Markowitz and ERC: these strategies solve a numerical optimisation every day.
  A last-bit difference in the inputs can change the solution enough to change
  the trades, so the results differ by about 1e-3 in the risk-free-adjusted
  Sharpe ratio (largest difference 0.007).
- RL: training is a long chain of floating-point operations, so a last-bit
  difference can change the trained policy.

Both NumPy and OpenBLAS must use the AVX2 kernels; one of them alone is not
sufficient. To select the AVX2 kernels on an AVX-512 CPU, set these variables
before you start Python (they have no effect on a CPU without AVX-512):

```bash
export OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
```

The thread count also changes the order of the sums: use 1 thread for non-RL
cells and 2 for RL cells (`--threads-per-cell`, `--threads`) to repeat the
study bit for bit.

### Results of the parity check

Computer: AMD Ryzen 7 7700 (AVX-512), Python 3.10.20, NumPy 2.2.6, pandas 2.2.3,
SciPy 1.15.3, scikit-learn 1.7.2, PyTorch 2.12.0, Stable-Baselines3 2.8.0
(November 2026).

Reference: the result files of the study (`summary.json` and
`merged_config.yaml`; daily portfolio values were kept only for the four tuned
signal-rule cells). Tolerance 1e-12.

| Check | AVX2 kernels (variables above) | Default kernels (AVX-512) |
|---|---|---|
| Deterministic cells, all 29 metrics bit-identical | 76/76 | 13/76 |
| Deterministic cells, all metrics within 1e-12 | 76/76 | 60/76 (largest difference 7.5e-15) |
| Deterministic cells outside the tolerance | 0 | 16: all Markowitz and ERC cells (largest difference in `sharpe_ratio_rf`: 0.007) |
| Daily portfolio values identical (all columns) | 4/4 | 0/4 (differences of 6e-14 or less) |
| Effective configuration (hash and every key) | 76/76 deterministic, 672/672 RL | same |
| Saved RL models of the study (seed 1), test metrics bit-identical | 27/27 | not run |

Five of the 32 seed-1 RL cells have no saved model in the reference files
(TD3), so 27 were evaluated.

One full RL training was repeated: PPO-MLP, PSE, F3, seed 1, 300,000 steps,
CPU, 2 threads, AVX2 kernels. The trained policy and optimizer state are
byte-identical to the study's model, and all 29 test metrics are
bit-identical. The training took 33 minutes on a fully loaded computer (the
study: 12 minutes). Only one RL training was repeated in full; the other RL
cells were checked with a budget of 2,048 steps (all eight configurations run).

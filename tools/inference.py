"""Statistical inference of the CIFEr 2027 paper.

The primary estimand of a contrast is

    Delta = Sharpe(non-RL strategy) - median over training seeds of Sharpe(RL seed)

with risk-free-adjusted (excess) daily returns on the F3 test window and
annualisation by sqrt(252). Uncertainty comes from a hierarchical paired
stationary bootstrap: each draw resamples the 21 RL seeds with replacement and
applies one stationary-bootstrap date sequence (Politis-Romano, mean block
length 21) jointly to the non-RL path and to every sampled RL path. The p-value
is null-centred, two-sided and plus-one corrected:

    p = (1 + #{b : |Delta*_b - Delta| >= |Delta|}) / (B + 1)

Holm's step-down correction is applied separately within each market.

Families (``FAMILIES``):

- ``primary``: 8 non-RL strategies x 4 RL configurations = 32 contrasts per
  market (the paper's primary analysis), with block-length sensitivity (10, 42)
  and the Monte-Carlo borderline rule (rerun at B = 100,000 if any Holm p is in
  [0.03, 0.07]).
- ``signal_tuned``: the primary 32 plus the two validation-tuned signal rules
  = 40 contrasts per market (post hoc).
- ``dividend``: the 40 plus the three dividend-inclusive risk-model variants =
  52 contrasts per market (post hoc).
- Secondary cadence control: CSM-LO versus monthly UCRP, one test per market,
  with the Ledoit-Wolf studentised stationary bootstrap (B = 1,000).

Every random-number seed is a fixed integer derived from the position of the
strategy in the family (``seed_resample_rng_seed``, ``date_resample_rng_seed``
columns), exactly as in the study, so that the p-values reproduce bit for bit
from the same daily portfolio values. The contrasts that the families share
(the 32 primary contrasts in all three families) therefore have identical raw
p-values in every family; only the Holm adjustment differs.

The code is a port of the study's analysis scripts in the private development
repository. Each function names its origin. Only the statistical functions and
the family definitions were ported; the provenance and scheduling machinery of
the study (campaign markers, release gates, cluster jobs) was not.
"""

from __future__ import annotations

import io
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from tools.rf_series import align_rf as _align_rf
from tools.rf_series import load_rf_series

# ── constants (origin: tools/stage3_statistical_inference.py) ───────────────

ANNUALISATION = {"nyse": 252, "pse": 252}
MARKETS = ("nyse", "pse")
HEADLINE_FRICTION = {"nyse": "F3", "pse": "F3"}
BLOCK_LENGTH = 21  # mean block length of the primary analysis (about one month)
BLOCK_LENGTH_SENSITIVITY = (10, 21, 42)
N_BOOTSTRAP = 10_000
# Monte-Carlo borderline rule: if any Holm-adjusted primary p is in [LOW, HIGH],
# all tests of the family are run again with MC_BORDERLINE_B draws.
MC_BORDERLINE_LOW = 0.03
MC_BORDERLINE_HIGH = 0.07
MC_BORDERLINE_B = 100_000
N_PAIRWISE_BOOTSTRAP = 1_000
EPS = 1e-12
ALPHA = 0.05

RL_ALGORITHMS = ("ppo_mlp", "ppo_eiie", "sac", "td3")
PRIMARY_NON_RL = ("ubah", "ucrp", "erc", "hrp", "markowitz", "zscore", "macd", "csm_lo")
EVALUATION_SEEDS = tuple(range(1, 22))


@dataclass(frozen=True)
class FamilyDesign:
    """One Holm family: the cross product of non-RL and RL strategies per market.

    The order of the tuples is part of the design: the bootstrap seeds of a
    contrast are derived from the positions of its two strategies.
    (origin: ``HeadlineDesign`` in tools/campaign_profile.py)
    """

    name: str
    non_rl_algorithms: Tuple[str, ...]
    rl_algorithms: Tuple[str, ...] = RL_ALGORITHMS
    evaluation_seeds: Tuple[int, ...] = EVALUATION_SEEDS
    post_hoc: bool = False
    description: str = field(default="", compare=False)

    @property
    def expected_contrasts_per_market(self) -> int:
        return len(self.non_rl_algorithms) * len(self.rl_algorithms)

    @property
    def family_label(self) -> str:
        return f"holm_within_market_{self.expected_contrasts_per_market}"


PRIMARY = FamilyDesign(
    "primary", PRIMARY_NON_RL,
    description="paper Section III: 8 non-RL x 4 RL contrasts per market",
)
# origin: NON_RL in tools/stage3_signal_tuned_inference.py
SIGNAL_TUNED = FamilyDesign(
    "signal_tuned", PRIMARY_NON_RL + ("zscore_tuned", "macd_tuned"), post_hoc=True,
    description="post hoc: primary family plus the two validation-tuned signal rules",
)
# origin: NON_RL in tools/cifer_followup_inference.py
DIVIDEND = FamilyDesign(
    "dividend", SIGNAL_TUNED.non_rl_algorithms + ("markowitz_tr", "hrp_tr", "erc_tr"), post_hoc=True,
    description="post hoc: signal-tuned family plus three dividend-inclusive risk models",
)
FAMILIES: Dict[str, FamilyDesign] = {d.name: d for d in (PRIMARY, SIGNAL_TUNED, DIVIDEND)}

# Secondary, separately declared cadence control: (strategy, control) per market.
CADENCE_CONTROLS: Tuple[Tuple[str, str], ...] = (("csm_lo", "ucrp_monthly"),)
# In the study the cadence test was one entry of an all-pairs test table over
# the 13 F3 strategies, and its random seed was ``i * 13 + j`` for the sorted
# positions i < j of the two strategies. The list is fixed here so that the
# seed does not depend on which other cells happen to be present.
PAIRWISE_ALGORITHMS: Tuple[str, ...] = tuple(sorted(PRIMARY_NON_RL + RL_ALGORITHMS + ("ucrp_monthly",)))


# ── bootstrap (origin: tools/stage3_statistical_inference.py) ────────────────


def sharpe_from_returns(returns: np.ndarray, ann: int) -> float:
    """Annualised Sharpe ratio of a 1-D daily return array (origin: _sharpe_from_returns)."""
    if len(returns) < 2:
        return float("nan")
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma == 0:
        return float("nan")
    return (mu / sigma) * math.sqrt(ann)


def vectorized_sharpes(boot_matrix: np.ndarray, ann: int) -> np.ndarray:
    """Annualised Sharpe ratio of each row (origin: _vectorized_sharpes)."""
    mu = boot_matrix.mean(axis=1)
    sig = boot_matrix.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(sig > 0, (mu / sig) * math.sqrt(ann), np.nan)
    return sh


def stationary_bootstrap_indices(
    n: int, block_length: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Stationary-bootstrap date indices, shape ``(n_boot, n)``.

    A new block starts with probability ``1 / block_length``; otherwise the
    next date follows the previous one (wrapping around).
    (origin: _stationary_bootstrap_indices; the order of the random draws is
    part of the reproducibility contract)
    """
    indices = np.empty((n_boot, n), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n, size=n_boot)
    restart = rng.random((n_boot, n - 1)) < (1.0 / block_length)
    starts = rng.integers(0, n, size=(n_boot, n - 1))
    for i in range(1, n):
        indices[:, i] = np.where(
            restart[:, i - 1], starts[:, i - 1], (indices[:, i - 1] + 1) % n
        )
    return indices


def bootstrap_median_ci(
    values: np.ndarray, *, n_boot: int = N_BOOTSTRAP, alpha: float = ALPHA, seed: int = 0
) -> dict:
    """Seed-resampling percentile interval for a median (origin: bootstrap_median_ci)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    observed = float(np.median(values))
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
    medians = np.median(sampled, axis=1)
    ci_low, ci_high = np.quantile(medians, [alpha / 2, 1 - alpha / 2])
    return {"median": observed, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def wilson_interval(successes: int, n: int, alpha: float = ALPHA) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (origin: _wilson_interval)."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _degenerate_contrast() -> dict:
    return {
        "non_rl_sharpe": float("nan"),
        "rl_median_sharpe": float("nan"),
        "delta": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "prob_delta_positive": float("nan"),
        "pvalue_uncorrected": float("nan"),
        "n_boot_effective": 0,
    }


def hierarchical_paired_bootstrap_delta(
    non_rl_returns: np.ndarray,
    rl_returns: np.ndarray,
    ann: int,
    *,
    block_length: int = BLOCK_LENGTH,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = 0,
    chunk_size: int = 250,
    shared_seed_indices: Optional[np.ndarray] = None,
) -> dict:
    """Non-RL Sharpe minus median RL-seed Sharpe, with a paired hierarchical bootstrap.

    ``rl_returns`` has one row per RL seed and the same dates as
    ``non_rl_returns``. Each draw resamples the seeds (``shared_seed_indices``,
    shape ``(n_boot, n_seeds)``, or drawn from ``seed``) and applies one
    stationary date sequence (drawn from ``seed``) to all paths. Draws are made
    in chunks of ``chunk_size``; the chunking is part of the random-number
    scheme. ``pvalue_uncorrected`` is NaN for a degenerate contrast (too few
    dates, non-finite Sharpe ratios, or any non-finite bootstrap draw).
    (origin: hierarchical_paired_bootstrap_delta)
    """
    non_rl_returns = np.asarray(non_rl_returns, dtype=float)
    rl_returns = np.asarray(rl_returns, dtype=float)
    if rl_returns.ndim == 1:
        rl_returns = rl_returns.reshape(1, -1)
    if rl_returns.ndim != 2 or rl_returns.shape[1] != len(non_rl_returns):
        raise ValueError("RL and non-RL return paths must share aligned dates")
    degenerate = _degenerate_contrast()
    if len(non_rl_returns) < 10 or rl_returns.shape[0] == 0:
        return dict(degenerate)

    non_rl_sharpe = sharpe_from_returns(non_rl_returns, ann)
    rl_sharpes = vectorized_sharpes(rl_returns, ann)
    rl_sharpes = rl_sharpes[np.isfinite(rl_sharpes)]
    if not np.isfinite(non_rl_sharpe) or rl_sharpes.size == 0:
        return dict(degenerate, non_rl_sharpe=non_rl_sharpe)

    rl_median_sharpe = float(np.median(rl_sharpes))
    observed = float(non_rl_sharpe - rl_median_sharpe)
    rng = np.random.default_rng(seed)
    n_seeds, n_dates = rl_returns.shape
    if shared_seed_indices is not None:
        shared_seed_indices = np.asarray(shared_seed_indices)
        if shared_seed_indices.shape != (n_boot, n_seeds):
            raise ValueError(
                f"shared_seed_indices shape {shared_seed_indices.shape} != "
                f"expected ({n_boot}, {n_seeds})"
            )
    boot_deltas = []
    for start in range(0, n_boot, chunk_size):
        draw_count = min(chunk_size, n_boot - start)
        date_indices = stationary_bootstrap_indices(n_dates, block_length, draw_count, rng)
        if shared_seed_indices is not None:
            seed_indices = shared_seed_indices[start : start + draw_count]
        else:
            seed_indices = rng.integers(0, n_seeds, size=(draw_count, n_seeds))
        non_rl_sharpes = vectorized_sharpes(non_rl_returns[date_indices], ann)
        sampled_rl_sharpes = np.empty((draw_count, n_seeds), dtype=float)
        for slot in range(n_seeds):
            sampled_paths = rl_returns[seed_indices[:, slot]]
            sampled_paths = np.take_along_axis(sampled_paths, date_indices, axis=1)
            sampled_rl_sharpes[:, slot] = vectorized_sharpes(sampled_paths, ann)
        delta = non_rl_sharpes - np.nanmedian(sampled_rl_sharpes, axis=1)
        boot_deltas.append(delta[np.isfinite(delta)])

    bootstrap = np.concatenate(boot_deltas) if boot_deltas else np.array([], dtype=float)
    # Valid only if every declared draw gives a finite statistic.
    if n_boot <= 0 or len(bootstrap) != n_boot:
        return dict(
            degenerate,
            non_rl_sharpe=non_rl_sharpe,
            rl_median_sharpe=rl_median_sharpe,
            delta=observed,
        )
    ci_low, ci_high = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    centred_extreme = np.abs(bootstrap - observed) >= abs(observed)
    pvalue = float((1 + int(np.sum(centred_extreme))) / (len(bootstrap) + 1))
    return {
        "non_rl_sharpe": non_rl_sharpe,
        "rl_median_sharpe": rl_median_sharpe,
        "delta": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_delta_positive": float(np.mean(bootstrap > 0)),
        "pvalue_uncorrected": pvalue,
        "n_boot_effective": int(len(bootstrap)),
    }


# ── Ledoit-Wolf HAC pairwise test (origin: tools/stage3_statistical_inference.py) ──


def _newey_west_lrv(values: np.ndarray, lag: Optional[int] = None) -> float:
    """Newey-West long-run variance of a series (origin: _newey_west_lrv)."""
    n = len(values)
    values = np.asarray(values, dtype=float) - float(np.mean(values))
    if lag is None:
        lag = max(1, int(4 * (n / 100) ** (2 / 9)))
    lrv = float(np.mean(values * values))
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1 - k / (lag + 1)
        lrv += 2 * weight * float(np.mean(values[k:] * values[:-k]))
    return max(lrv, 0.0)


def _sharpe_influence(returns: np.ndarray) -> Optional[np.ndarray]:
    """Influence function of the unannualised Sharpe ratio (origin: _sharpe_influence)."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=0))
    if sigma <= 0:
        return None
    centered = returns - mu
    return centered / sigma - (mu / (2 * sigma**3)) * (centered**2 - sigma**2)


def _hac_sharpe_diff_se(ret_a: np.ndarray, ret_b: np.ndarray, ann: int) -> float:
    """(origin: _hac_sharpe_diff_se)"""
    influence_a = _sharpe_influence(ret_a)
    influence_b = _sharpe_influence(ret_b)
    if influence_a is None or influence_b is None:
        return float("nan")
    influence = influence_a - influence_b
    return float(math.sqrt(_newey_west_lrv(influence) / len(ret_a)) * math.sqrt(ann))


def lw_sharpe_diff_test(
    ret_a: np.ndarray,
    ret_b: np.ndarray,
    ann: int,
    *,
    block_length: int = BLOCK_LENGTH,
    n_boot: int = N_PAIRWISE_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Ledoit-Wolf studentised stationary-bootstrap test of equal Sharpe ratios.

    Inputs are date-aligned paired returns. (origin: lw_sharpe_diff_test)
    """
    n = min(len(ret_a), len(ret_b))
    if n < 10:
        return {"diff": float("nan"), "pvalue": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "significant_0.05": False}
    a = np.asarray(ret_a[:n], dtype=float)
    b = np.asarray(ret_b[:n], dtype=float)
    observed = float(sharpe_from_returns(a, ann) - sharpe_from_returns(b, ann))
    observed_se = _hac_sharpe_diff_se(a, b, ann)
    if not np.isfinite(observed_se) or observed_se <= EPS:
        pvalue = 1.0 if abs(observed) <= EPS else float("nan")
        return {"diff": observed, "pvalue": pvalue, "ci_low": observed,
                "ci_high": observed, "significant_0.05": False}

    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(n, block_length, n_boot, rng)
    studentized = []
    for index in indices:
        boot_a = a[index]
        boot_b = b[index]
        boot_diff = sharpe_from_returns(boot_a, ann) - sharpe_from_returns(boot_b, ann)
        boot_se = _hac_sharpe_diff_se(boot_a, boot_b, ann)
        if np.isfinite(boot_diff) and np.isfinite(boot_se) and boot_se > EPS:
            studentized.append((boot_diff - observed) / boot_se)
    if len(studentized) < 100:
        return {"diff": observed, "pvalue": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "significant_0.05": False}
    studentized = np.asarray(studentized)
    observed_t = observed / observed_se
    pval = (1 + np.sum(np.abs(studentized) >= abs(observed_t))) / (len(studentized) + 1)
    q_low, q_high = np.quantile(studentized, [0.025, 0.975])
    return {
        "diff": observed,
        "pvalue": float(pval),
        "ci_low": float(observed - q_high * observed_se),
        "ci_high": float(observed - q_low * observed_se),
        "significant_0.05": pval < 0.05,
    }


# ── multiple comparisons (origin: tools/stage3_statistical_inference.py) ─────


def holm_bonferroni(pvalues: Sequence[float], alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjustment of one family.

    Returns ``(adjusted, reject)`` in input order. NaN p-values stay NaN, are
    not rejected and do not count in the family size. Adjusted values are
    capped at 1 and made monotone along the sorted order; ``reject`` is
    ``adjusted <= alpha`` (as statsmodels ``multipletests(method="holm")``).
    (origin: holm_bonferroni)
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    adjusted = np.full(n, np.nan, dtype=float)
    reject = np.zeros(n, dtype=bool)
    finite_idx = np.nonzero(np.isfinite(pvalues))[0]
    m = finite_idx.size
    if m == 0:
        return adjusted, reject
    order = finite_idx[np.argsort(pvalues[finite_idx], kind="mergesort")]
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((m - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    reject[finite_idx] = adjusted[finite_idx] <= alpha
    return adjusted, reject


# ── return paths (origin: tools/stage3_statistical_inference.py) ────────────


def load_excess_return_path(pv_path: Any, rf_series: pd.Series) -> pd.Series:
    """Daily excess returns of one ``portfolio_values.csv``, without trimming dates.

    Raises ``ValueError`` for a missing, unreadable or invalid file.
    (origin: _load_headline_return_path)
    """
    if pv_path is None or (not isinstance(pv_path, (str, Path)) and pd.isna(pv_path)):
        raise ValueError("portfolio_values.csv is missing")
    path = Path(str(pv_path))
    if not path.is_file():
        raise ValueError(f"portfolio_values.csv does not exist: {path}")
    try:
        pv = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"portfolio_values.csv is unreadable: {exc}") from exc
    missing = {"date", "portfolio_value"} - set(pv.columns)
    if missing:
        raise ValueError(f"portfolio_values.csv lacks columns {sorted(missing)}")
    if len(pv) < 2:
        raise ValueError("portfolio_values.csv has fewer than two observations")

    dates = pd.DatetimeIndex(pd.to_datetime(pv["date"], errors="coerce"))
    values = pd.to_numeric(pv["portfolio_value"], errors="coerce").to_numpy(dtype=float)
    if dates.hasnans:
        raise ValueError("portfolio_values.csv contains an invalid date")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("portfolio_values.csv dates are not unique and increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("portfolio_values.csv contains a non-finite value")
    if np.any(np.abs(values[:-1]) <= EPS):
        raise ValueError("portfolio_values.csv contains a zero return denominator")

    returns = np.diff(values) / values[:-1]
    return_dates = dates[1:]
    try:
        aligned_rf = _align_rf(rf_series, return_dates, coverage_threshold=0.9)
    except Exception as exc:
        raise ValueError(f"risk-free series cannot be aligned: {exc}") from exc
    returns = returns - np.asarray(aligned_rf, dtype=float)
    if not np.all(np.isfinite(returns)):
        raise ValueError("excess-return path contains a non-finite value")
    return pd.Series(returns, index=return_dates)


def load_ensemble_return_path(pv_paths: Sequence[Any], rf_series: Optional[pd.Series]) -> Optional[pd.Series]:
    """Equal-capital mean of the excess-return paths of a strategy's runs.

    For a deterministic strategy (one run) this is its own excess-return path.
    Raises ``ValueError`` if any declared run has no usable path (strict mode).
    (origin: get_aligned_return_paths_with_counts and get_aggregate_return_series
    with ``strict=True``)
    """
    return_paths = []
    for label, pv_path in enumerate(pv_paths):
        if pv_path is None:
            continue
        try:
            pv = pd.read_csv(pv_path)
            vals = pv["portfolio_value"].values.astype(float)
            if len(vals) < 2:
                continue
            rets = np.diff(vals) / vals[:-1]
            dates = pd.to_datetime(pv["date"]).iloc[1:]
            if rf_series is not None:
                rets = rets - _align_rf(rf_series, pd.DatetimeIndex(dates), coverage_threshold=0.9)
            return_paths.append(pd.Series(rets, index=dates, name=label))
        except Exception:
            continue
    if len(return_paths) < len(pv_paths):
        raise ValueError(
            f"declared-complete run family has {len(pv_paths)} runs but only "
            f"{len(return_paths)} loadable return paths"
        )
    if not return_paths:
        return None
    aligned = pd.concat(return_paths, axis=1, join="inner").dropna().sort_index()
    if aligned.empty:
        return None
    return aligned.mean(axis=1)


def median_seed_display_values(
    sharpe_values: np.ndarray, total_returns: np.ndarray, turnovers: np.ndarray
) -> Tuple[float, float]:
    """Total return and turnover of the observed median-Sharpe seed (display only).

    For an even number of seeds the lower-median seed is used.
    (origin: median_seed_display_values)
    """
    sharpe_values = np.asarray(sharpe_values, dtype=float)
    total_returns = np.asarray(total_returns, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)
    if not (sharpe_values.size == total_returns.size == turnovers.size):
        raise ValueError("Sharpe, return, and turnover arrays must have equal length")
    finite = np.isfinite(sharpe_values)
    if not finite.any():
        return float("nan"), float("nan")
    valid_indices = np.flatnonzero(finite)
    order = valid_indices[np.argsort(sharpe_values[finite], kind="mergesort")]
    mid = order[(len(order) - 1) // 2]
    return float(total_returns[mid]), float(turnovers[mid])


def common_support_metadata(paths: List[pd.Series], expected_dates: Optional[pd.DatetimeIndex] = None) -> dict:
    """Describe the common date support of the paths; never trims a path.

    (origin: _common_support_metadata)
    """
    expected = pd.DatetimeIndex(expected_dates) if expected_dates is not None else pd.DatetimeIndex([])
    expected_fields = {
        "expected_date_start": expected[0].isoformat() if len(expected) else None,
        "expected_date_end": expected[-1].isoformat() if len(expected) else None,
        "expected_date_count": int(len(expected)),
    }
    if not paths:
        return {"common_date_start": None, "common_date_end": None, "common_date_count": 0,
                "date_support_complete": False, **expected_fields}
    common = paths[0].index
    for path in paths[1:]:
        common = common.intersection(path.index, sort=False)
    paths_match = all(path.index.equals(paths[0].index) for path in paths[1:])
    matches_expected = bool(len(expected)) and all(path.index.equals(expected) for path in paths)
    return {
        "common_date_start": common[0].isoformat() if len(common) else None,
        "common_date_end": common[-1].isoformat() if len(common) else None,
        "common_date_count": int(len(common)),
        "date_support_complete": bool(paths_match and matches_expected),
        **expected_fields,
    }


# ── market inputs: risk-free rate and test calendar ─────────────────────────


def resolve_data_path(value: Any, data_root: Path) -> Path:
    """``data/<path>`` in a configuration -> ``<data_root>/<path>`` (as tools/run_cell.py)."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        path = Path(*path.parts[1:])
    return Path(data_root) / path


def expected_return_dates(test_csv: Path) -> pd.DatetimeIndex:
    """Return dates of the test split: its unique dates without the first one.

    (origin: expected_return_dates_from_base_config)
    """
    try:
        raw_dates = pd.read_csv(test_csv, usecols=["date"])["date"]
    except Exception as exc:
        raise ValueError(f"cannot load expected test calendar from {test_csv}: {exc}") from exc
    parsed = pd.to_datetime(raw_dates, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{test_csv} contains an invalid test date")
    dates = pd.DatetimeIndex(parsed.drop_duplicates().sort_values())
    if len(dates) < 2:
        raise ValueError(f"{test_csv} has fewer than two unique test dates")
    return dates[1:]


def load_market_inputs(
    manifest: Mapping[str, Any], data_root: Path, project_root: Path
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DatetimeIndex]]:
    """Daily risk-free series and test return calendar of each market.

    Read from the market base configuration named in the manifest.
    (origin: _daily_rf_from_base_config, expected_return_dates_from_base_config)
    """
    rf_by_market: Dict[str, pd.Series] = {}
    dates_by_market: Dict[str, pd.DatetimeIndex] = {}
    for market in MARKETS:
        cfg = yaml.safe_load((project_root / manifest["markets"][market]["base"]).read_text()) or {}
        rf = load_rf_series(cfg, resolve_fn=lambda value: resolve_data_path(value, data_root))
        if rf is None:
            raise ValueError(f"{market}: risk-free series {cfg.get('data', {}).get('rf_csv')} not loaded")
        rf_by_market[market] = rf
        dates_by_market[market] = expected_return_dates(resolve_data_path(cfg["data"]["test_csv"], data_root))
    return rf_by_market, dates_by_market


# ── cell results -> analysis input table ────────────────────────────────────

INPUT_COLUMNS = ["algo", "market", "friction", "seed", "sharpe_summary", "total_return", "turnover", "pv_path"]


def collect_inputs(
    cell_root: Path,
    algorithms: Iterable[str],
    *,
    rl_algorithms: Sequence[str] = RL_ALGORITHMS,
    seeds: Sequence[int] = EVALUATION_SEEDS,
    markets: Sequence[str] = MARKETS,
) -> pd.DataFrame:
    """One row per F3 cell result found under ``cell_root``.

    Cell directories are named ``<market>__<strategy>__F3__seed<n>`` (the
    runner's layout). RL strategies use seeds ``seeds``; deterministic
    strategies use seed 1 and get the seed label 0, as in the study. A missing
    cell is simply absent; the family builder then enters its contrasts at p = 1.
    (origin: _cell_row, load_stage2_rl_cells, load_stage1_grid_cells)
    """
    rows = []
    cell_root = Path(cell_root)
    for market in markets:
        friction = HEADLINE_FRICTION[market]
        for algo in algorithms:
            rl = algo in rl_algorithms
            for seed in (seeds if rl else (1,)):
                cell_dir = cell_root / f"{market}__{algo}__{friction}__seed{seed}"
                summary_path = cell_dir / "summary.json"
                if not summary_path.exists():
                    continue
                summary = json.loads(summary_path.read_text())
                bt = summary.get("backtest", {}) or {}
                pv_path = cell_dir / "backtest" / "portfolio_values.csv"
                rows.append({
                    "algo": algo,
                    "market": market,
                    "friction": friction,
                    "seed": int(summary.get("seed", seed)) if rl else 0,
                    "sharpe_summary": bt.get("sharpe_ratio_rf"),
                    "total_return": bt.get("total_return"),
                    "turnover": bt.get("total_turnover", bt.get("turnover")),
                    "pv_path": str(pv_path) if pv_path.exists() else None,
                })
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def csv_round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """Write a table to CSV text and read it back.

    The post hoc analyses of the study read their input table from a CSV
    file, which changes some values in the last bit (pandas' default float
    parser). The median-seed total return and turnover of those analyses
    depend on it; this reproduces it.
    """
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)))


# ── the family builder (origin: build_headline_typical_rl_comparisons) ──────

ContrastKey = Tuple[str, int, str, int, str, int, int]


def _contrast_job(args: tuple) -> Tuple[ContrastKey, dict]:
    key, non_rl, rl_matrix, ann, n_boot, block_length, date_seed, cell_seed = args
    n_rl = rl_matrix.shape[0]
    shared = np.random.default_rng(cell_seed).integers(0, n_rl, size=(n_boot, n_rl))
    result = hierarchical_paired_bootstrap_delta(
        non_rl, rl_matrix, ann, n_boot=n_boot, block_length=block_length,
        seed=date_seed, shared_seed_indices=shared,
    )
    return key, result


class ContrastCache:
    """Bootstrap results keyed by contrast, block length and B.

    The families share contrasts with identical seeds and inputs, so one
    computation serves all of them. ``workers > 1`` computes the contrasts of a
    family in parallel processes; the results do not depend on ``workers``.
    """

    def __init__(self, workers: int = 1):
        self.workers = max(1, int(workers))
        self.results: Dict[ContrastKey, dict] = {}

    def compute(self, jobs: List[tuple]) -> None:
        todo = [job for job in jobs if job[0] not in self.results]
        if not todo:
            return
        if self.workers == 1 or len(todo) == 1:
            for job in todo:
                key, result = _contrast_job(job)
                self.results[key] = result
            return
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            for key, result in pool.map(_contrast_job, todo, chunksize=1):
                self.results[key] = result


def build_family(
    inputs: pd.DataFrame,
    rf_by_market: Mapping[str, pd.Series],
    expected_dates_by_market: Mapping[str, pd.DatetimeIndex],
    design: FamilyDesign,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    block_length: int = BLOCK_LENGTH,
    alpha: float = ALPHA,
    cache: Optional[ContrastCache] = None,
    log: Callable[[str], None] = print,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Contrasts of one family (both markets) and the RL seed statistics.

    Returns ``(contrasts, seed_statistics)`` with the columns of the study's
    ``headline_non_rl_vs_typical_rl.csv`` (without the run-level columns) and
    ``headline_rl_seed_statistics.csv``. A contrast whose inputs are missing or
    invalid enters at p = 1 (``degenerate_contrast``), so the family size never
    shrinks. Holm is applied within each market.

    Random-number seeds (fixed integers, as in the study):

    - seed resample of RL configuration k: ``400000 + 1000 k`` (U.S.) or
      ``500000 + 1000 k`` (PSE), shared by all contrasts of that configuration;
    - date resample of contrast (non-RL j, RL k): ``100 k + j`` (U.S.) or
      ``10000 + 100 k + j`` (PSE);
    - median-Sharpe interval of RL configuration k: ``1000 + k`` (U.S.) or
      ``1100 + k`` (PSE).
    """
    cache = cache or ContrastCache()
    expected_rl_seeds = len(design.evaluation_seeds)
    family_size = design.expected_contrasts_per_market
    seed_rows: List[dict] = []
    pending: List[dict] = []  # contrast rows waiting for a bootstrap result
    jobs: List[tuple] = []

    for market in MARKETS:
        friction = HEADLINE_FRICTION[market]
        ann = ANNUALISATION[market]
        rf_series = rf_by_market.get(market)
        expected_dates = expected_dates_by_market.get(market)
        if {"market", "friction", "algo"}.issubset(inputs.columns):
            mkt_df = inputs[(inputs["market"] == market) & (inputs["friction"] == friction)]
        else:
            mkt_df = pd.DataFrame(columns=INPUT_COLUMNS)
        mkt_df = mkt_df.copy()
        for column in ("seed", "pv_path", "total_return", "turnover"):
            if column not in mkt_df.columns:
                mkt_df[column] = np.nan

        for rl_idx, rl_algo in enumerate(design.rl_algorithms):
            rl_group = mkt_df[mkt_df["algo"] == rl_algo]
            seed_count = len(rl_group)
            unique_seed_count = rl_group["seed"].nunique(dropna=False)
            rl_series: List[pd.Series] = []
            rl_labels: List[Any] = []
            rl_sorted_group = rl_group.sort_values("seed")
            cell_seed = (500000 if market == "pse" else 400000) + rl_idx * 1000
            median_ci_seed = 1000 + rl_idx + (0 if market == "nyse" else 100)
            rl_reason = None
            actual_seed_labels = tuple(rl_group["seed"].tolist())
            try:
                seed_labels_match = set(actual_seed_labels) == set(design.evaluation_seeds)
            except TypeError:
                seed_labels_match = False
            if expected_dates is None:
                rl_reason = "missing predeclared expected return-date calendar"
            elif rf_series is None:
                rl_reason = "missing risk-free-rate series"
            elif (seed_count != expected_rl_seeds or unique_seed_count != expected_rl_seeds
                  or not seed_labels_match):
                rl_reason = (
                    f"requires exact RL seed labels {list(design.evaluation_seeds)}; "
                    f"found {list(actual_seed_labels)} ({seed_count} rows, "
                    f"{unique_seed_count} unique)"
                )
            else:
                for _, row in rl_sorted_group.iterrows():
                    try:
                        rl_series.append(load_excess_return_path(row["pv_path"], rf_series))
                        rl_labels.append(row["seed"])
                    except ValueError as exc:
                        rl_reason = f"seed {row.get('seed')}: {exc}"
                        break
                support = common_support_metadata(rl_series, expected_dates)
                if rl_reason is None and not support["date_support_complete"]:
                    if not all(p.index.equals(rl_series[0].index) for p in rl_series[1:]):
                        rl_reason = "RL seed paths do not have identical date support"
                    else:
                        rl_reason = ("RL seed paths do not match the predeclared complete "
                                     "test return-date calendar")

            rl_support = common_support_metadata(rl_series, expected_dates)
            if len(rl_series) != expected_rl_seeds:
                rl_support = common_support_metadata([], expected_dates)
            rl_sharpes = np.array([], dtype=float)
            rl_matrix = None
            if rl_reason is None:
                rl_matrix = np.vstack([p.to_numpy() for p in rl_series])
                rl_sharpes = vectorized_sharpes(rl_matrix, ann)
                if not np.all(np.isfinite(rl_sharpes)):
                    rl_reason = "RL seed Sharpe values are non-finite"

            if rl_reason is not None:
                log(f"  [degenerate] {market} {friction} {rl_algo}: {rl_reason}; "
                    f"{len(design.non_rl_algorithms)} contrasts enter at p=1")
                seed_rows.append({
                    "market": market, "friction": friction, "rl_algo": rl_algo,
                    "n_seeds": len(rl_series), "n_seeds_declared": seed_count,
                    "rl_seed_sharpe_by_seed": "{}",
                    "rl_median_sharpe_rf_rl_window": float("nan"),
                    "rl_median_ci_low_95": float("nan"), "rl_median_ci_high_95": float("nan"),
                    "rl_iqm_sharpe_rf": float("nan"), "rl_mean_sharpe_rf": float("nan"),
                    "total_return_median_sharpe_seed": float("nan"),
                    "turnover_median_sharpe_seed": float("nan"),
                    "median_ci_rng_seed": median_ci_seed,
                    "degenerate_cell": True, "degenerate_reason": rl_reason,
                    **rl_support,
                })
            else:
                sharpe_by_seed = {
                    str(seed): float(sharpe)
                    for seed, sharpe in sorted(zip(rl_labels, rl_sharpes), key=lambda pair: pair[0])
                }
                median_ci = bootstrap_median_ci(rl_sharpes, n_boot=n_bootstrap, seed=median_ci_seed)
                median_total_return, median_turnover = median_seed_display_values(
                    rl_sharpes,
                    rl_sorted_group["total_return"].to_numpy(dtype=float),
                    rl_sorted_group["turnover"].to_numpy(dtype=float),
                )
                seed_rows.append({
                    "market": market, "friction": friction, "rl_algo": rl_algo,
                    "n_seeds": len(rl_series), "n_seeds_declared": seed_count,
                    "rl_seed_sharpe_by_seed": json.dumps(sharpe_by_seed),
                    "rl_median_sharpe_rf_rl_window": median_ci["median"],
                    "rl_median_ci_low_95": median_ci["ci_low"],
                    "rl_median_ci_high_95": median_ci["ci_high"],
                    "rl_iqm_sharpe_rf": float(stats.trim_mean(rl_sharpes, 0.25)),
                    "rl_mean_sharpe_rf": float(np.mean(rl_sharpes)),
                    "total_return_median_sharpe_seed": median_total_return,
                    "turnover_median_sharpe_seed": median_turnover,
                    "median_ci_rng_seed": median_ci_seed,
                    "degenerate_cell": False, "degenerate_reason": None,
                    **rl_support,
                })

            for non_idx, non_rl_algo in enumerate(design.non_rl_algorithms):
                date_seed = (10000 if market == "pse" else 0) + rl_idx * 100 + non_idx
                non_group = mkt_df[mkt_df["algo"] == non_rl_algo]
                reason = rl_reason
                non_path = None
                non_reason = None
                if len(non_group) != 1:
                    non_reason = f"requires one deterministic non-RL run; found {len(non_group)}"
                elif rf_series is None:
                    non_reason = "missing risk-free-rate series"
                else:
                    try:
                        non_path = load_excess_return_path(non_group.iloc[0]["pv_path"], rf_series)
                    except ValueError as exc:
                        non_reason = str(exc)
                if non_reason is not None:
                    reason = f"{reason}; {non_reason}" if reason is not None else non_reason

                support_paths = list(rl_series)
                if non_path is not None:
                    support_paths.insert(0, non_path)
                common_support = common_support_metadata(support_paths, expected_dates)
                if non_path is None or len(rl_series) != expected_rl_seeds:
                    common_support = common_support_metadata([], expected_dates)
                if reason is None and not common_support["date_support_complete"]:
                    if not all(p.index.equals(support_paths[0].index) for p in support_paths[1:]):
                        reason = "non-RL and RL paths do not have identical date support"
                    else:
                        reason = "paths do not match the predeclared complete test return-date calendar"

                key = (market, rl_idx, rl_algo, non_idx, non_rl_algo, block_length, n_bootstrap)
                if reason is None:
                    jobs.append((key, non_path.to_numpy(), rl_matrix, ann, n_bootstrap,
                                 block_length, date_seed, cell_seed))
                pending.append({
                    "key": key, "reason": reason, "rl_sharpes": rl_sharpes,
                    "n_rl_series": len(rl_series), "cell_seed": cell_seed,
                    "date_seed": date_seed, "support": common_support,
                })

    cache.compute(jobs)

    comparison_rows: List[dict] = []
    for item in pending:
        market, _, rl_algo, _, non_rl_algo, _, _ = item["key"]
        reason = item["reason"]
        rl_sharpes = item["rl_sharpes"]
        result = _degenerate_contrast()
        if reason is None:
            result = cache.results[item["key"]]
            if not np.isfinite(result.get("pvalue_uncorrected", np.nan)):
                reason = "bootstrap contrast is non-finite or degenerate"
        p_uncorrected = result.get("pvalue_uncorrected", float("nan"))
        degenerate = reason is not None or not np.isfinite(p_uncorrected)
        p_primary = 1.0 if degenerate else float(p_uncorrected)
        count_exceeding = int(np.sum(rl_sharpes > result["non_rl_sharpe"])) if not degenerate else 0
        prop_low, prop_high = wilson_interval(count_exceeding, len(rl_sharpes) if not degenerate else 0)
        if degenerate:
            log(f"  [degenerate] {market} {HEADLINE_FRICTION[market]} {non_rl_algo} vs {rl_algo}: "
                f"{reason}; contrast enters at p=1")
        comparison_rows.append({
            "market": market,
            "friction": HEADLINE_FRICTION[market],
            "non_rl_algo": non_rl_algo,
            "rl_algo": rl_algo,
            "n_rl_seeds": item["n_rl_series"],
            "non_rl_sharpe_rf": result["non_rl_sharpe"],
            "rl_median_sharpe_rf": result["rl_median_sharpe"],
            "delta_non_rl_minus_typical_rl": result["delta"],
            "delta_ci_low_95": result["ci_low"],
            "delta_ci_high_95": result["ci_high"],
            "prob_delta_positive": result["prob_delta_positive"],
            "pvalue_primary_uncorrected": p_primary,
            "degenerate_contrast": degenerate,
            "degenerate_reason": reason,
            "n_boot_effective": result.get("n_boot_effective", 0),
            "seed_resample_rng_seed": item["cell_seed"],
            "date_resample_rng_seed": item["date_seed"],
            "rl_seeds_exceeding_non_rl": count_exceeding,
            "rl_seed_exceeding_proportion": (
                count_exceeding / len(rl_sharpes) if not degenerate and len(rl_sharpes) else float("nan")
            ),
            "rl_seed_exceeding_ci_low_95": prop_low,
            "rl_seed_exceeding_ci_high_95": prop_high,
            **item["support"],
        })

    # Holm within each market's declared family only; never pooled.
    for market in MARKETS:
        family_rows = [r for r in comparison_rows if r["market"] == market]
        if len(family_rows) != family_size:
            raise AssertionError(f"{market} family has {len(family_rows)} rows, expected {family_size}")
        p_holm, reject = holm_bonferroni([r["pvalue_primary_uncorrected"] for r in family_rows], alpha=alpha)
        for k, r in enumerate(family_rows):
            r["p_holm"] = float(p_holm[k])
            r["significant_family_adjusted"] = bool(reject[k])
            r["correction_family"] = design.family_label
            r["family_size"] = family_size
    return pd.DataFrame(comparison_rows), pd.DataFrame(seed_rows)


def is_borderline(p_holm: Iterable[float]) -> bool:
    """True if any Holm p is in [0.03, 0.07] (the Monte-Carlo borderline rule)."""
    p = np.asarray(list(p_holm), dtype=float)
    return bool((np.isfinite(p) & (p >= MC_BORDERLINE_LOW) & (p <= MC_BORDERLINE_HIGH)).any())


# ── secondary cadence control (origin: run_analysis, pairwise section) ──────


def build_cadence_control(
    inputs: pd.DataFrame,
    rf_by_market: Mapping[str, pd.Series],
    *,
    comparisons: Sequence[Tuple[str, str]] = CADENCE_CONTROLS,
    pairwise_algorithms: Sequence[str] = PAIRWISE_ALGORITHMS,
    n_boot: int = N_PAIRWISE_BOOTSTRAP,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """The predeclared secondary cadence comparison(s), Holm within market.

    The test is ``lw_sharpe_diff_test`` on the date-aligned excess returns,
    with seed ``i * n + j`` for the sorted positions ``i < j`` of the two
    strategies in ``pairwise_algorithms`` (n = its length), as in the study's
    all-pairs table. A missing or invalid input raises an error: a declared
    secondary test cannot become a p = 1 row.
    """
    order = sorted(pairwise_algorithms)
    rows: List[dict] = []
    for market in MARKETS:
        friction = HEADLINE_FRICTION[market]
        ann = ANNUALISATION[market]
        mkt_df = inputs[(inputs["market"] == market) & (inputs["friction"] == friction)]
        market_rows = []
        for left, right in comparisons:
            i, j = order.index(left), order.index(right)
            a, b, sign = (left, right, 1.0) if i < j else (right, left, -1.0)
            seed = min(i, j) * len(order) + max(i, j)
            series = {}
            for algo in (a, b):
                group = mkt_df[mkt_df["algo"] == algo]
                ret = load_ensemble_return_path(list(group["pv_path"]), rf_by_market.get(market)) if len(group) else None
                if ret is None or len(ret) <= 10:
                    raise ValueError(f"cadence comparison {left!r} vs {right!r}: no complete "
                                     f"return path for {market} {algo}")
                series[algo] = ret
            paired = pd.concat([series[a].rename("a"), series[b].rename("b")], axis=1, join="inner").dropna()
            test = lw_sharpe_diff_test(paired["a"].to_numpy(), paired["b"].to_numpy(), ann,
                                       n_boot=n_boot, seed=seed)
            if not np.isfinite(float(test["pvalue"])):
                raise ValueError(f"cadence comparison {left!r} vs {right!r} has a non-finite p-value for {market}")
            market_rows.append({
                "analysis_scope": "secondary_predeclared_cadence_control",
                "market": market,
                "friction": friction,
                "strategy": left,
                "cadence_control": right,
                "sharpe_diff_strategy_minus_control": float(sign * test["diff"]),
                "ci_low_95": float(test["ci_low"]) if sign > 0 else float(-test["ci_high"]),
                "ci_high_95": float(test["ci_high"]) if sign > 0 else float(-test["ci_low"]),
                "pvalue_uncorrected": float(test["pvalue"]),
                "degenerate_comparison": False,
                "degenerate_reason": None,
            })
        p_holm, reject = holm_bonferroni([r["pvalue_uncorrected"] for r in market_rows], alpha=alpha)
        for k, r in enumerate(market_rows):
            r["p_holm"] = float(p_holm[k])
            r["significant_family_adjusted"] = bool(reject[k])
            r["correction_family"] = f"secondary_cadence_within_{market}_{len(market_rows)}"
            r["family_size"] = len(market_rows)
        rows.extend(market_rows)
    return pd.DataFrame(rows)


# ── complete analyses with the output files of the study ────────────────────

KEYS = ["market", "non_rl_algo", "rl_algo"]


def bootstrap_file_label(n_bootstrap: int) -> str:
    if n_bootstrap >= 1000 and n_bootstrap % 1000 == 0:
        return f"B{n_bootstrap // 1000}k"
    return f"B{n_bootstrap}"


def run_primary(
    inputs: pd.DataFrame,
    rf_by_market: Mapping[str, pd.Series],
    dates_by_market: Mapping[str, pd.DatetimeIndex],
    output_dir: Path,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    cache: Optional[ContrastCache] = None,
    cadence: bool = True,
    log: Callable[[str], None] = print,
) -> Dict[str, Path]:
    """The primary 32-contrast family with its robustness checks and the cadence control.

    Writes (formats as in the study):
    ``headline_non_rl_vs_typical_rl.csv`` and ``headline_rl_seed_statistics.csv``
    (``%.8f``), ``headline_block_length_sensitivity.csv`` (``%.6f``),
    ``secondary_cadence_control.csv`` (``%.8f``), and, only if the borderline
    rule fires, ``headline_non_rl_vs_typical_rl__B10k.csv`` and ``__B100k.csv``.
    (origin: run_analysis in tools/stage3_statistical_inference.py)
    """
    cache = cache or ContrastCache()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {}
    primary_b = n_bootstrap
    log(f"[primary] block {BLOCK_LENGTH}, B={primary_b}")
    headline, seeds = build_family(inputs, rf_by_market, dates_by_market, PRIMARY,
                                   n_bootstrap=primary_b, block_length=BLOCK_LENGTH, cache=cache, log=log)
    headline["block_length"] = BLOCK_LENGTH
    headline["n_bootstrap"] = primary_b
    if is_borderline(headline["p_holm"]) and primary_b < MC_BORDERLINE_B:
        log(f"[primary] borderline Holm p in [{MC_BORDERLINE_LOW}, {MC_BORDERLINE_HIGH}]: "
            f"all tests again with B={MC_BORDERLINE_B}")
        path = output_dir / f"headline_non_rl_vs_typical_rl__{bootstrap_file_label(primary_b)}.csv"
        headline.to_csv(path, index=False, float_format="%.8f")
        files["headline_initial_b"] = path
        primary_b = MC_BORDERLINE_B
        headline, seeds = build_family(inputs, rf_by_market, dates_by_market, PRIMARY,
                                       n_bootstrap=primary_b, block_length=BLOCK_LENGTH, cache=cache, log=log)
        headline["block_length"] = BLOCK_LENGTH
        headline["n_bootstrap"] = primary_b
        path = output_dir / f"headline_non_rl_vs_typical_rl__{bootstrap_file_label(primary_b)}.csv"
        headline.to_csv(path, index=False, float_format="%.8f")
        files["headline_high_b"] = path

    primary_decision = {tuple(r[c] for c in KEYS): bool(r["significant_family_adjusted"])
                        for _, r in headline.iterrows()}
    frames = []
    for block in BLOCK_LENGTH_SENSITIVITY:
        if block == BLOCK_LENGTH:
            continue
        log(f"[primary] sensitivity: block {block}, B={primary_b}")
        alt, _ = build_family(inputs, rf_by_market, dates_by_market, PRIMARY,
                              n_bootstrap=primary_b, block_length=block, cache=cache, log=log)
        merged = alt[KEYS].copy()
        merged["block_length"] = block
        merged["p_holm"] = alt["p_holm"].to_numpy()
        merged["significant_family_adjusted"] = alt["significant_family_adjusted"].to_numpy()
        merged["flips_vs_primary"] = [
            bool(alt.iloc[i]["significant_family_adjusted"]) != primary_decision[tuple(alt.iloc[i][c] for c in KEYS)]
            for i in range(len(alt))
        ]
        frames.append(merged)
    sensitivity = pd.concat(frames, ignore_index=True)
    flip_keys = {tuple(r[c] for c in KEYS) for _, r in sensitivity[sensitivity["flips_vs_primary"]].iterrows()}
    headline["block_length_sensitive"] = [tuple(r[c] for c in KEYS) in flip_keys for _, r in headline.iterrows()]

    files["sensitivity"] = output_dir / "headline_block_length_sensitivity.csv"
    sensitivity.to_csv(files["sensitivity"], index=False, float_format="%.6f")
    files["headline"] = output_dir / "headline_non_rl_vs_typical_rl.csv"
    headline.to_csv(files["headline"], index=False, float_format="%.8f")
    files["seed_statistics"] = output_dir / "headline_rl_seed_statistics.csv"
    seeds.to_csv(files["seed_statistics"], index=False, float_format="%.8f")
    if cadence:
        log("[primary] secondary cadence control")
        files["cadence"] = output_dir / "secondary_cadence_control.csv"
        build_cadence_control(inputs, rf_by_market).to_csv(files["cadence"], index=False, float_format="%.8f")
    return files


def run_post_hoc(
    design: FamilyDesign,
    inputs: pd.DataFrame,
    rf_by_market: Mapping[str, pd.Series],
    dates_by_market: Mapping[str, pd.DatetimeIndex],
    output_dir: Path,
    *,
    primary_headline: Optional[Path] = None,
    n_bootstrap: int = N_BOOTSTRAP,
    cache: Optional[ContrastCache] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Path]:
    """A post hoc family (``signal_tuned`` or ``dividend``) with its robustness checks.

    Writes, with full float precision as in the study,
    ``contrasts_block{10,21,42}_B<B>.csv``, ``seed_statistics_block*_B*.csv``,
    ``headline_non_rl_vs_typical_rl.csv`` and
    ``headline_block_length_sensitivity.csv``. Every contrast must be
    complete (no degenerate contrast), else ``ValueError``.

    ``signal_tuned`` (origin: run_block and finalise in
    tools/stage3_signal_tuned_inference.py): the headline is read back from the
    block-21 contrast file and gets the primary family's Holm p
    (``original_p_holm_32``, read from ``primary_headline``) and
    ``block_length_sensitive``. ``dividend`` (origin: run in
    tools/cifer_followup_inference.py): the headline is the in-memory block-21
    result. In both, the alternative block lengths use B = 10,000 even if the
    borderline rule raised the primary B.
    """
    if design.name not in ("signal_tuned", "dividend"):
        raise ValueError(f"no post hoc procedure for family {design.name!r}")
    if design.name == "signal_tuned" and primary_headline is None:
        raise ValueError("the signal-tuned family needs the primary headline file")
    cache = cache or ContrastCache()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = csv_round_trip(inputs)
    files: Dict[str, Path] = {}
    results: Dict[Tuple[int, int], pd.DataFrame] = {}
    n_contrasts = design.expected_contrasts_per_market * len(MARKETS)

    def block_result(block: int, draws: int) -> pd.DataFrame:
        log(f"[{design.name}] block {block}, B={draws}")
        contrasts, seeds = build_family(frame, rf_by_market, dates_by_market, design,
                                        n_bootstrap=draws, block_length=block, cache=cache, log=log)
        if (len(contrasts) != n_contrasts or contrasts["degenerate_contrast"].any()
                or not contrasts["date_support_complete"].all()
                or not (contrasts["n_boot_effective"] == draws).all()):
            raise ValueError(f"{design.name}: incomplete or degenerate family at block {block}, B={draws}")
        contrasts["block_length"], contrasts["n_bootstrap"] = block, draws
        contrasts["inference_role"] = "post_hoc_sensitivity"
        path = output_dir / f"contrasts_block{block}_B{draws}.csv"
        contrasts.to_csv(path, index=False)
        seeds.to_csv(output_dir / f"seed_statistics_block{block}_B{draws}.csv", index=False)
        files[f"contrasts_block{block}_B{draws}"] = path
        if design.name == "signal_tuned":
            contrasts = pd.read_csv(path)  # the study read the completed block files back
        results[block, draws] = contrasts
        return contrasts

    initial = block_result(BLOCK_LENGTH, n_bootstrap)
    borderline = is_borderline(initial["p_holm"])
    primary = block_result(BLOCK_LENGTH, MC_BORDERLINE_B) if borderline and n_bootstrap < MC_BORDERLINE_B else initial.copy()
    if design.name == "signal_tuned":
        original = pd.read_csv(primary_headline)
        primary = primary.merge(original[KEYS + ["p_holm"]].rename(columns={"p_holm": "original_p_holm_32"}),
                                on=KEYS, how="left", validate="one_to_one")
        primary["block_length_sensitive"] = False
    sensitivity = []
    for block in BLOCK_LENGTH_SENSITIVITY:
        if block == BLOCK_LENGTH:
            continue
        alternate = block_result(block, n_bootstrap)
        joined = alternate.merge(primary[KEYS + ["significant_family_adjusted"]], on=KEYS,
                                 suffixes=("", "_primary"), validate="one_to_one")
        joined["flips_vs_primary"] = joined["significant_family_adjusted"] != joined["significant_family_adjusted_primary"]
        if design.name == "signal_tuned":
            flip_keys = set(map(tuple, joined.loc[joined["flips_vs_primary"], KEYS].to_numpy()))
            primary["block_length_sensitive"] = primary["block_length_sensitive"] | np.array(
                [tuple(k) in flip_keys for k in primary[KEYS].to_numpy()], dtype=bool)
        sensitivity.append(joined)
    files["headline"] = output_dir / "headline_non_rl_vs_typical_rl.csv"
    primary.to_csv(files["headline"], index=False)
    files["sensitivity"] = output_dir / "headline_block_length_sensitivity.csv"
    pd.concat(sensitivity, ignore_index=True).to_csv(files["sensitivity"], index=False)
    return files


def required_algorithms(families: Iterable[str]) -> List[str]:
    """Strategies whose F3 cells the selected families need (in a fixed order)."""
    names: List[str] = []
    for family in families:
        for algo in FAMILIES[family].non_rl_algorithms + FAMILIES[family].rl_algorithms:
            if algo not in names:
                names.append(algo)
        if family == "primary":
            for pair in CADENCE_CONTROLS:
                for algo in pair:
                    if algo not in names:
                        names.append(algo)
    return names


def run_inference(
    cell_root: Path,
    data_root: Path,
    output_root: Path,
    *,
    families: Sequence[str] = ("primary", "signal_tuned"),
    manifest: Optional[Mapping[str, Any]] = None,
    project_root: Optional[Path] = None,
    n_bootstrap: int = N_BOOTSTRAP,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> Dict[str, Dict[str, Path]]:
    """Run the selected families on the cell results under ``cell_root``.

    Outputs go to ``<output_root>/<family>/``. The signal-tuned family needs the
    primary family (it reports the primary Holm p of the shared contrasts), so
    ``primary`` is added when ``signal_tuned`` is selected.
    """
    from tools.manifest import DEFAULT_MANIFEST, PROJECT_ROOT, load_manifest

    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; choose from {sorted(FAMILIES)}")
    selected = [f for f in FAMILIES if f in families or (f == "primary" and "signal_tuned" in families)]
    manifest = manifest or load_manifest(DEFAULT_MANIFEST)
    project_root = project_root or PROJECT_ROOT
    rf_by_market, dates_by_market = load_market_inputs(manifest, Path(data_root), Path(project_root))
    inputs = collect_inputs(Path(cell_root), required_algorithms(selected))
    log(f"[inference] {len(inputs)} cell results under {cell_root}")
    cache = ContrastCache(workers=workers)
    out: Dict[str, Dict[str, Path]] = {}
    output_root = Path(output_root)
    for family in selected:
        if family == "primary":
            out[family] = run_primary(inputs, rf_by_market, dates_by_market, output_root / family,
                                      n_bootstrap=n_bootstrap, cache=cache, log=log)
        else:
            out[family] = run_post_hoc(
                FAMILIES[family], inputs, rf_by_market, dates_by_market, output_root / family,
                primary_headline=out.get("primary", {}).get("headline"),
                n_bootstrap=n_bootstrap, cache=cache, log=log,
            )
    return out

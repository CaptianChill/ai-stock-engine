"""Overfitting statistics.

- Probabilistic / Deflated Sharpe Ratio: Bailey & Lopez de Prado (2014),
  "The Deflated Sharpe Ratio", Journal of Portfolio Management 40(5).
- Probability of Backtest Overfitting via CSCV: Bailey, Borwein, Lopez de Prado & Zhu (2017),
  Journal of Computational Finance 20(4).
All Sharpe ratios here are per-period (daily), not annualized, as the formulas require.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if len(r) > 1 and sd > 0 else 0.0


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), adjusting for sample length, skew and fat tails."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 3:
        return 0.0
    sr = sharpe(r)
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        denom = 1e-12
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """Expected best Sharpe among n_trials skill-less strategies (the luck hurdle)."""
    if n_trials <= 1 or trial_sharpe_var <= 0:
        return 0.0
    a = stats.norm.ppf(1 - 1 / n_trials)
    b = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(trial_sharpe_var) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(r: np.ndarray, n_trials: int, trial_sharpe_var: float | None = None) -> dict:
    """DSR = PSR measured against the luck hurdle for n_trials attempts.

    When the variance of Sharpe across trials is unknown, it defaults to the sampling
    variance of a daily Sharpe estimate, 1/(T-1) — a conservative stand-in.
    """
    r = np.asarray(r, dtype=float)
    if trial_sharpe_var is None:
        trial_sharpe_var = 1.0 / max(len(r) - 1, 1)
    hurdle = expected_max_sharpe(n_trials, trial_sharpe_var)
    return {"dsr": probabilistic_sharpe(r, hurdle), "sr_daily": sharpe(r), "hurdle_daily": hurdle,
            "n_trials": int(n_trials), "trial_sharpe_var": float(trial_sharpe_var)}


def pbo_cscv(m: np.ndarray, n_splits: int = 8) -> dict:
    """m: T x N matrix of per-period returns, one column per strategy variant tried.
    Returns the probability that the in-sample winner ranks below median out of sample."""
    m = np.asarray(m, dtype=float)
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least two variants")
    n_splits -= n_splits % 2
    usable = (t // n_splits) * n_splits
    if usable < n_splits * 2:
        raise ValueError("Not enough periods for CSCV")
    blocks = np.array_split(m[-usable:], n_splits)
    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_ = np.vstack([blocks[i] for i in combo])
        oos = np.vstack([blocks[i] for i in range(n_splits) if i not in combo])
        sr_is = np.array([sharpe(is_[:, k]) for k in range(n)])
        sr_oos = np.array([sharpe(oos[:, k]) for k in range(n)])
        best = int(np.argmax(sr_is))
        rank = stats.rankdata(sr_oos)[best]          # 1 = worst, n = best
        omega = rank / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": int(len(logits)),
            "n_variants": int(n), "median_logit": float(np.median(logits))}

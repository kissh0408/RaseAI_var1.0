"""Pair ticket probabilities from fused win probabilities (Stern)."""

from __future__ import annotations

import numpy as np

from prob_fusion.src.place_prob import stern_second_prob

PAIR_KEY = tuple[int, int]


def norm_pair(a: int, b: int) -> PAIR_KEY:
    return (a, b) if a < b else (b, a)


def stern_quinella_pair_prob(p_win: np.ndarray, i: int, j: int, lam2: float = 1.0) -> float:
    """Quinella probability for horses i and j."""
    return float(p_win[i] * stern_second_prob(p_win, i, lam2)[j] + p_win[j] * stern_second_prob(p_win, j, lam2)[i])


def stern_wide_pair_prob(p_win: np.ndarray, i: int, j: int, lam2: float, lam3: float) -> float:
    """Wide (exacta or quinella-style top2) approximation via Stern."""
    q = stern_quinella_pair_prob(p_win, i, j, lam2)
    n = len(p_win)
    wide = q
    for k in range(n):
        if k == i or k == j:
            continue
        wide += p_win[i] * stern_second_prob(p_win, i, lam2)[k] * stern_second_prob(p_win, k, lam3)[j]
        wide += p_win[j] * stern_second_prob(p_win, j, lam2)[k] * stern_second_prob(p_win, k, lam3)[i]
    return float(wide)


def all_pair_probs(p_win: np.ndarray, horse_nums: list[int], lam2: float = 1.0, lam3: float = 1.0) -> dict[PAIR_KEY, dict[str, float]]:
    """All pair probabilities for a race."""
    p = np.asarray(p_win, dtype=float)
    total = p.sum()
    if total > 0:
        p = p / total
    out: dict[PAIR_KEY, dict[str, float]] = {}
    n = len(horse_nums)
    for i in range(n):
        for j in range(i + 1, n):
            key = norm_pair(horse_nums[i], horse_nums[j])
            out[key] = {
                "quinella": stern_quinella_pair_prob(p, i, j, lam2),
                "wide": stern_wide_pair_prob(p, i, j, lam2, lam3),
            }
    return out


def calibration_max_error_pp(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Max |predicted - actual| in pp across bins."""
    if len(predicted) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    errors = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (predicted >= lo) & (predicted < hi if i < n_bins - 1 else predicted <= hi)
        if mask.sum() < 5:
            continue
        errors.append(abs(predicted[mask].mean() - actual[mask].mean()) * 100)
    return max(errors) if errors else float("nan")

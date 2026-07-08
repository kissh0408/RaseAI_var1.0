"""Stern place probabilities from fused win probabilities."""

from __future__ import annotations

import numpy as np
from scipy import optimize

_DENOM_EPS = 1e-12


def stern_second_prob(p_win: np.ndarray, winner_idx: int, lam2: float) -> np.ndarray:
    """Conditional 2nd-place distribution given winner_idx won."""
    p = np.asarray(p_win, dtype=float)
    n = len(p)
    out = np.zeros(n, dtype=float)
    pw = p[winner_idx]
    if pw <= _DENOM_EPS:
        return out
    for j in range(n):
        if j == winner_idx:
            continue
        denom = 1.0 - p[j]
        if denom >= _DENOM_EPS:
            out[j] = (p[j] / denom) ** lam2
    total = out.sum()
    if total > _DENOM_EPS:
        out /= total
    return out


def stern_third_prob(p_win: np.ndarray, first: int, second: int, lam3: float) -> np.ndarray:
    """Conditional 3rd-place distribution given first and second."""
    p = np.asarray(p_win, dtype=float)
    n = len(p)
    out = np.zeros(n, dtype=float)
    ps = p[second]
    if ps <= _DENOM_EPS:
        return out
    for k in range(n):
        if k == first or k == second:
            continue
        denom = 1.0 - p[k]
        if denom >= _DENOM_EPS:
            out[k] = (p[k] / denom) ** lam3
    total = out.sum()
    if total > _DENOM_EPS:
        out /= total
    return out


def stern_place_probs(p_win: np.ndarray, lam2: float, lam3: float) -> tuple[np.ndarray, np.ndarray]:
    """Stern 2nd and 3rd place probabilities."""
    p = np.asarray(p_win, dtype=float)
    n = len(p)
    p2 = np.zeros(n, dtype=float)
    p3 = np.zeros(n, dtype=float)
    for i in range(n):
        second_probs = stern_second_prob(p, i, lam2)
        for j in range(n):
            if j == i:
                continue
            sp_j = second_probs[j]
            if sp_j <= 0:
                continue
            p2[j] += p[i] * sp_j
            third_probs = stern_third_prob(p, i, j, lam3)
            for k in range(n):
                if k == i or k == j:
                    continue
                p3[k] += p[i] * sp_j * third_probs[k]
    return p2, p3


def place_prob_from_p_win(p_win: np.ndarray, lam2: float = 1.0, lam3: float = 1.0) -> np.ndarray:
    """P(place=top3) = p_win + p_2nd + p_3rd under Stern."""
    p2, p3 = stern_place_probs(p_win, lam2, lam3)
    return p_win + p2 + p3


def fit_stern_lambda(
    races_p_win: list[np.ndarray],
    races_place_outcome: list[np.ndarray],
    *,
    init_lam2: float = 1.0,
    init_lam3: float = 1.0,
) -> tuple[float, float]:
    """Fit lam2, lam3 on VALID by minimizing Brier score for place."""

    def brier(params: np.ndarray) -> float:
        lam2, lam3 = params
        if lam2 <= 0 or lam3 <= 0:
            return 1e6
        total = 0.0
        count = 0
        for p_win, outcome in zip(races_p_win, races_place_outcome):
            p_place = place_prob_from_p_win(p_win, lam2, lam3)
            total += float(np.mean((p_place - outcome) ** 2))
            count += 1
        return total / max(count, 1)

    res = optimize.minimize(
        brier,
        x0=np.array([init_lam2, init_lam3]),
        bounds=[(0.1, 3.0), (0.1, 3.0)],
        method="L-BFGS-B",
    )
    return float(res.x[0]), float(res.x[1])

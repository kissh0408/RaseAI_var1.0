"""Calibrated wide pair probabilities for Layer 1 inference (Step 2)."""
from __future__ import annotations

from typing import Literal

import numpy as np

try:
    from predict import (
        apply_bracket_isotonic,
        compute_race_probabilities,
        compute_race_probabilities_stern,
        load_bracket_calibration,
        softmax_with_temperature,
    )
except ModuleNotFoundError:
    from pure_rank.src.predict import (
        apply_bracket_isotonic,
        compute_race_probabilities,
        compute_race_probabilities_stern,
        load_bracket_calibration,
        softmax_with_temperature,
    )

try:
    from betting.src.pair_probs import PAIR_KEY, norm_pair
except ModuleNotFoundError:
    PAIR_KEY = tuple[int, int]

    def norm_pair(a: int, b: int) -> PAIR_KEY:
        return (a, b) if a < b else (b, a)


def get_pair_odds(race_id, h1, h2, lookup):
    """Stub: wide odds lookup removed from L1; Phase 4 uses betting layer."""
    return None


def compute_calibrated_wide_probs(
    scores: np.ndarray,
    horse_nums: list[int],
    *,
    T_opt: float,
    bracket_models: dict | None = None,
    wide_odds_lookup: dict | None = None,
    race_id: str | int | None = None,
    prob_method: Literal["harville", "stern"] = "harville",
    lam2: float | None = None,
    lam3: float | None = None,
    apply_bracket: bool = True,
) -> dict[PAIR_KEY, float]:
    """Convert LambdaRank scores to calibrated P_wide for all pairs in a race."""
    scores = np.asarray(scores, dtype=float)
    horse_nums = [int(h) for h in horse_nums]
    n = len(horse_nums)
    if n < 2:
        return {}

    if prob_method == "stern":
        if lam2 is None or lam3 is None:
            raise ValueError("stern requires lam2 and lam3")
        probs = compute_race_probabilities_stern(scores, float(T_opt), float(lam2), float(lam3))
    else:
        probs = compute_race_probabilities(scores, float(T_opt))

    wide_matrix = probs["wide_matrix"]
    out: dict[PAIR_KEY, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            p_raw = float(wide_matrix[i, j])
            if apply_bracket and bracket_models and wide_odds_lookup is not None and race_id is not None:
                prior = get_pair_odds(race_id, horse_nums[i], horse_nums[j], wide_odds_lookup)
                if prior is not None:
                    p_raw = apply_bracket_isotonic(p_raw, float(prior), bracket_models)
            out[norm_pair(horse_nums[i], horse_nums[j])] = p_raw
    return out


def load_bracket_models_from_config(cfg: dict, models_dir) -> dict | None:
    """Load bracket isotonic models if calibration.fitted is true."""
    cal = cfg.get("calibration", {})
    if not cal.get("fitted", False):
        return None
    bracket_models, _meta = load_bracket_calibration(models_dir)
    return bracket_models or None


def wide_probs_from_model_prob_frame(
    horse_nums: list[int],
    model_probs: np.ndarray,
) -> dict[PAIR_KEY, float]:
    """Layer 2: Harville wide from normalized model_prob."""
    try:
        from ev_filters import harville_wide_pair_prob
    except ModuleNotFoundError:
        from strategy.src.ev_filters import harville_wide_pair_prob

    horse_nums = [int(h) for h in horse_nums]
    p = np.asarray(model_probs, dtype=float)
    total = p.sum()
    if total <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    p_dict = {horse_nums[i]: float(p[i]) for i in range(len(horse_nums))}
    out: dict[PAIR_KEY, float] = {}
    for i in range(len(horse_nums)):
        for j in range(i + 1, len(horse_nums)):
            h1, h2 = horse_nums[i], horse_nums[j]
            out[norm_pair(h1, h2)] = harville_wide_pair_prob(p_dict, h1, h2)
    return out

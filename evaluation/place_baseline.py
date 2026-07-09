"""OOS place ROI comparison for model top1 vs market favorite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAKE = 100.0
PLACE_ROI_GATE = 100.0
PLACE_MIN_RACES = 200


KNOWN_PLACE_LIMITATIONS = [
    "confirmed HR payouts are settlement data, not pre-race place odds; use for ROI settlement and upper-bound diagnostics only",
    "refund/void handling depends on available HR fields; if refund flags are absent, scratched horses should be filtered upstream",
    "place payouts are not valid EV-threshold inputs because they are known only after race settlement",
]


def _normalize_lookup_value(value: object) -> tuple[float | None, int | None]:
    if isinstance(value, tuple):
        odds = float(value[0]) if value[0] is not None else None
        pop = int(value[1]) if len(value) > 1 and value[1] is not None else None
        return odds, pop
    if value is None:
        return None, None
    return float(value), None


def _bootstrap_roi_ci(
    payouts: list[float],
    *,
    stake: float = STAKE,
    samples: int = 1000,
    random_seed: int = 42,
) -> list[float | None]:
    if not payouts:
        return [None, None]
    rng = np.random.default_rng(random_seed)
    arr = np.asarray(payouts, dtype=float)
    rois = []
    for _ in range(samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        rois.append(float(sample.sum() / (stake * len(sample)) * 100.0))
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return [float(lo), float(hi)]


def compute_pick_place_roi(
    df: pd.DataFrame,
    picks: dict[str, int],
    place_lookup: dict[str, dict[int, int]],
    *,
    stake: float = STAKE,
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Settle flat place bets for race_id -> horse_num picks using HR payouts."""
    payouts: list[float] = []
    n_hits = 0
    for race_id in sorted(set(map(str, picks.keys()))):
        race_payouts = place_lookup.get(race_id)
        if not race_payouts:
            continue
        horse = int(picks[race_id])
        payout = float(race_payouts.get(horse, 0))
        payouts.append(payout)
        if payout > 0:
            n_hits += 1

    n_races = len(payouts)
    total_stake = stake * n_races
    total_payout = float(sum(payouts))
    roi = total_payout / total_stake * 100.0 if total_stake > 0 else None
    return {
        "n_races": n_races,
        "n_hits": n_hits,
        "hit_rate": n_hits / n_races if n_races else None,
        "total_stake": total_stake,
        "total_payout": total_payout,
        "roi_pct": roi,
        "bootstrap_ci_95": _bootstrap_roi_ci(
            payouts,
            stake=stake,
            samples=bootstrap_samples,
            random_seed=random_seed,
        ),
    }


def model_top1_picks(df: pd.DataFrame, score_col: str = "pure_score_z") -> dict[str, int]:
    picks: dict[str, int] = {}
    for race_id, grp in df.groupby("race_id", sort=False):
        best = grp.sort_values(score_col, ascending=False).iloc[0]
        picks[str(race_id)] = int(best["horse_num"])
    return picks


def favorite_picks(
    df: pd.DataFrame,
    win_odds_lookup: dict[str, dict[int, object]],
) -> dict[str, int]:
    picks: dict[str, int] = {}
    for race_id, grp in df.groupby("race_id", sort=False):
        rid = str(race_id)
        odds_map = win_odds_lookup.get(rid, {})
        candidates = []
        for horse in grp["horse_num"].astype(int).tolist():
            odds, pop = _normalize_lookup_value(odds_map.get(horse))
            if odds is None:
                continue
            candidates.append((horse, odds, pop if pop is not None else 9999))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[1], item[2], item[0]))
        picks[rid] = int(candidates[0][0])
    return picks


def compute_place_baseline_oos(
    df: pd.DataFrame,
    *,
    win_odds_lookup: dict[str, dict[int, object]],
    place_lookup: dict[str, dict[int, int]],
    test_start: str = "2025-01-01",
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compare OOS model-top1 place ROI against market favorite place ROI."""
    if not place_lookup:
        return {
            "status": "unavailable",
            "reason": "place payout lookup is empty",
            "protocol": "TEST 2025+ only; HR place payouts used for settlement, not EV thresholding",
            "test_start": test_start,
            "test_n_races": 0,
            "place_coverage_races": 0,
            "model_top1": None,
            "favorite": None,
            "gates": {
                "roi_above_100": False,
                "n_races_at_least_200": False,
                "place_coverage_positive": False,
                "phase3_place_pass": False,
            },
            "verdict": "FAIL",
            "known_limitations": KNOWN_PLACE_LIMITATIONS,
        }
    work = df.copy()
    work["race_id"] = work["race_id"].astype(str)
    if "horse_num" not in work.columns and "horse_number" in work.columns:
        work["horse_num"] = work["horse_number"]
    dates = pd.to_datetime(work["race_date"])
    test_df = work.loc[dates >= pd.Timestamp(test_start)].copy()

    model_picks = model_top1_picks(test_df)
    fav_picks = favorite_picks(test_df, win_odds_lookup)
    model = compute_pick_place_roi(
        test_df,
        model_picks,
        place_lookup,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    favorite = compute_pick_place_roi(
        test_df,
        fav_picks,
        place_lookup,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    if model["roi_pct"] is not None and favorite["roi_pct"] is not None:
        model["roi_minus_favorite_pp"] = float(model["roi_pct"] - favorite["roi_pct"])
    else:
        model["roi_minus_favorite_pp"] = None

    coverage_races = len({rid for rid in test_df["race_id"].astype(str).unique() if place_lookup.get(rid)})
    gates = {
        "roi_above_100": bool(model["roi_pct"] is not None and model["roi_pct"] > PLACE_ROI_GATE),
        "n_races_at_least_200": bool(model["n_races"] >= PLACE_MIN_RACES),
        "place_coverage_positive": bool(coverage_races > 0),
    }
    gates["phase3_place_pass"] = all(gates.values())

    return {
        "status": "measured",
        "protocol": "TEST 2025+ only; HR place payouts used for settlement, not EV thresholding",
        "test_start": test_start,
        "test_n_races": int(test_df["race_id"].nunique()),
        "place_coverage_races": coverage_races,
        "model_top1": model,
        "favorite": favorite,
        "gates": gates,
        "verdict": "PASS" if gates["phase3_place_pass"] else "FAIL",
        "known_limitations": KNOWN_PLACE_LIMITATIONS,
    }


def write_place_baseline_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

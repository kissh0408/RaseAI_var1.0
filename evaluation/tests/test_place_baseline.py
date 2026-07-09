"""Tests for model-top1 vs favorite place ROI baseline."""

from __future__ import annotations

import pandas as pd

from evaluation.place_baseline import compute_place_baseline_oos, compute_pick_place_roi


def _scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1", "R2", "R2", "R2"],
            "horse_num": [1, 2, 3, 1, 2, 3],
            "race_date": pd.to_datetime(["2025-01-05"] * 6),
            "finish_rank": [2, 1, 3, 2, 3, 1],
            "pure_score_z": [0.1, 1.2, -0.5, 1.1, 0.2, 0.9],
        }
    )


def test_compute_pick_place_roi_uses_payout_presence_not_finish_rank():
    df = _scores_frame()
    picks = {"R1": 2, "R2": 1}
    place_lookup = {"R1": {2: 150}, "R2": {3: 130}}

    result = compute_pick_place_roi(df, picks, place_lookup)

    assert result["n_races"] == 2
    assert result["n_hits"] == 1
    assert result["hit_rate"] == 0.5
    assert result["roi_pct"] == 75.0


def test_compute_place_baseline_oos_compares_model_top1_and_favorite():
    df = _scores_frame()
    win_odds_lookup = {
        "R1": {1: (2.0, 1), 2: (3.0, 2), 3: (8.0, 3)},
        "R2": {1: (5.0, 2), 2: (7.0, 3), 3: (1.5, 1)},
    }
    place_lookup = {"R1": {2: 150}, "R2": {3: 130}}

    report = compute_place_baseline_oos(
        df,
        win_odds_lookup=win_odds_lookup,
        place_lookup=place_lookup,
        bootstrap_samples=50,
        random_seed=1,
    )

    assert report["test_n_races"] == 2
    assert report["model_top1"]["roi_pct"] == 75.0
    assert report["favorite"]["roi_pct"] == 65.0
    assert report["model_top1"]["roi_minus_favorite_pp"] == 10.0
    assert report["model_top1"]["bootstrap_ci_95"][0] <= 75.0 <= report["model_top1"]["bootstrap_ci_95"][1]
    assert report["gates"] == {
        "roi_above_100": False,
        "n_races_at_least_200": False,
        "place_coverage_positive": True,
        "phase3_place_pass": False,
    }
    assert report["verdict"] == "FAIL"
    assert "confirmed HR payouts are settlement data" in report["known_limitations"][0]


def test_compute_place_baseline_oos_reports_unavailable_when_place_lookup_empty():
    df = _scores_frame()
    win_odds_lookup = {
        "R1": {1: (2.0, 1), 2: (3.0, 2), 3: (8.0, 3)},
        "R2": {1: (5.0, 2), 2: (7.0, 3), 3: (1.5, 1)},
    }

    report = compute_place_baseline_oos(df, win_odds_lookup=win_odds_lookup, place_lookup={})

    assert report["status"] == "unavailable"
    assert report["reason"] == "place payout lookup is empty"

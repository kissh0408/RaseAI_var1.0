"""Unit tests for going improvement feature engineering and constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_training.src.builders.going_delta import (
    add_going_delta_features,
    compute_going_delta_active_score,
)
from model_training.src.feature_groups import (
    assign_feature_group,
    build_interaction_constraints,
)


def test_assign_feature_group_going_vs_market() -> None:
    assert assign_feature_group("turf_condition") == "going_aptitude"
    assert assign_feature_group("going_x_turf_soft_winrate") == "going_aptitude"
    assert assign_feature_group("odds_rank_divergence") == "market_ability"
    assert assign_feature_group("course_code") == "course_distance"


def test_interaction_constraints_disabled_by_default() -> None:
    names = ["odds_rank_divergence", "turf_condition", "course_code"]
    assert build_interaction_constraints(names, {}) is None


def test_interaction_constraints_enabled() -> None:
    names = ["odds_rank_divergence", "turf_condition", "course_code"]
    cfg = {"going_improvement": {"interaction_constraints_enabled": True}}
    ic = build_interaction_constraints(names, cfg)
    assert ic is not None
    assert len(ic) == 3
    flat = {i for grp in ic for i in grp}
    assert flat == {0, 1, 2}


def test_going_delta_active_score_scenario() -> None:
    df = pd.DataFrame(
        {
            "track_code": [11, 24],
            "turf_condition": [1, 0],
            "dirt_condition": [0, 1],
            "delta_horse_turf_soft_aptitude": [0.05, 0.0],
            "delta_horse_dirt_soft_aptitude": [0.0, 0.04],
            "delta_horse_heavy_aptitude": [0.10, 0.08],
            "delta_horse_turf_very_heavy_aptitude": [0.15, 0.0],
            "delta_sire_heavy_aptitude": [0.02, 0.01],
            "delta_jockey_heavy_aptitude": [0.01, 0.02],
        }
    )
    good = compute_going_delta_active_score(df, 1)
    soft = compute_going_delta_active_score(df, 2)
    heavy = compute_going_delta_active_score(df, 3)
    assert good.iloc[0] == pytest.approx(0.0)
    assert soft.iloc[0] > 0.0
    assert heavy.iloc[0] > soft.iloc[0]


def test_add_going_delta_features_columns() -> None:
    df = pd.DataFrame(
        {
            "track_code": [11],
            "turf_condition": [1],
            "dirt_condition": [0],
            "horse_turf_heavy_win_rate": [0.2],
            "horse_turf_light_win_rate": [0.1],
            "horse_turf_soft_win_rate": [0.15],
            "horse_turf_very_heavy_win_rate": [0.25],
            "horse_dirt_heavy_win_rate": [np.nan],
            "horse_dirt_light_win_rate": [np.nan],
            "horse_dirt_soft_win_rate": [np.nan],
            "sire_turf_heavy_win_rate": [0.12],
            "sire_turf_soft_win_rate": [0.08],
            "sire_dirt_heavy_win_rate": [np.nan],
            "sire_dirt_soft_win_rate": [np.nan],
            "jockey_turf_heavy_win_rate": [0.11],
            "jockey_turf_light_win_rate": [0.09],
            "jockey_dirt_heavy_win_rate": [np.nan],
            "jockey_dirt_light_win_rate": [np.nan],
        }
    )
    out = add_going_delta_features(df)
    assert "delta_horse_turf_heavy_aptitude" in out.columns
    assert "going_delta_active_score" in out.columns
    assert out["delta_horse_turf_heavy_aptitude"].iloc[0] == pytest.approx(0.1, rel=1e-3)

"""Tests for apply_uniform_baba_jv_code track_condition sync and going_match 3/4 split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from main.pipeline.inference_pipeline import apply_uniform_baba_jv_code


def _base_turf_row(**overrides: float) -> dict:
    row = {
        "track_code": 11,
        "turf_condition": 1.0,
        "dirt_condition": 0.0,
        "track_condition_code": 1.0,
        "horse_turf_heavy_win_rate": 0.10,
        "horse_turf_very_heavy_win_rate": 0.20,
        "horse_turf_light_win_rate": 0.30,
        "horse_turf_soft_win_rate": 0.15,
        "horse_dirt_heavy_win_rate": 0.0,
        "horse_dirt_soft_win_rate": 0.0,
        "going_match_score_turf_imputed": 1.5,
        "going_change_lag1": 0.0,
        "going_worsening_flag": 0,
    }
    row.update(overrides)
    return row


def test_track_condition_code_synced_with_jv_code() -> None:
    df = pd.DataFrame(
        {
            "track_code": [11, 24],
            "turf_condition": [1.0, 0.0],
            "dirt_condition": [0.0, 1.0],
            "track_condition_code": [1.0, 1.0],
            "horse_turf_heavy_win_rate": [0.1, 0.1],
            "horse_turf_light_win_rate": [0.2, 0.2],
            "horse_turf_soft_win_rate": [0.15, 0.15],
            "horse_turf_very_heavy_win_rate": [0.05, 0.05],
            "horse_dirt_heavy_win_rate": [0.12, 0.12],
            "horse_dirt_soft_win_rate": [0.11, 0.11],
            "horse_dirt_very_heavy_win_rate": [0.08, 0.08],
        }
    )
    out = apply_uniform_baba_jv_code(df, 4)
    assert out.loc[0, "track_condition_code"] == 4.0
    assert out.loc[1, "track_condition_code"] == 4.0


def test_going_match_score_turf_differs_code3_vs_code4() -> None:
    df = pd.DataFrame([_base_turf_row()])
    h3 = apply_uniform_baba_jv_code(df, 3)["going_match_score_turf"].iloc[0]
    h4 = apply_uniform_baba_jv_code(df, 4)["going_match_score_turf"].iloc[0]
    assert np.isclose(h3, 0.10)
    assert np.isclose(h4, 0.20)


def test_going_match_score_turf_imputed_differs_code3_vs_code4() -> None:
    df = pd.DataFrame([_base_turf_row(going_match_score_turf_imputed=1.5)])
    im3 = apply_uniform_baba_jv_code(df, 3)["going_match_score_turf_imputed"].iloc[0]
    im4 = apply_uniform_baba_jv_code(df, 4)["going_match_score_turf_imputed"].iloc[0]
    assert np.isclose(im3, 0.10)
    assert np.isclose(im4, 0.20)
    assert im3 != im4


def test_going_change_lag1_updates_on_scenario() -> None:
    df = pd.DataFrame([_base_turf_row(turf_condition=3.0, going_change_lag1=1.0)])
    out = apply_uniform_baba_jv_code(df, 4)
    assert np.isclose(out["going_change_lag1"].iloc[0], 2.0)
    assert int(out["going_worsening_flag"].iloc[0]) == 1


def test_heavy_winrate_uses_code3_only_not_code4() -> None:
    df = pd.DataFrame([_base_turf_row()])
    out3 = apply_uniform_baba_jv_code(df, 3)
    out4 = apply_uniform_baba_jv_code(df, 4)
    assert np.isclose(out3["going_x_turf_heavy_winrate"].iloc[0], 0.10)
    assert np.isclose(out4["going_x_turf_heavy_winrate"].iloc[0], 0.0)

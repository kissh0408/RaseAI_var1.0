"""data-generator Phase4: リーク修正の回帰テスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from builders.pace import PaceFeatureBuilder
from builders.past_performance import PastPerformanceBuilder


class _DummyConn:
    pass


def test_pace_enrich_does_not_persist_agari_z_race() -> None:
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "horse_id": ["H1", "H2", "H1", "H2"],
            "race_date": pd.to_datetime(["2024-01-01"] * 2 + ["2024-02-01"] * 2),
            "agari3f": [35.0, 36.0, 34.5, 35.5],
            "finish_time": [120.0, 121.0, 119.0, 120.5],
            "distance": [1600] * 4,
            "lap_times": [np.nan] * 4,
        }
    )
    out = PaceFeatureBuilder(_DummyConn()).enrich(df)
    assert "agari_z_race" not in out.columns
    assert "agari_z_score" in out.columns
    assert pd.isna(out.loc[out["race_id"] == "R1", "agari_z_score"]).all()


def test_rpr_score_class_stats_exclude_future_races() -> None:
    """3走目のクラス平均に3走目以降の finish_time が混ざらない（DA-2）。"""
    df = pd.DataFrame(
        {
            "horse_id": ["H1"] * 4,
            "race_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "finish_time": [100.0, 102.0, 104.0, 90.0],
            "grade_code": [1] * 4,
            "distance": [1600] * 4,
            "surface_code": [1] * 4,
            "track_condition_code": [1] * 4,
            "finish_rank": [1] * 4,
            "time_diff": [0.0] * 4,
        }
    )
    out = PastPerformanceBuilder(_DummyConn())._rpr_score(df.copy())

    # 3走目 raw: prior class mean=(100+102)/2=101, ft=104
    prior_std = np.std([100.0, 102.0], ddof=1)
    expected_rpr3 = (101.0 - 104.0) / prior_std
    # 4走目 feature = shift 済み過去 raw の平均（1-3走目）→ 3走目 raw のみ非NaN
    assert out.iloc[3]["rpr_score"] == pytest.approx(expected_rpr3, rel=1e-5)

    # 全期間 merge なら 3走目 mean=(100+102+104+90)/4=99 となり値が変わる
    leaky_mean = df["finish_time"].mean()
    leaky_std = df["finish_time"].std(ddof=1)
    leaky_rpr3 = (leaky_mean - 104.0) / leaky_std
    assert expected_rpr3 != pytest.approx(leaky_rpr3, rel=1e-5)

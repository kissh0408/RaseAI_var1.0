"""race_filters 純粋関数の characterization テスト。"""
from __future__ import annotations

import pandas as pd

from betting.src.race_filters import (
    attach_race_num,
    filter_df_by_race_num,
    filter_df_exclude_surface,
    parse_race_num_from_race_id,
    race_num_in_range,
)


class TestParseRaceNumFromRaceId:
    def test_tail_two_digits(self):
        s = pd.Series(["2026061405010101"])
        assert parse_race_num_from_race_id(s).tolist() == [1.0]


class TestAttachRaceNum:
    def test_adds_race_num_from_race_id(self):
        df = pd.DataFrame({"race_id": ["2026061405010101", "2026061405010112"]})
        out = attach_race_num(df)
        assert out["race_num"].tolist() == [1.0, 12.0]

    def test_fills_missing_only_when_column_partial(self):
        df = pd.DataFrame(
            {
                "race_id": ["2026061405010101", "2026061405010112"],
                "race_num": [5.0, pd.NA],
            }
        )
        out = attach_race_num(df)
        assert out["race_num"].tolist() == [5.0, 12.0]


class TestRaceNumInRange:
    def test_within_bounds(self):
        assert race_num_in_range(5, race_num_min=3, race_num_max=10) is True

    def test_nan_with_no_bounds_is_true(self):
        assert race_num_in_range(float("nan")) is True


class TestFilterDfByRaceNum:
    def test_excludes_races_outside_band(self):
        df = pd.DataFrame(
            {
                "race_id": ["A01", "A01", "B12", "B12"],
                "x": [1, 2, 3, 4],
            }
        )
        out = filter_df_by_race_num(df, race_num_min=1, race_num_max=11)
        assert out["race_id"].tolist() == ["A01", "A01"]


class TestFilterDfExcludeSurface:
    def test_excludes_barrier_races(self):
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R2", "R2"],
                "surface_code": [1, 1, 3, 3],
            }
        )
        out = filter_df_exclude_surface(df, [3])
        assert out["race_id"].tolist() == ["R1", "R1"]

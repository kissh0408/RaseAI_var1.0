"""Tests for HR place payout loading."""

from __future__ import annotations

import pandas as pd

from evaluation.place_payout_loader import (
    attach_place_payout,
    build_place_payout_lookup,
    build_place_payout_lookup_from_csvs,
    make_race_id_from_row,
)


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "record_id": "HR",
        "year": 2025,
        "month_day": 101,
        "course_code": 5,
        "kai": 1,
        "nichi": 1,
        "race_num": 1,
    }
    for i in range(1, 6):
        row[f"place_{i}_horse"] = "00"
        row[f"place_{i}_money"] = "0"
    row.update(overrides)
    return row


def test_make_race_id_from_row_matches_canonical_16_digits():
    row = _base_row(month_day=708, course_code=4, kai=2, nichi=3, race_num=11)

    assert make_race_id_from_row(row) == "2025070804020311"


def test_build_place_payout_lookup_parses_normal_slots(tmp_path):
    path = tmp_path / "race_hr_2025.csv"
    pd.DataFrame(
        [
            _base_row(
                place_1_horse="01",
                place_1_money="140",
                place_2_horse="03",
                place_2_money="180",
                place_3_horse="07",
                place_3_money="220",
            )
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    lookup = build_place_payout_lookup_from_csvs(tmp_path)

    assert lookup["2025010105010101"] == {1: 140, 3: 180, 7: 220}


def test_build_place_payout_lookup_respects_two_paid_places_for_small_fields(tmp_path):
    path = tmp_path / "race_hr_2025.csv"
    pd.DataFrame(
        [
            _base_row(
                race_num=2,
                place_1_horse="02",
                place_1_money="110",
                place_2_horse="05",
                place_2_money="130",
                place_3_horse="00",
                place_3_money="0",
            )
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    lookup = build_place_payout_lookup_from_csvs(tmp_path)

    assert lookup["2025010105010102"] == {2: 110, 5: 130}
    assert 3 not in lookup["2025010105010102"]


def test_build_place_payout_lookup_parses_dead_heat_extra_slots(tmp_path):
    path = tmp_path / "race_hr_2025.csv"
    pd.DataFrame(
        [
            _base_row(
                race_num=3,
                place_1_horse="01",
                place_1_money="110",
                place_2_horse="02",
                place_2_money="120",
                place_3_horse="03",
                place_3_money="130",
                place_4_horse="04",
                place_4_money="150",
                place_5_horse="05",
                place_5_money="170",
            )
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    lookup = build_place_payout_lookup_from_csvs(tmp_path)

    assert lookup["2025010105010103"] == {1: 110, 2: 120, 3: 130, 4: 150, 5: 170}


def test_attach_place_payout_uses_payout_presence_as_hit_truth():
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "horse_num": [1, 2, 3],
        }
    )
    lookup = {"R1": {1: 120, 2: 0}}

    out = attach_place_payout(df, lookup)

    assert out["place_payout"].tolist() == [120, 0, 0]
    assert out["place_multiplier"].tolist() == [1.2, 0.0, 0.0]
    assert out["place_paid"].tolist() == [True, False, False]


def test_build_place_payout_lookup_falls_back_to_csv_when_parquet_has_no_place(tmp_path):
    hr_parquet = tmp_path / "HR_preprocessed.parquet"
    pd.DataFrame(
        {
            "race_id": ["R0"],
            "bet_type": ["win"],
            "horse_num_1": [1],
            "horse_num_2": [0],
            "payout": [200],
        }
    ).to_parquet(hr_parquet, index=False)
    pd.DataFrame(
        [
            _base_row(
                race_num=4,
                place_1_horse="06",
                place_1_money="160",
            )
        ]
    ).to_csv(tmp_path / "race_hr_2025.csv", index=False, encoding="utf-8-sig")

    lookup = build_place_payout_lookup(hr_parquet=hr_parquet, hr_dir=tmp_path)

    assert lookup["2025010105010104"] == {6: 160}

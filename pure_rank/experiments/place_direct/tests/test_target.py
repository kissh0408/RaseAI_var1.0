"""target_place の定義・フィルタ適用のテスト（仕様書 §8）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from place_lib import apply_base_filters, compute_target_place  # noqa: E402


def test_target_top3():
    finish_rank = pd.Series([1, 2, 3, 4, 5, 3, 4])
    target = compute_target_place(finish_rank)
    assert target.tolist() == [1, 1, 1, 0, 0, 1, 0]
    # 境界値 3/4 の確認
    assert compute_target_place(pd.Series([3])).iloc[0] == 1
    assert compute_target_place(pd.Series([4])).iloc[0] == 0


def test_target_no_shift():
    """target が「当該レース」の実着順と一致すること（shift されていない）。"""
    df = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r1", "r2", "r2"],
            "horse_id": ["h1", "h2", "h3", "h1", "h4"],
            "finish_rank": [1, 3, 4, 2, 5],
        }
    )
    df["target_place"] = compute_target_place(df["finish_rank"])
    # h1 が r1 で 1着、r2 で 2着でも、それぞれの当該レースの着順で判定されている
    r1 = df[df["race_id"] == "r1"].set_index("horse_id")["target_place"]
    r2 = df[df["race_id"] == "r2"].set_index("horse_id")["target_place"]
    assert r1.loc["h1"] == 1 and r1.loc["h2"] == 1 and r1.loc["h3"] == 0
    assert r2.loc["h1"] == 1 and r2.loc["h4"] == 0


def test_filters_applied():
    cfg = {
        "filters": {
            "exclude_grade_codes": [8, 9],
            "exclude_abnormal_codes": [1, 3, 4],
            "min_horse_count": 5,
        }
    }
    df = pd.DataFrame(
        {
            "grade_code": [1, 8, 1, 1, 1, 1],
            "abnormal_code": [0, 0, 1, 0, 0, 0],
            "horse_count": [10, 10, 10, 4, 10, 10],
            "finish_rank": [1, 1, 1, 1, 0, 1],
        }
    )
    out = apply_base_filters(df, cfg)
    # 残るのは index 0, 5（1=grade8 除外, 2=abnormal1 除外, 3=horse_count<5 除外, 4=finish_rank<=0 除外）
    assert len(out) == 2
    assert out.index.tolist() == [0, 5]

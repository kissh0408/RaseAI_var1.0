"""wide_ev_core.py の単体テスト。

注意（2026-07-09）: 当初「odds>9999.9は異常値」と誤判定してフィルタを追加したが、
WinOdds由来のHarville理論オッズと突き合わせて実データは正当（大頭数レースの
超大穴ペアは数万倍に達するのが数学的に正常）と判明したため、その前提のテストは
削除・訂正した。有効なフィルタは「odds<=1.0（無効値）を除外」のみ。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from betting.src.wide_ev_core import (
    compute_implied_prob,
    compute_pair_ev,
    load_wide_odds_lookup,
    norm_pair,
)


def _write_wide_odds_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_wide_odds_lookup_filters_zero_or_negative_odds(tmp_path: Path):
    _write_wide_odds_csv(
        tmp_path / "WideOdds_2025.csv",
        [
            {"race_id": 2025010101010101, "horse_num_1": 1, "horse_num_2": 2, "odds_status": "ok", "odds": 27.4},
            {"race_id": 2025010101010101, "horse_num_1": 2, "horse_num_2": 3, "odds_status": "ok", "odds": 0.5},
        ],
    )
    lookup = load_wide_odds_lookup([2025], tmp_path, odds_type="Wide")
    race = lookup["2025010101010101"]
    assert norm_pair(1, 2) in race
    assert norm_pair(2, 3) not in race  # 0.5 は1.0以下で無効


def test_load_wide_odds_lookup_keeps_large_field_extreme_longshot_odds(tmp_path: Path):
    """大頭数レースの超大穴ペアは数万倍に達するのが数学的に正常なので保持する。

    実測例（2026-07-09、16頭立てレースでWinOdds由来Harville理論値と照合済み）:
    ペア(11,13)の実オッズ60415.1は理論公正オッズ80553.4の約0.75倍
    （控除率25%相当）で、他の全ペアと同一比率だった。異常値ではない。
    """
    _write_wide_odds_csv(
        tmp_path / "WideOdds_2025.csv",
        [{"race_id": 2025010101010101, "horse_num_1": 11, "horse_num_2": 13, "odds_status": "ok", "odds": 60415.1}],
    )
    lookup = load_wide_odds_lookup([2025], tmp_path, odds_type="Wide")
    assert lookup["2025010101010101"][norm_pair(11, 13)] == pytest.approx(60415.1)


def test_compute_pair_ev_and_implied_prob():
    assert compute_pair_ev(0.1, 5.0) == pytest.approx(0.5)
    assert compute_implied_prob(5.0, overround=1.2) == pytest.approx((1 / 5.0) / 1.2)

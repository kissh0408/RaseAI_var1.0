"""fit/TEST 分割境界のテスト（仕様書 §8, §9）。

分割ロジックは prob_fusion/src/oos_protocol.py の split_oos_periods をそのまま
import して使う（本実験専用の分割関数を新規実装しない。fit期間はformal λ fitと
同一の FIT_START/FIT_END を再利用する。仕様書 §5）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prob_fusion.src.oos_protocol import (  # noqa: E402
    FIT_END,
    FIT_START,
    TEST_START,
    split_oos_periods,
)

DATA_DIR = EXP_DIR / "data"
CONFIG_PATH = EXP_DIR / "config.json"
PLACE_DIRECT_SCORES = (
    ROOT / "pure_rank" / "experiments" / "place_direct" / "scores" / "probs_place_direct_fold2_oos.parquet"
)


def _synthetic_df() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2026-05-24", freq="17D")
    return pd.DataFrame({"race_date": dates, "race_id": [f"r{i}" for i in range(len(dates))]})


def test_fit_test_boundaries():
    df = _synthetic_df()
    fit_df, test_df = split_oos_periods(df)

    assert fit_df["race_date"].min() >= pd.Timestamp(FIT_START)
    assert fit_df["race_date"].max() <= pd.Timestamp(FIT_END)
    assert test_df["race_date"].min() >= pd.Timestamp(TEST_START)

    # 重複ゼロ
    assert set(fit_df["race_id"]) & set(test_df["race_id"]) == set()

    # config.json の fit_period / test_period と整合していること（後出し変更禁止の固定値確認）
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["fit_period"]["start"] == FIT_START
    assert cfg["fit_period"]["end"] == FIT_END
    assert cfg["test_period"]["start"] == TEST_START


@pytest.mark.skipif(
    not (DATA_DIR / "test_2025.parquet").is_file(),
    reason="build_dataset.py 未実行（実データ統合テスト）",
)
def test_test_race_set():
    """TEST race_id 集合が既存 OOS の 4,775 レースと一致すること。"""
    test_df = pd.read_parquet(DATA_DIR / "test_2025.parquet")
    test_df["race_id"] = test_df["race_id"].astype(str)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert test_df["race_id"].nunique() == cfg["test_period"]["expected_n_races"]
    assert len(test_df) == cfg["test_period"]["expected_n_horses"]

    if PLACE_DIRECT_SCORES.is_file():
        ref = pd.read_parquet(PLACE_DIRECT_SCORES, columns=["race_id"])
        ref["race_id"] = ref["race_id"].astype(str)
        assert set(test_df["race_id"].unique()) == set(ref["race_id"].unique())

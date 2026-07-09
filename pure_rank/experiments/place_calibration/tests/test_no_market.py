"""市場情報混入禁止のテスト（仕様書 §8, §9, プロジェクト憲法）。

較正 fit の入力は p_win（既存 fold2 OOS コードパス由来）と複勝実績のみで、
odds / popularity / ninki / market_log_odds / init_score そのものを
較正入力・保存データに残さないことを確認する。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_DIR / "data"

BANNED_SUBSTRINGS = ["odds", "popularity", "ninki", "market_log_odds", "init_score"]


def _assert_clean_columns(df: pd.DataFrame) -> None:
    for col in df.columns:
        lowered = col.lower()
        for banned in BANNED_SUBSTRINGS:
            assert banned not in lowered, f"禁止パターン '{banned}' が列 '{col}' に含まれる"


@pytest.mark.skipif(
    not (DATA_DIR / "fit_2023_2024.parquet").is_file(),
    reason="build_dataset.py 未実行（実データ統合テスト）",
)
def test_calibration_inputs_clean():
    fit_df = pd.read_parquet(DATA_DIR / "fit_2023_2024.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_2025.parquet")
    _assert_clean_columns(fit_df)
    _assert_clean_columns(test_df)

    # 較正の入力変数は p_win と y_place のみであること（列の存在確認）
    for col in ("p_win", "y_place", "race_id", "horse_count"):
        assert col in fit_df.columns
        assert col in test_df.columns

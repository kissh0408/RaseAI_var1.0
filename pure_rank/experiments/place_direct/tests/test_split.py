"""fold2 分割境界のテスト（仕様書 §8, §9）。

分割ロジックは pure_rank/src/train.py の get_fold_split をそのまま import して使う
（本実験専用の分割関数を新規実装しない。market_leak_diagnostic と同一パターン）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
_PURE_RANK_SRC = str(ROOT / "pure_rank" / "src")
sys.path.insert(0, _PURE_RANK_SRC)

from train import get_fold_split  # noqa: E402

# pure_rank/src/common.py がトップレベル common パッケージをシャドウし、後続テスト
# （tests/test_realtime_jv_parsing.py 等）の `from common.data...` import を壊すため後始末する。
sys.path.remove(_PURE_RANK_SRC)
sys.modules.pop("common", None)

FOLD_VALID_YEARS = ["2022", "2023", "2024"]
FOLD = 2  # -> valid year "2023"


def _synthetic_df() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2026-05-24", freq="17D")
    return pd.DataFrame({"race_date": dates, "race_id": [f"r{i}" for i in range(len(dates))]})


def test_fold2_boundaries():
    df = _synthetic_df()
    train_df, valid_df = get_fold_split(df, FOLD, FOLD_VALID_YEARS)

    assert train_df["race_date"].max() < pd.Timestamp("2023-01-01")
    assert valid_df["race_date"].min() >= pd.Timestamp("2023-01-01")
    assert valid_df["race_date"].max() <= pd.Timestamp("2023-12-31")

    # ES データが 2023 年のみであること
    assert set(valid_df["race_date"].dt.year.unique().tolist()) == {2023}

    # 2024 年以降が学習系（train + ES）に一切含まれないこと
    combined_years = set(train_df["race_date"].dt.year.unique().tolist()) | set(
        valid_df["race_date"].dt.year.unique().tolist()
    )
    assert all(y < 2024 for y in combined_years)

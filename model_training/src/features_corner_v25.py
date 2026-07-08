"""
features_corner_v25.py — コーナー通過位置変化特徴量（v25）

corner_position_change_lag1: 前走の1コーナー→4コーナー位置取り変化
corner_advance_rate:          前走の追い込み指数（位置変化の絶対値/頭数）

リーク防止: shift(1) で当該レースを除外した前走値のみ使用。
データソース: SE.corner_1, SE.corner_4, SE.n_horses
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_NEW_COLS = [
    "corner_position_change_lag1",
    "corner_advance_rate",
]


def v25_corner_column_names() -> list[str]:
    return list(_NEW_COLS)


def add_corner_v25_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    corner_position_change_lag1 と corner_advance_rate を追加する。

    前提: df に corner_1, corner_4, n_horses, ketto_num の各列が存在すること。
    corner_1/corner_4 は数値型（0=欠損）。直線競馬など1コーナーなしのレースは NaN になる。
    また、df は (race_date, ketto_num) 順に時系列ソート済みであること。
    本関数は内部ソートを行わず sort=False でgroupby shiftを使用するため、
    呼び出し側でのソート保証が必要。（create_pastfeatures.pyのパイプラインは保証済み）
    """
    df = df.copy()

    # 冪等性ガード：既に列が存在する場合は早期リターン
    if all(c in df.columns for c in _NEW_COLS):
        return df

    has_c1 = "corner_1" in df.columns
    has_c4 = "corner_4" in df.columns
    has_nh = "n_horses" in df.columns

    if has_c1 and has_c4 and has_nh:
        c1 = pd.to_numeric(df["corner_1"], errors="coerce").replace(0, np.nan)
        c4 = pd.to_numeric(df["corner_4"], errors="coerce").replace(0, np.nan)
        nh = pd.to_numeric(df["n_horses"], errors="coerce").replace(0, np.nan)

        # 位置取り変化 = 4コーナー順位 - 1コーナー順位（負=追い上げ、正=沈み込み）
        pos_change = c4 - c1
        # 追い込み指数 = abs(位置変化) / 頭数（0〜1の無次元化）
        advance = pos_change.abs() / nh

        # shift(1) で前走値として使用（リーク防止）
        grp = df["ketto_num"]
        df["corner_position_change_lag1"] = (
            pos_change.groupby(grp, sort=False).shift(1).astype("float32")
        )
        df["corner_advance_rate"] = (
            advance.groupby(grp, sort=False).shift(1).astype("float32")
        )
    else:
        df["corner_position_change_lag1"] = np.nan
        df["corner_advance_rate"] = np.nan

    return df

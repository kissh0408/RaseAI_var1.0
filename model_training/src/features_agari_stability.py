"""
features_agari_stability.py — 上がり3F安定性・傾向特徴量

生成特徴量:
    agari3f_rank_stability          : 過去走の上がり3Fレース内順位の安定性（展開std、低いほど安定）
    agari3f_rank_trend              : 直近6走の上がり3F順位の改善傾向（線形回帰傾き、負=改善）
    agari3f_lag1_vs_course_avg_diff : 前走の上がり順位 vs 当該コース累積平均の乖離（コース適性差）

問題の根拠（v13バックテスト）:
    芝マイル（1201-1600m）で5歳69.1%、6歳31.5%と年齢が上がるほど急落。
    また差し・先行脚質で76%と回収率が低い。
    加齢に伴う上がり能力の衰退（agari3f_rank_trend で捉える）と、
    コース適性の偏り（agari3f_lag1_vs_course_avg_diff で捉える）が原因と推測。

リーク防止:
    agari3f_rank_in_race_lag1 はすでに shift(1) 済みの値。
    展開std は lag 値の expanding.std() を更に .shift(1) することで
    「現レースを除外した過去の安定性」を保証する。
    ただし expanding.std() 自体が lag 値（過去走分）のみを扱うため、
    shift 不要で正しくリーク防止済み（lag 値の展開stdは当該レース情報を含まない）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_MIN_PERIODS_STABILITY = 4
_MIN_PERIODS_TREND = 4
_TREND_WINDOW = 6


def add_agari_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    上がり3F安定性・傾向特徴量を df に追加して返す。

    前提: df に以下の列が存在すること:
        - agari3f_rank_in_race_lag1       : 前走の上がり3Fレース内順位（NaN=データなし）
        - horse_agari3f_course_rank_avg   : 馬×コースの上がり順位累積平均（features_agari_course由来）
        - ketto_num                        : 馬個体識別子
        - date                             : レース日（ソート用）

    Returns:
        新特徴量3列を追加した DataFrame（行数・順序は変更しない）
    """
    new_cols = [
        "agari3f_rank_stability",
        "agari3f_rank_trend",
        "agari3f_lag1_vs_course_avg_diff",
    ]
    if all(c in df.columns for c in new_cols):
        return df

    required = ["agari3f_rank_in_race_lag1", "ketto_num"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df

    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    lag_rank = pd.to_numeric(df["agari3f_rank_in_race_lag1"], errors="coerce")
    horse_key = df["ketto_num"].astype(str)

    # --- 1. 上がり順位の安定性（展開std）---
    # lag_rank は前走の値なので「前走までの全走の上がり順位」の expanding std が
    # 「当該レースを除外した安定性」を表す。shift(1) は不要。
    stability = (
        lag_rank.groupby(horse_key, sort=False)
        .transform(lambda x: x.expanding(min_periods=_MIN_PERIODS_STABILITY).std())
        .astype("float32")
    )
    df["agari3f_rank_stability"] = stability

    # --- 2. 上がり順位の改善傾向（直近6走の線形回帰傾き）---
    # 負の傾き = 順位が下がる（=改善）、正の傾き = 順位が上がる（=悪化）
    # rolling(window).apply(raw=True) で行ループを排除し pandas の C 実装に委ねる。
    # lag_rank は shift(1) 済みの値のため、当該レースは含まれない（リーク防止済み）。
    def _slope_from_window(arr: np.ndarray) -> float:
        """rolling window 内の有効値に対する線形回帰傾きを返す。"""
        valid_mask = ~np.isnan(arr)
        valid_vals = arr[valid_mask]
        if len(valid_vals) < _MIN_PERIODS_TREND:
            return np.nan
        positions = np.where(valid_mask)[0].astype("float64")
        if positions.std() < 1e-10:
            return 0.0
        return float(np.polyfit(positions, valid_vals, 1)[0])

    trend = (
        lag_rank.groupby(horse_key, sort=False)
        .transform(
            lambda x: x.rolling(
                window=_TREND_WINDOW, min_periods=_MIN_PERIODS_TREND
            ).apply(_slope_from_window, raw=True)
        )
        .astype("float32")
    )
    df["agari3f_rank_trend"] = trend

    # --- 3. 前走上がり順位 vs コース累積平均の乖離 ---
    if "horse_agari3f_course_rank_avg" in df.columns:
        course_avg = pd.to_numeric(df["horse_agari3f_course_rank_avg"], errors="coerce")
        diff = (lag_rank - course_avg).astype("float32")
        df["agari3f_lag1_vs_course_avg_diff"] = diff
    else:
        df["agari3f_lag1_vs_course_avg_diff"] = np.nan

    # 元のインデックス順に戻す（重複インデックスでも安全な位置ベース復元）
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df

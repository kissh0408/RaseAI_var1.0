"""
features_common.py — 特徴量エンジニアリング共通ユーティリティ

各 features_*.py モジュールで重複していたヘルパー関数を一箇所に集約し、
import コストと保守コストを削減する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bayesian_cumulative_rate(
    group_key: pd.Series,
    target_flag: pd.Series,
    count_flag: pd.Series,
    prior_n: float,
    prior_mean: float,
    min_periods: int,
) -> pd.Series:
    """
    グループ別・累積ベイズ平滑化率（当該行を cumsum-current で除外）。

    時系列順にソート済みの DataFrame に対して呼ぶこと。
    cumsum - current により当該行自体をカウントから除外するため、
    shift(1) なしでリーク防止が成立する。

    Args:
        group_key    : グループキー（str / category 推奨）
        target_flag  : 勝利/複勝フラグ（int8）
        count_flag   : 集計対象行フラグ（int8、対象条件に合致する行のみ 1）
        prior_n      : ベイズ事前出走数
        prior_mean   : ベイズ事前平均率
        min_periods  : この未満の過去出走数では NaN を返す（生サンプル数）

    Returns:
        float32 Series（行数・インデックスは入力と同一）
    """
    # 当該行を除いた累積出走数・的中数
    cum_runs = count_flag.groupby(group_key, sort=False).cumsum() - count_flag
    cum_wins = target_flag.groupby(group_key, sort=False).cumsum() - target_flag

    smoothed = (cum_wins + prior_n * prior_mean) / (cum_runs + prior_n)
    # 生サンプルが min_periods に満たない場合は情報不足として NaN
    return smoothed.where(cum_runs >= min_periods, np.nan).astype("float32")

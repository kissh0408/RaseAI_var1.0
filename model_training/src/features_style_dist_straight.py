"""
features_style_dist_straight.py — 脚質×距離帯 適性特徴量

生成特徴量:
    running_style_dist_band_win_rate   : 馬×脚質×距離帯の過去勝率（ベイズ平滑化, min_periods=3）
    running_style_dist_band_place_rate : 馬×脚質×距離帯の複勝率（3着以内率）
    jockey_style_dist_win_rate         : 騎手×脚質×距離帯の過去勝率（ベイズ平滑化, min_periods=10）

問題の根拠（v13バックテスト）:
    芝マイル（1201-1600m）で脚質別ROI: 逃げ145.2% vs 先行72.1% / 差し76.0% の格差。
    現行の horse_style_course_win_rate は距離帯を考慮しておらず、
    短距離得意の逃げ馬がマイルに出た場合の適性低下を学習できていない。

距離帯定義 (distance_band):
    1 = sprint  (     〜1200m)
    2 = mile    (1201〜1600m)
    3 = middle  (1601〜2000m)
    4 = long    (2001m〜)

リーク防止:
    cumsum - current で当該レースを除外したベイズ平滑化勝率を算出する。

Train-serving skew 修正 (c5):
    running_style_code は SE レコードのレース後確定フィールドのため推論時は全行 0。
    horse_modal_running_style（過去 N 走の最頻脚質コード）を使用することで
    学習時・推論時の両方で有効な脚質キーを生成する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_training.src.features_common import _bayesian_cumulative_rate

_HORSE_PRIOR_N = 10.0
_HORSE_PRIOR_MEAN = 0.072

_JOCKEY_PRIOR_N = 20.0
_JOCKEY_PRIOR_MEAN = 0.10


def _dist_band(distance: pd.Series) -> pd.Series:
    """距離をバンドコード（1〜4）に変換。"""
    d = pd.to_numeric(distance, errors="coerce")
    band = pd.Series(np.nan, index=distance.index, dtype="Int8")
    band = band.where(d.isna(), 4)          # デフォルト: long
    band = band.where(d.isna() | (d > 2000), 3)   # middle
    band = band.where(d.isna() | (d > 1600), 2)   # mile
    band = band.where(d.isna() | (d > 1200), 1)   # sprint
    return band


def add_style_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    脚質×距離帯 適性特徴量を df に追加して返す。

    Args:
        df: features_past_v13 など（ketto_num, jockey_code, horse_modal_running_style,
            distance, finish_rank, date 列を持つ DataFrame）
            horse_modal_running_style が存在しない場合は running_style_code にフォールバック。
    Returns:
        新特徴量3列を追加した DataFrame（行数・順序は変更しない）
    """
    new_cols = [
        "running_style_dist_band_win_rate",
        "running_style_dist_band_place_rate",
        "jockey_style_dist_win_rate",
    ]
    if all(c in df.columns for c in new_cols):
        return df

    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    place_flag = (finish <= 3).astype("int8")

    # horse_modal_running_style を優先使用（推論時も0以外の値が入る）
    # フォールバック: horse_modal_running_style 列が存在しない場合は running_style_code を使用
    if "horse_modal_running_style" in df.columns:
        style_num = pd.to_numeric(df["horse_modal_running_style"], errors="coerce")
    else:
        style_num = pd.to_numeric(df["running_style_code"], errors="coerce")
    valid_style = (style_num >= 1) & (style_num <= 4)

    dist_band = _dist_band(df["distance"])
    valid_dist = dist_band.notna()
    valid = valid_style & valid_dist

    style_str = style_num.astype("Int8").astype(str)
    band_str = dist_band.astype(str)
    style_dist_key = style_str + "_" + band_str
    style_dist_key = style_dist_key.where(valid, "__invalid__")

    # 馬×脚質×距離帯
    horse_key = df["ketto_num"].astype(str) + "_" + style_dist_key
    count = valid.astype("int8")

    df["running_style_dist_band_win_rate"] = _bayesian_cumulative_rate(
        horse_key, (win_flag * count).astype("int8"), count,
        _HORSE_PRIOR_N, _HORSE_PRIOR_MEAN, min_periods=3,
    ).where(valid, np.nan)

    df["running_style_dist_band_place_rate"] = _bayesian_cumulative_rate(
        horse_key, (place_flag * count).astype("int8"), count,
        _HORSE_PRIOR_N, _HORSE_PRIOR_MEAN * 3, min_periods=3,
    ).where(valid, np.nan)

    # 騎手×脚質×距離帯
    jockey_key = df["jockey_code"].astype(str) + "_" + style_dist_key
    jockey_count = valid.astype("int8")

    df["jockey_style_dist_win_rate"] = _bayesian_cumulative_rate(
        jockey_key, (win_flag * jockey_count).astype("int8"), jockey_count,
        _JOCKEY_PRIOR_N, _JOCKEY_PRIOR_MEAN, min_periods=10,
    ).where(valid, np.nan)

    # 元のインデックス順に戻す（重複インデックスでも安全な位置ベース復元）
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df

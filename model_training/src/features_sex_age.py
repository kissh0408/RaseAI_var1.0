"""
features_sex_age.py — 性別×距離帯・騸馬年齢特徴量

生成特徴量:
    sex_dist_band_win_rate   : 性別×距離帯での累積勝率（全馬集計、ベイズ平滑化、cumsum-current）
    sex_dist_band_place_rate : 同じく複勝率（3着以内率）
    age_peak_deviation       : |age - 4| — 平地競走のピーク年齢(4歳)からの乖離
    gelding_past_peak_flag   : 騸馬(sex_code==3)かつ age >= 5 のフラグ

問題の根拠（v13バックテスト）:
    性別別フラットベットROI: 牡1.43 vs 牝1.09 vs 騸0.78（n=1886）。
    騸馬は行動矯正のため去勢された馬で、高齢化に伴う能力低下傾向が顕著。
    現行モデルに性別×距離・騸馬×年齢の交互作用特徴量が欠如。

JV-Link sex_code: 1=牡, 2=牝, 3=騸
ピーク年齢: 4歳（平地競走の一般的なピーク）

リーク防止:
    sex_dist_band_win_rate / place_rate:
        全馬の実績を累積するがcumsum-currentで当該レースを除外する。
        ソート順（date, race_id, ketto_num）でデータを整列後に計算。
    age_peak_deviation / gelding_past_peak_flag:
        レース前に既知の値のみ使用。リーク非該当。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SEX_PRIOR_N = 30.0     # 全馬集計のため事前分布を標準的な強さに
_SEX_PRIOR_WIN = 0.072
_SEX_PRIOR_TOP3 = 0.22
_MIN_SEX_DIST = 10      # 10走以上で有効（全馬集計なので早期から十分なサンプル）

_PEAK_AGE = 4           # 平地競走のピーク年齢


def _dist_band(distance: pd.Series) -> pd.Series:
    """距離帯コード: 1=sprint(<=1200m), 2=mile(1201-1600m), 3=middle(1601-2000m), 4=long(>2000m)"""
    d = pd.to_numeric(distance, errors="coerce")
    band = pd.Series(4, index=distance.index, dtype="Int8")
    band = band.where(d.isna() | (d > 2000), 3)
    band = band.where(d.isna() | (d > 1600), 2)
    band = band.where(d.isna() | (d > 1200), 1)
    return band


def add_sex_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    性別×距離帯特徴量・騸馬年齢特徴量を df に追加して返す。

    Args:
        df: features_past_v14 相当（ketto_num, sex_code, age, distance,
            finish_rank, date 列を持つ）
    Returns:
        新特徴量4列を追加した DataFrame（行数・順序変更なし）
    """
    new_cols = [
        "sex_dist_band_win_rate",
        "sex_dist_band_place_rate",
        "age_peak_deviation",
        "gelding_past_peak_flag",
    ]
    if all(c in df.columns for c in new_cols):
        return df

    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    sex = pd.to_numeric(df["sex_code"], errors="coerce")
    age = pd.to_numeric(df["age"], errors="coerce")
    dist_band = _dist_band(df["distance"])

    win_flag = (finish == 1).astype("int8")
    top3_flag = (finish <= 3).astype("int8")
    run_flag = pd.Series(1, index=df.index, dtype="int8")

    # 性別×距離帯グループキー（例: "1_2" = 牡×マイル）
    sex_str = sex.astype("Int8").astype(str).fillna("na")
    dist_str = dist_band.astype(str)
    group_key = sex_str + "_" + dist_str

    # cumsum-current で当該行を除いた累積集計
    cum_runs = run_flag.groupby(group_key, sort=False).cumsum() - run_flag
    cum_wins = win_flag.groupby(group_key, sort=False).cumsum() - win_flag
    cum_top3 = top3_flag.groupby(group_key, sort=False).cumsum() - top3_flag

    sex_win_rate = (cum_wins + _SEX_PRIOR_N * _SEX_PRIOR_WIN) / (cum_runs + _SEX_PRIOR_N)
    sex_top3_rate = (cum_top3 + _SEX_PRIOR_N * _SEX_PRIOR_TOP3) / (cum_runs + _SEX_PRIOR_N)

    df["sex_dist_band_win_rate"] = sex_win_rate.where(
        cum_runs >= _MIN_SEX_DIST, np.nan
    ).astype("float32")
    df["sex_dist_band_place_rate"] = sex_top3_rate.where(
        cum_runs >= _MIN_SEX_DIST, np.nan
    ).astype("float32")

    # --- 年齢ピーク乖離（絶対値） ---
    df["age_peak_deviation"] = (age - _PEAK_AGE).abs().astype("float32")

    # --- 騸馬×高齢フラグ ---
    is_gelding = sex == 3
    is_past_peak = age >= 5
    df["gelding_past_peak_flag"] = (is_gelding & is_past_peak).astype("int8")

    # 元のインデックス順に戻す（重複インデックスでも安全な位置ベース復元）
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df

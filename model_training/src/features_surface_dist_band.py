"""
features_surface_dist_band.py — v16 追加特徴量

動機（v13バックテスト分析）:
    - 牝_turf_mile: ROI 0.93（n=1228レース） ← 損失
    - 牝_turf_long: ROI 0.82（n=307レース） ← 損失
    - 牝_dirt_long: ROI 0.55（n=115レース） ← 大きな損失
    - 騸_turf全般:  ROI 0.93以下
    - 芝_長距離(>=2001m): ROI 0.99（跛行）

    既存の sex_dist_band_win_rate は距離帯別だが芝/ダートを区別しない。
    牝馬・騸馬の 芝_マイル vs ダート_長距離 は全く別物なのに同じ特徴量に混在。
    芝/ダート × 距離帯 × 性別 の組み合わせ特徴量が欠如 → これを追加する。

追加特徴量（全馬集計、cumsum-current、Bayesian smoothing）:
    1. sex_surface_dist_band_win_rate   — 性別×芝/ダート×距離帯 累積勝率
    2. sex_surface_dist_band_place_rate — 性別×芝/ダート×距離帯 累積連対率（上位3着）

リーク防止:
    - 全馬集計（特定馬の情報ではなく母集団の傾向）
    - cumsum - 当該行 により当該レースを除外
    - sort_values("date") を使って時系列順を保証

使用方法:
    from features_surface_dist_band import add_surface_dist_band_features
    df = add_surface_dist_band_features(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# distance band boundaries (metres)
_DIST_BINS = [0, 1200, 1600, 2000, 100_000]
_DIST_LABELS = ["sprint", "mile", "middle", "long"]

# Bayesian smoothing hyper-parameters
_PRIOR_N = 30.0      # 仮想観測数（十分に安定した事前分布）
_PRIOR_WIN = 0.072   # 全クラス平均勝率
_PRIOR_PLACE = 0.215 # 全クラス平均連対率（3着以内）

# 少出走グループの最低件数（これ未満はNaN のまま）
_MIN_GROUP_RUNS = 5


def add_surface_dist_band_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    性別×芝/ダート×距離帯の累積勝率/連対率を追加する。

    Parameters
    ----------
    df : pd.DataFrame
        features_past_v15.parquet のスキーマを持つDF。
        必須列: date, race_id, ketto_num, sex_code, track_code, distance,
                finish_rank (1=1着, 2=2着, 3=3着以内)

    Returns
    -------
    pd.DataFrame
        元DFに以下を追加:
        - sex_surface_dist_band_win_rate   (float32)
        - sex_surface_dist_band_place_rate (float32)
    """
    # ------------------------------------------------------------------ #
    # 前処理                                                               #
    # ------------------------------------------------------------------ #
    # インデックス保存 → 時系列ソート保証（他の features_*.py と同じ標準実装）
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    sex = pd.to_numeric(df["sex_code"], errors="coerce").astype("Int8")

    # 芝フラグ: JV-Link track_code < 50 = 芝, >= 50 = ダート
    is_turf = (pd.to_numeric(df["track_code"], errors="coerce") < 50).astype("Int8")

    dist_num = pd.to_numeric(df["distance"], errors="coerce")
    dist_band_cat = pd.cut(dist_num, bins=_DIST_BINS, labels=_DIST_LABELS, right=True)

    # グループキー: "sex_surface_dist" e.g., "2_0_mile" = 牝×ダート×マイル
    group_key = (
        sex.astype(str) + "_"
        + is_turf.astype(str) + "_"
        + dist_band_cat.astype(str)
    )

    # 勝ち/3着以内フラグ
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    place_flag = (finish <= 3).astype("int8")
    run_flag = finish.notna().astype("int8")

    # ------------------------------------------------------------------ #
    # cumsum-current（当該行を除外した累積集計）                            #
    # ------------------------------------------------------------------ #
    cum_runs = run_flag.groupby(group_key, sort=False).cumsum() - run_flag
    cum_wins = win_flag.groupby(group_key, sort=False).cumsum() - win_flag
    cum_places = place_flag.groupby(group_key, sort=False).cumsum() - place_flag

    # ------------------------------------------------------------------ #
    # Bayesian smoothing                                                   #
    # ------------------------------------------------------------------ #
    win_rate = (cum_wins + _PRIOR_N * _PRIOR_WIN) / (cum_runs + _PRIOR_N)
    place_rate = (cum_places + _PRIOR_N * _PRIOR_PLACE) / (cum_runs + _PRIOR_N)

    # 出走実績が少なすぎるグループはNaN（事前分布に潰される → 情報なし）
    insufficient = cum_runs < _MIN_GROUP_RUNS
    win_rate = win_rate.where(~insufficient)
    place_rate = place_rate.where(~insufficient)

    df["sex_surface_dist_band_win_rate"] = win_rate.astype("float32")
    df["sex_surface_dist_band_place_rate"] = place_rate.astype("float32")

    # --- 元の順序・インデックスに復元 ---
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df

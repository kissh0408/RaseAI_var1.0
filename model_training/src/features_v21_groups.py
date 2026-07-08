"""
features_v21_groups.py — v21 Group A/B/C/D 特徴量の共通モジュール

build_features_v21.py と create_pastfeatures.py の両方から import して使用する。
DRY原則に従い、パラメータ（prior値等）を一箇所に集約して不整合を防ぐ。

追加特徴量（12列）:

    グループA: 洋芝・北海道適性（5列）
        youshiba_win_rate          : 馬×洋芝（札幌/函館 course_code in [1,2]）過去勝率
        youshiba_top3_rate         : 馬×洋芝 3着以内率
        sire_youshiba_win_rate     : 父馬×洋芝産駒勝率（ベイズ平滑化）
        horse_youshiba_exp_count   : 馬×洋芝累積出走数（NaN不可、初回=0.0）
        kokai_koban_win_rate       : 馬×小回り（course_code in [1,2,3,7]）勝率

    グループB: 芝稍重適性（4列）
        horse_soft_turf_win_rate   : 馬×芝×稍重 過去勝率（ベイズ平滑化）
        horse_soft_turf_top3_rate  : 馬×芝×稍重 3着以内率（ベイズ平滑化）
        sire_soft_turf_win_rate    : 父馬×芝×稍重産駒勝率（ベイズ平滑化）
        going_soft_exp_count       : 馬×芝×稍重累積出走数（NaN不可、初回=0.0）

    グループC: グローバルスピード指数（2列、speed_index_course_adjは当日タイム使用=リークのため除外）
        speed_index_3run_avg       : 馬ごとのスピード指数過去3走平均
        speed_index_trend          : 馬ごとのスピード指数過去3走の線形傾向

    グループD: ペース圧力×脚質交差（1列）
        pace_dist_style_win_rate   : running_style_code × 距離帯 × turf_condition 別勝率

使用方法:
    from model_training.src.features_v21_groups import (
        add_youshiba_features,
        add_sire_youshiba_features,
        add_kokai_koban_features,
        add_horse_soft_turf_features,
        add_sire_soft_turf_features,
        add_speed_index_features,
        add_pace_dist_style_features,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ===========================================================================
# ユーティリティ（build_features_v21.pyと同一パラメータ）
# ===========================================================================

def _bayesian_rate_horse(
    df: pd.DataFrame,
    horse_id_col: str,
    flag_series: pd.Series,
    win_series: pd.Series,
    prior_mean: float,
    prior_n: int,
    min_periods: int = 1,
    fill_zero: bool = False,
) -> pd.Series:
    """
    馬単位でのベイズ平滑化率を計算する。リーク防止のため cumsum-current を使用。

    flag_series : 条件フラグ（該当コース/馬場で出走した場合=1）
    win_series  : 勝利/上位着順フラグ（condition & win の場合=1）
    fill_zero   : True のとき条件出走0回でも NaN でなく先験値を返す
    """
    grp_key = df[horse_id_col].astype(str).fillna("__nan__")

    # 当該行除外の累積（cumsum - current）
    cum_runs = flag_series.groupby(grp_key, sort=False).cumsum() - flag_series
    cum_wins = win_series.groupby(grp_key, sort=False).cumsum() - win_series

    smoothed = (cum_wins + prior_n * prior_mean) / (cum_runs + prior_n)

    if fill_zero:
        # 未出走（cum_runs=0）でも先験率を返す（カウント特徴量では使わない）
        return smoothed.astype("float32")
    else:
        return smoothed.where(cum_runs >= min_periods, np.nan).astype("float32")


def _cumcount_horse(
    df: pd.DataFrame,
    horse_id_col: str,
    flag_series: pd.Series,
) -> pd.Series:
    """
    馬単位で条件出走の累積カウントを返す（当該行除外）。
    初出走は 0.0 を返す（NaN なし）。
    """
    grp_key = df[horse_id_col].astype(str).fillna("__nan__")
    cum_runs = flag_series.groupby(grp_key, sort=False).cumsum() - flag_series
    return cum_runs.astype("float32")


# ===========================================================================
# グループA: 洋芝・北海道適性特徴量
# （パラメータは build_features_v21.py と完全一致）
# ===========================================================================

def add_youshiba_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    洋芝（札幌・函館）コースでの適性を計算する。

    洋芝は野芝と異なりクッション性が高く、パワー型有利・スタミナ消費型。
    course_code in [1, 2] が洋芝開催の代理指標（コース情報から判定）。
    """
    if all(c in df.columns for c in ["youshiba_win_rate", "youshiba_top3_rate", "horse_youshiba_exp_count"]):
        return df

    # 洋芝フラグ: 札幌(1)・函館(2)
    youshiba_mask = df["course_code"].isin([1, 2]).astype("int8")
    # 3着以内フラグ
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    top3_flag = (finish <= 3).astype("int8")
    # 洋芝×勝利/3着以内
    youshiba_win = (youshiba_mask * win_flag).astype("int8")
    youshiba_top3 = (youshiba_mask * top3_flag).astype("int8")

    # 馬×洋芝 過去勝率（洋芝出走0回は NaN）
    df["youshiba_win_rate"] = _bayesian_rate_horse(
        df, "ketto_num",
        youshiba_mask, youshiba_win,
        prior_mean=0.0714, prior_n=20, min_periods=1,
    )

    # 馬×洋芝 3着以内率
    df["youshiba_top3_rate"] = _bayesian_rate_horse(
        df, "ketto_num",
        youshiba_mask, youshiba_top3,
        prior_mean=0.33, prior_n=20, min_periods=1,
    )

    # 馬×洋芝 累積出走数（初出走=0.0、NaN不可）
    df["horse_youshiba_exp_count"] = _cumcount_horse(df, "ketto_num", youshiba_mask)

    return df


def add_sire_youshiba_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    父馬ごとの洋芝産駒勝率を計算する。

    洋芝適性は遺伝性が高く、父馬によって産駒の洋芝成績に差が出やすい。
    min_periods=10: 産駒数が少ない種牡馬では NaN を返す。
    """
    if "sire_youshiba_win_rate" in df.columns:
        return df

    sire_col = "p_sire" if "p_sire" in df.columns else "sire_id"
    if sire_col not in df.columns:
        df["sire_youshiba_win_rate"] = np.nan
        return df

    youshiba_mask = df["course_code"].isin([1, 2]).astype("int8")
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    youshiba_win = (youshiba_mask * win_flag).astype("int8")

    sire_key = df[sire_col].astype(str).fillna("__nan__")
    _PRIOR_N = 30.0
    _PRIOR_MEAN = 0.0714
    _MIN_PERIODS = 10

    cum_runs = youshiba_mask.groupby(sire_key, sort=False).cumsum() - youshiba_mask
    cum_wins = youshiba_win.groupby(sire_key, sort=False).cumsum() - youshiba_win

    smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)
    df["sire_youshiba_win_rate"] = smoothed.where(
        cum_runs >= _MIN_PERIODS, np.nan
    ).astype("float32")

    return df


def add_kokai_koban_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    小回りコース（地方含む）での馬別過去勝率を計算する。

    小回り: 札幌(1)・函館(2)・福島(3)・小倉(7) が該当。
    新潟(4) は外回りの直線コースが主体であり小回りではないため除外。
    旧定義 [1,2,3,4,10] は新潟(4)を誤分類していたバグのため修正。
    """
    if "kokai_koban_win_rate" in df.columns:
        return df

    koban_mask = df["course_code"].isin([1, 2, 3, 7]).astype("int8")
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    koban_win = (koban_mask * win_flag).astype("int8")

    df["kokai_koban_win_rate"] = _bayesian_rate_horse(
        df, "ketto_num",
        koban_mask, koban_win,
        prior_mean=0.0714, prior_n=20, min_periods=1,
    )
    return df


# ===========================================================================
# グループB: 芝稍重適性特徴量
# （パラメータは build_features_v21.py と完全一致）
# ===========================================================================

def add_horse_soft_turf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとの芝×稍重（turf_condition=2）での適性を計算する。

    既存の horse_turf_soft_win_rate は _v10 suffix 版など実装が分散しており
    v20 では horse_turf_soft_win_rate_bayes が相当するが、本計算は稍重条件を
    明示的に独立した特徴量として管理するために再実装する。
    """
    if all(c in df.columns for c in ["horse_soft_turf_win_rate", "horse_soft_turf_top3_rate", "going_soft_exp_count"]):
        return df

    track_num = pd.to_numeric(df["track_code"], errors="coerce")
    turf_cond = pd.to_numeric(df["turf_condition"], errors="coerce")

    # 芝×稍重マスク（track_code<50 かつ turf_condition==2）
    soft_turf_mask = ((track_num < 50) & (turf_cond == 2)).astype("int8")

    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    top3_flag = (finish <= 3).astype("int8")

    soft_turf_win = (soft_turf_mask * win_flag).astype("int8")
    soft_turf_top3 = (soft_turf_mask * top3_flag).astype("int8")

    # 馬×芝×稍重 過去勝率
    df["horse_soft_turf_win_rate"] = _bayesian_rate_horse(
        df, "ketto_num",
        soft_turf_mask, soft_turf_win,
        prior_mean=0.0714, prior_n=15, min_periods=1,
    )

    # 馬×芝×稍重 3着以内率
    df["horse_soft_turf_top3_rate"] = _bayesian_rate_horse(
        df, "ketto_num",
        soft_turf_mask, soft_turf_top3,
        prior_mean=0.33, prior_n=15, min_periods=1,
    )

    # 馬×芝×稍重 累積出走数（NaN不可）
    df["going_soft_exp_count"] = _cumcount_horse(df, "ketto_num", soft_turf_mask)

    return df


def add_sire_soft_turf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    父馬ごとの芝×稍重産駒勝率を計算する。

    稍重適性は遺伝するケースがあり（特に欧州系血統）、産駒統計として
    馬個体の稍重実績を補完する役割を持つ。
    """
    if "sire_soft_turf_win_rate" in df.columns:
        return df

    sire_col = "p_sire" if "p_sire" in df.columns else "sire_id"
    if sire_col not in df.columns:
        df["sire_soft_turf_win_rate"] = np.nan
        return df

    track_num = pd.to_numeric(df["track_code"], errors="coerce")
    turf_cond = pd.to_numeric(df["turf_condition"], errors="coerce")
    soft_turf_mask = ((track_num < 50) & (turf_cond == 2)).astype("int8")

    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    soft_turf_win = (soft_turf_mask * win_flag).astype("int8")

    sire_key = df[sire_col].astype(str).fillna("__nan__")
    _PRIOR_N = 25.0
    _PRIOR_MEAN = 0.0714
    _MIN_PERIODS = 5

    cum_runs = soft_turf_mask.groupby(sire_key, sort=False).cumsum() - soft_turf_mask
    cum_wins = soft_turf_win.groupby(sire_key, sort=False).cumsum() - soft_turf_win

    smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)
    df["sire_soft_turf_win_rate"] = smoothed.where(
        cum_runs >= _MIN_PERIODS, np.nan
    ).astype("float32")

    return df


# ===========================================================================
# グループC: グローバルスピード指数
# （推論時は speed_index_course_adj を生成しない = リーク防止）
# ===========================================================================

def add_speed_index_features(
    df: pd.DataFrame,
    se_path: "str | Path",
) -> pd.DataFrame:
    """
    コース×距離×馬場ごとに基準タイムを累積中央値で正規化したスピード指数を計算する。

    SE_preprocessed.parquet から racetime を取得し、
    各行を「過去のみの基準タイム（中央値）＋標準偏差」で偏差化する。
    min_periods=20: 基準タイム推定に20件未満では NaN を返す。

    重要: speed_index_course_adj（当該レースの実走タイムを使う中間列）は
    生成後に即削除する。推論時は当日タイムが存在しないためリーク列。
    """
    _MIN_PERIODS = 20

    if all(c in df.columns for c in ["speed_index_3run_avg", "speed_index_trend"]):
        return df

    se_path = Path(se_path)

    _racetime_was_present = "racetime" in df.columns
    if not _racetime_was_present:
        # df に racetime がない場合のみ SE からマージする
        # （create_standardization_features が racetime を先に使う場合は既に存在する）
        if not se_path.exists():
            print(f"  [WARN] SE_preprocessed.parquet が見つかりません: {se_path}")
            df["speed_index_3run_avg"] = np.nan
            df["speed_index_trend"] = np.nan
            return df
        se = pd.read_parquet(se_path, columns=["race_id", "ketto_num", "racetime"])
        df = df.merge(se, on=["race_id", "ketto_num"], how="left")

    # 基準タイム計算キー: course_code × distance × turf_condition
    df["_grp_speed"] = (
        df["course_code"].astype(str).fillna("0") + "_"
        + df["distance"].astype(str).fillna("0") + "_"
        + df["turf_condition"].astype(str).fillna("0")
    )

    # 時系列ソートは呼び出し側で保証済みを前提とする
    racetime_num = pd.to_numeric(df["racetime"], errors="coerce")
    speed_idx = pd.Series(np.nan, index=df.index, dtype="float32")
    unique_grps = df["_grp_speed"].unique()

    grp_mask = df["_grp_speed"]
    for grp_key_val in unique_grps:
        idx = df.index[grp_mask == grp_key_val]
        g_racetime = racetime_num.loc[idx]
        if g_racetime.notna().sum() < _MIN_PERIODS:
            continue
        shifted = g_racetime.shift(1)
        med = shifted.expanding(min_periods=_MIN_PERIODS).median()
        std = shifted.expanding(min_periods=_MIN_PERIODS).std()
        # スピード指数: (基準タイム中央値 - 実走タイム) / std
        # 正値 = 基準より速い（良い）、負値 = 遅い（悪い）
        valid_std = std.replace(0, np.nan)
        idx_val = (med - g_racetime) / valid_std
        speed_idx.loc[idx] = idx_val.astype("float32")

    # speed_index_course_adj は中間計算用（当該レース実走タイム依存）
    # avg/trend 計算に使用後に必ず削除してリーク防止する
    df["speed_index_course_adj"] = speed_idx

    # 馬ごとの過去3走平均
    # shift(1) で当該レースを除外してから rolling(3) — リーク防止
    horse_key = df["ketto_num"].astype(str).fillna("__nan__")
    speed_adj = df["speed_index_course_adj"].copy()

    df["speed_index_3run_avg"] = (
        speed_adj
        .groupby(horse_key, sort=False)
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ).astype("float32")

    # 馬ごとの過去3走線形傾向
    # 等間隔 3 点 OLS の解析解: slope = (lag1 - lag3) / 2
    def _rolling_slope_vectorized(x: pd.Series) -> pd.Series:
        """
        shift(1) 後の過去3走に対して OLS 傾き（等間隔 3 点）を返す。
        t=[0,1,2] の OLS 解: slope = (y2 - y0) / 2
        3走全てが non-NaN の場合のみ値を返す。
        """
        shifted = x.shift(1)
        lag1 = shifted          # 直前走 (t=2)
        lag2 = shifted.shift(1) # 2走前 (t=1)
        lag3 = shifted.shift(2) # 3走前 (t=0)
        all_valid = lag1.notna() & lag2.notna() & lag3.notna()
        slope = (lag1 - lag3) / 2.0
        return slope.where(all_valid, np.nan)

    df["speed_index_trend"] = (
        df["speed_index_course_adj"]
        .groupby(horse_key, sort=False)
        .transform(_rolling_slope_vectorized)
    ).astype("float32")

    # 中間列を削除（speed_index_course_adj は当該レースの実走タイム＝リーク特徴量）
    # racetime はマージで追加した場合のみ削除する（元から df に存在した場合は remove_leak_features に委ねる）
    cols_to_drop = ["_grp_speed", "speed_index_course_adj"]
    if not _racetime_was_present:
        cols_to_drop.append("racetime")
    df.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    return df


# ===========================================================================
# グループD: ペース圧力×脚質交差
# （パラメータは build_features_v21.py と完全一致）
# ===========================================================================

def add_pace_dist_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    脚質 × 距離帯 × 馬場状態の3軸組み合わせ別過去勝率を計算する。

    先行脚質がスプリント良馬場では有利だが、差し脚質が長距離稍重では有利など
    ペース圧力・距離・馬場の交差効果を捉える。
    min_periods=10: 希薄な組み合わせでは NaN を返す。
    """
    if "pace_dist_style_win_rate" in df.columns:
        return df

    dist_num = pd.to_numeric(df["distance"], errors="coerce")
    dist_band = pd.cut(
        dist_num,
        bins=[0, 1400, 1800, 2200, 99999],
        labels=["sprint", "mile", "middle", "long"],
        right=True,
    ).astype(str)

    turf_cond = pd.to_numeric(df["turf_condition"], errors="coerce").fillna(0).astype(int).astype(str)
    # horse_modal_running_style を優先使用（推論時も 0 以外の値が入る）
    # フォールバック: horse_modal_running_style 列が存在しない場合は running_style_code を使用
    if "horse_modal_running_style" in df.columns:
        style = df["horse_modal_running_style"].astype(str).fillna("0")
    else:
        style = df["running_style_code"].astype(str).fillna("0")

    grp_key = style + "_" + dist_band + "_" + turf_cond

    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    _PRIOR_N = 20.0
    _PRIOR_MEAN = 0.0714
    _MIN_PERIODS = 10

    ones = pd.Series(np.ones(len(df), dtype="int8"), index=df.index)
    cum_runs = ones.groupby(grp_key, sort=False).cumsum() - ones
    cum_wins = win_flag.groupby(grp_key, sort=False).cumsum() - win_flag

    smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)
    df["pace_dist_style_win_rate"] = smoothed.where(
        cum_runs >= _MIN_PERIODS, np.nan
    ).astype("float32")

    return df

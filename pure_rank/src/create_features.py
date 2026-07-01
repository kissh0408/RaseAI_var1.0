"""
create_features.py — RaceAI_var1.0 特徴量生成スクリプト

01_preprocessed/ の Parquet から特徴量を生成し
02_features/features_{version}.parquet を出力する。
バージョンは pure_rank/config/train_config.json の features_version で管理する。

禁止事項:
- オッズ・人気 (odds, popularity) を特徴量に含めない
- market_log_odds / init_score を使わない
- shift(1) なしの全データ集計を hist_ 系特徴量に使わない
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── パス解決 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── 市場情報混入チェック ───────────────────────────────────────────────────────
FORBIDDEN_COLS = {
    "odds", "popularity", "win_odds", "place_odds",
    "quinella_odds", "market_prob", "market_log_odds",
    "init_score", "ninki",
}


def _check_no_market_features(df: pd.DataFrame) -> None:
    """DataFrame に市場情報列が含まれていないことを確認する。"""
    found = FORBIDDEN_COLS & set(df.columns)
    if found:
        raise ValueError(
            f"[FORBIDDEN] 市場情報が特徴量に混入しています: {sorted(found)}\n"
            f"即座に除去してください。"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: データ読み込み・結合
# ═══════════════════════════════════════════════════════════════════════════════

def _load_data(cfg: dict) -> pd.DataFrame:
    """SE / RA / SK の Parquet を読み込んで結合する。"""
    prep_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]

    se = pd.read_parquet(prep_dir / "SE_preprocessed.parquet")
    ra = pd.read_parquet(prep_dir / "RA_preprocessed.parquet")
    sk = pd.read_parquet(prep_dir / "SK_preprocessed.parquet")

    print(f"  SE: {len(se):,} rows, {len(se.columns)} cols")
    print(f"  RA: {len(ra):,} rows, {len(ra.columns)} cols")
    print(f"  SK: {len(sk):,} rows, {len(sk.columns)} cols")

    # SE + RA を race_id でマージ（RA の距離・馬場情報を SE に付加）
    ra_merge_cols = [
        "race_id", "grade_code", "distance", "track_code", "horse_count",
        "weather_code", "surface_code", "track_condition_code",
        "surface_condition", "distance_category", "race_date",
    ]
    # RA には race_date が既にある。SE の race_date と同一のはずだが RA のものを使う
    se = se.drop(columns=["race_date"], errors="ignore")
    ra_subset = ra[[c for c in ra_merge_cols if c in ra.columns]].copy()

    df = se.merge(ra_subset, on="race_id", how="inner")

    # SK（血統）をマージ
    sk_cols = ["ketto_num", "sire_id", "bms_id"]
    sk_subset = sk[[c for c in sk_cols if c in sk.columns]].copy()
    df = df.merge(sk_subset, on="ketto_num", how="left")

    print(f"  Merged: {len(df):,} rows, {len(df.columns)} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: フィルタ適用
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """必須フィルタを適用する。

    除外対象:
    - grade_code 8 (未格付け), 9 (障害)
    - abnormal_code 1 (取消), 3 (除外), 4 (落馬)
    - horse_count < 5 (少頭数レース)
    - finish_rank == 0 (着順無効)
    """
    f = cfg["filters"]
    n_before = len(df)

    mask = (
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    )
    df = df[mask].copy()

    print(f"  Filter applied: {n_before:,} → {len(df):,} rows "
          f"(removed {n_before - len(df):,})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: HISTORICAL FEATURES
# 全て shift(1) でリーク防止。horse_id × race_date でソート後に計算。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    """過去走成績ベースの特徴量を生成する。

    Notes
    -----
    - df は事前にフィルタ済みであること（DNF 等は除外済み）
    - sort_values(['ketto_num', 'race_date']) 後の順序で shift(1) を適用
    - groupby + transform(lambda x: x.shift(1).rolling/expanding) パターンを使用
    """
    # race_date でソートされた状態を保証する
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)

    # レース内の平均走破タイム（同レース内の相対評価用）
    # finish_rank > 0 の馬のみで計算（既にフィルタ済みなので全行有効）
    race_avg_time = df.groupby("race_id")["racetime"].transform("mean")
    df["_time_dev"] = df["racetime"] - race_avg_time

    # ─── 着順系 ───────────────────────────────────────────────────────────────
    grp_horse = df.groupby("ketto_num")

    df["hist_last_rank"] = grp_horse["finish_rank"].transform(
        lambda x: x.shift(1)
    )
    df["hist_avg_rank_3"] = grp_horse["finish_rank"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["hist_avg_rank_5"] = grp_horse["finish_rank"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["hist_win_rate"] = grp_horse["is_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["hist_place_rate"] = grp_horse["is_place"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # ─── タイム系 ──────────────────────────────────────────────────────────────
    df["hist_last_last3f"] = grp_horse["time_3f_after"].transform(
        lambda x: x.shift(1)
    )
    df["hist_avg_last3f_3"] = grp_horse["time_3f_after"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["hist_avg_last3f_5"] = grp_horse["time_3f_after"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["hist_last_time_dev"] = grp_horse["_time_dev"].transform(
        lambda x: x.shift(1)
    )
    df["hist_avg_time_dev_3"] = grp_horse["_time_dev"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["hist_avg_time_dev_5"] = grp_horse["_time_dev"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # ─── 馬場条件別 最速タイム ──────────────────────────────────────────────────
    # 同距離帯×同馬場種別×同馬場状態での過去最速タイム（shift(1) で当該レース除外）
    df["hist_best_time_same_cond"] = (
        df.groupby(
            ["ketto_num", "distance_category", "surface_code", "track_condition_code"]
        )["racetime"].transform(
            lambda x: x.shift(1).expanding().min()
        )
    )

    # ─── 馬場適性系 ───────────────────────────────────────────────────────────
    # 各グループ内で race_date 順に並んでいる前提（sort_values 済み）
    df["hist_same_surface_win_rate"] = (
        df.groupby(["ketto_num", "surface_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_condition_win_rate"] = (
        df.groupby(["ketto_num", "track_condition_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_surface_condition_win_rate"] = (
        df.groupby(["ketto_num", "surface_condition"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_course_win_rate"] = (
        df.groupby(["ketto_num", "course_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_dist_win_rate"] = (
        df.groupby(["ketto_num", "distance_category"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ─── 状態系 ───────────────────────────────────────────────────────────────
    # diff() は current - previous なので shift 不要（current - prev は過去情報）
    df["hist_days_since_last"] = grp_horse["race_date"].transform(
        lambda x: x.diff().dt.days
    )
    # 前走の馬体重変化（shift(1) で当該レース除外）
    df["hist_weight_change"] = grp_horse["horse_weight_change"].transform(
        lambda x: x.shift(1)
    )

    # ─── 賞金系 ───────────────────────────────────────────────────────────────
    df["hist_total_prize"] = grp_horse["hon_shokin"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    df["hist_avg_prize_3"] = grp_horse["hon_shokin"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # ─── 天候適性系 ───────────────────────────────────────────────────────────────
    df["hist_same_weather_win_rate"] = (
        df.groupby(["ketto_num", "weather_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_weather_avg_rank"] = (
        df.groupby(["ketto_num", "weather_code"])["finish_rank"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ─── コース×距離帯複合適性 ────────────────────────────────────────────────────
    df["hist_same_course_dist_win_rate"] = (
        df.groupby(["ketto_num", "course_code", "distance_category"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ─── グレード適性系 ───────────────────────────────────────────────────────────
    df["hist_same_grade_win_rate"] = (
        df.groupby(["ketto_num", "grade_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["_is_top_grade"] = (df["grade_code"] >= 5).astype(np.int8)
    df["hist_top_grade_exp_count"] = df.groupby("ketto_num")["_is_top_grade"].transform(
        lambda x: x.shift(1).expanding().sum()
    )

    # ─── 精細距離適性（100m単位） ──────────────────────────────────────────────────
    df["_dist_bin_100"] = (df["distance"] // 100) * 100
    df["hist_exact_dist_win_rate"] = (
        df.groupby(["ketto_num", "_dist_bin_100"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ─── クラス移動特徴量 ──────────────────────────────────────────────────────────
    # grade_code は大きいほど格上（=1: 条件戦, =5: OP, =7: 重賞）
    # 過去の最高格（最大 grade_code = 格上）
    df["hist_best_grade_ever"] = grp_horse["grade_code"].transform(
        lambda x: x.shift(1).expanding().max()
    )
    # 今回 grade_code と過去最高格の差（正=格下出走=有利, 負=格上挑戦=不利）
    df["hist_grade_diff"] = df["hist_best_grade_ever"] - df["grade_code"]

    # 重賞（grade_code >= 7）での過去平均着順（NaN率80〜90%は想定通り）
    df["_rank_top_grade"] = df["finish_rank"].where(df["grade_code"] >= 7, other=np.nan)
    df["hist_avg_rank_top_grade"] = df.groupby("ketto_num")["_rank_top_grade"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # ─── 先行傾向（running_style_code は過去走の値。shift(1) で当該レース除外） ───
    # running_style_code: 1=逃げ, 2=先行, 3=差し, 4=追込 / 先行系 = {1, 2}
    df["_is_front_runner"] = df["running_style_code"].isin([1, 2]).astype(np.int8)
    df["hist_front_running_pref"] = grp_horse["_is_front_runner"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 一時列を削除
    df = df.drop(columns=["_time_dev", "_is_top_grade", "_dist_bin_100", "_rank_top_grade", "_is_front_runner"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CURRENT FEATURES
# 当該レースの固定情報。リーク防止不要（レース前に観測可能な情報）。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_current_features(df: pd.DataFrame) -> pd.DataFrame:
    """当該レースの現状情報特徴量を生成する。"""
    # 季節 × 性別スコア: cos(2π × day_of_year/365) × sex_sign
    # sex_sign: 牝馬(sex_code=2)=+1, 牡馬(1)・騸馬(3)=-1
    day_of_year = df["race_date"].dt.dayofyear
    sex_sign = df["sex_code"].map({1: -1, 2: 1, 3: -1}).fillna(0).astype(float)
    df["season_sex_score"] = np.cos(2 * np.pi * day_of_year / 365) * sex_sign

    # 枠番 × 馬場種別 交互作用: 芝=+1, ダート=−1
    # 芝は内枠有利、ダートは大きな差なし（方向性を数値化）
    surface_sign = df["surface_code"].map({1: 1, 2: -1}).fillna(0).astype(float)
    df["wakuban_surface"] = df["wakuban"].astype(float) * surface_sign

    # ─── フィールド強度（SECTION 3完了後に依存） ─────────────────────────────────
    # 新馬(hist_win_rate=NaN)は 0 として fillna してから groupby で平均を取る
    df["_hist_win_rate_filled"] = df["hist_win_rate"].fillna(0)
    df["field_avg_win_rate"] = df.groupby("race_id")["_hist_win_rate_filled"].transform("mean")
    df["field_avg_prize"] = df.groupby("race_id")["hist_avg_prize_3"].transform("mean")
    df["win_rate_vs_field"] = df["hist_win_rate"] - df["field_avg_win_rate"]
    df["prize_vs_field"] = df["hist_avg_prize_3"] - df["field_avg_prize"]
    df = df.drop(columns=["_hist_win_rate_filled"])

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BLOODLINE FEATURES
# 父馬・母父の産駒成績（Phase 3: 時系列正確版）。
# 日次集計 → 累積 → shift(1) でリーク防止済み。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sire_features(df: pd.DataFrame) -> pd.DataFrame:
    """父馬・母父産駒の成績を時系列正確版で計算する。

    アプローチ: 日次集計 → 累積 → shift(1) → メイン df にマージ
    理由: 同一 sire_id の産駒が同日複数レースに出走しうるため、
         ketto_num 単位の shift(1) では同日他産駒の結果が混入する。
         日次集計後に shift(1) することで当日を含まない累計を保証する。
    """
    # 産駒数が少ない父馬（新種牡馬等）の累積勝率はS/N比が低くノイズになるため、
    # cum_races_prev < MIN_SIRE_RACES の場合は NaN を設定し、
    # LightGBM の欠損値分岐に処理を委ねる。
    MIN_SIRE_RACES = 30

    # ─── sire 特徴量 ──────────────────────────────────────────────────────────
    if "sire_id" not in df.columns or df["sire_id"].isna().all():
        for col in ["hist_sire_win_rate_ts", "hist_sire_surface_win_rate_ts",
                    "hist_sire_dist_win_rate_ts", "hist_sire_dist_diff"]:
            df[col] = np.nan
    else:
        # ── 通算勝率（sire × race_date） ──────────────────────────────────────
        sire_daily = (
            df.groupby(["sire_id", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "race_date"])
        )
        grp_s = sire_daily.groupby("sire_id", observed=True)
        sire_daily["cum_wins"]  = grp_s["d_wins"].cumsum()
        sire_daily["cum_races"] = grp_s["d_races"].cumsum()
        sire_daily["cum_wins_prev"]  = grp_s["cum_wins"].shift(1)
        sire_daily["cum_races_prev"] = grp_s["cum_races"].shift(1)
        sire_daily["hist_sire_win_rate_ts"] = (
            sire_daily["cum_wins_prev"] / sire_daily["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        sire_daily.loc[sire_daily["cum_races_prev"] < MIN_SIRE_RACES, "hist_sire_win_rate_ts"] = np.nan
        df = df.merge(
            sire_daily[["sire_id", "race_date", "hist_sire_win_rate_ts"]],
            on=["sire_id", "race_date"], how="left"
        )

        # ── 同馬場勝率（sire × surface_code × race_date） ───────────────────────
        sire_surf = (
            df.groupby(["sire_id", "surface_code", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "surface_code", "race_date"])
        )
        grp_ss = sire_surf.groupby(["sire_id", "surface_code"], observed=True)
        sire_surf["cum_wins"]  = grp_ss["d_wins"].cumsum()
        sire_surf["cum_races"] = grp_ss["d_races"].cumsum()
        sire_surf["cum_wins_prev"]  = grp_ss["cum_wins"].shift(1)
        sire_surf["cum_races_prev"] = grp_ss["cum_races"].shift(1)
        sire_surf["hist_sire_surface_win_rate_ts"] = (
            sire_surf["cum_wins_prev"] / sire_surf["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        sire_surf.loc[sire_surf["cum_races_prev"] < MIN_SIRE_RACES, "hist_sire_surface_win_rate_ts"] = np.nan
        df = df.merge(
            sire_surf[["sire_id", "surface_code", "race_date",
                        "hist_sire_surface_win_rate_ts"]],
            on=["sire_id", "surface_code", "race_date"], how="left"
        )

        # ── 同距離帯勝率（sire × distance_category × race_date） ─────────────────
        sire_dist = (
            df.groupby(["sire_id", "distance_category", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "distance_category", "race_date"])
        )
        grp_sd = sire_dist.groupby(["sire_id", "distance_category"], observed=True)
        sire_dist["cum_wins"]  = grp_sd["d_wins"].cumsum()
        sire_dist["cum_races"] = grp_sd["d_races"].cumsum()
        sire_dist["cum_wins_prev"]  = grp_sd["cum_wins"].shift(1)
        sire_dist["cum_races_prev"] = grp_sd["cum_races"].shift(1)
        sire_dist["hist_sire_dist_win_rate_ts"] = (
            sire_dist["cum_wins_prev"] / sire_dist["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        sire_dist.loc[sire_dist["cum_races_prev"] < MIN_SIRE_RACES, "hist_sire_dist_win_rate_ts"] = np.nan
        df = df.merge(
            sire_dist[["sire_id", "distance_category", "race_date",
                        "hist_sire_dist_win_rate_ts"]],
            on=["sire_id", "distance_category", "race_date"], how="left"
        )

        # ── 父産駒の平均勝ち距離との差（全期間統計、距離適性は安定のため許容） ─────
        sire_wins = df[df["is_win"] == 1]
        if len(sire_wins) > 0:
            sire_avg_dist = (
                sire_wins.groupby("sire_id", observed=True)["distance"]
                .mean()
                .reset_index()
                .rename(columns={"distance": "_sire_avg_win_dist"})
            )
            df = df.merge(sire_avg_dist, on="sire_id", how="left")
            df["hist_sire_dist_diff"] = (df["distance"] - df["_sire_avg_win_dist"]).abs()
            df = df.drop(columns=["_sire_avg_win_dist"])
        else:
            df["hist_sire_dist_diff"] = np.nan

    # ─── bms 特徴量 ──────────────────────────────────────────────────────────
    if "bms_id" not in df.columns or df["bms_id"].isna().all():
        df["hist_bms_win_rate_ts"] = np.nan
    else:
        bms_daily = (
            df.groupby(["bms_id", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["bms_id", "race_date"])
        )
        grp_b = bms_daily.groupby("bms_id", observed=True)
        bms_daily["cum_wins"]  = grp_b["d_wins"].cumsum()
        bms_daily["cum_races"] = grp_b["d_races"].cumsum()
        bms_daily["cum_wins_prev"]  = grp_b["cum_wins"].shift(1)
        bms_daily["cum_races_prev"] = grp_b["cum_races"].shift(1)
        bms_daily["hist_bms_win_rate_ts"] = (
            bms_daily["cum_wins_prev"] / bms_daily["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        bms_daily.loc[bms_daily["cum_races_prev"] < MIN_SIRE_RACES, "hist_bms_win_rate_ts"] = np.nan
        df = df.merge(
            bms_daily[["bms_id", "race_date", "hist_bms_win_rate_ts"]],
            on=["bms_id", "race_date"], how="left"
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.5: JOCKEY / TRAINER FEATURES
# 騎手・調教師の成績特徴量（Phase 4: 時系列正確版）。
# 日次集計 → 累積/rolling → shift(1)/closed='left' でリーク防止済み。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の成績特徴量を時系列正確版で計算する。

    アプローチ:
    - 通算勝率: 日次集計 → cumsum → shift(1) でリーク防止
    - 直近N日勝率: 日次集計 → GroupBy.rolling(ND, closed='left') でリーク防止

    騎手/調教師は同日に複数レースに関与しうるため（実測: 騎手76.7%・調教師74.8%）、
    エントリ単位の shift(1) では同日他レースの結果が混入する。
    日次集計後に処理することで当日を完全除外する。
    """
    # 分母が少ない場合は NaN としてノイズを抑制する（LightGBM の欠損値分岐に委ねる）
    MIN_JOCKEY_RACES = 10
    MIN_TRAINER_RACES = 10

    # ══════════════════════════════════════════════════════════════════════
    # 騎手特徴量
    # ══════════════════════════════════════════════════════════════════════

    # ─── Step J-1: 日次集計（jockey × date） ─────────────────────────────────
    jockey_daily = (
        df.groupby(["jockey_code", "race_date"], observed=True)
        .agg(
            d_wins=("is_win", "sum"),
            d_races=("is_win", "count"),
            d_place=("is_place", "sum"),
        )
        .reset_index()
        .sort_values(["jockey_code", "race_date"])
        .reset_index(drop=True)
    )

    # ─── Step J-2: 通算勝率（cumulative + shift(1) で当日を除外） ─────────────
    grp_j = jockey_daily.groupby("jockey_code", observed=True)
    jockey_daily["cum_wins"]       = grp_j["d_wins"].cumsum()
    jockey_daily["cum_races"]      = grp_j["d_races"].cumsum()
    jockey_daily["cum_wins_prev"]  = grp_j["cum_wins"].shift(1)
    jockey_daily["cum_races_prev"] = grp_j["cum_races"].shift(1)
    jockey_daily["hist_jockey_win_rate_cum"] = (
        jockey_daily["cum_wins_prev"] / jockey_daily["cum_races_prev"]
    )
    # 出走数が少ない場合はNaN（デビュー直後のノイズ抑制）
    jockey_daily.loc[
        jockey_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_win_rate_cum",
    ] = np.nan

    df = df.merge(
        jockey_daily[["jockey_code", "race_date", "hist_jockey_win_rate_cum"]],
        on=["jockey_code", "race_date"],
        how="left",
    )

    # ─── Step J-3: rolling 30D・60D 勝率（closed='left' で当日除外） ────────────
    # GroupBy.rolling を使うことで apply より効率的に時系列ウィンドウを計算する。
    # closed='left': ウィンドウ = [race_date - ND, race_date) → 当日を除外する。
    jd_idx = jockey_daily.set_index("race_date")

    for n_days in [30, 60]:
        roll = (
            jd_idx.groupby("jockey_code", observed=True)[["d_wins", "d_races", "d_place"]]
            .rolling(f"{n_days}D", closed="left")
            .sum()
            .reset_index()  # → columns: jockey_code, race_date, d_wins, d_races, d_place
            .rename(columns={
                "d_wins":  f"roll_wins_{n_days}d",
                "d_races": f"roll_races_{n_days}d",
                "d_place": f"roll_place_{n_days}d",
            })
        )

        # 勝率
        roll[f"hist_jockey_win_rate_{n_days}d"] = (
            roll[f"roll_wins_{n_days}d"] / roll[f"roll_races_{n_days}d"]
        )
        roll.loc[
            roll[f"roll_races_{n_days}d"] < MIN_JOCKEY_RACES,
            f"hist_jockey_win_rate_{n_days}d",
        ] = np.nan

        merge_cols = ["jockey_code", "race_date", f"hist_jockey_win_rate_{n_days}d"]

        # 30D のみ複勝率を追加（60D は重複情報となるため省略）
        if n_days == 30:
            roll["hist_jockey_place_rate_30d"] = (
                roll["roll_place_30d"] / roll["roll_races_30d"]
            )
            roll.loc[
                roll["roll_races_30d"] < MIN_JOCKEY_RACES,
                "hist_jockey_place_rate_30d",
            ] = np.nan
            merge_cols.append("hist_jockey_place_rate_30d")

        df = df.merge(roll[merge_cols], on=["jockey_code", "race_date"], how="left")

    # ─── Step J-4: 騎手×競馬場 通算勝率（cumulative + shift(1)） ────────────────
    # rolling ではなく cumulative を採用する理由: コース別は30日間のサンプルが
    # 極端に少なく（数レース程度）、累積の方が安定した適性スコアを提供する。
    jc_daily = (
        df.groupby(["jockey_code", "course_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["jockey_code", "course_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_jc = jc_daily.groupby(["jockey_code", "course_code"], observed=True)
    jc_daily["cum_wins"]       = grp_jc["d_wins"].cumsum()
    jc_daily["cum_races"]      = grp_jc["d_races"].cumsum()
    jc_daily["cum_wins_prev"]  = grp_jc["cum_wins"].shift(1)
    jc_daily["cum_races_prev"] = grp_jc["cum_races"].shift(1)
    jc_daily["hist_jockey_course_win_rate"] = (
        jc_daily["cum_wins_prev"] / jc_daily["cum_races_prev"]
    )
    jc_daily.loc[
        jc_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_course_win_rate",
    ] = np.nan

    df = df.merge(
        jc_daily[["jockey_code", "course_code", "race_date", "hist_jockey_course_win_rate"]],
        on=["jockey_code", "course_code", "race_date"],
        how="left",
    )

    # ══════════════════════════════════════════════════════════════════════
    # 調教師特徴量
    # ══════════════════════════════════════════════════════════════════════

    # ─── Step T-1: 日次集計（trainer × date） ─────────────────────────────────
    trainer_daily = (
        df.groupby(["trainer_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "race_date"])
        .reset_index(drop=True)
    )

    # ─── Step T-2: 通算勝率（cumulative + shift(1)） ─────────────────────────
    grp_t = trainer_daily.groupby("trainer_code", observed=True)
    trainer_daily["cum_wins"]       = grp_t["d_wins"].cumsum()
    trainer_daily["cum_races"]      = grp_t["d_races"].cumsum()
    trainer_daily["cum_wins_prev"]  = grp_t["cum_wins"].shift(1)
    trainer_daily["cum_races_prev"] = grp_t["cum_races"].shift(1)
    trainer_daily["hist_trainer_win_rate_cum"] = (
        trainer_daily["cum_wins_prev"] / trainer_daily["cum_races_prev"]
    )
    trainer_daily.loc[
        trainer_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_win_rate_cum",
    ] = np.nan

    df = df.merge(
        trainer_daily[["trainer_code", "race_date", "hist_trainer_win_rate_cum"]],
        on=["trainer_code", "race_date"],
        how="left",
    )

    # ─── Step T-3: rolling 30D・60D 勝率（closed='left' で当日除外） ────────────
    td_idx = trainer_daily.set_index("race_date")

    for n_days in [30, 60]:
        roll = (
            td_idx.groupby("trainer_code", observed=True)[["d_wins", "d_races"]]
            .rolling(f"{n_days}D", closed="left")
            .sum()
            .reset_index()
            .rename(columns={
                "d_wins":  f"roll_wins_{n_days}d",
                "d_races": f"roll_races_{n_days}d",
            })
        )
        roll[f"hist_trainer_win_rate_{n_days}d"] = (
            roll[f"roll_wins_{n_days}d"] / roll[f"roll_races_{n_days}d"]
        )
        roll.loc[
            roll[f"roll_races_{n_days}d"] < MIN_TRAINER_RACES,
            f"hist_trainer_win_rate_{n_days}d",
        ] = np.nan

        df = df.merge(
            roll[["trainer_code", "race_date", f"hist_trainer_win_rate_{n_days}d"]],
            on=["trainer_code", "race_date"],
            how="left",
        )

    # ─── Step T-4: 調教師×馬場種別 通算勝率（cumulative + shift(1)） ────────────
    # 芝・ダート適性は安定した長期特性のため cumulative を採用する。
    ts_daily = (
        df.groupby(["trainer_code", "surface_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "surface_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_ts = ts_daily.groupby(["trainer_code", "surface_code"], observed=True)
    ts_daily["cum_wins"]       = grp_ts["d_wins"].cumsum()
    ts_daily["cum_races"]      = grp_ts["d_races"].cumsum()
    ts_daily["cum_wins_prev"]  = grp_ts["cum_wins"].shift(1)
    ts_daily["cum_races_prev"] = grp_ts["cum_races"].shift(1)
    ts_daily["hist_trainer_surface_win_rate"] = (
        ts_daily["cum_wins_prev"] / ts_daily["cum_races_prev"]
    )
    ts_daily.loc[
        ts_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_surface_win_rate",
    ] = np.nan

    df = df.merge(
        ts_daily[["trainer_code", "surface_code", "race_date", "hist_trainer_surface_win_rate"]],
        on=["trainer_code", "surface_code", "race_date"],
        how="left",
    )

    # ─── Step J-5: 騎手×馬場種別 通算勝率（cumulative + shift(1)） ────────────────
    # 芝・ダート適性は安定した長期特性のため cumulative を採用する。
    js_daily = (
        df.groupby(["jockey_code", "surface_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["jockey_code", "surface_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_js = js_daily.groupby(["jockey_code", "surface_code"], observed=True)
    js_daily["cum_wins"]       = grp_js["d_wins"].cumsum()
    js_daily["cum_races"]      = grp_js["d_races"].cumsum()
    js_daily["cum_wins_prev"]  = grp_js["cum_wins"].shift(1)
    js_daily["cum_races_prev"] = grp_js["cum_races"].shift(1)
    js_daily["hist_jockey_surface_win_rate_ts"] = (
        js_daily["cum_wins_prev"] / js_daily["cum_races_prev"]
    )
    js_daily.loc[
        js_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_surface_win_rate_ts",
    ] = np.nan

    df = df.merge(
        js_daily[["jockey_code", "surface_code", "race_date", "hist_jockey_surface_win_rate_ts"]],
        on=["jockey_code", "surface_code", "race_date"],
        how="left",
    )

    # ─── Step T-5: 調教師×競馬場 通算勝率（cumulative + shift(1)） ────────────────
    # コース適性は安定した長期特性のため cumulative を採用する。
    tc_daily = (
        df.groupby(["trainer_code", "course_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "course_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_tc = tc_daily.groupby(["trainer_code", "course_code"], observed=True)
    tc_daily["cum_wins"]       = grp_tc["d_wins"].cumsum()
    tc_daily["cum_races"]      = grp_tc["d_races"].cumsum()
    tc_daily["cum_wins_prev"]  = grp_tc["cum_wins"].shift(1)
    tc_daily["cum_races_prev"] = grp_tc["cum_races"].shift(1)
    tc_daily["hist_trainer_course_win_rate_ts"] = (
        tc_daily["cum_wins_prev"] / tc_daily["cum_races_prev"]
    )
    tc_daily.loc[
        tc_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_course_win_rate_ts",
    ] = np.nan

    df = df.merge(
        tc_daily[["trainer_code", "course_code", "race_date", "hist_trainer_course_win_rate_ts"]],
        on=["trainer_code", "course_code", "race_date"],
        how="left",
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.6: SPEED INDEX FEATURES
# タイム速度指数（Phase 5: 歴史的条件別基準による標準化）。
# 日次集計 → cumsum → shift(1) で当日を除外したリーク防止済み計算。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_speed_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """歴史的条件別基準による速度指数特徴量を生成する。

    アプローチ:
    - 条件グループ (distance[m], surface_code, track_condition_code) 別に
      日次集計 → cumsum → shift(1) で当日を除外した平均・標準偏差を計算する
    - distance_category（粗い区分）ではなく distance（実距離m）を使う。
      カテゴリ内の異なる距離が混在すると speed_idx が距離バイアスを吸収してしまうため。
    - _speed_idx = (cond_avg_time - racetime) / cond_std_time を計算し、
      馬別に shift(1) を適用して horse-level 特徴量を生成する
    - _speed_idx 自体（当該レースの結果情報を含む）は最後に削除する

    Notes
    -----
    - df には racetime, distance, surface_code, track_condition_code,
      ketto_num, race_date が必要（_build_hist_features 後の df に全て存在する）
    - _build_hist_features 内で計算・削除済みの _time_dev は再計算しない
    - この関数は _build_hist_features の後・_build_current_features の前に呼ぶこと
    """
    # 速度指数の基準値を計算するのに必要な最低レース数
    # この値未満の条件では標準偏差が不安定なため NaN マスクを適用する
    MIN_COND_RACES = 20

    # ─── Step 1: 条件別・日次集計 ──────────────────────────────────────────────
    # 同じ条件（distance × surface_code × track_condition_code）で
    # 同日に複数レースが開催される場合があるため、日次で先に集約する。
    # distance_category（粗いカテゴリ）ではなく distance（実距離m）で集計する。
    # 理由: カテゴリ内で異なる距離（例: 1000m〜1400m）が混在すると
    #       speed_idx が「馬の能力」ではなく「どの距離を走ったか」を反映してしまう。
    cond_daily = (
        df.groupby(
            ["distance", "surface_code", "track_condition_code", "race_date"],
            observed=True,
        )
        .agg(
            d_sum_time=("racetime", "sum"),
            d_sum_sq_time=("racetime", lambda x: (x ** 2).sum()),
            d_count=("racetime", "count"),
        )
        .reset_index()
        .sort_values(
            ["distance", "surface_code", "track_condition_code", "race_date"]
        )
        .reset_index(drop=True)
    )

    # ─── Step 2: 条件グループ内での cumsum ────────────────────────────────────
    grp_cond = cond_daily.groupby(
        ["distance", "surface_code", "track_condition_code"],
        observed=True,
    )
    cond_daily["cum_sum"]   = grp_cond["d_sum_time"].cumsum()
    cond_daily["cum_sq"]    = grp_cond["d_sum_sq_time"].cumsum()
    cond_daily["cum_count"] = grp_cond["d_count"].cumsum()

    # ─── Step 3: shift(1) で当日を除いた前日以前の累積を取得 ──────────────────
    cond_daily["cum_sum_prev"]   = grp_cond["cum_sum"].shift(1)
    cond_daily["cum_sq_prev"]    = grp_cond["cum_sq"].shift(1)
    cond_daily["cum_count_prev"] = grp_cond["cum_count"].shift(1)

    # ─── Step 4: 平均・分散・標準偏差の計算 ────────────────────────────────────
    # Welford 公式: Var(X) = E[X^2] - (E[X])^2
    # 浮動小数点誤差で分散が微小な負値になることがあるため clip(lower=0) が必須
    cond_daily["cond_avg_time"] = (
        cond_daily["cum_sum_prev"] / cond_daily["cum_count_prev"]
    )
    cond_daily["cond_var_time"] = (
        cond_daily["cum_sq_prev"] / cond_daily["cum_count_prev"]
        - cond_daily["cond_avg_time"] ** 2
    )
    cond_daily["cond_std_time"] = np.sqrt(
        cond_daily["cond_var_time"].clip(lower=0)
    )

    # 最低レース数未満の条件は NaN マスク（標準偏差が不安定なためノイズ抑制）
    low_count_mask = cond_daily["cum_count_prev"] < MIN_COND_RACES
    cond_daily.loc[low_count_mask, "cond_avg_time"] = np.nan
    cond_daily.loc[low_count_mask, "cond_std_time"] = np.nan

    # ─── Step 5: df にマージ ──────────────────────────────────────────────────
    df = df.merge(
        cond_daily[
            [
                "distance", "surface_code", "track_condition_code",
                "race_date", "cond_avg_time", "cond_std_time",
            ]
        ],
        on=["distance", "surface_code", "track_condition_code", "race_date"],
        how="left",
    )

    # ─── Step 6: 速度指数の計算 ────────────────────────────────────────────────
    # 正の値 = 歴史的平均より速い = 高能力
    # cond_std_time == 0 の場合（全馬同タイム）は NaN を設定する
    df["_speed_idx"] = np.where(
        df["cond_std_time"] > 0,
        (df["cond_avg_time"] - df["racetime"]) / df["cond_std_time"],
        np.nan,
    )

    # ─── Step 7: 馬別 shift(1) で horse-level 特徴量を生成 ───────────────────
    # _build_hist_features の sort_values が継続している前提だが、念のため保証する
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
    grp_horse = df.groupby("ketto_num")

    # 前走の速度指数（最もリークから遠い、最重要候補）
    df["hist_speed_idx_last"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1)
    )

    # 過去最高速度指数（能力の上限値）
    df["hist_speed_idx_best"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1).expanding().max()
    )

    # 直近3走の速度指数平均（安定した能力推定。hist_avg_time_dev_3 の絶対スケール版）
    df["hist_speed_idx_avg3"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # 同条件（実距離×馬場種別）での過去最高速度指数（条件適性の絶対評価）
    # distance_category ではなく distance を使うことで他の speed_idx 系と一貫性を保つ
    df["hist_speed_idx_cond_best"] = (
        df.groupby(["ketto_num", "distance", "surface_code"])["_speed_idx"]
        .transform(lambda x: x.shift(1).expanding().max())
    )

    # ─── Step 8: 一時列を削除 ─────────────────────────────────────────────────
    # _speed_idx は当該レースの結果情報を含むため特徴量として残してはならない
    # cond_avg_time / cond_std_time も中間計算値であり不要
    df = df.drop(
        columns=["_speed_idx", "cond_avg_time", "cond_std_time"],
        errors="ignore",
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.7: RELATIVE FEATURES (within-race z-score + pace index)
# hist_speed_idx_avg3 生成後に呼び出すこと（依存関係）。
# ═══════════════════════════════════════════════════════════════════════════════

def _field_zscore(df: pd.DataFrame, col: str, z_col: str) -> None:
    """同レース内 z-score を in-place で追加する。"""
    race_mean = df.groupby("race_id")[col].transform("mean")
    race_std = df.groupby("race_id")[col].transform("std")
    df[z_col] = (df[col] - race_mean) / (race_std + 1e-6)


def _build_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量（within-race z-score + ペース指数）を生成する。"""
    z_pairs = [
        ("hist_last_time_dev", "field_z_time_dev"),
        ("hist_total_prize", "field_z_prize"),
        ("hist_last_last3f", "field_z_last3f"),
        ("hist_win_rate", "field_z_win_rate"),
        ("hist_speed_idx_avg3", "field_z_speed_idx"),
        ("hist_place_rate", "field_z_place_rate"),
    ]
    for src, dst in z_pairs:
        _field_zscore(df, src, dst)

    df["_front_pref_filled"] = df["hist_front_running_pref"].fillna(0)
    df["field_front_runner_density"] = df.groupby("race_id")["_front_pref_filled"].transform("mean")
    df = df.drop(columns=["_front_pref_filled"])

    df["relative_post_position"] = (
        df["wakuban"].astype(float) / df["horse_count"].astype(float)
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAINING FEATURES (HC/WC)
# 調教データは race_date より前のセッションのみを参照する（リーク防止）。
# ═══════════════════════════════════════════════════════════════════════════════

def _load_hc(cfg: dict) -> pd.DataFrame:
    """HC_preprocessed.parquet を読み込む。"""
    p = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "HC_preprocessed.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"HC_preprocessed.parquet が見つかりません: {p}\npreprocess.py を先に実行してください。"
        )
    df = pd.read_parquet(p)
    print(f"  HC: {len(df):,} rows")
    return df


def _load_wc(cfg: dict) -> pd.DataFrame:
    """WC_preprocessed.parquet を読み込む。なければ空 DataFrame を返す。"""
    p = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "WC_preprocessed.parquet"
    if not p.exists():
        print("  WC: ファイルなし（スキップ）")
        return pd.DataFrame(
            columns=["ketto_num", "training_date", "wc_3f_sec", "wc_4f_sec", "wc_1f_sec"]
        )
    df = pd.read_parquet(p)
    print(f"  WC: {len(df):,} rows")
    return df


def _add_training_features(
    df: pd.DataFrame,
    hc: pd.DataFrame,
    wc: pd.DataFrame,
) -> pd.DataFrame:
    """調教特徴量を df に追加して返す。

    カテゴリA: 絶対値系（最近接・最速・セッション数）
    カテゴリB: 同レース内相対比較（rank / zscore）
    カテゴリC: 過去走との差分（shift(1)）
    """
    # ketto_num を int64 に統一（SE parquet では object の場合がある）
    # merge_asof の by キーおよび後続の merge キーで dtype 一致が必要
    df["ketto_num"] = pd.to_numeric(df["ketto_num"], errors="coerce").astype(np.int64)
    keys = df[["race_id", "ketto_num", "race_date"]].copy()
    active_horses = set(keys["ketto_num"].unique())

    # ─── カテゴリA: HC 系 ──────────────────────────────────────────────────────
    if len(hc) > 0:
        hc_f = hc[hc["ketto_num"].isin(active_horses)].copy()
        # merge_asof は right_on キー（training_date）がグローバルソートされている必要がある
        hc_f = hc_f.sort_values("training_date").reset_index(drop=True)
        keys_sorted = keys.sort_values("race_date").reset_index(drop=True)

        # 最近接セッション (merge_asof: training_date < race_date かつ 14日以内)
        last_hc = pd.merge_asof(
            keys_sorted,
            hc_f[["ketto_num", "training_date", "hc_3f_sec", "hc_4f_sec", "hc_200_sec"]],
            left_on="race_date",
            right_on="training_date",
            by="ketto_num",
            direction="backward",
            tolerance=pd.Timedelta(days=14),
        ).rename(columns={
            "hc_3f_sec": "trn_hc_last_3f_sec",
            "hc_4f_sec": "trn_hc_last_4f_sec",
            "hc_200_sec": "trn_hc_last_200_sec",
        }).drop(columns=["training_date"])

        # 14日ウィンドウ集計（最速タイム・セッション数）
        merged_hc = keys.merge(
            hc_f[["ketto_num", "training_date", "hc_3f_sec", "hc_200_sec"]],
            on="ketto_num", how="left"
        )
        diff_days = (merged_hc["race_date"] - merged_hc["training_date"]).dt.days
        win_hc = merged_hc[(diff_days > 0) & (diff_days <= 14)].copy()
        hc_agg = win_hc.groupby(["race_id", "ketto_num"]).agg(
            trn_hc_best_3f_14d=("hc_3f_sec", "min"),
            trn_hc_best_200_14d=("hc_200_sec", "min"),
            trn_hc_count_14d=("training_date", "count"),
        ).reset_index()

        df = df.merge(
            last_hc[["race_id", "ketto_num", "trn_hc_last_3f_sec", "trn_hc_last_4f_sec", "trn_hc_last_200_sec"]],
            on=["race_id", "ketto_num"], how="left"
        )
        df = df.merge(hc_agg, on=["race_id", "ketto_num"], how="left")
    else:
        for col in ["trn_hc_last_3f_sec", "trn_hc_last_4f_sec", "trn_hc_last_200_sec",
                    "trn_hc_best_3f_14d", "trn_hc_best_200_14d", "trn_hc_count_14d"]:
            df[col] = np.nan

    # ─── カテゴリA: WC 系 ──────────────────────────────────────────────────────
    if len(wc) > 0:
        wc_f = wc[wc["ketto_num"].isin(active_horses)].copy()
        # merge_asof は right_on キー（training_date）がグローバルソートされている必要がある
        wc_f = wc_f.sort_values("training_date").reset_index(drop=True)
        keys_sorted = keys.sort_values("race_date").reset_index(drop=True)

        last_wc = pd.merge_asof(
            keys_sorted,
            wc_f[["ketto_num", "training_date", "wc_3f_sec", "wc_4f_sec", "wc_1f_sec"]],
            left_on="race_date",
            right_on="training_date",
            by="ketto_num",
            direction="backward",
            tolerance=pd.Timedelta(days=14),
        ).rename(columns={
            "wc_3f_sec": "trn_wc_last_3f_sec",
            "wc_4f_sec": "trn_wc_last_4f_sec",
            "wc_1f_sec": "trn_wc_last_1f_sec",
        }).drop(columns=["training_date"])

        merged_wc = keys.merge(
            wc_f[["ketto_num", "training_date", "wc_3f_sec", "wc_1f_sec"]],
            on="ketto_num", how="left"
        )
        diff_days = (merged_wc["race_date"] - merged_wc["training_date"]).dt.days
        win_wc = merged_wc[(diff_days > 0) & (diff_days <= 14)].copy()
        wc_agg = win_wc.groupby(["race_id", "ketto_num"]).agg(
            trn_wc_best_3f_14d=("wc_3f_sec", "min"),
            trn_wc_best_1f_14d=("wc_1f_sec", "min"),
            trn_wc_count_14d=("training_date", "count"),
        ).reset_index()

        df = df.merge(
            last_wc[["race_id", "ketto_num", "trn_wc_last_3f_sec", "trn_wc_last_4f_sec", "trn_wc_last_1f_sec"]],
            on=["race_id", "ketto_num"], how="left"
        )
        df = df.merge(wc_agg, on=["race_id", "ketto_num"], how="left")
    else:
        for col in ["trn_wc_last_3f_sec", "trn_wc_last_4f_sec", "trn_wc_last_1f_sec",
                    "trn_wc_best_3f_14d", "trn_wc_best_1f_14d", "trn_wc_count_14d"]:
            df[col] = np.nan

    # 合計セッション数（HC + WC）
    hc_cnt = df["trn_hc_count_14d"] if "trn_hc_count_14d" in df.columns else pd.Series(np.nan, index=df.index)
    wc_cnt = df["trn_wc_count_14d"] if "trn_wc_count_14d" in df.columns else pd.Series(np.nan, index=df.index)
    df["trn_total_count_14d"] = hc_cnt.fillna(0) + wc_cnt.fillna(0)
    # 両方NaNの場合はNaNに戻す
    both_nan = df["trn_hc_count_14d"].isna() & df["trn_wc_count_14d"].isna()
    df.loc[both_nan, "trn_total_count_14d"] = np.nan

    # ─── カテゴリB: 同レース内相対比較 ───────────────────────────────────────────

    def zscore_within_race(s: pd.Series) -> pd.Series:
        # 全馬NaNの場合はNaNを返す（0.0への暗黙補完を防ぐ）
        if s.isna().all():
            return pd.Series(np.nan, index=s.index)
        mean = s.mean()
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mean) / std

    df["trn_hc_rank_3f"] = df.groupby("race_id")["trn_hc_best_3f_14d"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    df["trn_hc_rank_200"] = df.groupby("race_id")["trn_hc_best_200_14d"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    df["trn_hc_zscore_3f"] = df.groupby("race_id")["trn_hc_best_3f_14d"].transform(zscore_within_race)

    df["trn_wc_rank_3f"] = df.groupby("race_id")["trn_wc_best_3f_14d"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    df["trn_wc_zscore_3f"] = df.groupby("race_id")["trn_wc_best_3f_14d"].transform(zscore_within_race)

    # ─── カテゴリC: 過去走との差分 (shift(1)) ─────────────────────────────────
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
    grp = df.groupby("ketto_num")

    df["trn_hc_3f_delta"]  = df["trn_hc_best_3f_14d"]  - grp["trn_hc_best_3f_14d"].shift(1)
    df["trn_hc_200_delta"] = df["trn_hc_best_200_14d"] - grp["trn_hc_best_200_14d"].shift(1)
    df["trn_wc_3f_delta"]  = df["trn_wc_best_3f_14d"]  - grp["trn_wc_best_3f_14d"].shift(1)
    df["trn_count_delta"]  = df["trn_total_count_14d"]  - grp["trn_total_count_14d"].shift(1)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ラベル生成
# ═══════════════════════════════════════════════════════════════════════════════

def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """LambdaRank / Binary 用ラベルを生成する。

    label_gain = [0, 1, 3, 7, 15, 31, 63]（7エントリ）の制約上、
    ラベルは 0〜6 の範囲に収める必要がある。

    着順 → ラベル対応:
        1着 → 6 (gain=63, 最高)
        2着 → 5 (gain=31)
        3着 → 4 (gain=15)
        4着 → 3 (gain=7)
        5着 → 2 (gain=3)
        6着 → 1 (gain=1)
        7着以下 → 0 (gain=0, 全て同等)

    公式: label = clip(7 - finish_rank, 0, 6)
    頭数に依存せず、着順のみで決まる絶対ラベル方式を採用する。
    """
    # clip(lower=0) で 7着以下は全て 0
    df["lr_label"] = (7 - df["finish_rank"]).clip(lower=0, upper=6).astype(np.int8)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: NaN 率レポート
# ═══════════════════════════════════════════════════════════════════════════════

def _report_nan_rates(df: pd.DataFrame, threshold: float = 0.3) -> None:
    """NaN 率が高い特徴量を報告する。"""
    n = len(df)
    nan_rates = (df.isnull().sum() / n).sort_values(ascending=False)
    high_nan = nan_rates[nan_rates > threshold]
    if len(high_nan) > 0:
        print("\n  [警告] NaN 率が高い列（新馬・初コースは許容）:")
        for col, rate in high_nan.items():
            print(f"    {col}: {rate:.1%}")
    else:
        print(f"\n  NaN 率 > {threshold:.0%} の列なし")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: マニフェスト保存
# ═══════════════════════════════════════════════════════════════════════════════

def _save_manifest(df: pd.DataFrame, out_dir: Path, version: str) -> None:
    """特徴量ファイルのメタ情報を manifest.json として保存する。"""
    n = len(df)
    manifest = {
        "version": version,
        "rows": n,
        "cols": len(df.columns),
        "columns": list(df.columns),
        "date_range": {
            "min": str(df["race_date"].min().date()),
            "max": str(df["race_date"].max().date()),
        },
        "race_count": df["race_id"].nunique(),
        "nan_rates": {
            col: float(f"{df[col].isnull().mean():.4f}")
            for col in df.columns
            if df[col].isnull().any()
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  manifest saved: {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: メイン
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = load_config()
    version = cfg["data"]["features_version"]
    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    out_path = feat_dir / f"features_{version}.parquet"

    # 既存ファイルのバックアップ（上書き前に必ず実行）
    if out_path.exists():
        bk_path = out_path.with_suffix(f".bak.parquet")
        shutil.copy2(out_path, bk_path)
        print(f"[backup] {out_path.name} → {bk_path.name}")

    print("\n[1] Loading preprocessed data...")
    df = _load_data(cfg)

    print("\n[2] Applying filters...")
    df = _apply_filters(df, cfg)

    # 市場情報混入チェック
    _check_no_market_features(df)

    print("\n[3] Building historical features (shift-1 leak prevention)...")
    df = _build_hist_features(df)

    print("\n[4] Building current race features...")
    df = _build_current_features(df)

    print("\n[5] Building bloodline features...")
    df = _build_sire_features(df)

    print("\n[5.5] Building jockey/trainer features...")
    df = _build_jockey_trainer_features(df)

    print("\n[5.6] Building speed index features (hist_speed_idx_*)...")
    df = _build_speed_index_features(df)

    print("\n[5.7] Building relative features (field_z_*, pace index)...")
    df = _build_relative_features(df)

    print("\n[6] Building training features (HC/WC)...")
    hc = _load_hc(cfg)
    wc = _load_wc(cfg)
    df = _add_training_features(df, hc, wc)

    print("\n[7] Building labels (lr_label)...")
    df = _build_labels(df)

    # 最終的な市場情報混入チェック
    _check_no_market_features(df)

    # NaN 率レポート
    print("\n[8] NaN rate report:")
    _report_nan_rates(df, threshold=0.3)

    # 保存前に行順序を LambdaRank グループ割り当て用に修正する。
    # 中間処理では ketto_num 順（shift(1) 効率化）を使うが、
    # parquet の行順序は (race_date, race_id, horse_num) でなければならない。
    # get_group_sizes(sort=False) が正しいグループを返す前提がこれ。
    sort_cols = ["race_date", "race_id", "horse_num"]
    available_sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(available_sort_cols).reset_index(drop=True)
    print(f"\n[8.5] Final row ordering: {available_sort_cols}")

    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    print(f"\n[9] Saved: {out_path}")
    print(f"  rows={len(df):,}, cols={len(df.columns)}")

    # 時系列の統計
    train_end = pd.Timestamp(cfg["training"]["train_end"])
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])
    train = df[df["race_date"] <= train_end]
    valid = df[(df["race_date"] > train_end) & (df["race_date"] <= valid_end)]
    test  = df[df["race_date"] > valid_end]
    print(f"\n  Train: {len(train):,} rows, {train['race_id'].nunique():,} races "
          f"({train['race_date'].min().date()} - {train['race_date'].max().date()})")
    print(f"  Valid: {len(valid):,} rows, {valid['race_id'].nunique():,} races "
          f"({valid['race_date'].min().date()} - {valid['race_date'].max().date()})")
    print(f"  Test:  {len(test):,} rows, {test['race_id'].nunique():,} races "
          f"({test['race_date'].min().date()} - {test['race_date'].max().date()})")

    # マニフェスト保存
    _save_manifest(df, feat_dir, version)

    print("\n[10] create_features Done.")


if __name__ == "__main__":
    main()

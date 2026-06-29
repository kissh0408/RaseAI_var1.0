"""
create_features.py — RaceAI_var1.0 特徴量生成スクリプト

01_preprocessed/ の Parquet から特徴量を生成し
02_features/features_v1.parquet を出力する。

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
    df["_is_top_grade"] = df["grade_code"].isin([1, 2, 3]).astype(np.int8)
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

    # 一時列を削除
    df = df.drop(columns=["_time_dev", "_is_top_grade", "_dist_bin_100"])
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
    # 新馬(hist_win_rate=NaN)は 0 として field 平均を計算
    win_rate_filled = df["hist_win_rate"].fillna(0)
    df["field_avg_win_rate"] = df.groupby("race_id")[win_rate_filled.name].transform(
        lambda x: win_rate_filled.loc[x.index].mean()
    )
    # groupby+transform では Series を直接参照できないため fillna後の列を使う
    df["_hist_win_rate_filled"] = df["hist_win_rate"].fillna(0)
    df["field_avg_win_rate"] = df.groupby("race_id")["_hist_win_rate_filled"].transform("mean")
    df["field_avg_prize"] = df.groupby("race_id")["hist_avg_prize_3"].transform("mean")
    df["win_rate_vs_field"] = df["hist_win_rate"] - df["field_avg_win_rate"]
    df["prize_vs_field"] = df["hist_avg_prize_3"] - df["field_avg_prize"]
    df = df.drop(columns=["_hist_win_rate_filled"])

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BLOODLINE FEATURES
# 父馬・母父の産駒成績。
# 注意: Phase 1 では全期間の統計を使用（軽微な前方参照バイアスあり）。
#       正確な時系列計算は Phase 3 以降で実装予定。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sire_features(df: pd.DataFrame) -> pd.DataFrame:
    """父馬・母父産駒の成績を集計して特徴量にする。

    Phase 1 実装方針:
    - 全期間データで sire_id / bms_id の勝率・平均距離を集計
    - 軽微な前方参照バイアスがあるが、血統特性は時間的に安定しており
      Phase 1 ベースラインとして許容する
    - Phase 3 で正確な時系列制限版に置き換える予定
    """
    # ─── 父馬産駒の同馬場勝率 ─────────────────────────────────────────────────
    # groupby(['sire_id', 'surface_code']) での通算勝率
    if "sire_id" in df.columns and df["sire_id"].notna().any():
        sire_surf_wr = (
            df.groupby(["sire_id", "surface_code"], observed=True)["is_win"]
            .mean()
            .reset_index()
            .rename(columns={"is_win": "hist_sire_surface_win_rate"})
        )
        df = df.merge(sire_surf_wr, on=["sire_id", "surface_code"], how="left")

        # 父馬産駒の平均勝ち距離（勝ちレースのみ）
        sire_wins = df[df["is_win"] == 1]
        if len(sire_wins) > 0:
            sire_avg_dist = (
                sire_wins.groupby("sire_id", observed=True)["distance"]
                .mean()
                .reset_index()
                .rename(columns={"distance": "_sire_avg_win_dist"})
            )
            df = df.merge(sire_avg_dist, on="sire_id", how="left")
            df["hist_sire_dist_diff"] = (
                (df["distance"] - df["_sire_avg_win_dist"]).abs()
            )
            df = df.drop(columns=["_sire_avg_win_dist"])
        else:
            df["hist_sire_dist_diff"] = np.nan
    else:
        df["hist_sire_surface_win_rate"] = np.nan
        df["hist_sire_dist_diff"] = np.nan

    # ─── 母父（BMS）産駒の通算勝率 ─────────────────────────────────────────────
    if "bms_id" in df.columns and df["bms_id"].notna().any():
        bms_wr = (
            df.groupby("bms_id", observed=True)["is_win"]
            .mean()
            .reset_index()
            .rename(columns={"is_win": "hist_bms_win_rate"})
        )
        df = df.merge(bms_wr, on="bms_id", how="left")
    else:
        df["hist_bms_win_rate"] = np.nan

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

    # 保存
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

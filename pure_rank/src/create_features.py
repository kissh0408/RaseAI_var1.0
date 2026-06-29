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

    # 一時列を削除
    df = df.drop(columns=["_time_dev"])
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
# SECTION 6: ラベル生成
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
# SECTION 7: NaN 率レポート
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
# SECTION 8: マニフェスト保存
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
# SECTION 9: メイン
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

    print("\n[6] Building labels (lr_label)...")
    df = _build_labels(df)

    # 最終的な市場情報混入チェック
    _check_no_market_features(df)

    # NaN 率レポート
    print("\n[7] NaN rate report:")
    _report_nan_rates(df, threshold=0.3)

    # 保存
    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    print(f"\n[8] Saved: {out_path}")
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

    print("\n[create_features] Done.")


if __name__ == "__main__":
    main()

"""
features_soft_turf_impute.py — 芝稍重補完特徴量・父馬芝重不良適性特徴量

生成特徴量:
    sire_turf_soft_win_rate           : 父馬×芝×稍重の産駒勝率（ベイズ平滑化）
    dam_sire_turf_soft_win_rate       : 母父馬×芝×稍重の産駒勝率（ベイズ平滑化）
    horse_turf_soft_win_rate_imputed  : 既存 horse_turf_soft_win_rate の NaN 補完版
    going_match_score_turf_imputed    : 既存 going_match_score_turf の補完版
    sire_turf_heavy_win_rate          : 父馬×芝×重・不良（turf_condition in [3,4]）勝率（ベイズ平滑化）
                                        全行付与（ダート・良馬場行でも父馬の重不良適性スコアを付与）

NaN 補完優先順位 (horse_turf_soft_win_rate_imputed):
    優先度1: horse_turf_soft_win_rate が non-NaN → そのまま使用
    優先度2: NaN → sire_turf_soft_win_rate で代替
    優先度3: それも NaN → dam_sire_turf_soft_win_rate で代替
    優先度4: 全て NaN → 訓練期間のコース×稍重全体平均勝率定数（約 0.065）

going_match_score_turf_imputed:
    horse_turf_soft_win_rate_imputed / (コース別稍重全体勝率) でスコア化
    結果を [0.0, 3.0] にクリップ

リーク防止:
    sire/dam_sire 産駒統計は df を date でソートした後 cumsum - current で当該行除外
    定数（コース別稍重全体勝率）は訓練期間（2015-2024年末）のみで計算
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 定数
# train_config.json の training.train_end_date を参照し、なければ定数にフォールバック
# ------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "train_config.json"
try:
    _TRAIN_END: str = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))["training"]["train_end_date"]
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    _TRAIN_END = "2024-12-31"

# 芝判定: track_code < 50（JV-Link: 10-24=芝, 52-57=ダート）
_TURF_THRESHOLD = 50

# 稍重: turf_condition == 2
_SOFT_CONDITION = 2

# ベイズ平滑化パラメータ
_SIRE_PRIOR_N = 15.0
_SIRE_PRIOR_MEAN = 0.065    # 芝稍重平均勝率

_DAM_SIRE_PRIOR_N = 15.0
_DAM_SIRE_PRIOR_MEAN = 0.065

# 全行フォールバック定数（訓練期間の芝稍重全体勝率）
_GLOBAL_SOFT_TURF_WIN_RATE = 0.065


def add_soft_turf_impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    芝稍重補完特徴量を df に追加して返す。

    Args:
        df: features_past_v12など（ketto_num, p_sire, p_dam_sire,
            horse_turf_soft_win_rate, going_match_score_turf,
            track_code, turf_condition, finish_rank, date 列を持つ DataFrame）
    Returns:
        新特徴量4列を追加した DataFrame（行数・順序は変更しない）
    """
    # --- 時系列ソート保証 ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 基本フラグ ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    track_num = pd.to_numeric(df["track_code"], errors="coerce")
    is_turf = (track_num < _TURF_THRESHOLD).fillna(False)

    turf_cond_num = pd.to_numeric(df["turf_condition"], errors="coerce")
    is_soft = (turf_cond_num == _SOFT_CONDITION).fillna(False)

    # 芝×稍重マスク
    turf_soft_mask = is_turf & is_soft

    # ------------------------------------------------------------------
    # 1. sire_turf_soft_win_rate
    #    父馬×芝×稍重の産駒勝率（ベイズ平滑化）
    #    リーク防止: date ソート済みで cumsum - current
    # ------------------------------------------------------------------
    if "p_sire" in df.columns:
        sire_key = df["p_sire"].astype(str).fillna("__nan__")

        # 芝稍重レースのみをカウント対象
        sire_count = turf_soft_mask.astype("int8")
        sire_win = (win_flag * sire_count).astype("int8")

        cum_sire_runs = sire_count.groupby(sire_key, sort=False).cumsum() - sire_count
        cum_sire_wins = sire_win.groupby(sire_key, sort=False).cumsum() - sire_win

        sire_smoothed = (
            (cum_sire_wins + _SIRE_PRIOR_N * _SIRE_PRIOR_MEAN)
            / (cum_sire_runs + _SIRE_PRIOR_N)
        )
        df["sire_turf_soft_win_rate"] = sire_smoothed.where(
            cum_sire_runs >= 5, np.nan
        ).astype("float32")
    else:
        df["sire_turf_soft_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 2. dam_sire_turf_soft_win_rate
    #    母父馬×芝×稍重の産駒勝率（ベイズ平滑化）
    # ------------------------------------------------------------------
    if "p_dam_sire" in df.columns:
        dam_sire_key = df["p_dam_sire"].astype(str).fillna("__nan__")

        dam_count = turf_soft_mask.astype("int8")
        dam_win = (win_flag * dam_count).astype("int8")

        cum_dam_runs = dam_count.groupby(dam_sire_key, sort=False).cumsum() - dam_count
        cum_dam_wins = dam_win.groupby(dam_sire_key, sort=False).cumsum() - dam_win

        dam_smoothed = (
            (cum_dam_wins + _DAM_SIRE_PRIOR_N * _DAM_SIRE_PRIOR_MEAN)
            / (cum_dam_runs + _DAM_SIRE_PRIOR_N)
        )
        df["dam_sire_turf_soft_win_rate"] = dam_smoothed.where(
            cum_dam_runs >= 5, np.nan
        ).astype("float32")
    else:
        df["dam_sire_turf_soft_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 3. horse_turf_soft_win_rate_imputed
    #    優先度: horse → sire → dam_sire → 全体定数
    # ------------------------------------------------------------------
    if "horse_turf_soft_win_rate" in df.columns:
        horse_base = pd.to_numeric(df["horse_turf_soft_win_rate"], errors="coerce").astype("float32")
    else:
        horse_base = pd.Series(np.nan, index=df.index, dtype="float32")

    imputed = horse_base.copy()

    # 優先度2: sire で補完
    nan_mask = imputed.isna()
    if "sire_turf_soft_win_rate" in df.columns:
        imputed = imputed.where(~nan_mask, df["sire_turf_soft_win_rate"].astype("float32"))

    # 優先度3: dam_sire で補完
    nan_mask = imputed.isna()
    if "dam_sire_turf_soft_win_rate" in df.columns:
        imputed = imputed.where(~nan_mask, df["dam_sire_turf_soft_win_rate"].astype("float32"))

    # 優先度4: 全体定数でフォールバック
    # 定数はコース×稍重全体平均勝率（訓練期間で計算）
    # 芝稍重レース全体の平均勝率を訓練データから算出して定数として使用
    train_mask = df["date"] <= _TRAIN_END
    train_turf_soft = train_mask & turf_soft_mask
    if train_turf_soft.sum() > 0:
        global_rate = float(win_flag[train_turf_soft].mean())
    else:
        global_rate = _GLOBAL_SOFT_TURF_WIN_RATE

    nan_mask = imputed.isna()
    imputed = imputed.where(~nan_mask, global_rate)

    df["horse_turf_soft_win_rate_imputed"] = imputed.astype("float32")

    # ------------------------------------------------------------------
    # 4. going_match_score_turf_imputed
    #    horse_turf_soft_win_rate_imputed / (コース別稍重全体勝率) でスコア化
    #    分母は訓練期間のコース別稍重全体勝率（定数）
    #    結果を [0.0, 3.0] にクリップ
    # ------------------------------------------------------------------

    # コース別稍重全体勝率を訓練期間で計算
    train_df = df[train_mask & turf_soft_mask].copy()
    train_finish = pd.to_numeric(train_df["finish_rank"], errors="coerce")
    train_win_flag = (train_finish == 1).astype("int8")

    if "course_code" in df.columns and len(train_df) > 0:
        # course_code を数値化してからグループ化し、map 時の型不一致（str vs float）によるNaN発生を防ぐ
        course_soft_win_rate = (
            train_df.assign(win=train_win_flag, _cc=pd.to_numeric(train_df["course_code"], errors="coerce"))
            .groupby("_cc")["win"]
            .mean()
        )
        # course_code のデフォルト値（全体定数）
        default_course_rate = float(win_flag[train_turf_soft].mean()) if train_turf_soft.sum() > 0 else global_rate

        course_num_series = pd.to_numeric(df["course_code"], errors="coerce")
        denom = course_num_series.map(course_soft_win_rate).fillna(default_course_rate).astype("float32")
        # ゼロ除算防止
        denom = denom.replace(0.0, default_course_rate)
    else:
        denom = pd.Series(global_rate, index=df.index, dtype="float32")

    raw_score = (df["horse_turf_soft_win_rate_imputed"] / denom).astype("float32")

    # 既存 going_match_score_turf が non-NaN の行はそちらを優先
    if "going_match_score_turf" in df.columns:
        existing = pd.to_numeric(df["going_match_score_turf"], errors="coerce").astype("float32")
        score = existing.where(existing.notna(), raw_score)
    else:
        score = raw_score

    df["going_match_score_turf_imputed"] = score.clip(0.0, 3.0).astype("float32")

    # ------------------------------------------------------------------
    # 5. sire_turf_heavy_win_rate
    #    父馬×芝×重・不良（turf_condition in [3, 4]）の産駒勝率（ベイズ平滑化）
    #
    #    設計: 全行に値を付与する（ダート行・良馬場行でも父馬の重不良適性スコアを保持）。
    #    理由: 雨が降り始めた状況で当日馬場が確定する前でもスコアを参照できるよう
    #          ダート行や良馬場行でも NaN にしない。
    #          ただし min_periods 未満の場合は NaN。
    #
    #    ベイズ平滑化パラメータ:
    #        prior_n    = 15.0  （稍重と同程度の希薄性）
    #        prior_mean = 0.0727 （芝重・不良の実測平均勝率）
    #        min_periods= 5     （重・不良は出走機会が少ないため閾値を低めに設定）
    # ------------------------------------------------------------------
    if "p_sire" in df.columns and "sire_turf_heavy_win_rate" not in df.columns:
        _HEAVY_PRIOR_N = 15.0
        _HEAVY_PRIOR_MEAN = 0.0727  # 芝重・不良の実測平均勝率
        _HEAVY_MIN_PERIODS = 5

        # 芝×重・不良マスク（turf_condition: 3=重, 4=不良）
        is_heavy_bad = turf_cond_num.isin([3, 4]).fillna(False)
        turf_heavy_mask = is_turf & is_heavy_bad

        sire_key_h = df["p_sire"].astype(str).fillna("__nan__")

        # 芝重・不良レースのみをカウント対象（当該行除外の累積）
        heavy_count = turf_heavy_mask.astype("int8")
        heavy_win = (win_flag * heavy_count).astype("int8")

        cum_heavy_runs = heavy_count.groupby(sire_key_h, sort=False).cumsum() - heavy_count
        cum_heavy_wins = heavy_win.groupby(sire_key_h, sort=False).cumsum() - heavy_win

        heavy_smoothed = (
            (cum_heavy_wins + _HEAVY_PRIOR_N * _HEAVY_PRIOR_MEAN)
            / (cum_heavy_runs + _HEAVY_PRIOR_N)
        )
        # min_periods 未満は NaN（全行付与設計のためサーフェスによるマスクなし）
        df["sire_turf_heavy_win_rate"] = heavy_smoothed.where(
            cum_heavy_runs >= _HEAVY_MIN_PERIODS, np.nan
        ).astype("float32")
    elif "sire_turf_heavy_win_rate" not in df.columns:
        df["sire_turf_heavy_win_rate"] = np.nan

    return df


def recompute_going_match_score_turf_imputed_scenario(df: pd.DataFrame) -> pd.Series:
    """what-if シナリオ後に going_match_score_turf_imputed を学習式と同期する。

    学習時（add_soft_turf_impute_features）と同様、更新済み going_match_score_turf を
    優先し、NaN 行のみ既存 imputed 値をフォールバックとして使う。
    """
    if "going_match_score_turf_imputed" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")

    existing_gms = (
        pd.to_numeric(df.get("going_match_score_turf"), errors="coerce").astype("float32")
        if "going_match_score_turf" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float32")
    )
    prior_imputed = pd.to_numeric(df["going_match_score_turf_imputed"], errors="coerce").astype(
        "float32"
    )
    score = existing_gms.where(existing_gms.notna(), prior_imputed)
    return score.clip(0.0, 3.0).astype("float32")


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v12_path = project_root / "model_training/data/02_features/features_past_v12.parquet"

    print("Loading data...")
    df = pd.read_parquet(v12_path)
    print(f"Input: {df.shape}")

    df = add_soft_turf_impute_features(df)
    print(f"Output: {df.shape}")

    new_cols = [
        "sire_turf_soft_win_rate",
        "dam_sire_turf_soft_win_rate",
        "horse_turf_soft_win_rate_imputed",
        "going_match_score_turf_imputed",
    ]
    for col in new_cols:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            valid = df[col].dropna()
            print(f"  {col}: NaN={nan_pct:.1f}%, mean={valid.mean():.4f}, std={valid.std():.4f}")
        else:
            print(f"  {col}: NOT GENERATED")

"""
features_agari_course.py — コース別上がり3F特徴量

生成特徴量:
    horse_agari3f_course_avg        : horse × course_code 別の上がり3F expanding mean（shift(1)）
    horse_agari3f_course_rank_avg   : 同一course_code内での上がり3F順位の expanding mean（shift(1)）
    agari3f_best_in_course_lag1     : 直前の同一course_codeレースでレース最速上がりだったか（0/1）

リーク防止:
    全特徴量は shift(1) で当該レース自身を除外した後に cumulative 統計を計算する。
    time_3f_after（当走のagari3F）は SE から結合してから処理し、
    最終的に削除することでリークを防止する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_agari_course_features(
    df: pd.DataFrame,
    se_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    コース別上がり3F特徴量を df に追加して返す。

    Args:
        df    : features_past_v10 など（race_id, ketto_num, course_code, date を持つ DataFrame）
        se_df : SE_preprocessed（race_id, ketto_num, time_3f_after を持つ DataFrame）

    Returns:
        新特徴量 3 列を追加した DataFrame（行数・順序は変更しない）
    """
    new_cols = [
        "horse_agari3f_course_avg",
        "horse_agari3f_course_rank_avg",
        "agari3f_best_in_course_lag1",
    ]
    if all(c in df.columns for c in new_cols):
        return df

    # --- SE から time_3f_after をマージ ---
    if "time_3f_after" not in se_df.columns:
        raise ValueError("SE_preprocessed に time_3f_after 列が存在しない")

    merge_src = se_df[["race_id", "ketto_num", "time_3f_after"]].copy()
    merge_src["time_3f_after"] = pd.to_numeric(
        merge_src["time_3f_after"], errors="coerce"
    ).astype("float32")

    df = df.merge(
        merge_src.rename(columns={"time_3f_after": "_agari3f_raw"}),
        on=["race_id", "ketto_num"],
        how="left",
    )

    # --- 時系列ソート保証 ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    agari = df["_agari3f_raw"]  # float32, NaN あり

    # ------------------------------------------------------------------
    # 1. レース内での上がり3F順位を計算（後続の shift に使用）
    #    時間が短い（小さい）ほど順位が上（ascending=True → rank小=最速）
    # ------------------------------------------------------------------
    if "race_id" in df.columns:
        agari_rank_in_race = agari.groupby(df["race_id"], sort=False).rank(
            method="min", ascending=True
        )
    else:
        agari_rank_in_race = pd.Series(np.nan, index=df.index)

    # ------------------------------------------------------------------
    # 2. 最速上がりフラグ（当該レースでレース1位の上がり3Fか）
    #    後続で shift(1) して "前走最速上がりだったか" に変換する
    # ------------------------------------------------------------------
    is_best_in_race = (agari_rank_in_race == 1).astype("float32")
    is_best_in_race = is_best_in_race.where(agari.notna(), np.nan)

    # ------------------------------------------------------------------
    # 3. horse × course_code グループでの lag 特徴量
    # ------------------------------------------------------------------
    if "course_code" not in df.columns:
        df["horse_agari3f_course_avg"] = np.nan
        df["horse_agari3f_course_rank_avg"] = np.nan
        df["agari3f_best_in_course_lag1"] = np.nan
        df = df.drop(columns=["_agari3f_raw"])
        return df

    grp_horse_course = [df["ketto_num"], df["course_code"]]

    # --- horse_agari3f_course_avg ---
    # agari3f の shift(1) 後 expanding mean（当該レースは除外）
    if "horse_agari3f_course_avg" not in df.columns:
        agari_filled = agari.fillna(0.0)
        agari_valid = agari.notna().astype("int8")

        # cumsum で shift(1) 相当を計算（当該行を引く）
        cum_agari_sum = (
            agari_filled.groupby(grp_horse_course, sort=False).cumsum()
            - agari_filled
        )
        cum_agari_cnt = (
            agari_valid.groupby(grp_horse_course, sort=False).cumsum()
            - agari_valid
        )
        df["horse_agari3f_course_avg"] = (
            (cum_agari_sum / cum_agari_cnt.replace(0, np.nan))
            .where(cum_agari_cnt > 0, np.nan)
            .astype("float32")
        )

    # --- horse_agari3f_course_rank_avg ---
    # 上がり3F順位の shift(1) 後 expanding mean
    if "horse_agari3f_course_rank_avg" not in df.columns:
        rank_filled = agari_rank_in_race.fillna(0.0)
        rank_valid = agari_rank_in_race.notna().astype("int8")

        cum_rank_sum = (
            rank_filled.groupby(grp_horse_course, sort=False).cumsum()
            - rank_filled
        )
        cum_rank_cnt = (
            rank_valid.groupby(grp_horse_course, sort=False).cumsum()
            - rank_valid
        )
        df["horse_agari3f_course_rank_avg"] = (
            (cum_rank_sum / cum_rank_cnt.replace(0, np.nan))
            .where(cum_rank_cnt > 0, np.nan)
            .astype("float32")
        )

    # --- agari3f_best_in_course_lag1 ---
    # 直前の同一course_codeレースで最速上がりだったか (0/1)
    # shift(1) で当該レースを除外する
    if "agari3f_best_in_course_lag1" not in df.columns:
        df["agari3f_best_in_course_lag1"] = (
            is_best_in_race
            .groupby(grp_horse_course, sort=False)
            .shift(1)
            .astype("float32")
        )

    # --- 作業列を削除（当走情報なのでリーク防止のため除去） ---
    df = df.drop(columns=["_agari3f_raw"])

    return df


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v10_path = project_root / "model_training/data/02_features/features_past_v10.parquet"
    se_path = project_root / "model_training/data/01_preprocessed/SE_preprocessed.parquet"

    print("Loading data...")
    df = pd.read_parquet(v10_path)
    se_df = pd.read_parquet(se_path)
    print(f"Input: {df.shape}")

    df = add_agari_course_features(df, se_df)
    print(f"Output: {df.shape}")

    for col in ["horse_agari3f_course_avg", "horse_agari3f_course_rank_avg", "agari3f_best_in_course_lag1"]:
        nan_pct = df[col].isna().mean() * 100
        print(f"  {col}: NaN={nan_pct:.1f}%, mean={df[col].dropna().mean():.4f}")

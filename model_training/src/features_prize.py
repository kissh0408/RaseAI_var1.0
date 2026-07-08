"""
features_prize.py — 馬別累積賞金特徴量

生成特徴量:
    horse_cumulative_prize_shifted  : 当該レース前までの累積賞金（hon_shokin + fuka_shokin, 千円単位）
    horse_prize_log1p_shifted       : 上記の log1p 変換版（スケール正規化）

リーク防止:
    shift(1) + expanding().sum() で当該レースの賞金を除外した累積値を計算する。
    hon_shokin / fuka_shokin は出走成績（SE）にのみ存在するため、
    race_id + ketto_num で v10 にマージしてから cumsum を適用する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_prize_features(df: pd.DataFrame, se_df: pd.DataFrame) -> pd.DataFrame:
    """
    馬別累積賞金特徴量を df に追加して返す。

    Args:
        df  : features_past_v10 など（race_id, ketto_num, date 列を持つ DataFrame）
        se_df : SE_preprocessed（race_id, ketto_num, hon_shokin, fuka_shokin を持つ DataFrame）

    Returns:
        新特徴量 2 列を追加した DataFrame（行数・順序は変更しない）
    """
    # --- 既に計算済みなら早期リターン ---
    if "horse_cumulative_prize_shifted" in df.columns and "horse_prize_log1p_shifted" in df.columns:
        return df

    # --- SE から賞金列だけ取り出してマージ ---
    prize_cols = ["race_id", "ketto_num", "hon_shokin", "fuka_shokin"]
    missing = [c for c in prize_cols if c not in se_df.columns]
    if missing:
        raise ValueError(f"SE_preprocessed に必要な列が不足: {missing}")

    prize_src = se_df[prize_cols].drop_duplicates(subset=["race_id", "ketto_num"]).copy()
    prize_src["_total_prize"] = (
        pd.to_numeric(prize_src["hon_shokin"], errors="coerce").fillna(0)
        + pd.to_numeric(prize_src["fuka_shokin"], errors="coerce").fillna(0)
    ).astype("float32")

    # --- df に賞金列をマージ（左結合: df の全行を維持） ---
    df = df.merge(
        prize_src[["race_id", "ketto_num", "_total_prize"]],
        on=["race_id", "ketto_num"],
        how="left",
    )

    # --- 時系列ソート保証 ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- shift(1) + expanding().sum() で当該レースを除外した累積賞金 ---
    # 賞金未取得（NaN）は 0 として扱う（未着・出走取消などの場合）
    prize_filled = df["_total_prize"].fillna(0.0)

    df["horse_cumulative_prize_shifted"] = (
        prize_filled
        .groupby(df["ketto_num"], sort=False)
        .transform(lambda x: x.shift(1).fillna(0).expanding().sum())
    ).astype("float32")

    # --- log1p 変換版（大賞金馬と平均的な馬のスケール差を圧縮） ---
    df["horse_prize_log1p_shifted"] = np.log1p(
        df["horse_cumulative_prize_shifted"]
    ).astype("float32")

    # --- 作業列を削除 ---
    df = df.drop(columns=["_total_prize"])

    return df


if __name__ == "__main__":
    # スタンドアロン実行でのテスト用
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v10_path = project_root / "model_training/data/02_features/features_past_v10.parquet"
    se_path = project_root / "model_training/data/01_preprocessed/SE_preprocessed.parquet"

    print("Loading data...")
    df = pd.read_parquet(v10_path)
    se_df = pd.read_parquet(se_path)

    print(f"Input: {df.shape}")
    df = add_prize_features(df, se_df)
    print(f"Output: {df.shape}")

    for col in ["horse_cumulative_prize_shifted", "horse_prize_log1p_shifted"]:
        nan_pct = df[col].isna().mean() * 100
        print(f"  {col}: NaN={nan_pct:.1f}%, mean={df[col].mean():.1f}, max={df[col].max():.1f}")

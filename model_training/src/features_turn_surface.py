"""
features_turn_surface.py — 騎手・調教師の回り×馬場別勝率特徴量

生成特徴量:
    jockey_turn_surface_win_rate  : 騎手別 race_type_code 別ベイズ平滑化勝率
    trainer_turn_surface_win_rate : 調教師別 race_type_code 別ベイズ平滑化勝率

race_type_code の意味（JV-Link）:
    11 = 右回り芝
    12 = 左回り芝（東京・新潟・中京）
    13 = 右回りダート
    14 = 左回りダート
    18/19 = 障害（統計には含まれるが NaN は出ない、単純に実績が集計される）

リーク防止:
    cumsum-current パターンで当該行を除外した累積勝率を算出する。
    df を date / race_id / ketto_num でソートしてから計算する。

ベイズ平滑化パラメータ:
    prior_n    = 30.0  （騎手・調教師ともにサンプル数は多いため強め事前）
    prior_mean = 0.065 （全 race_type_code 平均勝率）
    min_periods= 15    （15走未満の場合 NaN を返す）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ベイズ平滑化パラメータ
_PRIOR_N = 30.0
_PRIOR_MEAN = 0.065
_MIN_PERIODS = 15


def add_turn_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    騎手・調教師の回り×馬場別ベイズ平滑化勝率を df に追加して返す。

    Args:
        df: jockey_code, trainer_code, race_type_code, finish_rank,
            date, race_id, ketto_num 列を持つ DataFrame

    Returns:
        jockey_turn_surface_win_rate / trainer_turn_surface_win_rate 2列を追加した DataFrame
        （行数・順序は変更しない）
    """
    # --- インデックス保存 → ソート ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 勝利フラグ ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    # --- race_type_code を文字列化（グループキー用） ---
    race_type_str = pd.to_numeric(df["race_type_code"], errors="coerce").astype("Int8").astype(str)

    # ------------------------------------------------------------------
    # 1. jockey_turn_surface_win_rate
    #    グループキー: jockey_code + "_" + race_type_code
    # ------------------------------------------------------------------
    if "jockey_code" in df.columns and "jockey_turn_surface_win_rate" not in df.columns:
        jockey_str = df["jockey_code"].astype(str)
        grp_key = (jockey_str + "_" + race_type_str).rename("_jockey_ts_key")

        # 当該行を除外した累積出走数・累積勝利数（cumsum-current）
        cum_runs = win_flag.groupby(grp_key, sort=False).transform("cumcount")
        # transform("cumcount") は当該行を含まないカウントを返す（= cumsum - current に等価）
        # 勝利フラグの cumsum-current
        cum_wins = win_flag.groupby(grp_key, sort=False).cumsum() - win_flag

        smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)
        df["jockey_turn_surface_win_rate"] = (
            smoothed.where(cum_runs >= _MIN_PERIODS, np.nan).astype("float32")
        )
    elif "jockey_turn_surface_win_rate" not in df.columns:
        df["jockey_turn_surface_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 2. trainer_turn_surface_win_rate
    #    グループキー: trainer_code + "_" + race_type_code
    # ------------------------------------------------------------------
    if "trainer_code" in df.columns and "trainer_turn_surface_win_rate" not in df.columns:
        trainer_str = df["trainer_code"].astype(str)
        grp_key = (trainer_str + "_" + race_type_str).rename("_trainer_ts_key")

        cum_runs = win_flag.groupby(grp_key, sort=False).transform("cumcount")
        cum_wins = win_flag.groupby(grp_key, sort=False).cumsum() - win_flag

        smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)
        df["trainer_turn_surface_win_rate"] = (
            smoothed.where(cum_runs >= _MIN_PERIODS, np.nan).astype("float32")
        )
    elif "trainer_turn_surface_win_rate" not in df.columns:
        df["trainer_turn_surface_win_rate"] = np.nan

    # --- 元の順序・インデックスに復元 ---
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v16_path = project_root / "model_training/data/02_features/features_past_v16.parquet"

    print("Loading data...")
    df = pd.read_parquet(v16_path)
    print(f"Input: {df.shape}")
    df = add_turn_surface_features(df)
    print(f"Output: {df.shape}")

    new_cols = ["jockey_turn_surface_win_rate", "trainer_turn_surface_win_rate"]
    for col in new_cols:
        nan_pct = df[col].isna().mean() * 100
        valid = df[col].dropna()
        print(
            f"  {col}: NaN={nan_pct:.1f}%, "
            f"mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

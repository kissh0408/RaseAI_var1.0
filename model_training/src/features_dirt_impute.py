"""
features_dirt_impute.py — ダート初出走フラグ・going_match_score_dirt 補完・走法別ダート勝率

生成特徴量:
    horse_first_dirt_flag          : horse_dirt_n_runs==0 の行で 1（ダート初出走マーカー）
    going_match_score_dirt_imputed : going_match_score_dirt の NaN を
                                     running_style × ダート勝率で補完した版
    running_style_dirt_win_rate    : running_style_code × ダート（race_type_code==2 or track_code>=50）
                                     の過去勝率（ベイズ平滑化、beta=15, prior=0.103）

リーク防止:
    running_style_dirt_win_rate は cumcount / cumsum で当該行を除外した累積から計算する。
    going_match_score_dirt_imputed の補完値はすでにリーク安全な列どうしの演算のみ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_dirt_impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ダート関連補完・走法別特徴量を df に追加して返す。

    Args:
        df : features_past_v10 など（horse_dirt_n_runs, going_match_score_dirt,
             running_style_code, race_type_code, track_code, finish_rank,
             date, ketto_num 列を持つ DataFrame）

    Returns:
        新特徴量 3 列を追加した DataFrame（行数・順序は変更しない）
    """
    # --- 時系列ソート保証 ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. horse_first_dirt_flag
    #    horse_dirt_n_runs == 0（ダート出走未経験）のフラグ
    # ------------------------------------------------------------------
    if "horse_first_dirt_flag" not in df.columns:
        if "horse_dirt_n_runs" in df.columns:
            dirt_runs = pd.to_numeric(df["horse_dirt_n_runs"], errors="coerce")
            df["horse_first_dirt_flag"] = (
                (dirt_runs == 0).astype("int8")
            )
        else:
            # horse_dirt_n_runs がない場合はダートレース当走かつ過去ダート実績なしを
            # horse_dirt_win_rate から推定する（代替: 全行 NaN）
            df["horse_first_dirt_flag"] = np.nan

    # ------------------------------------------------------------------
    # 2. running_style_dirt_win_rate
    #    running_style_code × ダート（track_code>=50 or race_type_code==2）の過去勝率
    #    ベイズ平滑化: beta=15, prior=0.103（ダート平均勝率）
    # ------------------------------------------------------------------
    if "running_style_dirt_win_rate" not in df.columns:
        if "running_style_code" in df.columns:
            _BETA_RSD = 15.0
            _PRIOR_RSD = 0.103  # ダート平均勝率（経験的事前値）

            # ダート判定: JV-Link track_code 20-29 がダートコース（23=右ダート, 24=左ダート）
            # 旧閾値 >=50 は砂特殊コース(52-57)のみ捕捉でデータの3%しか対象外だった
            if "track_code" in df.columns:
                track_num = pd.to_numeric(df["track_code"], errors="coerce")
                is_dirt = track_num.between(20, 29).fillna(False).astype("int8")
            elif "race_type_code" in df.columns:
                rt_num = pd.to_numeric(df["race_type_code"], errors="coerce")
                is_dirt = (rt_num == 2).fillna(False).astype("int8")
            else:
                is_dirt = pd.Series(0, index=df.index, dtype="int8")

            # 勝利フラグ × ダートフラグ
            finish = pd.to_numeric(df["finish_rank"], errors="coerce")
            win_flag = (finish == 1).astype("int8")
            dirt_win = (win_flag * is_dirt).astype("int8")

            # ランニングスタイル × ダート出走数・勝利数の累積（当該行除外）
            grp_col = df["running_style_code"].astype(str)
            cum_dirt_runs = (
                is_dirt.groupby(grp_col, sort=False).cumsum() - is_dirt
            )
            cum_dirt_wins = (
                dirt_win.groupby(grp_col, sort=False).cumsum() - dirt_win
            )

            df["running_style_dirt_win_rate"] = (
                (cum_dirt_wins + _BETA_RSD * _PRIOR_RSD)
                / (cum_dirt_runs + _BETA_RSD)
            ).where(cum_dirt_runs > 0, _PRIOR_RSD).astype("float32")
        else:
            df["running_style_dirt_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 3. going_match_score_dirt_imputed
    #    going_match_score_dirt の NaN を以下の優先順で補完:
    #      a) 芝レース（ダートでないレース）: NaN のまま維持（非ダートでは無意味）
    #      b) ダートレースの NaN: running_style_dirt_win_rate で線形補完
    #         （going_match_score_dirt の値域は概ね -3〜3 のため、
    #          勝率を (win_rate - prior_mean) / std_estimate * scale でスケーリング）
    # ------------------------------------------------------------------
    if "going_match_score_dirt_imputed" not in df.columns:
        if "going_match_score_dirt" in df.columns:
            gmd = pd.to_numeric(df["going_match_score_dirt"], errors="coerce").astype("float32")

            # ダート判定（上で計算済みの is_dirt を再利用）
            # JV-Link track_code 20-29 がダートコース（23=右ダート, 24=左ダート）
            if "track_code" in df.columns:
                track_num = pd.to_numeric(df["track_code"], errors="coerce")
                is_dirt_for_impute = track_num.between(20, 29).fillna(False)
            elif "race_type_code" in df.columns:
                rt_num = pd.to_numeric(df["race_type_code"], errors="coerce")
                is_dirt_for_impute = (rt_num == 2).fillna(False)
            else:
                is_dirt_for_impute = pd.Series(False, index=df.index)

            # 補完: ダートレースで NaN の場合のみ running_style_dirt_win_rate をスケールして代入
            # スケーリング: (x - 0.103) / 0.050 * 0.5
            # （win_rate の標準偏差 ~0.05 を going_match_score の単位 0.5 に変換する経験的スケール）
            if "running_style_dirt_win_rate" in df.columns:
                rs_rate = pd.to_numeric(df["running_style_dirt_win_rate"], errors="coerce")
                impute_val = ((rs_rate - 0.103) / 0.050 * 0.5).astype("float32")
            else:
                impute_val = pd.Series(np.nan, index=df.index, dtype="float32")

            is_nan_and_dirt = gmd.isna() & is_dirt_for_impute
            df["going_match_score_dirt_imputed"] = gmd.copy()
            df.loc[is_nan_and_dirt, "going_match_score_dirt_imputed"] = impute_val[is_nan_and_dirt]
            df["going_match_score_dirt_imputed"] = df["going_match_score_dirt_imputed"].astype("float32")
        else:
            df["going_match_score_dirt_imputed"] = np.nan

    return df


def recompute_going_match_score_dirt_imputed_scenario(df: pd.DataFrame) -> pd.Series:
    """what-if シナリオ後に going_match_score_dirt_imputed を学習式と同期する。"""
    if "going_match_score_dirt_imputed" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")

    if "going_match_score_dirt" not in df.columns:
        return pd.to_numeric(df["going_match_score_dirt_imputed"], errors="coerce").astype(
            "float32"
        )

    gmd = pd.to_numeric(df["going_match_score_dirt"], errors="coerce").astype("float32")

    if "track_code" in df.columns:
        track_num = pd.to_numeric(df["track_code"], errors="coerce")
        is_dirt = track_num >= 23
    elif "race_type_code" in df.columns:
        is_dirt = pd.to_numeric(df["race_type_code"], errors="coerce") == 2
    else:
        is_dirt = pd.Series(False, index=df.index)

    if "running_style_dirt_win_rate" in df.columns:
        rs_rate = pd.to_numeric(df["running_style_dirt_win_rate"], errors="coerce")
        impute_val = ((rs_rate - 0.103) / 0.050 * 0.5).astype("float32")
    else:
        impute_val = pd.Series(np.nan, index=df.index, dtype="float32")

    is_nan_and_dirt = gmd.isna() & is_dirt.fillna(False)
    out = gmd.copy()
    out.loc[is_nan_and_dirt] = impute_val[is_nan_and_dirt]
    return out.astype("float32")


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v10_path = project_root / "model_training/data/02_features/features_past_v10.parquet"

    print("Loading data...")
    df = pd.read_parquet(v10_path)
    print(f"Input: {df.shape}")

    df = add_dirt_impute_features(df)
    print(f"Output: {df.shape}")

    for col in ["horse_first_dirt_flag", "going_match_score_dirt_imputed", "running_style_dirt_win_rate"]:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            print(f"  {col}: NaN={nan_pct:.1f}%, mean={df[col].dropna().mean():.4f}")
        else:
            print(f"  {col}: NOT GENERATED")

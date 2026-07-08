"""RunningStyleBuilder: 脚質・コーナー通過順位 + レース内展開集計。

Per-horse特徴量（shift済み）:
  - corner_pos_mean   : 過去走の平均コーナー通過順位（4コーナー平均）
  - corner_pos_last   : 過去走の最終コーナー（4コーナー）平均通過順位
  - running_style_mode: 過去走での最頻脚質コード（1=逃, 2=先, 3=差, 4=追）
  - front_rate        : 過去走での「先行以内（1or2）」率

Race-level特徴量（同レース全馬の過去脚質を集計）:
  - race_front_count   : レース内の逃げ・先行馬数（前走脚質コード1or2の馬数）
  - race_front_ratio   : 同 / 出走頭数
  - race_style_entropy : 脚質分布のエントロピー（値が高いほど多様な展開）
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


class RunningStyleBuilder:
    """脚質・コーナー通過順位・レース内展開集計を担当する。"""

    # JV-Link の running_style_code 定義
    FRONT_STYLES = {1, 2}  # 1=逃げ, 2=先行

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._attach_corner_data(df)
        df = self._per_horse_features(df)
        df = self._race_level_features(df)
        return df

    # ------------------------------------------------------------------
    # コーナー通過順位・脚質コードをSEから取得して結合
    # ------------------------------------------------------------------

    def _attach_corner_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """SEテーブルからコーナー順位と脚質コードを取得する。

        テーブルに該当カラムがない場合はNaNのまま続行する。
        """
        try:
            corner_df = pd.read_sql_query(
                """SELECT race_id, horse_id,
                          corner_1, corner_2, corner_3, corner_4,
                          running_style_code
                   FROM SE
                   WHERE finish_rank > 0""",
                self.conn,
            )
            df = df.merge(corner_df, on=["race_id", "horse_id"], how="left")
        except Exception as e:
            print(f"  [WARN] コーナー/脚質データ取得失敗（SEテーブル確認要）: {e}")
            for col in ["corner_1", "corner_2", "corner_3", "corner_4", "running_style_code"]:
                df[col] = np.nan
        return df

    # ------------------------------------------------------------------
    # Per-horse特徴量（shift済み）
    # ------------------------------------------------------------------

    def _per_horse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["horse_id", "race_date"]).copy()

        # 4コーナーの平均通過順位（1走あたり）
        corner_cols = ["corner_1", "corner_2", "corner_3", "corner_4"]
        existing = [c for c in corner_cols if c in df.columns and df[c].notna().any()]

        if existing:
            df["corner_pos_avg_race"] = df[existing].mean(axis=1)
            df["corner_pos_mean"] = (
                df.groupby("horse_id")["corner_pos_avg_race"]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
            df["corner_pos_last"] = (
                df.groupby("horse_id")["corner_4" if "corner_4" in existing else existing[-1]]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
            df = df.drop(columns=["corner_pos_avg_race"], errors="ignore")
        else:
            df["corner_pos_mean"] = np.nan
            df["corner_pos_last"] = np.nan

        # 脚質コードの最頻値（最も多く使った戦法）
        if "running_style_code" in df.columns and df["running_style_code"].notna().any():
            df["running_style_mode"] = (
                df.groupby("horse_id")["running_style_code"]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1)
                           .apply(lambda w: pd.Series(w).mode().iloc[0] if len(w) > 0 else np.nan, raw=False))
            )
            # 先行率（過去走で先行以内だった割合）
            df["is_front"] = df["running_style_code"].isin(self.FRONT_STYLES).astype(float)
            df["front_rate"] = (
                df.groupby("horse_id")["is_front"]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
            df = df.drop(columns=["is_front"], errors="ignore")
        else:
            df["running_style_mode"] = np.nan
            df["front_rate"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Race-level特徴量（同レース全馬の過去脚質を集計）
    # ------------------------------------------------------------------

    def _race_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """同じレースに出走する馬全員の「前走脚質」を集計し、ペース展開を数値化する。

        front_rate（各馬の過去先行率）をレース内で集計する。
        これはshift済みのため未来情報を含まない。
        """
        df = df.sort_values(["race_id", "horse_num"]).copy()

        # レース内の逃げ・先行傾向馬数（front_rate > 0.5 を「先行傾向あり」と判定）
        df["is_front_tendency"] = (df["front_rate"] > 0.5).astype(float)

        race_agg = (
            df.groupby("race_id")
            .agg(
                race_front_count=("is_front_tendency", "sum"),
                race_front_ratio=("is_front_tendency", "mean"),
            )
            .reset_index()
        )
        df = df.merge(race_agg, on="race_id", how="left")
        df = df.drop(columns=["is_front_tendency"], errors="ignore")

        # 脚質分布のエントロピー（1=逃, 2=先, 3=差, 4=追 の比率から計算）
        def _style_entropy(group: pd.DataFrame) -> pd.Series:
            styles = group["running_style_mode"].dropna()
            if len(styles) < 3:
                return pd.Series(np.nan, index=group.index)
            counts = styles.value_counts(normalize=True).values
            ent = float(scipy_entropy(counts, base=2)) if len(counts) > 1 else 0.0
            return pd.Series(ent, index=group.index)

        if df["running_style_mode"].notna().any():
            df["race_style_entropy"] = df.groupby("race_id", group_keys=False).apply(_style_entropy)
        else:
            df["race_style_entropy"] = np.nan

        return df

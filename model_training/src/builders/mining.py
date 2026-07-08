"""MiningFeatureBuilder: JRA公式データマイニング予測タイム・タイム指数特徴量。

データソース:
  DM テーブル: JRAデータマイニング予測タイム（MSSCC秒）+ 誤差幅
  TM テーブル: JRA公式タイム指数スコア（高いほど能力が高い）

出力カラム:
  jra_tm_score       : JRA公式タイム指数スコア（生値）
  jra_tm_rank        : レース内ランク（1=最高スコア）
  jra_tm_implied_prob: Softmaxによる暗黙の勝率（市場プライアー近似）
  jra_tm_log_odds    : log(p/(1-p)) → base_margin として使用
  jra_dm_pred_time_s : JRA予測走破タイム（秒）
  jra_dm_rank        : 予測タイムのレース内ランク（1=最速予測）
  jra_dm_gap_to_best : 最速予測との差（秒）
  jra_dm_uncertainty : 予測誤差幅 (error_plus + error_minus) / 2
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


class MiningFeatureBuilder:
    """JRA公式マイニングデータ（DM/TM）から特徴量を生成する。

    これらは全てレース当日の事前予測データであるためデータリークは発生しない。
    jra_tm_log_odds はモデルのbase_marginとして使用することで残差学習を実現する。
    """

    # Softmax 温度（TM指数を確率に変換する際のシャープネス制御）
    TM_TEMPERATURE = 100.0  # TM指数の単位（0-1000程度）に合わせた温度

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._dm_df: pd.DataFrame | None = None
        self._tm_df: pd.DataFrame | None = None

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        self._load_tables()
        df = self._attach_tm(df)
        df = self._attach_dm(df)
        df = self._calc_race_level_features(df)
        return df

    # ------------------------------------------------------------------
    # テーブル読み込み
    # ------------------------------------------------------------------

    def _load_tables(self) -> None:
        try:
            self._tm_df = pd.read_sql_query(
                "SELECT race_id, horse_num, jra_tm_score FROM TM", self.conn
            )
        except Exception as e:
            print(f"  [WARN] TMテーブル読み込み失敗: {e}")
            self._tm_df = pd.DataFrame(columns=["race_id", "horse_num", "jra_tm_score"])

        try:
            self._dm_df = pd.read_sql_query(
                "SELECT race_id, horse_num, dm_pred_time_s, dm_error_plus_s, dm_error_minus_s FROM DM",
                self.conn,
            )
        except Exception as e:
            print(f"  [WARN] DMテーブル読み込み失敗: {e}")
            self._dm_df = pd.DataFrame(
                columns=["race_id", "horse_num", "dm_pred_time_s", "dm_error_plus_s", "dm_error_minus_s"]
            )

    # ------------------------------------------------------------------
    # TM特徴量の付与
    # ------------------------------------------------------------------

    def _attach_tm(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._tm_df is None or len(self._tm_df) == 0:
            df["jra_tm_score"] = np.nan
            return df

        # TM も重複排除してから merge
        tm_dedup = (
            self._tm_df
            .sort_values("jra_tm_score", ascending=False)
            .drop_duplicates(subset=["race_id", "horse_num"], keep="first")
        )
        df = df.merge(tm_dedup, on=["race_id", "horse_num"], how="left")
        return df

    # ------------------------------------------------------------------
    # DM特徴量の付与
    # ------------------------------------------------------------------

    def _attach_dm(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._dm_df is None or len(self._dm_df) == 0:
            df["jra_dm_pred_time_s"] = np.nan
            df["jra_dm_error_plus_s"] = np.nan
            df["jra_dm_error_minus_s"] = np.nan
            return df

        # DM は (race_id, horse_num) が重複している場合があるため先に dedup する
        dm_dedup = (
            self._dm_df
            .sort_values("dm_pred_time_s")
            .drop_duplicates(subset=["race_id", "horse_num"], keep="first")
            .rename(columns={
                "dm_pred_time_s": "jra_dm_pred_time_s",
                "dm_error_plus_s": "jra_dm_error_plus_s",
                "dm_error_minus_s": "jra_dm_error_minus_s",
            })
        )
        df = df.merge(dm_dedup, on=["race_id", "horse_num"], how="left")
        return df

    # ------------------------------------------------------------------
    # レース内正規化特徴量
    # ------------------------------------------------------------------

    def _calc_race_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # TM: レース内ランクと暗黙確率
        if "jra_tm_score" in df.columns and df["jra_tm_score"].notna().any():
            df["jra_tm_rank"] = df.groupby("race_id")["jra_tm_score"].rank(
                ascending=False, method="min"
            )
            # Softmaxで暗黙確率を計算（TM指数が高い馬ほど高確率）
            df["jra_tm_implied_prob"] = df.groupby("race_id")["jra_tm_score"].transform(
                self._softmax_within_race
            )
            # log-oddsに変換（base_margin 用）
            p = df["jra_tm_implied_prob"].clip(1e-6, 1 - 1e-6)
            df["jra_tm_log_odds"] = np.log(p / (1 - p))
        else:
            df["jra_tm_rank"] = np.nan
            df["jra_tm_implied_prob"] = np.nan
            df["jra_tm_log_odds"] = np.nan

        # DM: 予測タイムのレース内ランクとギャップ
        if "jra_dm_pred_time_s" in df.columns and df["jra_dm_pred_time_s"].notna().any():
            df["jra_dm_rank"] = df.groupby("race_id")["jra_dm_pred_time_s"].rank(
                ascending=True, method="min"  # タイムは低いほど良い
            )
            df["jra_dm_gap_to_best"] = df.groupby("race_id")["jra_dm_pred_time_s"].transform(
                lambda x: x - x.min()
            )
            df["jra_dm_uncertainty"] = (
                df.get("jra_dm_error_plus_s", pd.Series(np.nan, index=df.index)).fillna(0)
                + df.get("jra_dm_error_minus_s", pd.Series(np.nan, index=df.index)).fillna(0)
            ) / 2
        else:
            df["jra_dm_rank"] = np.nan
            df["jra_dm_gap_to_best"] = np.nan
            df["jra_dm_uncertainty"] = np.nan

        # 中間カラムの削除
        df = df.drop(
            columns=["jra_dm_error_plus_s", "jra_dm_error_minus_s"], errors="ignore"
        )
        return df

    def _softmax_within_race(self, scores: pd.Series) -> pd.Series:
        """レース内で TM スコアを Softmax 確率に変換する。"""
        s = scores.fillna(scores.mean() if scores.notna().any() else 0)
        s_shifted = s - s.max()
        exp_s = np.exp(s_shifted / self.TM_TEMPERATURE)
        total = exp_s.sum()
        if total == 0:
            return pd.Series(1.0 / len(scores), index=scores.index)
        return exp_s / total

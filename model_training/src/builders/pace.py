"""PaceFeatureBuilder: PCI（ペース特性指数）・真Zスコア。

PCI (Pace Character Index):
  前半ペース（先頭通過タイム）と後半ペース（上がり3F）の相対差。
  高い値 = ハイペース = スタミナ消耗型 = 追込み有利
  低い値 = スローペース = 瞬発力勝負 = 先行有利

真Zスコア（ペース順応指数）:
  同じペース環境で他馬と比較したときの相対的な上がり速度。
  ハイペースでの速い上がりとスローでの速い上がりを同等に評価する。

出力カラム:
  pci_past_mean    : 過去走のPCI平均（この馬がどんなペースで走ってきたか）
  agari_z_score    : 過去走の真Zスコア平均（ペース補正後の末脚評価）
  race_pci         : 今回レースのPCI予測値（出走馬の過去PCI平均）
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


class PaceFeatureBuilder:
    """PCI・真Zスコアを担当する。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._attach_lap_data(df)
        df = self._calc_pci(df)
        df = self._per_horse_pace_features(df)
        df = self._race_level_pace(df)
        return df

    # ------------------------------------------------------------------
    # ラップタイム取得
    # ------------------------------------------------------------------

    def _attach_lap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """RAテーブルからラップタイムを取得する。

        JV-Linkのラップは前半ペース算出のために使用する。
        取得不可の場合は推定値（finish_time - agari3f）で代替する。
        """
        try:
            lap_df = pd.read_sql_query(
                "SELECT race_id, lap_times FROM RA WHERE lap_times IS NOT NULL",
                self.conn,
            )
            df = df.merge(lap_df, on="race_id", how="left")
        except Exception as e:
            print(f"  [WARN] ラップタイム取得失敗: {e}")
            df["lap_times"] = np.nan

        # 前半タイム推定: finish_time - agari3f（3ハロン換算）
        # 完全なラップがない場合のフォールバック
        df["first_half_time"] = np.where(
            df["lap_times"].notna(),
            df["lap_times"].apply(self._parse_first_half),
            df["finish_time"] - df["agari3f"],
        )
        return df

    @staticmethod
    def _parse_first_half(lap_str) -> float:
        """ラップタイム文字列から前半タイム（後半3ハロン以外の合計）を返す。

        JV-Link形式: "12.3 11.5 12.0 ..." のようなスペース区切り文字列を想定。
        前半 = 全体 - 後半3ハロン分
        """
        if not isinstance(lap_str, str) or not lap_str.strip():
            return np.nan
        try:
            laps = [float(x) for x in lap_str.strip().split()]
            if len(laps) < 4:
                return np.nan
            total = sum(laps)
            last_3f = sum(laps[-3:]) if len(laps) >= 3 else laps[-1]
            return total - last_3f
        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # PCI 計算
    # ------------------------------------------------------------------

    def _calc_pci(self, df: pd.DataFrame) -> pd.DataFrame:
        """PCI = (前半ペース - 後半ペース) / 全体タイム × 100

        前半ペース = first_half_time / (distance - 600) * 600  ← 600m換算
        後半ペース = agari3f（600m）
        """
        front_dist = df["distance"] - 600  # 後半600m（3ハロン）を除いた距離
        # 前半タイムを600m換算に正規化
        front_normalized = np.where(
            front_dist > 0,
            df["first_half_time"] / front_dist * 600,
            np.nan,
        )
        rear_normalized = df["agari3f"]  # すでに600m（3ハロン）

        total_time = df["finish_time"]
        df["pci"] = np.where(
            total_time.notna() & (total_time > 0),
            (front_normalized - rear_normalized) / total_time * 100,
            np.nan,
        )
        return df

    # ------------------------------------------------------------------
    # Per-horse 過去走集計（shift済み）
    # ------------------------------------------------------------------

    @staticmethod
    def _race_agari_z_series(df: pd.DataFrame) -> pd.Series:
        """レース内上がり3F Zスコア（確定値）。特徴量化は shift(1) 経由のみ。"""
        grp = df.groupby("race_id")["agari3f"]
        mu = grp.transform("mean")
        sigma = grp.transform("std").replace(0.0, np.nan)
        return (mu - df["agari3f"]) / sigma

    def _per_horse_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """過去走のPCI・Zスコアの平均（shift(1)でリーク防止）。"""
        df = df.sort_values(["horse_id", "race_date"]).copy()
        grp = df.groupby("horse_id")

        df["pci_past_mean"] = grp["pci"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        # agari_z_race 列は df に載せず、過去走集計のみ出力する（DA-1）
        z_race = self._race_agari_z_series(df)
        df["agari_z_score"] = (
            z_race.groupby(df["horse_id"])
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        df = df.drop(columns=["pci", "first_half_time", "lap_times"], errors="ignore")
        return df

    # ------------------------------------------------------------------
    # Race-level ペース予測
    # ------------------------------------------------------------------

    def _race_level_pace(self, df: pd.DataFrame) -> pd.DataFrame:
        """同レース出走馬の過去PCI平均 → 今回のペース予測指標。

        高い値 = 逃げ先行馬が多くハイペース傾向
        """
        # mergeではなくtransformでインデックスを保ったままブロードキャストする
        df["race_pci"] = df.groupby("race_id")["pci_past_mean"].transform("mean")
        return df

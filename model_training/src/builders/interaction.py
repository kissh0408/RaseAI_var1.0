"""InteractionFeatureBuilder: 交互作用特徴量。

枠順×馬場面・直線コース補正・夏牝馬サイクルを担当する。

出力カラム:
  gate_surface_cross  : 枠番 × 馬場面（芝=-1, ダート=1）の交互作用
                        内枠有利/不利が馬場面で異なることを捉える
  gate_straight_cross : 直線コースでの枠番の有利不利
                        新潟1000m芝など内枠不利な直線コースを識別
  summer_mare_sin     : 夏牝馬の季節周期（sin成分）
  summer_mare_cos     : 夏牝馬の季節周期（cos成分）
                        牝馬(sex_code=2)の夏(7-8月)のパフォーマンス向上を捉える
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd


# 直線コース: course_code + distance + surface_code の組み合わせ
# 新潟1000m芝（course_code=10, distance=1000, surface_code=1）など
STRAIGHT_COURSES: frozenset[tuple[int, int, int]] = frozenset(
    [
        (10, 1000, 1),   # 新潟1000m芝
        (9, 1000, 1),    # 小倉1000m芝（直線部分を含む）
    ]
)

# JV-Link sex_code: 1=牡, 2=牝, 3=セン
# 馬場面 surface_code: 1=芝, 2=ダート
SURFACE_MAP = {1: -1.0, 2: 1.0}


class InteractionFeatureBuilder:
    """交互作用特徴量を担当する。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._gate_surface_cross(df)
        df = self._gate_straight_cross(df)
        df = self._summer_mare(df)
        return df

    # ------------------------------------------------------------------
    # 枠番 × 馬場面
    # ------------------------------------------------------------------

    def _gate_surface_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        """枠番と馬場面の交互作用項。

        芝=-1 / ダート=1 にエンコードし、枠番と掛け合わせる。
        内枠(小さい値)×芝 → 負 = 芝内枠有利を符号で表す。
        """
        surface_num = df["surface_code"].map(SURFACE_MAP).fillna(0.0)
        gate = pd.to_numeric(df["gate_number"], errors="coerce").fillna(0.0)
        df["gate_surface_cross"] = gate * surface_num
        return df

    # ------------------------------------------------------------------
    # 直線コース 枠番
    # ------------------------------------------------------------------

    def _gate_straight_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        """直線コースの場合のみ枠番を特徴量として付与する。

        直線コース以外は NaN（モデルは欠損として扱う）。
        直線コースでは大外枠が有利なため、枠番の正の値が効く。
        """
        course_key = list(zip(
            df["course_code"].astype(int, errors="ignore"),
            df["distance"].astype(int, errors="ignore"),
            df["surface_code"].astype(int, errors="ignore"),
        ))
        is_straight = pd.Series(
            [tuple(k) in STRAIGHT_COURSES for k in course_key],
            index=df.index,
        )
        gate = pd.to_numeric(df["gate_number"], errors="coerce")
        df["gate_straight_cross"] = gate.where(is_straight, other=np.nan)
        return df

    # ------------------------------------------------------------------
    # 夏牝馬サイクル（sin/cos）
    # ------------------------------------------------------------------

    def _summer_mare(self, df: pd.DataFrame) -> pd.DataFrame:
        """牝馬の夏季パフォーマンス向上を周期的特徴量で表現する。

        sex_multiplier: 牝馬=1, それ以外=0
        season_signal: cos(2π × day_of_year / 365) + 1  [0 ~ 2、夏に最大]
          → cos(0) = -1（1月1日）、cos(π) = -1（7月2日付近）→ 夏に信号が最大

        注: cos のみだと正弦波の位相が曖昧なため sin/cos の両成分を出力する。
        """
        race_date = pd.to_datetime(df["race_date"], errors="coerce")
        day_of_year = race_date.dt.dayofyear.fillna(1).astype(float)
        angle = 2 * np.pi * day_of_year / 365.25

        # 夏(7-8月)に高い値: -cos でシフト (夏=最大値)
        summer_sin = np.sin(angle)       # 春〜夏に正
        summer_cos = -np.cos(angle) + 1  # 夏に最大（0〜2）

        # 牝馬フラグ（sex_code=2 のみ1、その他0）
        is_mare = (pd.to_numeric(df.get("horse_sex_code", pd.Series(0, index=df.index)), errors="coerce") == 2).astype(float)

        df["summer_mare_sin"] = is_mare * summer_sin
        df["summer_mare_cos"] = is_mare * summer_cos
        return df

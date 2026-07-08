"""PastPerformanceBuilder: 過去走特徴量 + EMA時間減衰 + RPRクラス調整。

新規追加:
  - ema_rank / ema_time_diff : 指数平滑移動平均で直近成績をより重視
  - rpr_score               : クラス基準タイム偏差（RPR相当）
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from pipeline_common import load_pastfeatures_config


class PastPerformanceBuilder:
    """勝率・上がり3F・騎手・血統・EMA・RPRを担当する。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.pcfg = load_pastfeatures_config()

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        nan = self.pcfg["nan_fill"]
        df = df.sort_values(["horse_id", "race_date"]).copy()
        df["is_win"] = (df["finish_rank"] == 1).astype(int)

        df = self._performance(df)
        df = self._condition_win_rates(df)
        df = self._jockey_trainer(df)
        df = self._pedigree(df)
        df = self._tm_index(df)
        df = self._ema_features(df)
        df = self._rpr_score(df)

        # NaN埋め
        for col in [
            "last3_agari3f_mean", "last5_rank_mean", "last5_time_diff_mean",
            "last5_rank_std", "top3_rate_career", "top3_rate_class",
            "career_win_rate", "distance_win_rate", "course_win_rate",
            "surface_win_rate", "condition_win_rate",
            "jockey_30d_win_rate", "trainer_30d_win_rate", "jockey_course_win_rate",
            "sire_surface_win_rate", "bms_win_rate",
            "ema_rank", "ema_time_diff", "rpr_score",
        ]:
            if col in df.columns:
                fill = nan.get("agari3f" if "agari3f" in col else "win_rate", 0.0)
                df[col] = df[col].fillna(fill)

        return df

    # ------------------------------------------------------------------
    # 直近N走平均（shift済み）
    # ------------------------------------------------------------------

    def _performance(self, df: pd.DataFrame) -> pd.DataFrame:
        w5, w3 = self.pcfg["past_window"], self.pcfg["agari3f_window"]
        grp = df.groupby("horse_id")
        df["last3_agari3f_mean"] = grp["agari3f"].transform(
            lambda x: x.shift(1).rolling(w3, min_periods=1).mean()
        )
        df["last5_rank_mean"] = grp["finish_rank"].transform(
            lambda x: x.shift(1).rolling(w5, min_periods=1).mean()
        )
        df["last5_time_diff_mean"] = grp["time_diff"].transform(
            lambda x: x.shift(1).rolling(w5, min_periods=1).mean()
        )
        df["last5_rank_std"] = grp["finish_rank"].transform(
            lambda x: x.shift(1).rolling(w5, min_periods=1).std()
        )
        df["is_top3"] = (df["finish_rank"] <= 3).astype(int)
        df["top3_rate_career"] = grp["is_top3"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df = df.drop(columns=["is_top3"], errors="ignore")
        df["career_win_rate"] = grp["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        return df

    # ------------------------------------------------------------------
    # 条件別勝率（shift済み）
    # ------------------------------------------------------------------

    def _condition_win_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["distance_band"] = (df["distance"] // 200) * 200

        for col, keys in [
            ("distance_win_rate", ["horse_id", "distance_band"]),
            ("course_win_rate",   ["horse_id", "course_code"]),
            ("surface_win_rate",  ["horse_id", "surface_code"]),
            ("condition_win_rate",["horse_id", "track_condition_code"]),
            ("top3_rate_class",   ["horse_id", "grade_code"]),
        ]:
            if col == "top3_rate_class":
                df[col] = (
                    df.assign(_t3=(df["finish_rank"] <= 3).astype(int))
                    .sort_values("race_date")
                    .groupby(keys, group_keys=False)["_t3"]
                    .transform(lambda x: x.shift(1).expanding().mean())
                )
            else:
                df[col] = (
                    df.sort_values("race_date")
                    .groupby(keys, group_keys=False)["is_win"]
                    .transform(lambda x: x.shift(1).expanding().mean())
                )
        return df

    # ------------------------------------------------------------------
    # 騎手・調教師（shift + 30日rolling）
    # ------------------------------------------------------------------

    def _jockey_trainer(self, df: pd.DataFrame) -> pd.DataFrame:
        # 日数ウィンドウの代わりにレース件数ウィンドウを使用（30日≈20件）
        # pandas の時系列ローリングはインデックス整合が複雑なためカウント方式で近似
        w_races = 20
        df = df.sort_values("race_date").copy()

        for col, id_col in [
            ("jockey_30d_win_rate", "jockey_id"),
            ("trainer_30d_win_rate", "trainer_id"),
        ]:
            df[col] = df.groupby(id_col)["is_win"].transform(
                lambda x: x.shift(1).rolling(w_races, min_periods=1).mean()
            )

        df["jockey_course_win_rate"] = (
            df.sort_values("race_date")
            .groupby(["jockey_id", "course_code"], group_keys=False)["is_win"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        return df

    # ------------------------------------------------------------------
    # 血統（PEDテーブル）
    # ------------------------------------------------------------------

    def _pedigree(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            ped = pd.read_sql_query("SELECT horse_id, sire_id, bms_id FROM PED", self.conn)
            df = df.merge(ped, on="horse_id", how="left")
        except Exception as e:
            print(f"  [WARN] PEDテーブル取得失敗: {e}")
            df["sire_id"] = np.nan
            df["bms_id"] = np.nan
            df["sire_surface_win_rate"] = np.nan
            df["bms_win_rate"] = np.nan
            return df

        if df["sire_id"].notna().any():
            df["sire_surface_win_rate"] = (
                df.sort_values("race_date")
                .groupby(["sire_id", "surface_code"], group_keys=False)["is_win"]
                .transform(lambda x: x.shift(1).expanding().mean())
            )
        else:
            df["sire_surface_win_rate"] = np.nan

        if df["bms_id"].notna().any():
            df["bms_win_rate"] = (
                df.sort_values("race_date")
                .groupby("bms_id", group_keys=False)["is_win"]
                .transform(lambda x: x.shift(1).expanding().mean())
            )
        else:
            df["bms_win_rate"] = np.nan

        return df

    # ------------------------------------------------------------------
    # タイム指数（TMテーブル）
    # ------------------------------------------------------------------

    def _tm_index(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            tm = pd.read_sql_query(
                "SELECT race_id, horse_id, tm_index FROM TM WHERE tm_index IS NOT NULL", self.conn
            )
            df = df.merge(tm, on=["race_id", "horse_id"], how="left")
        except Exception:
            df["tm_index"] = np.nan
        return df

    # ------------------------------------------------------------------
    # EMA 時間減衰加重平均（新規）
    # ------------------------------------------------------------------

    def _ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """直近の成績を重視するEMA特徴量。shift(1)でリーク防止。

        EMA(t) = α × x(t-1) + (1-α) × EMA(t-2)
        α=0.4: 直近1走に40%の重みを置く。
        """
        alpha = 0.4

        def _ema(series: pd.Series) -> pd.Series:
            # shift(1) して EMA を計算（当該レースを除外）
            shifted = series.shift(1)
            return shifted.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

        grp = df.sort_values("race_date").groupby("horse_id")
        df["ema_rank"] = grp["finish_rank"].transform(_ema)
        df["ema_time_diff"] = grp["time_diff"].transform(_ema)
        return df

    # ------------------------------------------------------------------
    # RPR相当: クラス基準タイム偏差（新規）
    # ------------------------------------------------------------------

    def _rpr_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """クラス×距離×馬場の基準タイムからの偏差をZスコア化。

        下級条件での大差勝ちと上級条件での惜敗を適切に比較できる。
        クラス統計は expanding + shift(1) で当該レース以前のみ使用（DA-2）。
        """
        df = df.sort_values("race_date").copy()
        grp_key = ["grade_code", "distance", "surface_code", "track_condition_code"]

        def _class_rpr_raw(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("race_date")
            ft = g["finish_time"]
            class_mean = ft.shift(1).expanding().mean()
            class_std = ft.shift(1).expanding().std()
            g = g.copy()
            g["_rpr_raw"] = np.where(
                class_std.notna() & (class_std > 0),
                (class_mean - ft) / class_std,
                np.nan,
            )
            return g

        parts = [_class_rpr_raw(g) for _, g in df.groupby(grp_key, sort=False)]
        df = pd.concat(parts).sort_index()

        df["rpr_score"] = (
            df.sort_values(["horse_id", "race_date"])
            .groupby("horse_id")["_rpr_raw"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        return df.drop(columns=["_rpr_raw"], errors="ignore")

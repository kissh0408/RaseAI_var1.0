"""BasicFeatureBuilder: スピード指数・枠順バイアス・レース基本情報。

create_features.py から移植 + FeatureCreator Composer パターンに対応。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from pipeline_common import load_config, load_pastfeatures_config, make_target


class BasicFeatureBuilder:
    """ベーステーブル生成と基本特徴量を担当する。

    他のビルダーが前提とするカラム（race_id, horse_id, race_date, finish_rank,
    target, surface_code, distance 等）はすべてここで生成する。
    """

    def __init__(self, conn: sqlite3.Connection, include_pending: bool = False) -> None:
        """include_pending=True で未確定レース（finish_rank=0、当日出馬表）も含める。

        学習時は False（確定レースのみ）。本番の当日特徴量生成時のみ True にする。
        当日行は時系列の最後尾のため、shift/expanding ベースの過去特徴量は
        履歴のみから計算され、リークは発生しない。
        """
        self.conn = conn
        self.include_pending = include_pending
        self.cfg = load_config()
        self.pcfg = load_pastfeatures_config()

    def _rank_filter(self) -> str:
        return "se.finish_rank >= 0" if self.include_pending else "se.finish_rank > 0"

    # ------------------------------------------------------------------
    # ベーステーブル取得
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """ベーステーブルを生成して基本特徴量を付与し返す。"""
        t_cfg = self.cfg["training"]

        df = self._fetch_base()
        df = df[~df["abnormal_code"].isin(t_cfg["exclude_abnormal_codes"])].copy()
        df = df[~df["grade_code"].isin(t_cfg["exclude_grade_codes"])].copy()
        df = df[df["horse_count"] >= t_cfg["min_horse_count"]].copy()

        df = self._attach_horse_meta(df)
        df["target"] = make_target(df["finish_rank"])
        df["speed_index"] = self._calc_speed_index(df)
        df["draw_bias_score"] = self._calc_draw_bias(df)
        df = self._calc_weight_features(df)
        df["days_since_last_race"] = self._calc_days_since_last(df)
        df["class_change"] = self._calc_class_change(df)

        # speed_index は現在レースのfinish_timeから算出するためリーク源になる。
        # 特徴量として使う場合は過去走のshift済み平均のみ使用する。
        df = self._calc_past_speed_index(df)

        return df.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composer インタフェース（BasicBuilder は build() のみ）。"""
        return df

    # ------------------------------------------------------------------
    # 内部実装
    # ------------------------------------------------------------------

    def _fetch_base(self) -> pd.DataFrame:
        query = """
            SELECT
                se.race_id, se.horse_id, se.finish_rank, se.abnormal_code,
                se.horse_num, se.gate_num AS gate_number, se.carry_weight,
                se.finish_time, se.agari3f, se.time_diff,
                se.horse_weight, se.horse_weight_diff,
                se.jockey_id, se.trainer_id,
                se.odds,
                ra.race_date, ra.course_code, ra.race_num, ra.distance,
                ra.surface_code, ra.track_condition_code, ra.grade_code,
                ra.horse_count, ra.base_time, ra.standard_weight
            FROM SE se
            JOIN RA ra ON se.race_id = ra.race_id
            WHERE {rank_filter}
        """.format(rank_filter=self._rank_filter())
        df = pd.read_sql_query(query, self.conn, parse_dates=["race_date"])
        # 単勝オッズ → 市場確率（raw = 1/odds）
        raw_market = (1.0 / df["odds"].clip(lower=1.01)).where(df["odds"] > 0)
        df["market_prob"] = raw_market
        # レース内正規化済み市場確率（オーバーラウンド補正）
        df["market_prob_norm"] = df.groupby("race_id")["market_prob"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )
        # 市場 log-odds（base_margin として使用可能）
        p = df["market_prob_norm"].clip(1e-6, 1 - 1e-6)
        df["market_log_odds"] = np.log(p / (1 - p))
        return df.sort_values(["race_date", "race_id", "horse_num"]).reset_index(drop=True)

    def _attach_horse_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬の年齢・性別を付与する。取得失敗時はNaNで続行。"""
        try:
            age_df = pd.read_sql_query(
                """SELECT se.race_id, se.horse_id,
                          (julianday(ra.race_date) - julianday(um.birth_date)) / 365.25 AS horse_age,
                          um.sex_code AS horse_sex_code
                   FROM SE se
                   JOIN RA ra ON se.race_id = ra.race_id
                   JOIN UM um ON se.horse_id = um.horse_id
                   WHERE {rank_filter}""".format(rank_filter=self._rank_filter()),
                self.conn,
            )
            df = df.merge(age_df, on=["race_id", "horse_id"], how="left")
        except Exception as e:
            print(f"  [WARN] 馬メタ取得失敗（UMテーブル確認要）: {e}")
            df["horse_age"] = np.nan
            df["horse_sex_code"] = 0
        return df

    def _calc_speed_index(self, df: pd.DataFrame) -> pd.Series:
        si_cfg = self.pcfg["speed_index"]
        base_const = si_cfg["base_constant"]
        weight_corr = si_cfg["weight_correction_per_kg"]
        std_weight = si_cfg["standard_weight_kg"]

        dist_coefs = {int(k): v for k, v in si_cfg["distance_coefficients"].items()}

        def _nearest_coef(d: int) -> float:
            return dist_coefs[min(dist_coefs, key=lambda x: abs(x - d))]

        # 距離のユニーク値（数十種）だけ最近傍探索し、mapで全行に展開する
        coef_map = {d: _nearest_coef(d) for d in df["distance"].dropna().unique()}
        dist_coef = df["distance"].map(coef_map)
        cond_adj = pd.Series(si_cfg["track_condition_adjustments"])
        track_adj = df["track_condition_code"].astype(str).map(cond_adj).fillna(0)
        weight_adj = (df["carry_weight"].fillna(std_weight) - std_weight) * weight_corr
        time_diff = df["base_time"] - df["finish_time"]

        si = time_diff * dist_coef + track_adj + weight_adj + base_const
        si[df["finish_time"].isnull()] = np.nan
        return si

    def _calc_draw_bias(self, df: pd.DataFrame) -> pd.Series:
        min_samples = self.pcfg["draw_bias"]["min_samples_for_bias"]
        alpha = self.pcfg["draw_bias"]["smoothing_factor"]

        df = df.sort_values("race_date").copy()
        df["in_top3"] = (df["finish_rank"] <= 3).astype(int)
        df["gate_block"] = pd.cut(
            df["gate_number"], bins=[0, 4, 8, 12, 18], labels=[1, 2, 3, 4]
        ).astype("Int64")

        global_mean = 1 / 3

        def _expanding_bias(g: pd.DataFrame) -> pd.Series:
            shifted = g["in_top3"].shift(1)
            n = shifted.expanding().count()
            cum_mean = shifted.expanding().mean()
            return (cum_mean * n + global_mean * alpha * min_samples) / (n + alpha * min_samples)

        return df.groupby(["course_code", "distance", "gate_block"], group_keys=False).apply(
            _expanding_bias
        ).rename("draw_bias_score")

    def _calc_weight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["horse_id", "race_date"]).copy()
        grp = df.groupby("horse_id")["horse_weight"]
        df["weight_diff"] = grp.diff()
        # 直近3走平均（shift済み）と3走前体重の差 = 体重トレンド
        shifted = grp.shift(1)
        rolling_mean = (
            shifted.groupby(df["horse_id"])
            .rolling(3, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["weight_diff_trend"] = rolling_mean - grp.shift(3)
        return df

    def _calc_days_since_last(self, df: pd.DataFrame) -> pd.Series:
        fill = self.pcfg["nan_fill"]["days_since_last_race"]
        return (
            df.sort_values(["horse_id", "race_date"])
            .groupby("horse_id")["race_date"]
            .diff()
            .dt.days
            .fillna(fill)
        )

    def _calc_class_change(self, df: pd.DataFrame) -> pd.Series:
        diff = (
            df.sort_values(["horse_id", "race_date"])
            .groupby("horse_id")["grade_code"]
            .diff()
        )
        # 昇級=1 / 降級=-1 / 同一・初走=0
        return np.sign(diff).fillna(0).astype(int)

    def _calc_past_speed_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """過去走のスピード指数を集計する（shift(1)でリーク防止）。

        生のspeed_indexは現在レースのfinish_timeを使うためリーク源になる。
        このメソッドでshift(1)済みの過去5走平均を生成し、rawは特徴量から除外する。
        """
        df = df.sort_values(["horse_id", "race_date"]).copy()
        shifted = df.groupby("horse_id")["speed_index"].shift(1)
        df["past_speed_index_mean"] = (
            shifted.groupby(df["horse_id"])
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # 当該レース finish_time 由来の生指数は学習特徴量に残さない（DA-4）
        return df.drop(columns=["speed_index"], errors="ignore")

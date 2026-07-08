"""
features_rank1_context_v24.py — Rank1 向けコンテキスト特徴量（v24）

多頭数トラフィック、斤量/ハンデ、馬場悪化適性、レース内混戦度を
リーク防止（cumsum - current / レース前情報のみ）で追加する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_training.src.features_common import _bayesian_cumulative_rate

_NEW_COLS = [
    "traffic_risk_score",
    "horse_large_field_win_rate",
    "horse_large_field_top3_rate",
    "burden_vs_race_mean",
    "burden_vs_race_z",
    "horse_handicap_win_rate",
    "horse_worse_going_top3_rate",
    "race_tm_score_std",
    "horse_tm_score_pct_in_race",
    "race_mining_rank_std",
    "field_competition_score",
]

_LARGE_FIELD_MIN = 16
_HANDICAP_WEIGHT_TYPE = 1
_CLOSER_STYLES = {3, 4}


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols).reset_index(drop=True)


def _running_style_series(df: pd.DataFrame) -> pd.Series:
    if "horse_modal_running_style" in df.columns:
        return pd.to_numeric(df["horse_modal_running_style"], errors="coerce").fillna(0)
    if "running_style_code" in df.columns:
        return pd.to_numeric(df["running_style_code"], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype=float)


def _add_traffic_and_large_field(df: pd.DataFrame) -> pd.DataFrame:
    style = _running_style_series(df)
    is_closer = style.isin(_CLOSER_STYLES).astype(float)
    closer_w = style.map({3: 0.55, 4: 1.0}).fillna(0.0).astype(float)

    n_h = pd.to_numeric(df.get("n_horses"), errors="coerce")
    field_scale = (n_h / 18.0).clip(0.0, 1.2)

    corner_back = pd.to_numeric(df.get("corner4_normalized_lag1"), errors="coerce").fillna(0.5)
    corner_back = corner_back.clip(0.0, 1.0)

    df["traffic_risk_score"] = (field_scale * closer_w * corner_back).astype("float32")

    finish = pd.to_numeric(df.get("finish_rank"), errors="coerce")
    win_flag = (finish == 1).astype("int8")
    top3_flag = ((finish >= 1) & (finish <= 3)).astype("int8")
    horse_key = df["ketto_num"].astype(str)

    if "n_horses" in df.columns:
        large_mask = (n_h >= _LARGE_FIELD_MIN).astype("int8")
        df["horse_large_field_win_rate"] = _bayesian_cumulative_rate(
            horse_key,
            win_flag * large_mask,
            large_mask,
            prior_n=8.0,
            prior_mean=0.08,
            min_periods=2,
        )
        df["horse_large_field_top3_rate"] = _bayesian_cumulative_rate(
            horse_key,
            top3_flag * large_mask,
            large_mask,
            prior_n=8.0,
            prior_mean=0.22,
            min_periods=2,
        )
    else:
        df["horse_large_field_win_rate"] = np.nan
        df["horse_large_field_top3_rate"] = np.nan

    return df


def _add_burden_context(df: pd.DataFrame) -> pd.DataFrame:
    if "burden_weight" not in df.columns or "race_id" not in df.columns:
        df["burden_vs_race_mean"] = np.nan
        df["burden_vs_race_z"] = np.nan
        return df

    bw = pd.to_numeric(df["burden_weight"], errors="coerce")
    race_mean = bw.groupby(df["race_id"].astype(str), sort=False).transform("mean")
    race_std = bw.groupby(df["race_id"].astype(str), sort=False).transform("std")
    df["burden_vs_race_mean"] = (bw - race_mean).astype("float32")
    df["burden_vs_race_z"] = (
        (bw - race_mean) / race_std.replace(0, np.nan)
    ).astype("float32")
    return df


def _add_handicap_history(df: pd.DataFrame) -> pd.DataFrame:
    if "weight_type" not in df.columns or "finish_rank" not in df.columns:
        df["horse_handicap_win_rate"] = np.nan
        return df

    wt = pd.to_numeric(df["weight_type"], errors="coerce")
    hc_mask = (wt == _HANDICAP_WEIGHT_TYPE).astype("int8")
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    horse_key = df["ketto_num"].astype(str)

    df["horse_handicap_win_rate"] = _bayesian_cumulative_rate(
        horse_key,
        win_flag * hc_mask,
        hc_mask,
        prior_n=10.0,
        prior_mean=0.07,
        min_periods=2,
    )
    return df


def _add_worse_going_history(df: pd.DataFrame) -> pd.DataFrame:
    if "going_change_lag1" not in df.columns or "finish_rank" not in df.columns:
        df["horse_worse_going_top3_rate"] = np.nan
        return df

    going_chg = pd.to_numeric(df["going_change_lag1"], errors="coerce")
    worse_mask = (going_chg > 0).astype("int8")
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    top3_flag = ((finish >= 1) & (finish <= 3)).astype("int8")
    horse_key = df["ketto_num"].astype(str)

    df["horse_worse_going_top3_rate"] = _bayesian_cumulative_rate(
        horse_key,
        top3_flag * worse_mask,
        worse_mask,
        prior_n=8.0,
        prior_mean=0.20,
        min_periods=2,
    )
    return df


def _add_field_competition(df: pd.DataFrame) -> pd.DataFrame:
    if "race_id" not in df.columns:
        for c in ("race_tm_score_std", "horse_tm_score_pct_in_race", "race_mining_rank_std", "field_competition_score"):
            df[c] = np.nan
        return df

    rid = df["race_id"].astype(str)

    if "tm_score" in df.columns:
        tm = pd.to_numeric(df["tm_score"], errors="coerce")
        df["race_tm_score_std"] = tm.groupby(rid, sort=False).transform("std").astype("float32")
        df["horse_tm_score_pct_in_race"] = (
            tm.groupby(rid, sort=False).rank(pct=True, method="average").astype("float32")
        )
    else:
        df["race_tm_score_std"] = np.nan
        df["horse_tm_score_pct_in_race"] = np.nan

    if "mining_predicted_rank" in df.columns:
        mr = pd.to_numeric(df["mining_predicted_rank"], errors="coerce")
        df["race_mining_rank_std"] = mr.groupby(rid, sort=False).transform("std").astype("float32")
    else:
        df["race_mining_rank_std"] = np.nan

    tm_std = pd.to_numeric(df["race_tm_score_std"], errors="coerce").fillna(0)
    mr_std = pd.to_numeric(df["race_mining_rank_std"], errors="coerce").fillna(0)
    df["field_competition_score"] = (
        (tm_std / 200.0).clip(0, 1) * 0.6 + (mr_std / 5.0).clip(0, 1) * 0.4
    ).astype("float32")

    return df


def add_rank1_context_v24_features(df: pd.DataFrame) -> pd.DataFrame:
    """v24 Rank1 コンテキスト特徴量を追加（既存列があればスキップ）。"""
    if all(c in df.columns for c in _NEW_COLS):
        return df

    out = _sort_df(df.copy())
    out = _add_traffic_and_large_field(out)
    out = _add_burden_context(out)
    out = _add_handicap_history(out)
    out = _add_worse_going_history(out)
    out = _add_field_competition(out)

    for col in _NEW_COLS:
        if col in out.columns:
            out[col] = out[col].astype("float32")

    return out


def v24_new_column_names() -> list[str]:
    return list(_NEW_COLS)

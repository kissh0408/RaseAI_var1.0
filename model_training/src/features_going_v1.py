"""P1 going_track 特徴量: 馬×芝/ダート×馬場状態の Bayes 平滑勝率。

domain-planner: docs/specs/domain_planner_spec_going_v1.md
全列 shift(1) 相当（cumsum − 当該行）でリーク防止。v6 の surface_code / track_condition_code を使用。

追加列:
  horse_turf_heavy_win_rate   芝 & 重/不良(track>=3) 過去勝率
  horse_dirt_heavy_win_rate   ダート & 重/不良 過去勝率
  condition_win_rate_bayes    当該馬場コードと同一 condition の Bayes 勝率
  going_active_heavy_rate     当日馬場が重以上のとき surface に応じた heavy 勝率
  horse_going_preference      同一 surface 内 heavy − light 勝率差
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GOING_V1_COLS: tuple[str, ...] = (
    "horse_turf_heavy_win_rate",
    "horse_dirt_heavy_win_rate",
    "condition_win_rate_bayes",
    "going_active_heavy_rate",
    "horse_going_preference",
)

_PRIOR_TURF = 0.07
_PRIOR_DIRT = 0.08
_BETA = 5.0


def going_v1_column_names() -> list[str]:
    return list(GOING_V1_COLS)


def _bayes_shifted_rate(
    df: pd.DataFrame,
    *,
    horse_col: str,
    mask: pd.Series,
    prior: float,
    beta: float = _BETA,
) -> pd.Series:
    """条件 mask に合致する過去走のみで Bayes 平滑勝率（当該レース除外）。"""
    win = (df["finish_rank"] == 1).astype(float).where(df["finish_rank"] > 0, 0.0)
    m = mask.fillna(False).astype(float)
    cond_win = win * m
    grp = df[horse_col]
    cum_runs = m.groupby(grp, sort=False).cumsum() - m
    cum_wins = cond_win.groupby(grp, sort=False).cumsum() - cond_win
    rate = (cum_wins + beta * prior) / (cum_runs + beta)
    return rate.where(cum_runs > 0, np.nan).astype("float32")


def _condition_bayes_rate(df: pd.DataFrame, horse_col: str = "horse_id") -> pd.Series:
    """馬×track_condition_code ごとの Bayes 勝率（shift 済み）。"""
    win = (df["finish_rank"] == 1).astype(float).where(df["finish_rank"] > 0, 0.0)
    keys = [horse_col, "track_condition_code"]
    grp = df.groupby(keys, sort=False)
    cum_runs = grp.cumcount().astype(float)
    cum_wins = win.groupby([df[horse_col], df["track_condition_code"]], sort=False).cumsum() - win
    rate = (cum_wins + _BETA * _PRIOR_TURF) / (cum_runs + _BETA)
    return rate.where(cum_runs > 0, np.nan).astype("float32")


def add_going_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    """features_v6 に P1 going 列を追加する。"""
    out = df.copy()
    if "race_date" in out.columns:
        out["race_date"] = pd.to_datetime(out["race_date"])
        out = out.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    surface = pd.to_numeric(out["surface_code"], errors="coerce")
    tcond = pd.to_numeric(out["track_condition_code"], errors="coerce")
    is_turf = surface == 1
    is_dirt = surface == 2
    is_heavy_cond = tcond >= 3
    is_light_cond = tcond == 1

    out["horse_turf_heavy_win_rate"] = _bayes_shifted_rate(
        out,
        horse_col="horse_id",
        mask=is_turf & is_heavy_cond,
        prior=_PRIOR_TURF,
    )
    out["horse_dirt_heavy_win_rate"] = _bayes_shifted_rate(
        out,
        horse_col="horse_id",
        mask=is_dirt & is_heavy_cond,
        prior=_PRIOR_DIRT,
    )
    turf_light = _bayes_shifted_rate(
        out, horse_col="horse_id", mask=is_turf & is_light_cond, prior=_PRIOR_TURF
    )
    dirt_light = _bayes_shifted_rate(
        out, horse_col="horse_id", mask=is_dirt & is_light_cond, prior=_PRIOR_DIRT
    )
    out["condition_win_rate_bayes"] = _condition_bayes_rate(out)

    heavy_now = is_heavy_cond.fillna(False)
    out["going_active_heavy_rate"] = np.where(
        heavy_now & is_turf,
        out["horse_turf_heavy_win_rate"],
        np.where(
            heavy_now & is_dirt,
            out["horse_dirt_heavy_win_rate"],
            np.nan,
        ),
    ).astype("float32")

    turf_pref = out["horse_turf_heavy_win_rate"] - turf_light
    dirt_pref = out["horse_dirt_heavy_win_rate"] - dirt_light
    out["horse_going_preference"] = np.where(
        is_turf,
        turf_pref,
        np.where(is_dirt, dirt_pref, np.nan),
    ).astype("float32")

    return out

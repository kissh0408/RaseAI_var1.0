"""馬場 what-if シナリオ用の推論時特徴量再計算（学習式と同一ロジック）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIRT_TRACK_CODE_MIN = 23


def unified_going_condition(
    df: pd.DataFrame,
    *,
    is_turf: pd.Series | None = None,
    is_dirt: pd.Series | None = None,
) -> pd.Series:
    """芝/ダート混在を1列に統合（create_pastfeatures v10 と同一）。"""
    turf_cond = pd.to_numeric(df.get("turf_condition"), errors="coerce")
    dirt_cond = pd.to_numeric(df.get("dirt_condition"), errors="coerce")
    if is_turf is None or is_dirt is None:
        tc = pd.to_numeric(df.get("track_code"), errors="coerce").fillna(0).astype(np.int64)
        is_dirt = tc >= DIRT_TRACK_CODE_MIN
        is_turf = ~is_dirt
    unified = turf_cond.where(is_turf, dirt_cond)
    return unified.replace(0, np.nan)


def recompute_going_change_features_scenario(
    df: pd.DataFrame,
    base_unified: pd.Series,
) -> pd.DataFrame:
    """シナリオ適用後の unified_cond から going_change_lag1 / going_worsening_flag を更新。"""
    out = df
    if "going_change_lag1" not in out.columns:
        return out

    tc = pd.to_numeric(out["track_code"], errors="coerce").fillna(0).astype(np.int64)
    is_dirt = tc >= DIRT_TRACK_CODE_MIN
    is_turf = ~is_dirt

    new_unified = unified_going_condition(out, is_turf=is_turf, is_dirt=is_dirt)
    old_change = pd.to_numeric(out["going_change_lag1"], errors="coerce").fillna(0.0)
    prev_cond = base_unified - old_change
    going_change = (new_unified - prev_cond).astype("float32")
    out.loc[:, "going_change_lag1"] = going_change.fillna(0.0)
    if "going_worsening_flag" in out.columns:
        out.loc[:, "going_worsening_flag"] = (going_change > 0).astype("int8")
    return out

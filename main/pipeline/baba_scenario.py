"""馬場 what-if シナリオ特徴量更新（model_training 非依存）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIRT_TRACK_CODE_MIN = 23

_TURF_WIN_RATE_FALLBACK: float = 0.07
_TURF_WIN_RATE_FALLBACK_HEAVY: float = 0.06
_DIRT_WIN_RATE_FALLBACK: float = 0.08
_DIRT_WIN_RATE_FALLBACK_HEAVY: float = 0.07
_TURF_TOP3_FALLBACK_LIGHT: float = 0.30
_TURF_TOP3_FALLBACK_SOFT: float = 0.20
_TURF_TOP3_FALLBACK_HEAVY: float = 0.22


def unified_going_condition(
    df: pd.DataFrame,
    *,
    is_turf: pd.Series | None = None,
    is_dirt: pd.Series | None = None,
) -> pd.Series:
    turf_cond = pd.to_numeric(df.get("turf_condition"), errors="coerce")
    dirt_cond = pd.to_numeric(df.get("dirt_condition"), errors="coerce")
    if is_turf is None or is_dirt is None:
        tc = pd.to_numeric(df.get("track_code"), errors="coerce").fillna(0).astype(np.int64)
        is_dirt = tc >= DIRT_TRACK_CODE_MIN
        is_turf = ~is_dirt
    unified = turf_cond.where(is_turf, dirt_cond)
    return unified.replace(0, np.nan)


def recompute_going_change_features_scenario(df: pd.DataFrame, base_unified: pd.Series) -> pd.DataFrame:
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


def recompute_going_match_score_turf_imputed_scenario(df: pd.DataFrame) -> pd.Series:
    if "going_match_score_turf_imputed" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")

    existing_gms = (
        pd.to_numeric(df.get("going_match_score_turf"), errors="coerce").astype("float32")
        if "going_match_score_turf" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float32")
    )
    prior_imputed = pd.to_numeric(df["going_match_score_turf_imputed"], errors="coerce").astype("float32")
    score = existing_gms.where(existing_gms.notna(), prior_imputed)
    return score.clip(0.0, 3.0).astype("float32")


def recompute_going_match_score_dirt_imputed_scenario(df: pd.DataFrame) -> pd.Series:
    if "going_match_score_dirt_imputed" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")

    existing_gms = (
        pd.to_numeric(df.get("going_match_score_dirt"), errors="coerce").astype("float32")
        if "going_match_score_dirt" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float32")
    )
    prior_imputed = pd.to_numeric(df["going_match_score_dirt_imputed"], errors="coerce").astype("float32")
    score = existing_gms.where(existing_gms.notna(), prior_imputed)
    return score.clip(0.0, 3.0).astype("float32")


def apply_uniform_baba_jv_code(df: pd.DataFrame, jv_code: int) -> pd.DataFrame:
    """全行を同一の馬場シナリオ（JV コード 1–4）に置き換える。"""
    out = df.copy()
    if "track_code" not in out.columns:
        return out
    tc = pd.to_numeric(out["track_code"], errors="coerce").fillna(0).astype(np.int64)
    is_dirt = tc >= DIRT_TRACK_CODE_MIN
    is_turf = ~is_dirt
    n = len(out)

    base_unified = unified_going_condition(out, is_turf=is_turf, is_dirt=is_dirt)

    if "turf_condition" in out.columns:
        arr = np.zeros(n, dtype=np.float64)
        arr[is_turf.to_numpy()] = float(jv_code)
        out.loc[:, "turf_condition"] = arr
    if "dirt_condition" in out.columns:
        arr = np.zeros(n, dtype=np.float64)
        arr[is_dirt.to_numpy()] = float(jv_code)
        out.loc[:, "dirt_condition"] = arr

    if "track_condition_code" in out.columns:
        tc_arr = np.zeros(n, dtype=np.float64)
        tc_arr[is_turf.to_numpy()] = float(jv_code)
        tc_arr[is_dirt.to_numpy()] = float(jv_code)
        out.loc[:, "track_condition_code"] = tc_arr

    for code in [2, 3, 4]:
        col_t = f"turf_cond_{code}"
        col_d = f"dirt_cond_{code}"
        arr_t = np.zeros(n, dtype=np.float32)
        arr_t[is_turf.to_numpy()] = float(jv_code == code)
        out.loc[:, col_t] = arr_t
        arr_d = np.zeros(n, dtype=np.float32)
        arr_d[is_dirt.to_numpy()] = float(jv_code == code)
        out.loc[:, col_d] = arr_d

    t2 = out["turf_cond_2"] if "turf_cond_2" in out.columns else pd.Series(0.0, index=out.index)
    t3 = out["turf_cond_3"] if "turf_cond_3" in out.columns else pd.Series(0.0, index=out.index)
    t4 = out["turf_cond_4"] if "turf_cond_4" in out.columns else pd.Series(0.0, index=out.index)
    d2 = out["dirt_cond_2"] if "dirt_cond_2" in out.columns else pd.Series(0.0, index=out.index)
    d3 = out["dirt_cond_3"] if "dirt_cond_3" in out.columns else pd.Series(0.0, index=out.index)
    d4 = out["dirt_cond_4"] if "dirt_cond_4" in out.columns else pd.Series(0.0, index=out.index)

    def _col(col: str) -> pd.Series:
        return out[col] if col in out.columns else pd.Series(0.0, index=out.index)

    heavy_wr = _col("horse_turf_heavy_win_rate")
    vheavy_wr = _col("horse_turf_very_heavy_win_rate")
    out.loc[:, "going_x_turf_heavy_winrate"] = (t3 * heavy_wr).astype(np.float32)

    light_flag = (1 - t2 - t3 - t4).clip(0, 1)
    light_wr = _col("horse_turf_light_win_rate")
    out.loc[:, "going_x_turf_light_winrate"] = (light_flag * light_wr).astype(np.float32)

    soft_wr = _col("horse_turf_soft_win_rate")
    out.loc[:, "going_x_turf_soft_winrate"] = (t2 * soft_wr).astype(np.float32)

    d_heavy_wr = _col("horse_dirt_heavy_win_rate")
    d_vheavy_wr = _col("horse_dirt_very_heavy_win_rate")
    out.loc[:, "going_x_dirt_heavy_winrate"] = (d3 * d_heavy_wr).astype(np.float32)

    out.loc[:, "going_match_score_turf"] = (
        t2 * soft_wr + t3 * heavy_wr + t4 * vheavy_wr.fillna(heavy_wr)
    ).astype(np.float32)

    d_soft_wr = _col("horse_dirt_soft_win_rate")
    out.loc[:, "going_match_score_dirt"] = (
        d2 * d_soft_wr + d3 * d_heavy_wr + d4 * d_vheavy_wr.fillna(d_heavy_wr)
    ).astype(np.float32)

    bayes_wr = _col("horse_turf_soft_win_rate_bayes")
    if "going_x_soft_win_rate_imputed" in out.columns:
        out.loc[:, "going_x_soft_win_rate_imputed"] = (t2 * bayes_wr).astype(np.float32)

    turf_idx = is_turf.to_numpy()
    dirt_idx = is_dirt.to_numpy()

    def _get(col: str) -> pd.Series:
        return out[col] if col in out.columns else pd.Series(np.nan, index=out.index)

    if "current_going_win_rate_turf" in out.columns:
        light_wr2 = _get("horse_turf_light_win_rate")
        soft_wr10 = _get("horse_turf_soft_win_rate_v10")
        hv3_wr = _get("horse_turf_heavy3_win_rate")
        vh_wr = _get("horse_turf_very_heavy_win_rate")

        relay_turf = pd.Series(np.nan, index=out.index, dtype="float32")
        if jv_code == 1:
            relay_turf[turf_idx] = light_wr2[turf_idx].fillna(_TURF_WIN_RATE_FALLBACK)
        elif jv_code == 2:
            relay_turf[turf_idx] = soft_wr10[turf_idx].fillna(light_wr2[turf_idx]).fillna(_TURF_WIN_RATE_FALLBACK)
        elif jv_code == 3:
            relay_turf[turf_idx] = hv3_wr[turf_idx].fillna(soft_wr10[turf_idx]).fillna(light_wr2[turf_idx]).fillna(_TURF_WIN_RATE_FALLBACK)
        elif jv_code == 4:
            relay_turf[turf_idx] = vh_wr[turf_idx].fillna(hv3_wr[turf_idx]).fillna(soft_wr10[turf_idx]).fillna(light_wr2[turf_idx]).fillna(_TURF_WIN_RATE_FALLBACK_HEAVY)
        out.loc[:, "current_going_win_rate_turf"] = relay_turf

    if "current_going_win_rate_dirt" in out.columns:
        d_light_wr = _get("horse_dirt_light_win_rate")
        d_soft_wr2 = _get("horse_dirt_soft_win_rate")
        d_hv3_wr = _get("horse_dirt_heavy3_win_rate")
        d_vh_wr = _get("horse_dirt_very_heavy_win_rate")

        relay_dirt = pd.Series(np.nan, index=out.index, dtype="float32")
        if jv_code == 1:
            relay_dirt[dirt_idx] = d_light_wr[dirt_idx].fillna(_DIRT_WIN_RATE_FALLBACK)
        elif jv_code == 2:
            relay_dirt[dirt_idx] = d_soft_wr2[dirt_idx].fillna(d_light_wr[dirt_idx]).fillna(_DIRT_WIN_RATE_FALLBACK)
        elif jv_code == 3:
            relay_dirt[dirt_idx] = d_hv3_wr[dirt_idx].fillna(d_soft_wr2[dirt_idx]).fillna(d_light_wr[dirt_idx]).fillna(_DIRT_WIN_RATE_FALLBACK)
        elif jv_code == 4:
            relay_dirt[dirt_idx] = d_vh_wr[dirt_idx].fillna(d_hv3_wr[dirt_idx]).fillna(d_soft_wr2[dirt_idx]).fillna(d_light_wr[dirt_idx]).fillna(_DIRT_WIN_RATE_FALLBACK_HEAVY)
        out.loc[:, "current_going_win_rate_dirt"] = relay_dirt

    if "current_going_top3_rate_turf" in out.columns:
        soft_top3_10 = _get("horse_turf_soft_top3_rate_v10")
        soft_top3_v9 = _get("horse_turf_soft_top3_rate")
        hv3_top3 = _get("horse_turf_heavy3_top3_rate")

        relay_top3 = pd.Series(np.nan, index=out.index, dtype="float32")
        if jv_code == 1:
            relay_top3[turf_idx] = _TURF_TOP3_FALLBACK_LIGHT
        elif jv_code == 2:
            relay_top3[turf_idx] = soft_top3_10[turf_idx].fillna(soft_top3_v9[turf_idx]).fillna(_TURF_TOP3_FALLBACK_SOFT)
        elif jv_code in (3, 4):
            relay_top3[turf_idx] = hv3_top3[turf_idx].fillna(soft_top3_10[turf_idx]).fillna(soft_top3_v9[turf_idx]).fillna(_TURF_TOP3_FALLBACK_HEAVY)
        out.loc[:, "current_going_top3_rate_turf"] = relay_top3

    if "going_match_score_turf_imputed" in out.columns:
        out.loc[:, "going_match_score_turf_imputed"] = recompute_going_match_score_turf_imputed_scenario(out)
    if "going_match_score_dirt_imputed" in out.columns:
        out.loc[:, "going_match_score_dirt_imputed"] = recompute_going_match_score_dirt_imputed_scenario(out)

    recompute_going_change_features_scenario(out, base_unified)
    return out

# NOTE: standalone ablation では REJECT だが、create_pastfeatures v26 と本番 inference_pipeline
# の going_delta_active_score 再計算で使用中。削除不可。
# REJECTED EXPERIMENT (2026-06): バックテスト ROI 改善は未確認。
"""馬場適性 delta 特徴量（良 vs 重/稍重の差分ベクトル）。

全 delta は shift(1) 済みの過去集計列から算出する（当該レース結果は不使用）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DELTA_COLS = (
    "delta_sire_turf_heavy_aptitude",
    "delta_sire_dirt_heavy_aptitude",
    "delta_sire_heavy_aptitude",
    "delta_horse_turf_heavy_aptitude",
    "delta_horse_turf_soft_aptitude",
    "delta_horse_dirt_heavy_aptitude",
    "delta_horse_dirt_soft_aptitude",
    "delta_horse_heavy_aptitude",
    "delta_horse_turf_very_heavy_aptitude",
    "delta_jockey_turf_heavy_aptitude",
    "delta_jockey_dirt_heavy_aptitude",
    "delta_jockey_heavy_aptitude",
    "going_delta_active_score",
)


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _delta(h: pd.Series, l: pd.Series) -> pd.Series:
    return (h - l).astype("float32")


def add_going_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """v26: 血統・馬・騎手の馬場適性 delta 列と going_delta_active_score を追加。"""
    out = df.copy()

    sire_t_h = _num(out, "sire_turf_heavy_win_rate")
    sire_t_l = _num(out, "sire_turf_soft_win_rate", _num(out, "sire_turf_light_win_rate"))
    sire_d_h = _num(out, "sire_dirt_heavy_win_rate")
    sire_d_l = _num(out, "sire_dirt_soft_win_rate", _num(out, "sire_dirt_light_win_rate"))
    out["delta_sire_turf_heavy_aptitude"] = _delta(sire_t_h, sire_t_l)
    out["delta_sire_dirt_heavy_aptitude"] = _delta(sire_d_h, sire_d_l)
    out["delta_sire_heavy_aptitude"] = (
        out["delta_sire_turf_heavy_aptitude"].fillna(0)
        + out["delta_sire_dirt_heavy_aptitude"].fillna(0)
    ).astype("float32") / 2.0

    h_t_h = _num(out, "horse_turf_heavy_win_rate")
    h_t_l = _num(out, "horse_turf_light_win_rate")
    h_t_s = _num(out, "horse_turf_soft_win_rate")
    h_t_vh = _num(out, "horse_turf_very_heavy_win_rate")
    h_d_h = _num(out, "horse_dirt_heavy_win_rate")
    h_d_l = _num(out, "horse_dirt_light_win_rate")
    h_d_s = _num(out, "horse_dirt_soft_win_rate")
    out["delta_horse_turf_heavy_aptitude"] = _delta(h_t_h, h_t_l)
    out["delta_horse_turf_soft_aptitude"] = _delta(h_t_s, h_t_l)
    out["delta_horse_dirt_heavy_aptitude"] = _delta(h_d_h, h_d_l)
    out["delta_horse_dirt_soft_aptitude"] = _delta(h_d_s, h_d_l)
    out["delta_horse_heavy_aptitude"] = (
        out["delta_horse_turf_heavy_aptitude"].fillna(0)
        + out["delta_horse_dirt_heavy_aptitude"].fillna(0)
    ).astype("float32") / 2.0
    out["delta_horse_turf_very_heavy_aptitude"] = _delta(h_t_vh, h_t_l)

    j_t_h = _num(out, "jockey_turf_heavy_win_rate", _num(out, "jockey_heavy_win_rate"))
    j_t_l = _num(out, "jockey_turf_light_win_rate")
    j_d_h = _num(out, "jockey_dirt_heavy_win_rate")
    j_d_l = _num(out, "jockey_dirt_light_win_rate")
    out["delta_jockey_turf_heavy_aptitude"] = _delta(j_t_h, j_t_l)
    out["delta_jockey_dirt_heavy_aptitude"] = _delta(j_d_h, j_d_l)
    out["delta_jockey_heavy_aptitude"] = (
        out["delta_jockey_turf_heavy_aptitude"].fillna(0)
        + out["delta_jockey_dirt_heavy_aptitude"].fillna(0)
    ).astype("float32") / 2.0

    tc = pd.to_numeric(out.get("track_code", 0), errors="coerce").fillna(0)
    is_dirt = tc >= 23
    tcond = pd.to_numeric(out.get("turf_condition", 1), errors="coerce").fillna(1).astype(int)
    dcond = pd.to_numeric(out.get("dirt_condition", 0), errors="coerce").fillna(0).astype(int)
    jv = np.where(is_dirt.to_numpy(), dcond.to_numpy(), tcond.to_numpy())
    out["going_delta_active_score"] = compute_going_delta_active_score_from_arrays(
        jv,
        out["delta_horse_turf_soft_aptitude"].to_numpy(dtype=float),
        out["delta_horse_dirt_soft_aptitude"].to_numpy(dtype=float),
        out["delta_horse_heavy_aptitude"].to_numpy(dtype=float),
        out["delta_horse_turf_very_heavy_aptitude"].to_numpy(dtype=float),
        out["delta_sire_heavy_aptitude"].to_numpy(dtype=float),
        out["delta_jockey_heavy_aptitude"].to_numpy(dtype=float),
        is_dirt.to_numpy(),
    )
    return out


def compute_going_delta_active_score_from_arrays(
    jv_codes: np.ndarray,
    horse_turf_soft: np.ndarray,
    horse_dirt_soft: np.ndarray,
    horse_heavy: np.ndarray,
    horse_vheavy: np.ndarray,
    sire_heavy: np.ndarray,
    jockey_heavy: np.ndarray,
    is_dirt: np.ndarray,
) -> np.ndarray:
    n = len(jv_codes)
    active = np.zeros(n, dtype=np.float32)
    for i in range(n):
        code = int(jv_codes[i])
        if code == 1:
            continue
        if code == 2:
            active[i] = float(
                horse_dirt_soft[i] if is_dirt[i] else horse_turf_soft[i]
            ) + float(sire_heavy[i]) * 0.5 + float(jockey_heavy[i]) * 0.5
        elif code == 3:
            active[i] = float(horse_heavy[i]) + float(sire_heavy[i]) + float(jockey_heavy[i])
        elif code == 4:
            active[i] = (
                float(horse_heavy[i])
                + float(horse_vheavy[i])
                + float(sire_heavy[i])
                + float(jockey_heavy[i])
            )
    return active


def compute_going_delta_active_score(df: pd.DataFrame, jv_code: int) -> pd.Series:
    """what-if 推論: 全行を同一 JV シナリオにした going_delta_active_score。"""
    n = len(df)
    jv = np.full(n, int(jv_code), dtype=int)
    tc = pd.to_numeric(df.get("track_code", 0), errors="coerce").fillna(0)
    is_dirt = (tc >= 23).to_numpy()
    return pd.Series(
        compute_going_delta_active_score_from_arrays(
            jv,
            _num(df, "delta_horse_turf_soft_aptitude", 0.0).to_numpy(dtype=float),
            _num(df, "delta_horse_dirt_soft_aptitude", 0.0).to_numpy(dtype=float),
            _num(df, "delta_horse_heavy_aptitude", 0.0).to_numpy(dtype=float),
            _num(df, "delta_horse_turf_very_heavy_aptitude", 0.0).to_numpy(dtype=float),
            _num(df, "delta_sire_heavy_aptitude", 0.0).to_numpy(dtype=float),
            _num(df, "delta_jockey_heavy_aptitude", 0.0).to_numpy(dtype=float),
            is_dirt,
        ),
        index=df.index,
        dtype="float32",
    )


def going_delta_feature_names() -> tuple[str, ...]:
    return _DELTA_COLS

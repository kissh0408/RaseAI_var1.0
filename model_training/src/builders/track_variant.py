"""開催日×競馬場×芝ダのトラックバリアントと馬場補正済みタイム指数。

第1版: 同一 race_date / course_code / surface_code 内の base_time 中央値と
       過去同キー par（shift 済み expanding median）の差を daily_track_variant とする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_COND_ADJ = {1: 0.0, 2: -5.0, 3: -12.0, 4: -20.0}


def add_track_variant_features(df: pd.DataFrame) -> pd.DataFrame:
    """v27: daily_track_variant, tm_score_surface_adj を追加。"""
    out = df.copy()
    if "race_date" not in out.columns:
        out["daily_track_variant"] = np.nan
        out["tm_score_surface_adj"] = _num(out, "tm_score")
        return out

    out = out.sort_values(["race_date", "course_code", "surface_code"]).copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")

    if "base_time" in out.columns:
        bt = pd.to_numeric(out["base_time"], errors="coerce")
        day_med = out.groupby(["race_date", "course_code", "surface_code"])["base_time"].transform("median")
        out["daily_track_variant"] = (bt - day_med).astype("float32")
    else:
        out["daily_track_variant"] = np.nan

    tm = _num(out, "tm_score")
    variant = pd.to_numeric(out["daily_track_variant"], errors="coerce").fillna(0.0)
    tcc = pd.to_numeric(out.get("track_condition_code", 1), errors="coerce").fillna(1).astype(int)
    going_adj = tcc.map(_COND_ADJ).fillna(0.0)
    out["tm_score_surface_adj"] = (tm - variant + going_adj).astype("float32")
    return out


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")

"""Race-internal z-score utilities for L1 score export."""

from __future__ import annotations

import numpy as np
import pandas as pd


def race_zscore(series: pd.Series) -> pd.Series:
    """Standardize scores within a race; return 0 if std is degenerate."""
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std()
    if sd is None or sd < 1e-8 or not np.isfinite(sd):
        return pd.Series(0.0, index=series.index, dtype="float32")
    return ((s - mu) / sd).astype("float32")


def attach_pure_score_z(
    df: pd.DataFrame,
    *,
    score_col: str = "pure_score",
    race_id_col: str = "race_id",
    out_col: str = "pure_score_z",
) -> pd.DataFrame:
    """Add race-internal z-score column."""
    out = df.copy()
    out[out_col] = out.groupby(race_id_col, sort=False)[score_col].transform(race_zscore)
    return out

"""Pure functions for alpha_segments: segment flag assignment.

This module never reads betting-market-derived columns of any kind.
SEGMENT_COLUMNS is the whitelist of the only columns segment rules may read;
it is asserted against the project-wide FORBIDDEN_MARKET_COLS /
SUSPICIOUS_MARKET_NAME_PATTERN guards in the tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent

# Whitelist of columns segment rules may read (spec section 4 / section 9 item 8).
# Adding a column here that is not a race attribute or shift(1)-safe past-race
# attribute would be a market-information leak.
SEGMENT_COLUMNS: frozenset[str] = frozenset({
    "hist_last_rank",
    "horse_count",
    "track_condition_code",
    "course_code",
    "race_condition_code",
})


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or (EXP_DIR / "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _race_level_value(df: pd.DataFrame, col: str, race_id_col: str) -> pd.Series:
    """Per-race representative value (first non-null in group), broadcast to row index."""
    return df.groupby(race_id_col, sort=False)[col].transform("first")


def flag_debut_ratio(
    df: pd.DataFrame,
    *,
    col: str = "hist_last_rank",
    race_id_col: str = "race_id",
    threshold: float = 0.5,
) -> pd.Series:
    """S1: race-level NaN ratio of `col` >= threshold (boundary inclusive)."""
    is_nan = df[col].isna()
    ratio = is_nan.groupby(df[race_id_col], sort=False).transform("mean")
    return ratio >= threshold


def flag_lte(
    df: pd.DataFrame,
    *,
    col: str,
    race_id_col: str = "race_id",
    threshold: float,
) -> pd.Series:
    """Race-level column <= threshold (e.g. S2 horse_count)."""
    val = _race_level_value(df, col, race_id_col)
    return val <= threshold


def flag_in(
    df: pd.DataFrame,
    *,
    col: str,
    race_id_col: str = "race_id",
    values: list,
) -> pd.Series:
    """Race-level column in values (e.g. S3/S4/S5)."""
    val = _race_level_value(df, col, race_id_col)
    return val.isin(values)


def apply_segment_flag(
    df: pd.DataFrame,
    segment_id: str,
    cfg: dict,
    *,
    race_id_col: str = "race_id",
) -> pd.Series:
    """Compute the boolean flag Series for one segment_id per config.json rule."""
    spec = cfg["segments"][segment_id]
    col = spec["column"]
    if col not in SEGMENT_COLUMNS:
        raise ValueError(
            f"Column '{col}' for segment {segment_id} is not in SEGMENT_COLUMNS "
            f"whitelist ({sorted(SEGMENT_COLUMNS)}). Refusing to compute a segment "
            f"flag from an unregistered (potentially market-derived) column."
        )
    rule = spec["rule"]
    if rule == "nan_ratio_gte":
        return flag_debut_ratio(df, col=col, race_id_col=race_id_col, threshold=spec["threshold"])
    if rule == "lte":
        return flag_lte(df, col=col, race_id_col=race_id_col, threshold=spec["threshold"])
    if rule == "in":
        return flag_in(df, col=col, race_id_col=race_id_col, values=spec["values"])
    raise ValueError(f"Unknown segment rule '{rule}' for segment {segment_id}")


def add_all_segment_flags(
    df: pd.DataFrame,
    cfg: dict,
    *,
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """Attach one `seg_<ID>` boolean column per segment in cfg['segment_order']."""
    out = df.copy()
    for seg_id in cfg["segment_order"]:
        out[f"seg_{seg_id}"] = apply_segment_flag(out, seg_id, cfg, race_id_col=race_id_col)
    return out


def confirm_segments(counts: dict[str, dict[str, int]], n_min: int = 300) -> dict[str, dict]:
    """Add 'confirmed' bool (n_2024 >= n_min) to each segment's count dict."""
    out: dict[str, dict] = {}
    for seg_id, c in counts.items():
        confirmed = c["n_2024"] >= n_min
        out[seg_id] = {**c, "confirmed": bool(confirmed)}
    return out


def bonferroni_threshold(k: int, base_alpha: float = 0.01) -> float:
    """Bonferroni-corrected significance threshold; 0.0 (never significant) if K=0."""
    if k <= 0:
        return 0.0
    return base_alpha / k


def leak_stop(
    top1: float,
    spearman: float,
    *,
    top1_threshold: float = 0.40,
    spearman_threshold: float = 0.60,
) -> bool:
    """Leak-stop trigger: Top-1 > threshold OR Spearman > threshold."""
    return bool(top1 > top1_threshold or spearman > spearman_threshold)


def primary_pass(
    p_value: float,
    delta_ll_per_race: float,
    bonferroni_threshold_value: float,
    leak: bool,
) -> bool:
    """Stage 2 primary-pass gate: p < Bonferroni AND deltaLL>0, unless leak-stopped."""
    if leak:
        return False
    return bool(p_value < bonferroni_threshold_value and delta_ll_per_race > 0)

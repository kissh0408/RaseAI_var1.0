"""Pure(ish) functions for track_b_training: HC/WC training-time-series candidates.

This module never reads betting-related columns of any kind. Inputs are
restricted to HC/WC (JRA official workout timing) and race keys
(race_id, horse_num, ketto_num, race_date). See the project README's market
boundary section for the full guard-list this module is checked against.

All window/count parameters are passed explicitly by callers (build_candidates.py
reads them from config.json) -- no hardcoded thresholds live in this module
except the pure-math defaults mirrored from config for standalone testability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or (EXP_DIR / "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────
# Generic math primitives
# ─────────────────────────────────────────────────────────────────────────

def slope_ols(t_days: np.ndarray, y: np.ndarray, *, min_n: int = 3) -> float:
    """OLS slope of y on t_days (t_days typically negative day offsets from race_date).

    Returns NaN if fewer than min_n finite points remain, or t_days has ~zero
    variance (all-same-day). Returns 0.0 (not NaN) for a genuinely flat fit.
    """
    t = np.asarray(t_days, dtype=float)
    v = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if len(t) < min_n:
        return float("nan")
    t_var = np.sum((t - t.mean()) ** 2)
    if t_var < 1e-9:
        return float("nan")
    slope = float(np.sum((t - t.mean()) * (v - v.mean())) / t_var)
    return slope


def expanding_median_shift1(series: pd.Series, *, min_periods: int) -> pd.Series:
    """shift(1) + expanding median counting only non-NaN prior observations.

    Unlike pandas' native `.shift(1).expanding(min_periods=k).median()`, this
    counts *valid* (non-NaN) prior values toward min_periods, not raw window
    length, matching the spec's "過去レースの n_interval 標本数 < 3 -> NaN".
    """
    out = np.full(len(series), np.nan, dtype=float)
    valid_history: list[float] = []
    for i, v in enumerate(series.to_numpy(dtype=float)):
        if len(valid_history) >= min_periods:
            out[i] = float(np.median(valid_history))
        if np.isfinite(v):
            valid_history.append(v)
    return pd.Series(out, index=series.index)


def freq_ratio(n_interval: float, baseline: float) -> float:
    """n_interval / baseline; NaN if either input is NaN or baseline == 0."""
    if n_interval is None or baseline is None:
        return float("nan")
    if not np.isfinite(n_interval) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return float(n_interval / baseline)


def recent_minus_career(values: np.ndarray, *, recent_n: int, min_career_n: int) -> float:
    """mean(last recent_n) - mean(all); NaN if fewer than min_career_n valid values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < min_career_n:
        return float("nan")
    career = float(v.mean())
    recent = float(v[-recent_n:].mean())
    return recent - career


def compute_fade(hc_200_sec: pd.Series, hc_3f_sec: pd.Series) -> pd.Series:
    """fade = hc_200_sec / (hc_3f_sec / 3); NaN where hc_3f_sec is NaN/<=0 or hc_200_sec NaN."""
    t200 = pd.to_numeric(hc_200_sec, errors="coerce")
    t3f = pd.to_numeric(hc_3f_sec, errors="coerce")
    valid = t200.notna() & t3f.notna() & (t3f > 0)
    fade = t200 / (t3f / 3.0)
    return fade.where(valid)


def wc_share(n_wc: float, n_total: float) -> float:
    """n_wc / n_total; NaN if n_total <= 0."""
    if n_total is None or not np.isfinite(n_total) or n_total <= 0:
        return float("nan")
    return float(n_wc / n_total)


def share_diff(recent_share: float, career_share: float) -> float:
    if recent_share is None or career_share is None:
        return float("nan")
    if not np.isfinite(recent_share) or not np.isfinite(career_share):
        return float("nan")
    return float(recent_share - career_share)


def fill_race_mean(
    df: pd.DataFrame,
    *,
    score_col: str = "cand_score",
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """Fill NaN cand_score with the race-internal mean of non-NaN values (spec section 3).

    Rows in a race where every horse is NaN get filled with 0.0 (spec item 2).
    """
    out = df.copy()
    race_mean = out.groupby(race_id_col, sort=False)[score_col].transform("mean")
    filled = out[score_col].where(out[score_col].notna(), race_mean)
    filled = filled.fillna(0.0)
    out[score_col] = filled.astype(float)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Candidate builders (B-1 .. B-5). Each takes small already-prepared frames
# and returns raw (NaN-bearing) cand_score, one row per race_keys row.
# Callers (build_candidates.py) are responsible for restricting HC/WC to the
# relevant ketto_num set before calling these, for performance.
# ─────────────────────────────────────────────────────────────────────────

def _hc_grouped(hc: pd.DataFrame, value_col: str | None, *, date_col: str = "training_date") -> dict:
    """ketto_num -> sorted-by-date sub-DataFrame (or array if value_col is None)."""
    hc = hc.copy()
    hc[date_col] = pd.to_datetime(hc[date_col])
    out = {}
    for k, g in hc.groupby("ketto_num", sort=False):
        g = g.sort_values(date_col)
        out[k] = g
    return out


def build_b1_intensity_trend(
    hc: pd.DataFrame,
    race_keys: pd.DataFrame,
    *,
    window_days: int = 30,
    min_n: int = 3,
) -> pd.DataFrame:
    """B-1: 30-day OLS slope of hc_4f_sec (per horse); cand_score = -slope."""
    hc_valid = hc.dropna(subset=["hc_4f_sec"]).copy()
    hc_by_horse = _hc_grouped(hc_valid, "hc_4f_sec")

    rk = race_keys.copy()
    rk["race_date"] = pd.to_datetime(rk["race_date"])

    scores = []
    for row in rk.itertuples(index=False):
        g = hc_by_horse.get(row.ketto_num)
        if g is None:
            scores.append(float("nan"))
            continue
        lo = row.race_date - pd.Timedelta(days=window_days)
        window = g[(g["training_date"] >= lo) & (g["training_date"] < row.race_date)]
        if len(window) < min_n:
            scores.append(float("nan"))
            continue
        t = (window["training_date"] - row.race_date).dt.days.to_numpy(dtype=float)
        y = window["hc_4f_sec"].to_numpy(dtype=float)
        slope = slope_ols(t, y, min_n=min_n)
        scores.append(float("nan") if np.isnan(slope) else -slope)

    out = rk[["race_id", "horse_num"]].copy()
    out["cand_score"] = scores
    return out


def build_b2_freq_change(
    hc: pd.DataFrame,
    race_history: pd.DataFrame,
    race_keys: pd.DataFrame,
    *,
    min_baseline_n: int = 3,
) -> pd.DataFrame:
    """B-2: n_interval (HC row count since prev race) / individual baseline (expanding median).

    race_history must carry the horse's *full career* race_date sequence
    (from features parquet) so the baseline is computed over true past races,
    not just the fold2-OOS-scored subset.
    """
    hc_by_horse = {
        k: g.sort_values("training_date")["training_date"].to_numpy()
        for k, g in hc.assign(training_date=pd.to_datetime(hc["training_date"])).groupby("ketto_num", sort=False)
    }

    rh = race_history.drop_duplicates(subset=["ketto_num", "race_date"]).copy()
    rh["race_date"] = pd.to_datetime(rh["race_date"])
    rh = rh.sort_values(["ketto_num", "race_date"], kind="stable").reset_index(drop=True)
    rh["prev_race_date"] = rh.groupby("ketto_num")["race_date"].shift(1)

    def _n_interval(ketto_num, prev_date, race_date) -> float:
        if pd.isna(prev_date):
            return float("nan")
        dates = hc_by_horse.get(ketto_num)
        if dates is None or len(dates) == 0:
            return 0.0
        mask = (dates >= np.datetime64(prev_date)) & (dates < np.datetime64(race_date))
        return float(mask.sum())

    rh["n_interval"] = [
        _n_interval(k, p, r) for k, p, r in zip(rh["ketto_num"], rh["prev_race_date"], rh["race_date"])
    ]
    rh["baseline"] = rh.groupby("ketto_num", sort=False)["n_interval"].transform(
        lambda s: expanding_median_shift1(s, min_periods=min_baseline_n)
    )
    rh["cand_score"] = [
        freq_ratio(n, b) for n, b in zip(rh["n_interval"], rh["baseline"])
    ]

    rk = race_keys.copy()
    rk["race_date"] = pd.to_datetime(rk["race_date"])
    merged = rk.merge(
        rh[["ketto_num", "race_date", "cand_score"]],
        on=["ketto_num", "race_date"],
        how="left",
    )
    return merged[["race_id", "horse_num", "cand_score"]]


def build_b3_accel_profile(
    hc: pd.DataFrame,
    race_keys: pd.DataFrame,
    *,
    recent_n: int = 3,
    min_career_n: int = 6,
) -> pd.DataFrame:
    """B-3: recent3(hc_accel_sec) mean - career mean, both strictly before race_date."""
    hc_valid = hc.dropna(subset=["hc_accel_sec"]).copy()
    hc_by_horse = _hc_grouped(hc_valid, "hc_accel_sec")

    rk = race_keys.copy()
    rk["race_date"] = pd.to_datetime(rk["race_date"])

    scores = []
    for row in rk.itertuples(index=False):
        g = hc_by_horse.get(row.ketto_num)
        if g is None:
            scores.append(float("nan"))
            continue
        past = g.loc[g["training_date"] < row.race_date, "hc_accel_sec"].to_numpy(dtype=float)
        scores.append(recent_minus_career(past, recent_n=recent_n, min_career_n=min_career_n))

    out = rk[["race_id", "horse_num"]].copy()
    out["cand_score"] = scores
    return out


def build_b4_fade_trend(
    hc: pd.DataFrame,
    race_keys: pd.DataFrame,
    *,
    window_days: int = 30,
    min_n: int = 3,
) -> pd.DataFrame:
    """B-4: 30-day OLS slope of fade = hc_200_sec/(hc_3f_sec/3); cand_score = -slope."""
    hc_work = hc.copy()
    hc_work["fade"] = compute_fade(hc_work["hc_200_sec"], hc_work["hc_3f_sec"])
    hc_valid = hc_work.dropna(subset=["fade"]).copy()
    hc_by_horse = _hc_grouped(hc_valid, "fade")

    rk = race_keys.copy()
    rk["race_date"] = pd.to_datetime(rk["race_date"])

    scores = []
    for row in rk.itertuples(index=False):
        g = hc_by_horse.get(row.ketto_num)
        if g is None:
            scores.append(float("nan"))
            continue
        lo = row.race_date - pd.Timedelta(days=window_days)
        window = g[(g["training_date"] >= lo) & (g["training_date"] < row.race_date)]
        if len(window) < min_n:
            scores.append(float("nan"))
            continue
        t = (window["training_date"] - row.race_date).dt.days.to_numpy(dtype=float)
        y = window["fade"].to_numpy(dtype=float)
        slope = slope_ols(t, y, min_n=min_n)
        scores.append(float("nan") if np.isnan(slope) else -slope)

    out = rk[["race_id", "horse_num"]].copy()
    out["cand_score"] = scores
    return out


def build_b5_wc_switch(
    hc: pd.DataFrame,
    wc: pd.DataFrame,
    race_keys: pd.DataFrame,
    *,
    window_days: int = 30,
    wc_start: str,
    min_window_n: int = 2,
    min_career_n: int = 5,
) -> pd.DataFrame:
    """B-5: wc_share_recent(30d) - wc_share_career, both restricted to >= wc_start."""
    wc_start_ts = pd.Timestamp(wc_start)

    hc_work = hc.copy()
    hc_work["training_date"] = pd.to_datetime(hc_work["training_date"])
    hc_work = hc_work.loc[hc_work["training_date"] >= wc_start_ts]

    wc_work = wc.copy()
    wc_work["training_date"] = pd.to_datetime(wc_work["training_date"])
    wc_work = wc_work.loc[wc_work["training_date"] >= wc_start_ts]

    hc_by_horse = {
        k: g.sort_values("training_date")["training_date"].to_numpy()
        for k, g in hc_work.groupby("ketto_num", sort=False)
    }
    wc_by_horse = {
        k: g.sort_values("training_date")["training_date"].to_numpy()
        for k, g in wc_work.groupby("ketto_num", sort=False)
    }

    rk = race_keys.copy()
    rk["race_date"] = pd.to_datetime(rk["race_date"])

    empty = np.array([], dtype="datetime64[ns]")
    scores = []
    for row in rk.itertuples(index=False):
        race_date = np.datetime64(row.race_date)
        lo = np.datetime64(row.race_date - pd.Timedelta(days=window_days))
        hc_dates = hc_by_horse.get(row.ketto_num, empty)
        wc_dates = wc_by_horse.get(row.ketto_num, empty)

        n_hc_recent = int(((hc_dates >= lo) & (hc_dates < race_date)).sum())
        n_wc_recent = int(((wc_dates >= lo) & (wc_dates < race_date)).sum())
        n_hc_career = int((hc_dates < race_date).sum())
        n_wc_career = int((wc_dates < race_date).sum())

        recent_total = n_hc_recent + n_wc_recent
        career_total = n_hc_career + n_wc_career
        if recent_total < min_window_n or career_total < min_career_n:
            scores.append(float("nan"))
            continue
        recent_share = wc_share(n_wc_recent, recent_total)
        career_share = wc_share(n_wc_career, career_total)
        scores.append(share_diff(recent_share, career_share))

    out = rk[["race_id", "horse_num"]].copy()
    out["cand_score"] = scores
    return out

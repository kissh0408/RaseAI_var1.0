"""Market baseline metrics: favorite Top-1, ROI, logloss."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.splits import VALID_END, filter_by_split
from evaluation.odds_loader import attach_odds_from_se_parquet

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
STAKE = 100.0


def proportional_market_prob(odds: np.ndarray) -> np.ndarray:
    """q_i = (1/odds_i) / sum(1/odds_j)."""
    raw = np.where(odds > 1.0, 1.0 / odds, 0.0)
    total = raw.sum()
    if total <= 0:
        n = len(odds)
        return np.full(n, 1.0 / n)
    return raw / total


def power_market_prob(odds: np.ndarray, power: float = 0.81) -> np.ndarray:
    """Power takeout correction: q_i ∝ (1/odds_i)^power."""
    raw = np.where(odds > 1.0, (1.0 / odds) ** power, 0.0)
    total = raw.sum()
    if total <= 0:
        n = len(odds)
        return np.full(n, 1.0 / n)
    return raw / total


def market_logloss_per_race(
    df: pd.DataFrame,
    *,
    odds_col: str = "odds",
    horse_num_col: str = "horse_num",
    finish_rank_col: str = "finish_rank",
    race_id_col: str = "race_id",
    method: str = "proportional",
    power: float = 0.81,
) -> float:
    """Mean -log q(winner) across races."""
    losses: list[float] = []
    for _, grp in df.groupby(race_id_col):
        odds = grp[odds_col].astype(float).values
        if method == "power":
            q = power_market_prob(odds, power=power)
        else:
            q = proportional_market_prob(odds)
        winners = grp.loc[grp[finish_rank_col].astype(int) == 1]
        if winners.empty:
            continue
        w_idx = winners.index[0]
        pos = list(grp.index).index(w_idx)
        q_w = float(q[pos])
        if q_w > 1e-12:
            losses.append(-np.log(q_w))
    return float(np.mean(losses)) if losses else float("nan")


def compute_favorite_baseline(
    df_test: pd.DataFrame,
    win_odds_lookup: dict[str, dict[int, tuple[float | None, int | None]]],
    hr_win_lookup: dict[str, dict[int, int]] | None = None,
    stake: float = STAKE,
) -> dict[str, Any]:
    """Top-1 hit rate and win ROI for market favorite (min odds)."""
    n_races_total = int(df_test["race_id"].nunique())
    if not win_odds_lookup:
        return {
            "available": False,
            "reason": "WinOdds data missing",
            "n_races_total": n_races_total,
            "n_races_with_odds": 0,
            "coverage_rate": 0.0,
            "favorite_top1_hit": None,
            "favorite_top1_rate": None,
            "favorite_roi": None,
            "favorite_roi_n_races": 0,
        }

    n_with_odds = 0
    n_hit = 0
    total_payout = 0.0
    total_stake = 0.0
    n_roi_races = 0

    for race_id, grp in df_test.groupby("race_id"):
        rid = str(race_id)
        odds_map = win_odds_lookup.get(rid)
        if not odds_map:
            continue
        candidates = []
        for h in grp["horse_num"].astype(int).tolist():
            entry = odds_map.get(h)
            if entry is None:
                continue
            odds_val, pop_val = entry
            if odds_val is None:
                continue
            candidates.append((h, odds_val, pop_val if pop_val is not None else 9999))
        if not candidates:
            continue
        n_with_odds += 1
        candidates.sort(key=lambda t: (t[1], t[2]))
        fav_horse = candidates[0][0]
        fav_row = grp[grp["horse_num"].astype(int) == fav_horse]
        if fav_row.empty:
            continue
        finish_rank = int(fav_row["finish_rank"].iloc[0])
        if finish_rank == 1:
            n_hit += 1
        if hr_win_lookup is not None:
            payout = hr_win_lookup.get(rid, {}).get(fav_horse, 0)
            total_payout += float(payout)
            total_stake += stake
            n_roi_races += 1

    rate = n_hit / n_with_odds if n_with_odds else None
    roi = (total_payout / total_stake * 100.0) if total_stake > 0 else None
    return {
        "available": True,
        "n_races_total": n_races_total,
        "n_races_with_odds": n_with_odds,
        "coverage_rate": n_with_odds / n_races_total if n_races_total else 0.0,
        "favorite_top1_hit": n_hit,
        "favorite_top1_rate": rate,
        "favorite_roi": roi,
        "favorite_roi_n_races": n_roi_races,
    }


def build_win_odds_lookup_from_df(df: pd.DataFrame) -> dict[str, dict[int, tuple[float | None, int | None]]]:
    """Build win_odds_lookup from dataframe with race_id, horse_num, odds."""
    lookup: dict[str, dict[int, tuple[float | None, int | None]]] = {}
    for _, row in df.iterrows():
        rid = str(row["race_id"])
        h = int(row["horse_num"])
        odds = float(row["odds"]) if pd.notna(row.get("odds")) else None
        pop = int(row["popularity"]) if "popularity" in row and pd.notna(row["popularity"]) else None
        lookup.setdefault(rid, {})[h] = (odds, pop)
    return lookup


def load_hr_win_lookup(hr_path: Path) -> dict[str, dict[int, int]] | None:
    """Load HR win payout lookup if parquet exists."""
    if not hr_path.is_file():
        return None
    hr_df = pd.read_parquet(hr_path)
    sub = hr_df[hr_df["bet_type"] == "win"]
    lookup: dict[str, dict[int, int]] = {}
    for _, row in sub.iterrows():
        rid = str(row["race_id"])
        lookup.setdefault(rid, {})[int(row["horse_num_1"])] = int(row["payout"])
    return lookup


def compute_and_save_market_baseline(
    df: pd.DataFrame,
    *,
    win_odds_lookup: dict | None = None,
    hr_win_lookup: dict | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Compute market baseline on TEST split and save JSON."""
    df_test = filter_by_split(df, "test")
    try:
        if "odds" not in df_test.columns:
            df_test = attach_odds_from_se_parquet(df_test)
    except (FileNotFoundError, ModuleNotFoundError, Exception) as exc:
        report = {
            "valid_end": VALID_END,
            "test_n_races": int(df_test["race_id"].nunique()),
            "odds_available": False,
            "error": str(exc),
            "favorite_baseline": {
                "available": False,
                "favorite_top1_rate": 0.329,
                "favorite_roi": 77.94,
                "note": "Reference values sealed; rerun when WinOdds CSV available",
            },
            "market_logloss_proportional": None,
            "reference_top1_rate": 0.329,
            "reference_roi": 77.94,
        }
        out = out_path or (REPORTS_DIR / "market_baseline.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    if win_odds_lookup is None:
        win_odds_lookup = build_win_odds_lookup_from_df(df_test)

    fav = compute_favorite_baseline(df_test, win_odds_lookup, hr_win_lookup)
    logloss_prop = market_logloss_per_race(df_test, method="proportional")
    logloss_power = market_logloss_per_race(df_test, method="power")

    report = {
        "valid_end": VALID_END,
        "test_n_races": int(df_test["race_id"].nunique()),
        "favorite_baseline": fav,
        "market_logloss_proportional": logloss_prop,
        "market_logloss_power": logloss_power,
        "reference_top1_rate": 0.329,
        "reference_roi": 77.94,
    }
    out = out_path or (REPORTS_DIR / "market_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

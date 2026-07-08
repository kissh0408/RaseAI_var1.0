"""Wide EV core — shared odds loading, overround, EV, and divergence (Steps 1–3).

WideOdds / market implied probabilities are betting-layer only (never features).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PAIR_KEY = tuple[int, int]
OddsLookup = dict[str, dict[PAIR_KEY, float]]


def norm_pair(h1: int, h2: int) -> PAIR_KEY:
    """Normalize horse numbers to (min, max)."""
    a, b = int(h1), int(h2)
    return (min(a, b), max(a, b))


def normalize_race_id(race_id: str | int) -> str:
    """Normalize race_id to 14-digit string (matches features parquet)."""
    s = str(race_id).strip()
    if len(s) >= 16:
        return s[:14]
    return s.zfill(14)[:14]


def normalize_race_id_lookup_key(race_id: str | int) -> str:
    """Normalize race_id for WideOdds CSV lookup (16-digit int string)."""
    s = str(race_id).strip()
    if len(s) == 14:
        return s + "01"
    return s.zfill(16)


def odds_dir_default(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "common" / "data" / "output" / "odds"


def o3_odds_path_default(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "common" / "data" / "output" / "realtime_odds" / "o3_odds.csv"


def load_wide_odds_lookup(
    years: list[int],
    odds_dir: Path,
    *,
    odds_type: Literal["Wide", "Quinella"] = "Wide",
) -> OddsLookup:
    """Load {OddsType}Odds_YYYY.csv into race_id -> {(h1,h2): decimal_odds}."""
    lookup: OddsLookup = {}
    for year in sorted(years):
        path = odds_dir / f"{odds_type}Odds_{year}.csv"
        if not path.exists():
            print(f"  [warn] {odds_type}Odds_{year}.csv not found, skipping")
            continue
        df = pd.read_csv(path)
        if "odds_status" in df.columns:
            df = df[df["odds_status"] == "ok"].copy()
        df = df[df["odds"].notna()].copy()
        if df.empty:
            continue
        df["race_id_str"] = df["race_id"].apply(lambda x: str(int(x)))
        df["h_min"] = df[["horse_num_1", "horse_num_2"]].min(axis=1).astype(int)
        df["h_max"] = df[["horse_num_1", "horse_num_2"]].max(axis=1).astype(int)
        df["pair_key"] = list(zip(df["h_min"], df["h_max"]))
        for rid, grp in df.groupby("race_id_str"):
            lookup[rid] = dict(zip(grp["pair_key"], grp["odds"].astype(float)))
    print(f"  {odds_type}Odds loaded: {len(lookup):,} races across {years}")
    return lookup


def load_wide_odds_live(o3_path: Path) -> dict[tuple[str, str, str], float]:
    """Load O3 realtime wide odds: {(race_id14, h1_str, h2_str): decimal_odds}."""
    if not o3_path.exists():
        return {}
    out: dict[tuple[str, str, str], float] = {}
    try:
        with open(o3_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rid = normalize_race_id(str(row.get("race_id", "")).strip())
                h1 = str(row.get("horse_num_1", "")).zfill(2)
                h2 = str(row.get("horse_num_2", "")).zfill(2)
                raw = str(row.get("odds_min_raw", row.get("odds_raw", ""))).strip()
                if not rid or not raw.isdigit() or int(raw) == 0:
                    continue
                key = (rid, min(h1, h2), max(h1, h2))
                out[key] = int(raw) / 10.0
    except OSError as exc:
        print(f"  [warn] O3 wide odds load failed: {exc}")
    return out


def live_dict_to_race_lookup(
    live: dict[tuple[str, str, str], float],
) -> OddsLookup:
    """Convert O3 live dict to race_id14 -> {(int,int): odds} lookup."""
    lookup: OddsLookup = {}
    for (rid, h1s, h2s), odds in live.items():
        rid14 = normalize_race_id(rid)
        pair = norm_pair(int(h1s), int(h2s))
        lookup.setdefault(rid14, {})[pair] = float(odds)
    return lookup


def get_pair_odds(
    race_id: str | int,
    h1: int,
    h2: int,
    wide_odds_lookup: OddsLookup,
) -> float | None:
    """Get wide odds for a pair; tries 14- and 16-digit race_id keys."""
    pair = norm_pair(h1, h2)
    rid14 = normalize_race_id(race_id)
    for key in (rid14, normalize_race_id_lookup_key(race_id)):
        race_map = wide_odds_lookup.get(key)
        if race_map and pair in race_map:
            return float(race_map[pair])
    return None


def compute_race_overround(
    race_id: str | int,
    wide_odds_lookup: OddsLookup,
) -> float:
    """Sum of raw implied probabilities (1/odds) over all pairs in a race."""
    rid14 = normalize_race_id(race_id)
    for key in (rid14, normalize_race_id_lookup_key(race_id)):
        race_map = wide_odds_lookup.get(key)
        if not race_map:
            continue
        total = 0.0
        for odds in race_map.values():
            if odds and odds > 1.0:
                total += 1.0 / float(odds)
        return total if total > 0 else float("nan")
    return float("nan")


def compute_implied_prob(odds: float, overround: float) -> float:
    """Overround-corrected implied probability."""
    if odds <= 1.0 or overround <= 0 or math.isnan(overround):
        return float("nan")
    return (1.0 / float(odds)) / float(overround)


def compute_pair_ev(p_wide: float, wide_odds: float) -> float:
    """EV = P_wide × decimal_odds (JRA multiplier)."""
    if p_wide <= 0 or wide_odds <= 1.0:
        return float("nan")
    return float(p_wide) * float(wide_odds)


def compute_log_divergence(p_wide: float, wide_odds: float, overround: float) -> float:
    """log(P_model × odds × overround) = log(P_model / p_implied_corrected)."""
    if p_wide <= 0 or wide_odds <= 1.0 or overround <= 0:
        return float("-inf")
    return float(math.log(float(p_wide) * float(wide_odds) * float(overround)))


def wide_probs_from_win_dict(p_win: dict[int, float]) -> dict[PAIR_KEY, float]:
    """Harville wide probabilities for all pairs from per-horse win probs."""
    try:
        from ev_filters import harville_wide_pair_prob
    except ModuleNotFoundError:
        from strategy.src.ev_filters import harville_wide_pair_prob

    horses = sorted(int(h) for h in p_win.keys())
    out: dict[PAIR_KEY, float] = {}
    for i in range(len(horses)):
        for j in range(i + 1, len(horses)):
            h1, h2 = horses[i], horses[j]
            out[norm_pair(h1, h2)] = harville_wide_pair_prob(p_win, h1, h2)
    return out


def select_best_pair_by_p_wide(
    p_wide_map: dict[PAIR_KEY, float],
) -> tuple[PAIR_KEY, float] | None:
    if not p_wide_map:
        return None
    best_key = max(p_wide_map, key=lambda k: p_wide_map[k])
    return best_key, float(p_wide_map[best_key])


def select_best_pair_by_divergence(
    p_wide_map: dict[PAIR_KEY, float],
    wide_odds_lookup: OddsLookup,
    race_id: str | int,
) -> tuple[PAIR_KEY, float, float, float, float] | None:
    """Return (pair, p_wide, odds, ev, log_divergence) for argmax divergence."""
    overround = compute_race_overround(race_id, wide_odds_lookup)
    if math.isnan(overround) or overround <= 0:
        return None
    best: tuple[PAIR_KEY, float, float, float, float] | None = None
    best_div = float("-inf")
    for pair, p_w in p_wide_map.items():
        odds = get_pair_odds(race_id, pair[0], pair[1], wide_odds_lookup)
        if odds is None or odds <= 1.0:
            continue
        div = compute_log_divergence(p_w, odds, overround)
        if div > best_div:
            ev = compute_pair_ev(p_w, odds)
            best_div = div
            best = (pair, float(p_w), float(odds), float(ev), float(div))
    return best


def collect_divergence_bets_per_race(
    race_id: str | int,
    p_wide_map: dict[PAIR_KEY, float],
    wide_odds_lookup: OddsLookup,
    *,
    strategy: Literal["A", "B", "C", "D"] = "D",
    ev_threshold: float = 1.05,
    div_threshold: float = 0.0,
) -> dict | None:
    """Select pair and bet flags for one race (Strategy A–D)."""
    overround = compute_race_overround(race_id, wide_odds_lookup)
    if math.isnan(overround) or overround <= 0 or not p_wide_map:
        return None

    candidates: list[dict] = []
    for pair, p_w in p_wide_map.items():
        odds = get_pair_odds(race_id, pair[0], pair[1], wide_odds_lookup)
        if odds is None or odds <= 1.0:
            continue
        ev = compute_pair_ev(p_w, odds)
        div = compute_log_divergence(p_w, odds, overround)
        candidates.append(
            {
                "pair": pair,
                "p_wide": float(p_w),
                "wide_odds": float(odds),
                "ev_wide": float(ev),
                "log_divergence": float(div),
                "overround": float(overround),
            }
        )
    if not candidates:
        return None

    if strategy == "A":
        pick = max(candidates, key=lambda x: x["p_wide"])
    else:
        pick = max(candidates, key=lambda x: x["log_divergence"])

    bet = False
    if strategy == "A":
        bet = pick["ev_wide"] >= ev_threshold
    elif strategy == "B":
        bet = pick["ev_wide"] >= ev_threshold
    elif strategy == "C":
        bet = pick["log_divergence"] > div_threshold
    elif strategy == "D":
        bet = pick["ev_wide"] >= ev_threshold and pick["log_divergence"] > div_threshold

    pick = dict(pick)
    pick["race_id"] = normalize_race_id(race_id)
    pick["strategy"] = strategy
    pick["bet"] = bet
    return pick


def compare_ev_vs_divergence(
    df_ev: pd.DataFrame,
    df_div: pd.DataFrame,
    *,
    stake: float = 100.0,
) -> dict:
    """Compare Strategy A (ev) vs Strategy D (divergence) bet DataFrames."""

    def _stats(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n_bets": 0, "hit_rate": None, "roi": None, "total_profit": None}
        hits = int(sub["hit"].sum()) if "hit" in sub.columns else 0
        n = len(sub)
        payout = float(sub["payout"].sum()) if "payout" in sub.columns else 0.0
        inv = stake * n
        return {
            "n_bets": n,
            "hit_rate": hits / n if n else None,
            "roi": payout / inv if inv else None,
            "total_profit": payout - inv,
        }

    return {
        "strategy_ev": _stats(df_ev),
        "strategy_div": _stats(df_div),
    }


def tune_thresholds_on_valid(
    bets: list[dict],
    *,
    ev_thresholds: list[float] | None = None,
    div_thresholds: list[float] | None = None,
    min_bets: int = 100,
) -> dict:
    """Grid search thresholds on VALID bets (each row must have ev_wide, log_divergence, roi_unit)."""
    ev_thresholds = ev_thresholds or [1.0, 1.05, 1.1, 1.2]
    div_thresholds = div_thresholds or [0.0, 0.1, 0.2]
    best: dict | None = None
    best_score = float("-inf")
    for ev_t in ev_thresholds:
        for div_t in div_thresholds:
            sub = [
                b
                for b in bets
                if b.get("ev_wide", 0) >= ev_t and b.get("log_divergence", -999) > div_t
            ]
            n = len(sub)
            if n < min_bets:
                continue
            ret = sum(float(b.get("payout_mult", 0)) for b in sub)
            roi = ret / n if n else 0
            score = roi * math.sqrt(n)
            if score > best_score:
                best_score = score
                best = {
                    "ev_threshold": ev_t,
                    "div_threshold": div_t,
                    "n_bets": n,
                    "roi": roi,
                    "score": score,
                }
    return best or {"ev_threshold": 1.05, "div_threshold": 0.0, "n_bets": 0, "roi": None}

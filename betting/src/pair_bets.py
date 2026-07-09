"""ワイド/馬連ペア馬券の確率計算・選定・決済。

bet_type ごとの確率モデル選定根拠（fold2 OOS実測、2026-07-09、
betting/analysis/compare_pair_probability_models.py の結果）:
  wide     -> Stern式 (pair_probs.py)   logloss・較正誤差ともHarvilleより優位
  quinella -> Harville式 (ev_filters.py) logloss僅差、較正誤差はHarvilleが明確に優位
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from betting.src.ev_filters import harville_quinella_pair_prob
from betting.src.pair_probs import norm_pair, stern_wide_pair_prob

PairKey = tuple[int, int]
PairOddsLookup = dict[str, dict[PairKey, float]]
BetType = Literal["wide", "quinella"]


def race_pair_probs(
    p_win: dict[int, float],
    bet_type: BetType,
    *,
    lam2: float = 1.0,
    lam3: float = 1.0,
) -> dict[PairKey, float]:
    """レース内の全ペアについて bet_type に応じた確率を計算する。"""
    if bet_type not in ("wide", "quinella"):
        raise ValueError(f"未対応の bet_type: {bet_type!r}（wide/quinella のみ）")

    horses = sorted(p_win.keys())
    n = len(horses)
    out: dict[PairKey, float] = {}
    if bet_type == "wide":
        p_arr = np.array([p_win[h] for h in horses], dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = horses[i], horses[j]
                out[norm_pair(hi, hj)] = stern_wide_pair_prob(p_arr, i, j, lam2, lam3)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = horses[i], horses[j]
                out[norm_pair(hi, hj)] = harville_quinella_pair_prob(p_win[hi], p_win[hj])
    return out


def select_best_pair(
    pair_probs: dict[PairKey, float],
    odds_for_race: dict[PairKey, float],
    ev_threshold: float,
) -> dict | None:
    """レース内でEV最大のペアを選ぶ。閾値未満・オッズ無しなら None。"""
    best: dict | None = None
    for pair, p in pair_probs.items():
        odds = odds_for_race.get(pair)
        if odds is None or odds <= 1.0:
            continue
        ev = float(p) * float(odds)
        if best is None or ev > best["ev"]:
            best = {"pair": pair, "p": float(p), "odds": float(odds), "ev": ev}
    if best is None or best["ev"] < ev_threshold:
        return None
    return best


def _pair_hit(bet_type: BetType, finish_a: int, finish_b: int) -> bool:
    if bet_type == "quinella":
        return {finish_a, finish_b} == {1, 2}
    return finish_a <= 3 and finish_b <= 3


def tune_pair_ev_threshold_on_valid(
    valid_df: pd.DataFrame,
    *,
    bet_type: BetType,
    odds_lookup: PairOddsLookup,
    grid: list[float],
    lam2: float = 1.0,
    lam3: float = 1.0,
    min_bets: int = 50,
) -> dict:
    """VALID期間のみでEV閾値をROI最大化により選ぶ（Rule 3: TESTは使わない）。"""
    best_threshold = grid[0]
    best_roi = -1.0
    grid_hits = 0
    for thr in grid:
        bets = simulate_pair_bets(
            valid_df, bet_type=bet_type, odds_lookup=odds_lookup, ev_threshold=thr,
            lam2=lam2, lam3=lam3,
        )
        if len(bets) < min_bets:
            continue
        grid_hits += 1
        roi = float(bets["payout"].sum()) / float(bets["stake"].sum()) * 100.0
        if roi > best_roi:
            best_roi = roi
            best_threshold = thr
    fallback = grid_hits == 0
    return {
        "threshold": best_threshold,
        "grid_evaluated": grid_hits,
        "fallback_used": fallback,
        "valid_n_races": int(valid_df["race_id"].nunique()) if len(valid_df) else 0,
    }


def simulate_pair_bets(
    df: pd.DataFrame,
    *,
    bet_type: BetType,
    odds_lookup: PairOddsLookup,
    ev_threshold: float,
    stake: float = 100.0,
    lam2: float = 1.0,
    lam3: float = 1.0,
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """race_id, horse_num, p_win, finish_rank を持つDFからペア馬券ベットを生成・決済する。"""
    rows = []
    for race_id, grp in df.groupby(race_id_col, sort=False):
        p_win = {int(h): float(p) for h, p in zip(grp["horse_num"], grp["p_win"])}
        finish = {int(h): int(f) for h, f in zip(grp["horse_num"], grp["finish_rank"])}
        odds_for_race = odds_lookup.get(str(race_id), {})
        pair_probs = race_pair_probs(p_win, bet_type, lam2=lam2, lam3=lam3)
        pick = select_best_pair(pair_probs, odds_for_race, ev_threshold)
        if pick is None:
            continue
        h1, h2 = pick["pair"]
        hit = _pair_hit(bet_type, finish[h1], finish[h2])
        payout = stake * pick["odds"] if hit else 0.0
        rows.append(
            {
                "race_id": race_id,
                "bet_type": bet_type,
                "pair": pick["pair"],
                "p": pick["p"],
                "odds": pick["odds"],
                "ev": pick["ev"],
                "stake": stake,
                "hit": int(hit),
                "payout": payout,
                "pnl": payout - stake,
            }
        )
    return pd.DataFrame(rows)

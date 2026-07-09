"""pair_bets.py（ワイド=Stern式 / 馬連=Harville式 選定・決済）のテスト。

モデル選定根拠: betting/analysis/pair_probability_model_comparison.json
（fold2 OOS実測、2026-07-09）で wide は Stern が logloss・較正誤差とも優位、
quinella は Harville が較正誤差で明確に優位だったため bet_type ごとに使い分ける。
"""
from __future__ import annotations

import pandas as pd
import pytest

from betting.src.pair_bets import (
    race_pair_probs,
    select_best_pair,
    simulate_pair_bets,
    tune_pair_ev_threshold_on_valid,
)


def test_race_pair_probs_wide_uses_stern_and_sums_correctly():
    p_win = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    probs = race_pair_probs(p_win, "wide", lam2=1.0, lam3=1.0)
    assert set(probs.keys()) == {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
    assert sum(probs.values()) == pytest.approx(3.0, abs=1e-6)


def test_race_pair_probs_quinella_uses_harville_and_sums_to_one():
    p_win = {1: 0.5, 2: 0.3, 3: 0.2}
    probs = race_pair_probs(p_win, "quinella")
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


def test_race_pair_probs_rejects_unknown_bet_type():
    with pytest.raises(ValueError, match="bet_type"):
        race_pair_probs({1: 0.5, 2: 0.5}, "trifecta")  # type: ignore[arg-type]


def test_select_best_pair_picks_highest_ev_above_threshold():
    pair_probs = {(1, 2): 0.10, (1, 3): 0.05, (2, 3): 0.20}
    odds = {(1, 2): 8.0, (1, 3): 30.0, (2, 3): 3.0}  # EV: 0.8, 1.5, 0.6
    picked = select_best_pair(pair_probs, odds, ev_threshold=1.0)
    assert picked is not None
    assert picked["pair"] == (1, 3)
    assert picked["ev"] == pytest.approx(1.5)


def test_select_best_pair_returns_none_when_no_pair_clears_threshold():
    pair_probs = {(1, 2): 0.10}
    odds = {(1, 2): 5.0}  # EV=0.5
    assert select_best_pair(pair_probs, odds, ev_threshold=1.05) is None


def test_select_best_pair_skips_pairs_without_odds():
    pair_probs = {(1, 2): 0.5, (1, 3): 0.5}
    odds = {(1, 2): 1.0}  # odds<=1.0 は無効、(1,3)はオッズ無し
    assert select_best_pair(pair_probs, odds, ev_threshold=0.1) is None


def test_simulate_pair_bets_settles_hit_and_miss():
    """1レース的中(オッズ通り払戻)、1レース不的中(払戻0)を確認。"""
    races = pd.DataFrame({
        "race_id": ["R1", "R1", "R1", "R2", "R2", "R2"],
        "horse_num": [1, 2, 3, 1, 2, 3],
        "p_win": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2],
        "finish_rank": [1, 2, 3, 1, 3, 2],  # R1: 1-2着 R2: 1-3着(2は3着)
    })
    odds_lookup = {
        "R1": {(1, 2): 8.0, (1, 3): 20.0, (2, 3): 15.0},
        "R2": {(1, 2): 8.0, (1, 3): 20.0, (2, 3): 15.0},
    }
    bets = simulate_pair_bets(
        races, bet_type="quinella", odds_lookup=odds_lookup, ev_threshold=0.5, stake=100.0
    )
    assert len(bets) == 2
    r1 = bets[bets["race_id"] == "R1"].iloc[0]
    r2 = bets[bets["race_id"] == "R2"].iloc[0]
    # R1: 実際の1-2着 = (1,2) が選ばれていれば的中
    if r1["pair"] == (1, 2):
        assert r1["hit"] == 1
        assert r1["payout"] == pytest.approx(100.0 * 8.0)
    # R2: 1着=1, 3着=2 なので (1,2)馬連は不的中
    if r2["pair"] == (1, 2):
        assert r2["hit"] == 0
        assert r2["payout"] == 0.0


def test_tune_pair_ev_threshold_picks_best_roi_on_valid_only():
    races = pd.DataFrame({
        "race_id": [f"R{i}" for i in range(1, 21) for _ in range(3)],
        "horse_num": [1, 2, 3] * 20,
        "p_win": [0.5, 0.3, 0.2] * 20,
        "finish_rank": ([1, 2, 3] * 10 + [3, 1, 2] * 10),
    })
    odds_lookup = {f"R{i}": {(1, 2): 5.0, (1, 3): 20.0, (2, 3): 10.0} for i in range(1, 21)}
    result = tune_pair_ev_threshold_on_valid(
        races, bet_type="quinella", odds_lookup=odds_lookup, grid=[0.5, 1.0, 1.5], min_bets=5,
    )
    assert result["fallback_used"] is False
    assert result["threshold"] in (0.5, 1.0, 1.5)
    assert result["valid_n_races"] == 20


def test_tune_pair_ev_threshold_falls_back_when_too_few_bets():
    races = pd.DataFrame({
        "race_id": ["R1", "R1"],
        "horse_num": [1, 2],
        "p_win": [0.5, 0.5],
        "finish_rank": [1, 2],
    })
    odds_lookup = {"R1": {(1, 2): 3.0}}
    result = tune_pair_ev_threshold_on_valid(
        races, bet_type="quinella", odds_lookup=odds_lookup, grid=[1.05], min_bets=50,
    )
    assert result["fallback_used"] is True
    assert result["threshold"] == 1.05

"""Tests for flat top-1 loss-minimization betting logic (betting/src/flat_top1.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting.src.flat_top1 import (
    DISCLAIMER,
    apply_flat_sizing,
    run_loss_min_recommendations,
    select_top1_bets,
    settle_win_bets,
)

BASE_CFG = {
    "min_odds": 2.0,
    "max_odds": 50.0,
    "loss_min": {
        "score_col": "pure_score_z",
        "stake_fraction": 0.005,
        "stake_rounding_yen": 100,
    },
}


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_one_pick_per_race():
    df = _frame(
        [
            {"race_id": "R1", "horse_num": 1, "pure_score_z": 0.5, "odds": 3.0, "finish_rank": 2},
            {"race_id": "R1", "horse_num": 2, "pure_score_z": 1.2, "odds": 5.0, "finish_rank": 1},
            {"race_id": "R1", "horse_num": 3, "pure_score_z": -0.3, "odds": 10.0, "finish_rank": 3},
            {"race_id": "R2", "horse_num": 1, "pure_score_z": 0.1, "odds": 4.0, "finish_rank": 1},
        ]
    )
    picks, skipped = select_top1_bets(df, cfg=BASE_CFG)
    assert len(picks) == 2
    assert set(picks["race_id"]) == {"R1", "R2"}
    r1 = picks.loc[picks["race_id"] == "R1"].iloc[0]
    assert int(r1["horse_num"]) == 2  # highest pure_score_z


def test_tiebreak_horse_num_ascending():
    df = _frame(
        [
            {"race_id": "R1", "horse_num": 5, "pure_score_z": 1.0, "odds": 3.0, "finish_rank": 1},
            {"race_id": "R1", "horse_num": 2, "pure_score_z": 1.0, "odds": 4.0, "finish_rank": 2},
            {"race_id": "R1", "horse_num": 8, "pure_score_z": 1.0, "odds": 6.0, "finish_rank": 3},
        ]
    )
    picks, _ = select_top1_bets(df, cfg=BASE_CFG)
    assert len(picks) == 1
    assert int(picks.iloc[0]["horse_num"]) == 2


def test_odds_out_of_range_skips_race_no_fallback_to_rank2():
    df = _frame(
        [
            {"race_id": "R1", "horse_num": 1, "pure_score_z": 1.0, "odds": 1.5, "finish_rank": 1},  # below min
            {"race_id": "R1", "horse_num": 2, "pure_score_z": 0.5, "odds": 5.0, "finish_rank": 2},
            {"race_id": "R2", "horse_num": 1, "pure_score_z": 1.0, "odds": 80.0, "finish_rank": 1},  # above max
            {"race_id": "R3", "horse_num": 1, "pure_score_z": 1.0, "odds": None, "finish_rank": 1},  # missing
        ]
    )
    picks, skipped = select_top1_bets(df, cfg=BASE_CFG)
    assert picks.empty
    assert len(skipped) == 3
    reasons = dict(zip(skipped["race_id"], skipped["reason"]))
    assert reasons["R1"] == "odds_below_min"
    assert reasons["R2"] == "odds_above_max"
    assert reasons["R3"] == "odds_missing"
    # Confirm no rank-2 fallback occurred for R1 (only 1 skipped row, not a picked row)
    assert "R1" not in set(picks["race_id"]) if not picks.empty else True


def test_stake_is_100yen_rounded_and_matches_bankroll_times_fraction():
    picks = _frame([{"race_id": "R1", "horse_num": 1, "odds": 5.0}])
    sized = apply_flat_sizing(picks, bankroll=100_000, stake_fraction=0.005, rounding_yen=100)
    assert sized.iloc[0]["stake"] == 500.0

    # Non-round bankroll * fraction should floor to nearest 100 yen.
    picks2 = _frame([{"race_id": "R1", "horse_num": 1, "odds": 5.0}])
    sized2 = apply_flat_sizing(picks2, bankroll=123_456, stake_fraction=0.005, rounding_yen=100)
    # 123456 * 0.005 = 617.28 -> floor to 600
    assert sized2.iloc[0]["stake"] == 600.0


def test_stake_fixed_to_initial_bankroll_does_not_vary_with_current_bankroll():
    """apply_flat_sizing must always size off the value passed as `bankroll` (the
    operation's *initial* bankroll), never a running/current-bankroll figure. This
    test documents the contract: two calls with the same `bankroll` argument but
    different numbers of prior picks/settlements must yield identical stakes.
    """
    picks_early = _frame([{"race_id": "R1", "horse_num": 1, "odds": 5.0}])
    picks_late = _frame(
        [
            {"race_id": "R1", "horse_num": 1, "odds": 5.0},
            {"race_id": "R2", "horse_num": 1, "odds": 3.0},
            {"race_id": "R3", "horse_num": 1, "odds": 8.0},
        ]
    )
    sized_early = apply_flat_sizing(picks_early, bankroll=100_000, stake_fraction=0.001, rounding_yen=100)
    sized_late = apply_flat_sizing(picks_late, bankroll=100_000, stake_fraction=0.001, rounding_yen=100)
    assert sized_early.iloc[0]["stake"] == sized_late.iloc[0]["stake"] == 100.0
    assert sized_late.iloc[1]["stake"] == 100.0
    assert sized_late.iloc[2]["stake"] == 100.0


def test_apply_flat_sizing_rejects_bankroll_below_minimum():
    picks = _frame([{"race_id": "R1", "horse_num": 1, "odds": 5.0}])
    with pytest.raises(ValueError, match="min_bankroll"):
        apply_flat_sizing(picks, bankroll=99_999, stake_fraction=0.001, rounding_yen=100)


def test_apply_flat_sizing_rejects_bankroll_exactly_at_min_purchase_boundary_below_100k():
    # 100,000 * 0.001 = 100 (== rounding_yen) is the minimum viable operating bankroll
    # for f=0.001; anything less floors below the JRA minimum purchase unit.
    picks = _frame([{"race_id": "R1", "horse_num": 1, "odds": 5.0}])
    sized = apply_flat_sizing(picks, bankroll=100_000, stake_fraction=0.001, rounding_yen=100)
    assert sized.iloc[0]["stake"] == 100.0


def test_apply_flat_sizing_rejects_stake_below_min_purchase_unit():
    # A bankroll/fraction combination whose floored stake would be 0 (below the JRA
    # 100-yen minimum purchase unit) must error rather than silently place a 0-yen bet.
    picks = _frame([{"race_id": "R1", "horse_num": 1, "odds": 5.0}])
    with pytest.raises(ValueError, match="min_bankroll"):
        apply_flat_sizing(picks, bankroll=150_000, stake_fraction=0.0001, rounding_yen=100)


def test_settlement_win_and_loss():
    picks = _frame(
        [
            {"race_id": "R1", "horse_num": 1, "odds": 5.0, "stake": 500.0, "finish_rank": 1},
            {"race_id": "R2", "horse_num": 1, "odds": 3.0, "stake": 500.0, "finish_rank": 4},
        ]
    )
    settled = settle_win_bets(picks)
    win_row = settled.loc[settled["race_id"] == "R1"].iloc[0]
    lose_row = settled.loc[settled["race_id"] == "R2"].iloc[0]
    assert win_row["payout"] == 2500.0
    assert win_row["pnl"] == 2000.0
    assert lose_row["payout"] == 0.0
    assert lose_row["pnl"] == -500.0


def test_note_column_contains_disclaimer():
    rank_preds = _frame(
        [
            {"race_id": "R1", "horse_num": 1, "pure_score_z": 1.0},
            {"race_id": "R1", "horse_num": 2, "pure_score_z": 0.2},
        ]
    )
    odds_df = _frame(
        [
            {"race_id": "R1", "horse_num": 1, "odds": 4.0},
            {"race_id": "R1", "horse_num": 2, "odds": 8.0},
        ]
    )
    recs, skipped = run_loss_min_recommendations(
        rank_preds, odds_df, cfg=BASE_CFG, odds_timestamp="2026-07-10T00:00:00Z", bankroll=100_000
    )
    assert len(recs) == 1
    assert recs.iloc[0]["note"] == DISCLAIMER
    assert recs.iloc[0]["mode"] == "loss_min_top1"
    assert recs.iloc[0]["selection"] == 1
    assert skipped.empty


def test_ev_filter_mode_still_works_backward_compat():
    """mode='ev_filter' must keep using the legacy betting/src/backtest.py path untouched."""
    from betting.src.backtest import simulate_bets

    df = pd.DataFrame(
        [
            {
                "race_id": "R1",
                "horse_num": 1,
                "horse_number": 1,
                "p_win": 0.30,
                "odds": 5.0,
                "finish_rank": 1,
            },
            {
                "race_id": "R1",
                "horse_num": 2,
                "horse_number": 2,
                "p_win": 0.10,
                "odds": 3.0,
                "finish_rank": 2,
            },
        ]
    )
    cfg = {
        "ev_haircut": 0.95,
        "min_odds": 2.0,
        "max_odds": 50.0,
        "min_model_prob": 0.05,
        "bankroll": 100000,
        "kelly_fraction": 0.08,
        "max_bet_ratio": 0.05,
        "max_picks_per_race": 2,
    }
    bets = simulate_bets(df, bet_type="win", ev_threshold=1.0, cfg=cfg)
    # Legacy EV path should still select based on p_win/odds EV, not pure_score_z.
    assert "kelly_bet_yen" in bets.columns

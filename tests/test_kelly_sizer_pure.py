"""kelly_sizer 純粋関数の characterization テスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.src.kelly_sizer import apply_kelly_sizing, kelly_bet_amount, kelly_fraction


class TestKellyFraction:
    def test_known_prob_odds(self):
        # 観測値: p=0.35, odds=5.0 → fractional kelly = 0.015
        assert np.isclose(kelly_fraction(0.35, 5.0), 0.015)

    def test_positive_edge_at_higher_odds(self):
        assert np.isclose(kelly_fraction(0.2, 8.0), 0.0068571428571428585)

    def test_negative_kelly_is_zero(self):
        assert kelly_fraction(0.05, 2.0) == 0.0

    def test_capped_at_max_bet_ratio(self):
        frac = kelly_fraction(0.80, 2.0, kelly_frac=0.08, max_bet_ratio=0.05)
        assert frac <= 0.05

    def test_zero_when_odds_one_or_less(self):
        assert kelly_fraction(0.5, 1.0) == 0.0


class TestKellyBetAmount:
    def test_matches_fraction_times_bankroll_rounded_down(self):
        assert kelly_bet_amount(0.35, 5.0, 100_000.0) == 1400.0


class TestApplyKellySizing:
    def test_vectorized_matches_scalar(self):
        df = pd.DataFrame({"model_prob": [0.35], "odds": [5.0]})
        out = apply_kelly_sizing(df, bankroll=100_000.0)
        expected_ratio = kelly_fraction(0.35, 5.0)
        assert np.isclose(out["kelly_ratio"].iloc[0], expected_ratio)
        assert out["kelly_bet_yen"].iloc[0] == 1400.0

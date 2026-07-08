"""Tests for betting kelly_sizer (ported from tests/test_kelly_sizer_pure.py)."""

from __future__ import annotations

from betting.src.kelly_sizer import kelly_bet_amount, kelly_fraction


def test_kelly_zero_for_negative_edge():
    assert kelly_fraction(0.1, 2.0) == 0.0


def test_kelly_positive_edge():
    f = kelly_fraction(0.5, 3.0, kelly_frac=0.08, max_bet_ratio=0.05)
    assert 0 < f <= 0.05


def test_kelly_bet_amount_rounds_to_100():
    amt = kelly_bet_amount(0.5, 3.0, bankroll=100000)
    assert amt >= 0
    assert amt % 100 == 0

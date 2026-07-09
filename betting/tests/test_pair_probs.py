"""pair_probs.py（Stern式ワイド/馬連ペア確率）の単体テスト。"""
from __future__ import annotations

import numpy as np
import pytest

from betting.src.pair_probs import (
    all_pair_probs,
    calibration_max_error_pp,
    norm_pair,
    stern_quinella_pair_prob,
    stern_wide_pair_prob,
)


def test_norm_pair_is_order_invariant():
    assert norm_pair(3, 1) == (1, 3)
    assert norm_pair(1, 3) == (1, 3)


def test_quinella_pair_prob_symmetric_in_i_j():
    p = np.array([0.5, 0.3, 0.2])
    assert stern_quinella_pair_prob(p, 0, 1, lam2=1.0) == pytest.approx(
        stern_quinella_pair_prob(p, 1, 0, lam2=1.0)
    )


def test_quinella_probs_sum_to_one_for_three_horses():
    """3頭レースでは top-2 の組は必ずどれか1組なので、全ペアの馬連確率合計は1.0。"""
    p = np.array([0.5, 0.3, 0.2])
    total = (
        stern_quinella_pair_prob(p, 0, 1, lam2=1.0)
        + stern_quinella_pair_prob(p, 0, 2, lam2=1.0)
        + stern_quinella_pair_prob(p, 1, 2, lam2=1.0)
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_quinella_equal_probs_matches_harville_formula():
    """3頭均等(1/3)ならどのペアも馬連確率は1/3（Harville解析解と一致、lam2=1）。"""
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    got = stern_quinella_pair_prob(p, 0, 1, lam2=1.0)
    assert got == pytest.approx(1 / 3, abs=1e-9)


def test_wide_pair_prob_at_least_quinella_for_three_plus_horses():
    """ワイド（3着内）は馬連（1-2着）を包含する条件なので必ず以上になる。"""
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = stern_quinella_pair_prob(p, 0, 1, lam2=1.0)
    w = stern_wide_pair_prob(p, 0, 1, lam2=1.0, lam3=1.0)
    assert w >= q - 1e-9


def test_wide_probs_sum_to_three_for_four_horses():
    """4頭レースでは3着以内に入る2頭組は必ず3組（3C2 - 1組を除く数の期待値=3）になる。

    厳密には各レースで実現する3着以内ペアはC(3,2)=3組なので、
    全ペアのワイド確率合計の期待値は3.0に収束するはず。
    """
    p = np.array([0.4, 0.3, 0.2, 0.1])
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    total = sum(stern_wide_pair_prob(p, i, j, lam2=1.0, lam3=1.0) for i, j in pairs)
    assert total == pytest.approx(3.0, abs=1e-6)


def test_all_pair_probs_matches_pairwise_functions():
    p = np.array([0.5, 0.3, 0.2])
    horse_nums = [1, 2, 3]
    out = all_pair_probs(p, horse_nums, lam2=1.0, lam3=1.0)
    assert set(out.keys()) == {(1, 2), (1, 3), (2, 3)}
    assert out[(1, 2)]["quinella"] == pytest.approx(
        stern_quinella_pair_prob(p, 0, 1, lam2=1.0)
    )
    assert out[(1, 2)]["wide"] == pytest.approx(
        stern_wide_pair_prob(p, 0, 1, lam2=1.0, lam3=1.0)
    )


def test_all_pair_probs_normalizes_input_win_probs():
    """p_win の合計が1でなくても内部で正規化される。"""
    p = np.array([2.0, 1.0, 1.0])  # sum=4, 正規化後 [0.5, 0.25, 0.25]
    out = all_pair_probs(p, [1, 2, 3], lam2=1.0, lam3=1.0)
    normalized = np.array([0.5, 0.25, 0.25])
    expected = stern_quinella_pair_prob(normalized, 0, 1, lam2=1.0)
    assert out[(1, 2)]["quinella"] == pytest.approx(expected)


def test_calibration_max_error_pp_perfect_calibration_is_zero():
    rng = np.random.default_rng(42)
    predicted = rng.uniform(0, 1, 5000)
    actual = (rng.uniform(0, 1, 5000) < predicted).astype(float)
    err = calibration_max_error_pp(predicted, actual, n_bins=10)
    assert err < 5.0  # サンプリング誤差の範囲内


def test_calibration_max_error_pp_detects_miscalibration():
    predicted = np.array([0.9] * 100)
    actual = np.array([0.0] * 100)  # 予測0.9なのに実現率0%
    err = calibration_max_error_pp(predicted, actual, n_bins=10)
    assert err == pytest.approx(90.0, abs=1.0)


def test_calibration_max_error_pp_empty_returns_nan():
    assert np.isnan(calibration_max_error_pp(np.array([]), np.array([])))

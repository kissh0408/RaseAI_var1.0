"""ev_filters.py（Harville式ワイド/馬連ペア確率）の単体テスト。

pair_probs.py（Stern式）と同じ不変条件で検証し、両モデルの健全性を並行担保する。
"""
from __future__ import annotations

import pytest

from betting.src.ev_filters import harville_quinella_pair_prob, harville_wide_pair_prob


def test_quinella_symmetric_in_i_j():
    assert harville_quinella_pair_prob(0.5, 0.3) == pytest.approx(
        harville_quinella_pair_prob(0.3, 0.5)
    )


def test_quinella_probs_sum_to_one_for_three_horses():
    p = {1: 0.5, 2: 0.3, 3: 0.2}
    total = (
        harville_quinella_pair_prob(p[1], p[2])
        + harville_quinella_pair_prob(p[1], p[3])
        + harville_quinella_pair_prob(p[2], p[3])
    )
    assert total == pytest.approx(1.0, abs=1e-9)


def test_quinella_equal_probs_matches_one_third():
    p = 1 / 3
    assert harville_quinella_pair_prob(p, p) == pytest.approx(1 / 3, abs=1e-9)


def test_wide_at_least_quinella():
    p = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    q = harville_quinella_pair_prob(p[1], p[2])
    w = harville_wide_pair_prob(p, 1, 2)
    assert w >= q - 1e-9


def test_wide_probs_sum_to_three_for_four_horses():
    """4頭レースでは3着以内ペアは常に3組 → 全ペアのワイド確率合計は3.0（分解が正しければ）。"""
    p = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    total = sum(harville_wide_pair_prob(p, h1, h2) for h1, h2 in pairs)
    assert total == pytest.approx(3.0, abs=1e-6)


def test_wide_probs_sum_to_six_for_five_horses():
    """5頭なら3着以内ペアは常にC(3,2)=3組 × ... 実際には5頭でも3組(3着以内3頭からの組)。

    C(3,2)=3 は着順に入る頭数(3)に依存するため、頭数が増えても3のまま。
    """
    p = {1: 0.35, 2: 0.25, 3: 0.2, 4: 0.12, 5: 0.08}
    horses = list(p.keys())
    pairs = [(horses[i], horses[j]) for i in range(len(horses)) for j in range(i + 1, len(horses))]
    total = sum(harville_wide_pair_prob(p, h1, h2) for h1, h2 in pairs)
    assert total == pytest.approx(3.0, abs=1e-6)

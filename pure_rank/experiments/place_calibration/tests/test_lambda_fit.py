"""λ logloss fit のテスト（仕様書 §8）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from calib_lib import (  # noqa: E402
    band_5to7_8plus,
    fit_lambda_logloss,
    fit_lambda_logloss_banded,
    logloss_objective,
)

from prob_fusion.src.place_prob import place_prob_from_p_win  # noqa: E402


def _hand_logloss(p_place: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p_place, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_logloss_objective_value():
    """小さな合成レース集合で目的関数値が手計算 logloss と一致する。"""
    rng = np.random.default_rng(1)
    races_p_win = []
    races_y = []
    for _ in range(5):
        n = rng.integers(5, 9)
        raw = rng.uniform(0.3, 2.0, size=n)
        p_win = raw / raw.sum()
        y = (rng.uniform(size=n) < 0.3).astype(float)
        races_p_win.append(p_win)
        races_y.append(y)

    lam2, lam3 = 0.7, 1.2
    got = logloss_objective(np.array([lam2, lam3]), races_p_win, races_y)

    hand_losses = []
    for p_win, y in zip(races_p_win, races_y):
        p_place = place_prob_from_p_win(p_win, lam2, lam3)
        hand_losses.append(_hand_logloss(p_place, y))
    expected = float(np.mean(hand_losses))

    assert abs(got - expected) < 1e-9


def test_recovers_known_lambda():
    """既知 λ で生成した合成データ（soft label = 真の place 確率）から
    logloss fit が λ を近似回復することを確認する。

    注意（実測に基づく設計判断）: Stern の place 確率は λ に対して非常に平坦な
    loss landscape を持つ（実測: 40〜300レースの合成データで真値と全く異なる λ
    （例 lam2=1.05 vs 真値0.8）でも loss差が1e-4未満）。そのため λ 自体の
    ピンポイント一致ではなく、(1) fit が真の λ における loss 以下（同等）の
    loss に到達すること（最適化の正しさ）、(2) 真の λ 近傍から出発した場合は
    ほぼ真値に留まること、の2点で検証する。許容誤差はこの実測に基づき設定。
    """
    rng = np.random.default_rng(0)
    true_lam2, true_lam3 = 0.8, 1.3
    races_p_win = []
    races_y = []
    for _ in range(300):
        n = rng.integers(6, 16)
        raw = rng.uniform(0.1, 5.0, size=n)
        p_win = raw / raw.sum()
        y_true = place_prob_from_p_win(p_win, true_lam2, true_lam3)
        races_p_win.append(p_win)
        races_y.append(y_true)

    loss_at_true = logloss_objective(np.array([true_lam2, true_lam3]), races_p_win, races_y)

    # (1) 最適化の正しさ: 生成に使った真の λ の loss を下回れない（オプティマイザが
    # 真値より真に悪い解に収束するのはバグ）。tol はソルバーの数値誤差許容分。
    lam2, lam3 = fit_lambda_logloss(races_p_win, races_y, init_lam2=1.0, init_lam3=1.0)
    loss_at_fit = logloss_objective(np.array([lam2, lam3]), races_p_win, races_y)
    assert loss_at_fit <= loss_at_true + 1e-3, (
        f"fit後のlossが真値のlossを大きく上回る: fit={loss_at_fit}, true={loss_at_true}"
    )

    # (2) 真値近傍から出発すればほぼ真値に留まる（flat landscape でも局所的に安定）。
    lam2_near, lam3_near = fit_lambda_logloss(
        races_p_win, races_y, init_lam2=true_lam2, init_lam3=true_lam3
    )
    assert abs(lam2_near - true_lam2) < 0.05, f"lam2={lam2_near} != {true_lam2}"
    assert abs(lam3_near - true_lam3) < 0.1, f"lam3={lam3_near} != {true_lam3}"


def test_bounds_respected():
    """fit 結果が bounds [0.1, 3.0] 内であること（極端なy分布でも逸脱しない）。"""
    rng = np.random.default_rng(2)
    races_p_win = []
    races_y = []
    for _ in range(10):
        n = rng.integers(5, 10)
        raw = rng.uniform(0.3, 2.0, size=n)
        p_win = raw / raw.sum()
        # 極端: 常に最下位馬だけがplace（現実にはあり得ないがboundsを試すため）
        y = np.zeros(n)
        y[np.argmin(p_win)] = 1.0
        races_p_win.append(p_win)
        races_y.append(y)

    lam2, lam3 = fit_lambda_logloss(races_p_win, races_y, init_lam2=0.6, init_lam3=0.6)
    assert 0.1 <= lam2 <= 3.0
    assert 0.1 <= lam3 <= 3.0


def test_band_split():
    """A2 の頭数帯割当: 5–7頭 / 8頭以上の境界（7頭→帯1、8頭→帯2）。"""
    horse_count = np.array([4, 5, 6, 7, 8, 9, 18])
    bands = band_5to7_8plus(horse_count)
    expected = np.array(["le7", "le7", "le7", "le7", "ge8", "ge8", "ge8"])
    assert (bands == expected).all()


def test_fit_lambda_logloss_banded_splits_by_band():
    rng = np.random.default_rng(3)
    races_p_win, races_y, races_band = [], [], []
    for i in range(20):
        n = rng.integers(5, 12)
        raw = rng.uniform(0.3, 2.0, size=n)
        p_win = raw / raw.sum()
        y = place_prob_from_p_win(p_win, 0.6, 0.6)
        races_p_win.append(p_win)
        races_y.append(y)
        races_band.append("le7" if n <= 7 else "ge8")

    out = fit_lambda_logloss_banded(races_p_win, races_y, races_band)
    assert set(out.keys()) == {"le7", "ge8"}
    for band, res in out.items():
        assert res["n_races"] == races_band.count(band)
        if res["n_races"] > 0:
            assert 0.1 <= res["lam2"] <= 3.0
            assert 0.1 <= res["lam3"] <= 3.0

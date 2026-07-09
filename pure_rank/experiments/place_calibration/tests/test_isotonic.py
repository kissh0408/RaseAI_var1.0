"""isotonic 事後較正のテスト（仕様書 §8）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from calib_lib import fit_isotonic, top1_index  # noqa: E402


def test_monotonic():
    """isotonic 出力が入力順序に対し単調非減少であること。"""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=500)
    # 真の関係に単調な歪みとノイズを乗せる
    y = (rng.uniform(size=500) < np.clip(p ** 1.5, 0, 1)).astype(float)
    iso = fit_isotonic(p, y)

    test_x = np.linspace(0, 1, 50)
    preds = iso.predict(test_x)
    diffs = np.diff(preds)
    assert (diffs >= -1e-12).all(), "isotonic出力が単調非減少でない"


def test_output_range():
    """出力が [0, 1]。out_of_bounds='clip' の挙動（fit範囲外入力）。"""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, size=200)  # fit範囲を [0.2, 0.8] に限定
    y = (rng.uniform(size=200) < p).astype(float)
    iso = fit_isotonic(p, y)

    # fit範囲外（0.0, 1.0付近）を含めて予測
    test_x = np.array([-0.5, 0.0, 0.1, 0.5, 0.9, 1.0, 1.5])
    preds = iso.predict(test_x)
    assert (preds >= 0.0).all()
    assert (preds <= 1.0).all()


def test_rank_preserved():
    """較正後のレース内 top1 が較正前と一致すること（タイは元の p_stern 順で安定）。"""
    rng = np.random.default_rng(2)
    p_fit = rng.uniform(0, 1, size=300)
    y_fit = (rng.uniform(size=300) < p_fit).astype(float)
    iso = fit_isotonic(p_fit, y_fit)

    # 1レース分: 明確に異なる p_stern値
    p_stern_race = np.array([0.1, 0.9, 0.5, 0.3])
    p_calibrated = iso.predict(p_stern_race)

    top1_before = top1_index(p_stern_race)
    top1_after = top1_index(p_calibrated, tiebreak=p_stern_race)
    assert top1_before == top1_after

    # isotonicの平坦区間で複数馬が同値になるケース（タイブレークで元順位を保つか）
    p_flat_input = np.array([0.05, 0.05, 0.05, 0.99])  # 最初の3頭は同一入力
    p_flat_calibrated = np.array([0.2, 0.2, 0.2, 0.8])  # isotonic後も同値（平坦区間を模擬）
    top1_flat = top1_index(p_flat_calibrated, tiebreak=p_flat_input)
    # 最後の馬(0.99 stern, 0.8 calibrated)が最高値なのでそれが選ばれる
    assert top1_flat == 3

    # 完全な同値タイ（calibrated値もtiebreak値も同じ）の場合は最初のインデックスを安定選択
    p_tied = np.array([0.5, 0.5])
    tie_tb = np.array([0.5, 0.5])
    assert top1_index(p_tied, tiebreak=tie_tb) == 0

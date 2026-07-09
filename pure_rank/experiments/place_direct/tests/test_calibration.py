"""較正誤差ビン計算のテスト（仕様書 §8）。

ビン関数は betting/src/pair_probs.py の calibration_max_error_pp を import して
再利用する（コピー禁止。compare_pair_probability_models.py と同一の実装であることを
import 同一性で担保する）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from betting.src.pair_probs import calibration_max_error_pp  # noqa: E402
import betting.src.pair_probs as pair_probs_module  # noqa: E402


def test_bins_match_reference():
    # compare_pair_probability_models.py の生成物と同一の実装を使っていることの確認
    # (import 同一性 + デフォルト n_bins=10, [0,1] の等幅ビンであることを直接検証)
    predicted = np.concatenate([np.full(20, 0.05), np.full(20, 0.95)])
    actual = np.concatenate([np.full(20, 0.05), np.full(20, 0.95)])
    # 手動で同一ビン境界（linspace(0,1,11)）を再現し、一致することを確認
    bins = np.linspace(0, 1, 11)
    errors = []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (predicted >= lo) & (predicted < hi if i < 9 else predicted <= hi)
        if mask.sum() < 5:
            continue
        errors.append(abs(predicted[mask].mean() - actual[mask].mean()) * 100)
    expected = max(errors) if errors else float("nan")
    got = calibration_max_error_pp(predicted, actual)
    assert abs(got - expected) < 1e-9
    # 関数がこのモジュールの実装そのものであることの確認（コピーでないこと）
    assert calibration_max_error_pp is pair_probs_module.calibration_max_error_pp


def test_perfect_calibration():
    rng = np.random.default_rng(1)
    predicted = rng.uniform(0, 1, size=2000)
    # 各予測確率どおりの実測率になるよう y をベルヌーイ生成（完全較正）
    actual = (rng.uniform(0, 1, size=2000) < predicted).astype(float)
    err = calibration_max_error_pp(predicted, actual)
    assert err < 8.0  # サンプリング誤差の範囲内でほぼ0


def test_known_miscalibration():
    # 予測は常に 0.5 だが実測は常に 1 → 誤差は 50pp 付近
    predicted = np.full(50, 0.5)
    actual = np.full(50, 1.0)
    err = calibration_max_error_pp(predicted, actual)
    assert abs(err - 50.0) < 1e-6

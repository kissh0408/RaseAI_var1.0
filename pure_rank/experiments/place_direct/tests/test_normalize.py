"""レース内正規化（合計3、clip+再配分）のテスト（仕様書 §8）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from place_lib import normalize_place_probs  # noqa: E402


def test_sum_to_three():
    rng = np.random.default_rng(0)
    race_id = np.repeat(["r1", "r2", "r3"], [8, 10, 6])
    p_raw = rng.uniform(0.05, 0.6, size=len(race_id))
    p_norm, _ = normalize_place_probs(p_raw, race_id)
    for rid in ["r1", "r2", "r3"]:
        mask = race_id == rid
        assert abs(p_norm[mask].sum() - 3.0) < 1e-9


def test_clip_redistribution():
    # r1: 1頭が生予測 0.95 で圧倒的に高く、正規化するとその馬が 1.0 超えになるケース
    race_id = np.array(["r1"] * 5)
    p_raw = np.array([0.95, 0.02, 0.02, 0.02, 0.02])
    p_norm, clip_count = normalize_place_probs(p_raw, race_id)
    assert clip_count >= 1
    assert p_norm.max() <= 1.0 + 1e-9
    assert abs(p_norm.sum() - 3.0) < 1e-9


def test_uniform_case():
    # 全馬同確率 n 頭 → 各 3/n
    for n in [5, 8, 12]:
        race_id = np.array([f"r_{n}"] * n)
        p_raw = np.full(n, 0.3)
        p_norm, clip_count = normalize_place_probs(p_raw, race_id)
        assert clip_count == 0
        np.testing.assert_allclose(p_norm, np.full(n, 3.0 / n), atol=1e-9)

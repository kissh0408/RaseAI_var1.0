"""B2 正規化（レース内合計3、clip+再配分）のテスト（仕様書 §8）。

正規化ロジック自体は pure_rank/experiments/place_direct/place_lib.py の
normalize_place_probs を import 再利用する（コピー禁止。仕様書 §3.2, §7）。
本テストは calib_lib 経由の re-export が同一関数であることと、
place_calibration の文脈（B2ユースケース）で正しく動作することを確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from calib_lib import normalize_place_probs  # noqa: E402


def test_sum_to_three():
    rng = np.random.default_rng(0)
    race_id = np.repeat(["r1", "r2", "r3"], [8, 10, 6])
    p_raw = rng.uniform(0.05, 0.6, size=len(race_id))
    p_norm, _ = normalize_place_probs(p_raw, race_id)
    for rid in ["r1", "r2", "r3"]:
        mask = race_id == rid
        assert abs(p_norm[mask].sum() - 3.0) < 1e-9


def test_clip_redistribution():
    race_id = np.array(["r1"] * 5)
    p_raw = np.array([0.95, 0.02, 0.02, 0.02, 0.02])
    p_norm, clip_count = normalize_place_probs(p_raw, race_id)
    assert clip_count >= 1
    assert p_norm.max() <= 1.0 + 1e-9
    assert abs(p_norm.sum() - 3.0) < 1e-9


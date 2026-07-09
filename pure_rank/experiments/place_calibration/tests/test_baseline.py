"""S0 再現検証（仕様書 §4, §8）。

S0（現行 Stern、Brier fit λ 固定 0.6018/0.6381）を本実験のパイプラインで
TEST(2025+) 上で再計算し、基準値（logloss=0.4003477616722795±0.0005、
calibration_max_error_pp=4.528241481640338±0.05）と一致することを確認する。
この検証に合格してから A/B 系列の評価に進む（実データ fit の前段ゲート）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from calib_lib import place_logloss  # noqa: E402

from betting.src.pair_probs import calibration_max_error_pp  # noqa: E402
from prob_fusion.src.place_prob import place_prob_from_p_win  # noqa: E402

DATA_DIR = EXP_DIR / "data"
CONFIG_PATH = EXP_DIR / "config.json"
EPS = 1e-12


@pytest.mark.skipif(
    not (DATA_DIR / "test_2025.parquet").is_file(),
    reason="build_dataset.py 未実行（実データ統合テスト）",
)
def test_s0_reproduction():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    s0 = cfg["s0_baseline"]

    test_df = pd.read_parquet(DATA_DIR / "test_2025.parquet")
    test_df["race_id"] = test_df["race_id"].astype(str)

    p_s0_list = []
    y_list = []
    for _, grp in test_df.groupby("race_id", sort=False):
        p_win = grp["p_win"].astype(float).to_numpy()
        p_place = place_prob_from_p_win(p_win, s0["lam2"], s0["lam3"])
        p_s0_list.append(p_place)
        y_list.append(grp["y_place"].astype(float).to_numpy())

    p_s0 = np.concatenate(p_s0_list)
    y = np.concatenate(y_list)
    p_s0_clipped = np.clip(p_s0, EPS, 1 - EPS)

    logloss = place_logloss(p_s0_clipped, y, eps=EPS)
    calib_err = calibration_max_error_pp(p_s0_clipped, y)

    assert abs(logloss - s0["expected_logloss"]) <= s0["logloss_tol"], (
        f"S0 logloss 不一致: got={logloss}, expected={s0['expected_logloss']}±{s0['logloss_tol']}"
    )
    assert abs(calib_err - s0["expected_calibration_max_error_pp"]) <= s0["calibration_tol_pp"], (
        f"S0 較正誤差不一致: got={calib_err}, "
        f"expected={s0['expected_calibration_max_error_pp']}±{s0['calibration_tol_pp']}"
    )

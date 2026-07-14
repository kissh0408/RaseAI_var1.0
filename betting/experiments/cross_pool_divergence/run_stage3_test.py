"""cross_pool_divergence: run_stage3_test.py

Stage 3（TEST 2025+、1回のみ。evaluator 承認済み 2026-07-10）:
一次通過セグメント **place POP1×FS_L のみ** を TEST 期間で二次判定する。
他セグメントの TEST 値は計算・出力しない。

- 二次判定基準（事前登録）: TEST D_adj ≥ +0.03。
- ROI_flat（均一全張り診断）を報告するため、§7 の payout 集中度ゲート
  （top1_payout_share ≤ 0.30 かつ n_hits ≥ 10）を適用する。
- TEST ユニットは fit 用 parquet とは別ファイル（data/units_place_test.parquet）に
  保存し、fit 期間データを汚染しない。
- 実行は 1 回のみ。結果がどうであれ再実行・パラメータ変更は禁止。

出力:
    betting/experiments/cross_pool_divergence/results/divergence_test.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import divergence_lib as dl  # noqa: E402

from build_dataset import (  # noqa: E402
    _build_place_units,
    _load_base_frame,
    _load_config,
    _load_lambda_params,
)

DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
OUT_PATH = RESULTS_DIR / "divergence_test.json"
TEST_UNITS_PATH = DATA_DIR / "units_place_test.parquet"

# 一次通過セグメント（Stage 2 確定・evaluator 承認）。後出し変更禁止。
TARGET_SEGMENT = {"bet_type": "place", "pop_band": "POP1", "fs_band": "FS_L"}


def run_stage3() -> dict:
    if OUT_PATH.exists():
        raise RuntimeError(
            f"Stage 3 は 1 回のみ実行可能。既に結果が存在します: {OUT_PATH}"
        )

    cfg = _load_config()
    lam2, lam3 = _load_lambda_params(cfg)
    test_start = cfg["protocol"]["test_start"]
    t_place = cfg["place"]["t_place"]
    d_adj_threshold = cfg["thresholds"]["d_adj_pass"]
    gate_cfg = cfg["payout_concentration_gate"]

    # TEST 期間（2025-01-01〜、上限なし）のベースフレームを構築（fit parquet 非接触）
    base = _load_base_frame(cfg, start=test_start, end=None)
    assert pd.to_datetime(base["race_date"]).min() >= pd.Timestamp(test_start), "TEST開始前の行が混入"

    place_units = _build_place_units(base, cfg, lam2, lam3)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    place_units.to_parquet(TEST_UNITS_PATH, index=False, compression="snappy")

    # 対象セグメントのみ抽出（他セグメントの TEST 値は計算しない）
    sub = place_units[
        (place_units["pop_band"] == TARGET_SEGMENT["pop_band"])
        & (place_units["fs_band"] == TARGET_SEGMENT["fs_band"])
    ]

    y = sub["y"].to_numpy(dtype=float)
    p_theo = sub["p_theo"].to_numpy(dtype=float)
    o = sub["O"].to_numpy(dtype=float)

    h = dl.seg_h(y)
    p_bar_theo = dl.seg_p_bar_theo(p_theo)
    d_cal = dl.seg_d_cal(h, p_bar_theo)
    roi_flat = dl.seg_roi_flat(y, o)
    o_bar_hit = dl.seg_o_bar_hit(y, o)
    d_adj = dl.seg_d_adj_place(h, o_bar_hit, t_place)
    n_hits = int(y.sum())

    secondary_pass = bool(np.isfinite(d_adj) and d_adj >= d_adj_threshold)

    # §7 payout 集中度ゲート（均一全張り ROI 診断の妥当性検査）
    hit_payouts = o[np.isfinite(o) & (y > 0)]
    gates = dl.payout_concentration_gate(
        hit_payouts.tolist(),
        n_hits,
        top1_share_max=gate_cfg["top1_payout_share_max"],
        n_hits_min=gate_cfg["n_hits_min"],
    )

    report = {
        "stage": "divergence_test",
        "executed_once": True,
        "segment": TARGET_SEGMENT,
        "test_period": f"{test_start}..",
        "n_units": int(len(sub)),
        "n_races": int(sub["race_id"].nunique()),
        "n_hits": n_hits,
        "h": h,
        "p_bar_theo": p_bar_theo,
        "d_cal": d_cal,
        "o_bar_hit": o_bar_hit,
        "roi_flat": roi_flat,
        "d_adj": d_adj,
        "d_adj_threshold": d_adj_threshold,
        "secondary_pass": secondary_pass,
        "gates": gates,
        "protocol": {
            "q_method": cfg["q_method"],
            "lam_source": cfg["fusion_params_source"],
            "lam2": lam2,
            "lam3": lam3,
            "t_place": t_place,
        },
        "interpretation_fixed_in_advance": (
            "TEST通過でも結論は『較正乖離の頑健性確認』であり、ROI_flat≈86%水準（fit実測）"
            "で収益化不可は確定済み。D_adj>0 は複勝プールの POP1 過小評価が控除率の壁の"
            "内側に留まることと両立する。"
        ),
        "caveats": [
            "確定払戻は事前オッズではない。購入時点で乖離が消えている可能性は残る（上限診断）。",
            "複勝 D_adj は Ō_hit セグメントレベル近似（調和平均とのバイアスあり）。",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_PATH}")
    return report


if __name__ == "__main__":
    run_stage3()

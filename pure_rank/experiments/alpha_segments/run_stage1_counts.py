"""alpha_segments: run_stage1_counts.py

Stage 1（TEST 非接触）: gate_dataset.parquet を 2024-12-31 以前でフィルタし、
2023 / 2024 のセグメント別レース数を集計する。2024 年 n>=300 のセグメントのみ
「確定」とし、K（確定セグメント数）と Bonferroni 閾値（0.01/K）を出力する。

このスクリプトは 2025 年以降のデータを一切読み込まない
（date フィルタを io 直後に適用。仕様書 §7 Stage 1-1）。

出力:
    pure_rank/experiments/alpha_segments/results/stage1_counts.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import segments_lib as sl  # noqa: E402

DATA_PATH = EXP_DIR / "data" / "gate_dataset.parquet"
RESULTS_DIR = EXP_DIR / "results"
OUT_PATH = RESULTS_DIR / "stage1_counts.json"

STAGE1_CUTOFF = "2024-12-31"  # TEST(2025+) 非接触ガード


def run_stage1() -> dict:
    cfg = sl.load_config()
    n_min = cfg["thresholds"]["n_min_eval_2024"]

    df = pd.read_parquet(DATA_PATH)
    df["race_date"] = pd.to_datetime(df["race_date"])

    # TEST 非接触ガード: 2024-12-31 以前しか読み込まない
    df = df.loc[df["race_date"] <= pd.Timestamp(STAGE1_CUTOFF)].copy()
    assert df["race_date"].max() <= pd.Timestamp(STAGE1_CUTOFF), "Stage1にTEST期間が混入"

    df = sl.add_all_segment_flags(df, cfg)

    fit_year = cfg["protocol"]["fit_year"]
    eval_year = cfg["protocol"]["eval_year"]

    counts: dict[str, dict[str, int]] = {}
    for seg_id in cfg["segment_order"]:
        seg_col = f"seg_{seg_id}"
        seg_races = df.loc[df[seg_col], "race_id"].drop_duplicates()
        seg_df = df.loc[df["race_id"].isin(seg_races)]
        n_2023 = seg_df.loc[seg_df["race_date"].dt.year == fit_year, "race_id"].nunique()
        n_2024 = seg_df.loc[seg_df["race_date"].dt.year == eval_year, "race_id"].nunique()
        counts[seg_id] = {"n_2023": int(n_2023), "n_2024": int(n_2024)}

    confirmed = sl.confirm_segments(counts, n_min=n_min)
    k = sum(1 for c in confirmed.values() if c["confirmed"])
    bonferroni = sl.bonferroni_threshold(k, base_alpha=cfg["thresholds"]["lrt_alpha"])

    report = {
        "stage": "stage1_counts",
        "cutoff": STAGE1_CUTOFF,
        "fit_year": fit_year,
        "eval_year": eval_year,
        "n_min_eval_2024": n_min,
        "segments": confirmed,
        "K": k,
        "bonferroni_threshold": bonferroni,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_PATH}")
    return report


if __name__ == "__main__":
    run_stage1()

"""alpha_segments: run_stage3_test.py

Stage 3（TEST 二次判定。一次通過セグメントのみ、各1回。事前登録済み手順。
仕様書 §7 Stage 3）。

一次通過セグメントが 0 の場合、このスクリプトは TEST データを一切読み込まずに
終了する（TEST 完全非接触を維持する）。

実行する場合の手順（固定）:
  1. split_oos_periods で fit(2023-01-01..2024-12-31) / TEST(2025-01-01..) に分割し、
     セグメントフィルタ適用。
  2. fit 期間セグメント races で H1（alpha, beta）と H0（market_only=True の beta）を再フィット。
  3. TEST セグメント df で test_logloss_fusion / test_logloss_market を各1回だけ計算。
  4. 二次判定合格 = test_logloss_fusion < test_logloss_market かつ
     TEST セグメント Top-1 <= 0.40（かつ Spearman <= 0.60）。

出力:
    pure_rank/experiments/alpha_segments/results/alpha_segments_test.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import segments_lib as sl  # noqa: E402
from prob_fusion.src.fit_fusion import (  # noqa: E402
    build_race_tuples,
    fit_fusion_mle,
    fusion_probs,
    mean_logloss,
    top1_hit_rate,
)
from prob_fusion.src.oos_protocol import split_oos_periods  # noqa: E402

DATA_PATH = EXP_DIR / "data" / "gate_dataset.parquet"
RESULTS_DIR = EXP_DIR / "results"
STAGE2_PATH = RESULTS_DIR / "alpha_segments.json"
OUT_PATH = RESULTS_DIR / "alpha_segments_test.json"


def _race_spearman(df: pd.DataFrame, alpha: float, beta: float) -> float:
    rhos: list[float] = []
    for _, grp in df.groupby("race_id"):
        if len(grp) < 3:
            continue
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        p = fusion_probs(z, ln_q, alpha, beta)
        finish = grp["finish_rank"].astype(int).values
        if np.std(p) < 1e-12 or np.std(finish) < 1e-12:
            continue
        rho, _ = scipy_stats.spearmanr(p, -finish)
        if np.isfinite(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def run_stage3() -> dict:
    cfg = sl.load_config()
    stage2 = json.loads(STAGE2_PATH.read_text(encoding="utf-8"))
    primary_pass_segments = stage2.get("primary_pass_segments", [])

    if not primary_pass_segments:
        report = {
            "stage": "stage3_test",
            "executed": False,
            "reason": "no primary_pass segments in Stage 2; TEST left untouched per spec section 7",
            "segments": {},
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("No primary_pass segments; Stage 3 not executed (TEST left untouched).")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    leak_top1 = cfg["thresholds"]["leak_top1"]
    leak_spearman = cfg["thresholds"]["leak_spearman"]
    alpha_bounds = tuple(cfg["fusion_bounds"]["alpha_bounds"])
    beta_bounds = tuple(cfg["fusion_bounds"]["beta_bounds"])
    fit_start = cfg["protocol"]["fit_start_test_stage"]
    fit_end = cfg["protocol"]["fit_end_test_stage"]
    test_start = cfg["protocol"]["test_start"]

    df = pd.read_parquet(DATA_PATH)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = sl.add_all_segment_flags(df, cfg)

    results: dict[str, dict] = {}
    for seg_id in primary_pass_segments:
        seg_col = f"seg_{seg_id}"
        seg_races = df.loc[df[seg_col], "race_id"].drop_duplicates()
        seg_df = df.loc[df["race_id"].isin(seg_races)].copy()

        fit_df, test_df = split_oos_periods(
            seg_df, fit_start=fit_start, fit_end=fit_end, test_start=test_start
        )
        fit_races = build_race_tuples(fit_df)

        fitted = fit_fusion_mle(fit_races, alpha_bounds=alpha_bounds, beta_bounds=beta_bounds)
        h0 = fit_fusion_mle(
            fit_races, alpha_bounds=alpha_bounds, beta_bounds=beta_bounds, market_only=True
        )

        test_logloss_fusion = mean_logloss(test_df, fitted.alpha, fitted.beta)
        test_logloss_market = mean_logloss(test_df, 0.0, h0.beta)
        test_top1 = top1_hit_rate(test_df, fitted.alpha, fitted.beta)
        test_spearman = _race_spearman(test_df, fitted.alpha, fitted.beta)

        leak = sl.leak_stop(
            test_top1, test_spearman,
            top1_threshold=leak_top1, spearman_threshold=leak_spearman,
        )
        secondary_pass = bool(
            (test_logloss_fusion < test_logloss_market)
            and (test_top1 <= leak_top1)
            and (test_spearman <= leak_spearman)
            and not leak
        )

        results[seg_id] = {
            "n_fit_races": len(fit_races),
            "n_test_races": test_df["race_id"].nunique(),
            "alpha": fitted.alpha,
            "beta": fitted.beta,
            "h0_beta": h0.beta,
            "test_logloss_fusion": test_logloss_fusion,
            "test_logloss_market": test_logloss_market,
            "test_top1": test_top1,
            "test_spearman": test_spearman,
            "leak_stop": leak,
            "secondary_pass": secondary_pass,
        }

    report = {
        "stage": "stage3_test",
        "executed": True,
        "fit_period": f"{fit_start}..{fit_end}",
        "test_start": test_start,
        "segments": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_PATH}")
    return report


if __name__ == "__main__":
    run_stage3()

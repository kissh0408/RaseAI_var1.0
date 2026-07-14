"""alpha_segments: run_stage2_lrt.py

Stage 2（fit=2023 -> eval=2024、alphaゲートと同一プロトコル）: Stage 1 で確定
したセグメントごとに、セグメント内 races で条件付きロジット H1（alpha, beta 自由）
と H0（market_only=True）を fit=2023 の races で推定し、alpha の LRT p 値・
eval=2024 での DeltaLL/race・Top-1・Spearman を計算する。

一次判定: p < 0.01/K かつ DeltaLL/race > 0（K は stage1_counts.json の確定値）。
リーク停止: eval Top-1 > 0.40 または eval Spearman > 0.60 のセグメントがあれば
即座に leak_stop=true を記録し、後続の一次判定・Stage3対象から外す。

出力:
    pure_rank/experiments/alpha_segments/results/alpha_segments.json
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
from evaluation.alpha_gate import split_alpha_gate_cv  # noqa: E402
from prob_fusion.src.fit_fusion import (  # noqa: E402
    build_race_tuples,
    fit_fusion_mle,
    fusion_probs,
    likelihood_ratio_test,
    mean_logloss,
    top1_hit_rate,
)

DATA_PATH = EXP_DIR / "data" / "gate_dataset.parquet"
RESULTS_DIR = EXP_DIR / "results"
STAGE1_PATH = RESULTS_DIR / "stage1_counts.json"
OUT_PATH = RESULTS_DIR / "alpha_segments.json"

STAGE2_CUTOFF = "2024-12-31"  # TEST(2025+) 非接触ガード（Stage2もTESTには触れない）


def _race_spearman(df: pd.DataFrame, alpha: float, beta: float) -> float:
    """Mean per-race Spearman correlation between fusion probability rank
    and finish_rank (higher fusion prob should correspond to better/lower
    finish_rank, so we correlate prob against -finish_rank for an intuitive
    sign, matching spec's "fusion 確率の順位 vs finish_rank")."""
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


def run_stage2() -> dict:
    cfg = sl.load_config()
    lrt_alpha = cfg["thresholds"]["lrt_alpha"]
    leak_top1 = cfg["thresholds"]["leak_top1"]
    leak_spearman = cfg["thresholds"]["leak_spearman"]
    alpha_bounds = tuple(cfg["fusion_bounds"]["alpha_bounds"])
    beta_bounds = tuple(cfg["fusion_bounds"]["beta_bounds"])

    stage1 = json.loads(STAGE1_PATH.read_text(encoding="utf-8"))
    k = stage1["K"]
    bonferroni = stage1["bonferroni_threshold"]
    confirmed_segments = [
        seg_id for seg_id, c in stage1["segments"].items() if c["confirmed"]
    ]
    print(f"Confirmed segments (K={k}): {confirmed_segments}, bonferroni_threshold={bonferroni}")

    df = pd.read_parquet(DATA_PATH)
    df["race_date"] = pd.to_datetime(df["race_date"])
    # Stage 2 も TEST(2025+) 非接触（仕様書 §7 Stage 2 は fit=2023/eval=2024 のみ）
    df = df.loc[df["race_date"] <= pd.Timestamp(STAGE2_CUTOFF)].copy()
    assert df["race_date"].max() <= pd.Timestamp(STAGE2_CUTOFF), "Stage2にTEST期間が混入"

    df = sl.add_all_segment_flags(df, cfg)

    results: dict[str, dict] = {}
    any_leak_stop = False

    if k == 0:
        print("K=0: no confirmed segments; Stage 2 ends with an empty run (per spec section 7).")

    for seg_id in confirmed_segments:
        seg_col = f"seg_{seg_id}"
        seg_races = df.loc[df[seg_col], "race_id"].drop_duplicates()
        seg_df = df.loc[df["race_id"].isin(seg_races)].copy()

        fit_df, eval_df = split_alpha_gate_cv(seg_df)
        fit_races = build_race_tuples(fit_df)
        eval_races = build_race_tuples(eval_df)

        n_fit = len(fit_races)
        n_eval = len(eval_races)

        fitted = fit_fusion_mle(fit_races, alpha_bounds=alpha_bounds, beta_bounds=beta_bounds)
        lrt = likelihood_ratio_test(fit_races, fitted, alpha_bounds=alpha_bounds, beta_bounds=beta_bounds)
        h0_beta = lrt["h0_beta"]

        ll_h1 = mean_logloss(eval_df, fitted.alpha, fitted.beta)
        ll_h0 = mean_logloss(eval_df, 0.0, h0_beta)
        delta_ll_per_race = ll_h0 - ll_h1

        eval_top1 = top1_hit_rate(eval_df, fitted.alpha, fitted.beta)
        eval_spearman = _race_spearman(eval_df, fitted.alpha, fitted.beta)

        leak = sl.leak_stop(
            eval_top1, eval_spearman,
            top1_threshold=leak_top1, spearman_threshold=leak_spearman,
        )
        if leak:
            any_leak_stop = True

        p_value = lrt["p_value"]
        primary = sl.primary_pass(p_value, delta_ll_per_race, bonferroni, leak)

        results[seg_id] = {
            "n_fit": n_fit,
            "n_eval": n_eval,
            "alpha": fitted.alpha,
            "beta": fitted.beta,
            "h0_beta": h0_beta,
            "lrt_statistic": lrt["lr_statistic"],
            "lrt_p": p_value,
            "bonferroni_threshold": bonferroni,
            "delta_ll_per_race": delta_ll_per_race,
            "eval_top1": eval_top1,
            "eval_spearman": eval_spearman,
            "primary_pass": primary,
            "leak_stop": leak,
        }
        print(
            f"{seg_id}: n_fit={n_fit} n_eval={n_eval} alpha={fitted.alpha:.4f} beta={fitted.beta:.4f} "
            f"p={p_value:.6g} deltaLL/race={delta_ll_per_race:.6f} top1={eval_top1:.4f} "
            f"spearman={eval_spearman:.4f} primary_pass={primary} leak_stop={leak}"
        )

    primary_pass_segments = [s for s, r in results.items() if r["primary_pass"]]
    verdict = None
    if any_leak_stop:
        verdict = "leak_stop"
    elif k == 0:
        verdict = "no_confirmed_segments"
    elif not primary_pass_segments:
        verdict = cfg["verdict"]["all_fail"]
    else:
        verdict = "primary_pass_found"

    report = {
        "stage": "stage2_lrt",
        "fit_year": cfg["protocol"]["fit_year"],
        "eval_year": cfg["protocol"]["eval_year"],
        "K": k,
        "bonferroni_threshold": bonferroni,
        "segments": results,
        "primary_pass_segments": primary_pass_segments,
        "any_leak_stop": any_leak_stop,
        "verdict": verdict,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_PATH}")
    if any_leak_stop:
        print("!!! LEAK STOP TRIGGERED. Report to evaluator immediately. Not proceeding to Stage 3. !!!")
    return report


if __name__ == "__main__":
    run_stage2()

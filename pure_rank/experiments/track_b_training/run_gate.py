"""track_b_training: run_gate.py

Runs the standing alpha-gate (`evaluation.alpha_gate.run_alpha_gate`) for one
already-built candidate parquet, computes the eval(2024) race-internal
Spearman correlation the gate itself does not compute (spec section 4.3), and
appends the result to results/track_b_summary.json.

`run_alpha_gate` is always called with out_dir=results/ (never
evaluation/reports/) to keep this experiment isolated from production.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import training_lib as tl  # noqa: E402
from evaluation.alpha_gate import (  # noqa: E402
    load_candidate_dataset,
    run_alpha_gate_on_dataframe,
    split_alpha_gate_cv,
)
from prob_fusion.src.fit_fusion import fusion_probs  # noqa: E402

RESULTS_DIR = EXP_DIR / "results"
SUMMARY_PATH = RESULTS_DIR / "track_b_summary.json"


def _eval_spearman(eval_df: pd.DataFrame, alpha: float, beta: float, gamma: float) -> float:
    """Mean per-race Spearman between fusion probability and finish_rank (2024 eval set).

    Mirrors alpha_segments/run_stage2_lrt.py::_race_spearman, extended with
    the candidate term x=cand_score_z, gamma.
    """
    rhos: list[float] = []
    for _, grp in eval_df.groupby("race_id"):
        if len(grp) < 3:
            continue
        z = grp["pure_score_z"].astype(float).to_numpy()
        ln_q = grp["ln_market_q"].astype(float).to_numpy()
        x = grp["cand_score_z"].astype(float).to_numpy()
        p = fusion_probs(z, ln_q, alpha, beta, x=x, gamma=gamma)
        finish = grp["finish_rank"].astype(int).to_numpy()
        if np.std(p) < 1e-12 or np.std(finish) < 1e-12:
            continue
        rho, _ = scipy_stats.spearmanr(p, -finish)
        if np.isfinite(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def run_one(candidate_key: str, cfg: dict) -> dict:
    cand_meta = cfg["candidates"][candidate_key]
    name = cand_meta["name"]
    candidate_path = EXP_DIR / "data" / cand_meta["output"]
    scores_path = ROOT / cfg["paths"]["scores_fold2_oos"]
    features_path = ROOT / cfg["paths"]["features_v39_course_slim"]

    df = load_candidate_dataset(candidate_path, scores_path, features_path)
    report = run_alpha_gate_on_dataframe(df, candidate_name=name)

    _, eval_df = split_alpha_gate_cv(df)
    spearman = _eval_spearman(eval_df, report["alpha"], report["beta"], report["gamma"])
    leak_spearman_threshold = cfg["gate_thresholds"]["leak_spearman"]
    leak_top1_threshold = cfg["gate_thresholds"]["leak_top1"]

    spearman_leak = bool(np.isfinite(spearman) and spearman > leak_spearman_threshold)
    top1_leak = bool(report["eval_top1"] > leak_top1_threshold)
    leak_stop = bool(spearman_leak or top1_leak or report["leak_warnings"]["leak_warning"])

    report["eval_spearman"] = spearman
    report["leak_stop"] = leak_stop
    report["leak_stop_reasons"] = {
        "spearman_gt_0_60": spearman_leak,
        "top1_gt_0_40": top1_leak,
        "corr_leak_warning": report["leak_warnings"]["leak_warning"],
    }
    report["primary_pass"] = bool(report["pass"] and not leak_stop)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"alpha_gate_{name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{candidate_key}] wrote {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    _append_summary(candidate_key, report, cfg)
    return report


def _append_summary(candidate_key: str, report: dict, cfg: dict) -> None:
    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary.setdefault("candidates", {})
    summary["candidates"][candidate_key] = {
        "id": cfg["candidates"][candidate_key]["id"],
        "name": report["candidate"],
        "gamma": report["gamma"],
        "gamma_lrt_p": report["gamma_lrt_p"],
        "delta_ll_per_race": report["delta_ll_per_race"],
        "eval_top1": report["eval_top1"],
        "eval_spearman": report["eval_spearman"],
        "leak_stop": report["leak_stop"],
        "primary_pass": report["primary_pass"],
        "gates": report["gates"],
        "generated_at": report["generated_at"],
    }
    order = cfg["candidate_order"]
    done = [k for k in order if k in summary["candidates"]]
    if len(done) == len(order):
        any_leak = any(summary["candidates"][k]["leak_stop"] for k in order)
        any_pass = any(summary["candidates"][k]["primary_pass"] for k in order)
        if any_leak:
            summary["verdict"] = "leak_stop"
        elif any_pass:
            summary["verdict"] = "primary_pass_found"
        else:
            summary["verdict"] = cfg["verdict"]["all_fail"]
    else:
        summary["verdict"] = "in_progress"
    summary["candidates_completed"] = done
    summary["candidates_total"] = order
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated: {SUMMARY_PATH}")
    if report["leak_stop"]:
        print("!!! LEAK STOP TRIGGERED. Report to evaluator immediately. Halt before next candidate. !!!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alpha-gate for one track_b_training candidate")
    parser.add_argument("--candidate", choices=["b1", "b2", "b3", "b4", "b5"], required=True)
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.json")
    args = parser.parse_args()
    cfg = tl.load_config(args.config)
    run_one(args.candidate, cfg)


if __name__ == "__main__":
    main()

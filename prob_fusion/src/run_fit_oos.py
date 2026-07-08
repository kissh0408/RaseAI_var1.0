"""CLI: OOS 正式測定（Phase 2）— fold2 スコアで fit=2023-2024 / TEST=2025+。

正式判定: fit=2023+2024（2023 は fold2 の early stopping のみ接触 — caveat 記録）
感度分析: fit=2024 単独（完全 OOS）
判定は evaluate_oos_gates（logloss 市場超え AND Top-1 ≥ 33%）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prob_fusion.src.fit_fusion import (
    build_race_tuples,
    calibration_bins,
    fit_fusion_mle,
    fusion_probs,
    likelihood_ratio_test,
    mean_logloss,
    top1_hit_rate,
)
from prob_fusion.src.manifest import file_sha256, write_manifest
from prob_fusion.src.market_prob import attach_market_q
from prob_fusion.src.oos_protocol import (
    FIT_END,
    FIT_START,
    TEST_START,
    evaluate_oos_gates,
    split_oos_periods,
)
from prob_fusion.src.place_prob import fit_stern_lambda
from prob_fusion.src.predict_fusion import fuse_dataframe, load_fusion_config
from prob_fusion.src.run_fit import load_scored_dataset

VERSION = "benter_oos_fold2"


def _fit_and_eval(
    fit_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    label: str,
) -> dict:
    bounds_a = tuple(cfg["fit"]["alpha_bounds"])
    bounds_b = tuple(cfg["fit"]["beta_bounds"])
    races = build_race_tuples(fit_df)
    fitted = fit_fusion_mle(races, alpha_bounds=bounds_a, beta_bounds=bounds_b)
    lrt = likelihood_ratio_test(races, fitted, alpha_bounds=bounds_a, beta_bounds=bounds_b)
    market_only = fit_fusion_mle(races, market_only=True, beta_bounds=bounds_b)

    p_win_races = []
    place_outcomes = []
    for _, grp in fit_df.groupby("race_id"):
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        p_win_races.append(fusion_probs(z, ln_q, fitted.alpha, fitted.beta))
        place_outcomes.append((grp["finish_rank"].astype(int).values <= 3).astype(float))
    lam2, lam3 = fit_stern_lambda(p_win_races, place_outcomes)

    metrics = {
        "test_logloss_fusion": mean_logloss(test_df, fitted.alpha, fitted.beta),
        "test_logloss_market": mean_logloss(test_df, 0.0, market_only.beta),
        "test_top1": top1_hit_rate(test_df, fitted.alpha, fitted.beta),
    }
    calib = calibration_bins(test_df, fitted.alpha, fitted.beta)
    return {
        "label": label,
        "alpha": fitted.alpha,
        "beta": fitted.beta,
        "market_only_beta": market_only.beta,
        "lam2": lam2,
        "lam3": lam3,
        "lrt_p_value": lrt["p_value"],
        "fit_n_races": len(races),
        **metrics,
        "calibration_max_error_pp": calib["max_error_pp"],
        "gates": evaluate_oos_gates(metrics),
    }


def run_fit_oos(scores_path: Path, features_path: Path | None, out_dir: Path) -> dict:
    cfg = load_fusion_config()
    q_method = cfg.get("q_method", "proportional")
    q_power = cfg.get("q_power", 0.81)

    df = load_scored_dataset(scores_path, features_path)
    df = attach_market_q(df, method=q_method, power=q_power)

    fit_df, test_df = split_oos_periods(df)
    if fit_df.empty or test_df.empty:
        raise ValueError(
            f"OOS 期間が空です: fit={len(fit_df)} rows, test={len(test_df)} rows。"
            f"scores parquet の期間を確認してください: {scores_path}"
        )

    formal = _fit_and_eval(fit_df, test_df, cfg, label="formal_fit_2023_2024")

    fit_2024 = fit_df[pd.to_datetime(fit_df["race_date"]) >= pd.Timestamp("2024-01-01")]
    sensitivity = _fit_and_eval(fit_2024, test_df, cfg, label="sensitivity_fit_2024_only")

    fused = fuse_dataframe(
        df,
        alpha=formal["alpha"],
        beta=formal["beta"],
        lam2=formal["lam2"],
        lam3=formal["lam3"],
        q_method=q_method,
        q_power=q_power,
        model_version=VERSION,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = out_dir / f"probs_{VERSION}.parquet"
    fused.to_parquet(probs_path, index=False)

    report = {
        "version": VERSION,
        "protocol": {
            "l1_scores": "fold2-only 5-seed ensemble (train<2023; 2024/2025 fully OOS)",
            "fit_period": f"{FIT_START}..{FIT_END}",
            "test_period": f"{TEST_START}..",
            "caveat": "2023 was fold2 early-stopping year (weak contamination, model selection only)",
            "formal_judgment": "formal_fit_2023_2024",
        },
        "formal": formal,
        "sensitivity": sensitivity,
        "config": cfg,
    }
    report_path = ROOT / "evaluation" / "reports" / "fusion_oos_fold2.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    inputs = {"scores": file_sha256(scores_path)}
    if features_path and features_path.exists():
        inputs["features"] = file_sha256(features_path)
    write_manifest(out_dir, inputs=inputs, config=cfg, extra={"oos_report": report["protocol"], "formal": formal})

    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS formal fusion measurement (fold2 scores)")
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "prob_fusion" / "data")
    args = parser.parse_args()
    run_fit_oos(args.scores, args.features, args.out)


if __name__ == "__main__":
    main()

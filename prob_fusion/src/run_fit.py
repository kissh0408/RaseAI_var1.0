"""CLI: fit Benter fusion and export probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.splits import filter_fold, get_walkforward_folds
from evaluation.odds_loader import attach_odds_from_se_parquet
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
from prob_fusion.src.place_prob import fit_stern_lambda
from prob_fusion.src.predict_fusion import fuse_dataframe, load_fusion_config


def load_scored_dataset(scores_path: Path, features_path: Path | None) -> pd.DataFrame:
    """Load scores and join odds + finish_rank from features if needed."""
    scores = pd.read_parquet(scores_path)
    scores["race_id"] = scores["race_id"].astype(str)
    if "horse_number" not in scores.columns:
        scores["horse_number"] = scores["horse_num"] if "horse_num" in scores.columns else scores["ketto_num"]
    scores["horse_num"] = scores["horse_number"].astype(int)

    if "odds" in scores.columns and "finish_rank" in scores.columns and "race_date" in scores.columns:
        return scores

    if features_path is not None and features_path.exists():
        feat = pd.read_parquet(features_path, columns=["race_id", "horse_num", "finish_rank", "race_date"])
        feat["race_id"] = feat["race_id"].astype(str)
        scores = scores.merge(feat, on=["race_id", "horse_num"], how="inner", suffixes=("", "_feat"))
        if "race_date_feat" in scores.columns and "race_date" not in scores.columns:
            scores["race_date"] = scores["race_date_feat"]
            scores = scores.drop(columns=["race_date_feat"])

    if "odds" not in scores.columns:
        scores = attach_odds_from_se_parquet(scores)
    if "race_date" not in scores.columns:
        raise KeyError("race_date required for walk-forward splits")
    return scores


def run_fit(scores_path: Path, features_path: Path | None, out_dir: Path) -> dict:
    cfg = load_fusion_config()
    df = load_scored_dataset(scores_path, features_path)
    q_method = cfg.get("q_method", "proportional")
    df = attach_market_q(df, method=q_method, power=cfg.get("q_power", 0.81))

    folds = get_walkforward_folds()
    fold_results = []
    all_test_frames: list[pd.DataFrame] = []

    for fold in folds:
        train_df = filter_fold(df, fold, "train")
        valid_df = filter_fold(df, fold, "valid")
        test_df = filter_fold(df, fold, "test")
        fit_df = pd.concat([train_df, valid_df], ignore_index=True)
        fit_df = attach_market_q(fit_df, method=q_method, power=cfg.get("q_power", 0.81))
        races = build_race_tuples(fit_df)
        bounds_a = tuple(cfg["fit"]["alpha_bounds"])
        bounds_b = tuple(cfg["fit"]["beta_bounds"])
        fitted = fit_fusion_mle(races, alpha_bounds=bounds_a, beta_bounds=bounds_b)
        lrt = likelihood_ratio_test(races, fitted, alpha_bounds=bounds_a, beta_bounds=bounds_b)

        valid_df = attach_market_q(valid_df, method=q_method, power=cfg.get("q_power", 0.81))
        p_win_races = []
        place_outcomes = []
        for _, grp in valid_df.groupby("race_id"):
            z = grp["pure_score_z"].astype(float).values
            ln_q = grp["ln_market_q"].astype(float).values
            pw = fusion_probs(z, ln_q, fitted.alpha, fitted.beta)
            p_win_races.append(pw)
            finish = grp["finish_rank"].astype(int).values
            place_outcomes.append((finish <= 3).astype(float))

        lam2, lam3 = fit_stern_lambda(p_win_races, place_outcomes)

        market_only = fit_fusion_mle(build_race_tuples(fit_df), market_only=True, beta_bounds=bounds_b)
        test_df = attach_market_q(test_df, method=q_method, power=cfg.get("q_power", 0.81))
        test_ll_fusion = mean_logloss(test_df, fitted.alpha, fitted.beta)
        test_ll_market = mean_logloss(test_df, 0.0, market_only.beta)
        test_top1 = top1_hit_rate(test_df, fitted.alpha, fitted.beta)
        calib = calibration_bins(test_df, fitted.alpha, fitted.beta)

        tier = "formal" if fold["fold"] == 3 else "reference_l1_contaminated"
        l1_note = (
            None
            if fold["fold"] == 3
            else "L1 scores from v39 production model (train<=2023, ES uses 2024); TEST period is in-sample for L1"
        )

        fused = fuse_dataframe(
            test_df,
            alpha=fitted.alpha,
            beta=fitted.beta,
            lam2=lam2,
            lam3=lam3,
            q_method=q_method,
            q_power=cfg.get("q_power", 0.81),
        )
        all_test_frames.append(fused)

        fold_results.append(
            {
                "fold": fold["fold"],
                "judgment_tier": tier,
                "l1_contamination_note": l1_note,
                "alpha": fitted.alpha,
                "beta": fitted.beta,
                "market_beta_fit_on": "train+valid",
                "market_beta": market_only.beta,
                "lam2": lam2,
                "lam3": lam3,
                "lrt_p_value": lrt["p_value"],
                "test_logloss_fusion": test_ll_fusion,
                "test_logloss_market": test_ll_market,
                "test_logloss_beats_market": test_ll_fusion < test_ll_market,
                "test_top1": test_top1,
                "calibration_max_error_pp": calib["max_error_pp"],
            }
        )

    out_probs = pd.concat(all_test_frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = out_dir / f"probs_{cfg.get('version', 'benter_v1')}.parquet"
    out_probs.to_parquet(probs_path, index=False)

    report = {
        "folds": fold_results,
        "config": cfg,
        "formal_judgment_fold": 3,
        "formal_fold_summary": next((f for f in fold_results if f["fold"] == 3), None),
    }
    report_path = ROOT / "evaluation" / "reports" / f"fusion_{cfg.get('version', 'benter_v1')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    params_path = ROOT / "prob_fusion" / "data" / "fusion_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    inputs = {"scores": file_sha256(scores_path) if scores_path.exists() else "missing"}
    if features_path and features_path.exists():
        inputs["features"] = file_sha256(features_path)
    write_manifest(out_dir, inputs=inputs, config=cfg, extra={"fold_results": fold_results})

    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Benter conditional logit fusion")
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "prob_fusion" / "data")
    args = parser.parse_args()
    run_fit(args.scores, args.features, args.out)


if __name__ == "__main__":
    main()

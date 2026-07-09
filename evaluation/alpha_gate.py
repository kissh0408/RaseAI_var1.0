"""Alpha-gate harness for candidate signals.

Primary gate:
  - fit H0/H1 on 2023 only
  - evaluate logloss improvement on 2024 only
  - never use TEST 2025+ during candidate screening
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.odds_loader import attach_odds_from_se_parquet
from prob_fusion.src.fit_fusion import (
    build_race_tuples,
    fit_fusion_mle,
    gamma_likelihood_ratio_test,
    mean_logloss,
    top1_hit_rate,
)
from prob_fusion.src.market_prob import attach_market_q

FIT_YEAR = 2023
EVAL_YEAR = 2024
TEST_START = "2025-01-01"
GAMMA_LRT_ALPHA = 0.01
DELTA_LL_GATE = 0.0
TOP1_REGRESSION_FLOOR = 0.2999
CORR_STOP = 0.7


def _race_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    sd = s.std()
    if sd is None or not np.isfinite(sd) or sd < 1e-8:
        return pd.Series(0.0, index=series.index, dtype="float32")
    return ((s - s.mean()) / sd).astype("float32")


def attach_candidate_z(
    df: pd.DataFrame,
    *,
    score_col: str = "cand_score",
    race_id_col: str = "race_id",
    out_col: str = "cand_score_z",
) -> pd.DataFrame:
    """Attach race-internal candidate z-score."""
    out = df.copy()
    out[out_col] = out.groupby(race_id_col, sort=False)[score_col].transform(_race_zscore)
    return out


def split_alpha_gate_cv(
    df: pd.DataFrame,
    *,
    race_date_col: str = "race_date",
    fit_year: int = FIT_YEAR,
    eval_year: int = EVAL_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fit_df, eval_df) for 2023 fit and 2024 eval."""
    dates = pd.to_datetime(df[race_date_col])
    fit_df = df.loc[dates.dt.year == fit_year].copy()
    eval_df = df.loc[dates.dt.year == eval_year].copy()
    return fit_df, eval_df


def _safe_abs_corr(a: pd.Series, b: pd.Series) -> float | None:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if mask.sum() < 3:
        return None
    if aa.loc[mask].std() < 1e-12 or bb.loc[mask].std() < 1e-12:
        return None
    return float(abs(aa.loc[mask].corr(bb.loc[mask])))


def evaluate_leak_warnings(
    df: pd.DataFrame,
    *,
    x_col: str = "cand_score_z",
) -> dict[str, Any]:
    """Return correlation-based warnings for candidate screening."""
    finish_corr = _safe_abs_corr(df[x_col], df["finish_rank"]) if "finish_rank" in df.columns else None
    market_corr = _safe_abs_corr(df[x_col], df["ln_market_q"]) if "ln_market_q" in df.columns else None
    pure_corr = _safe_abs_corr(df[x_col], df["pure_score_z"]) if "pure_score_z" in df.columns else None
    leak_warning = any(
        corr is not None and corr >= CORR_STOP
        for corr in (finish_corr, market_corr, pure_corr)
    )
    return {
        "finish_rank_abs_corr": finish_corr,
        "market_abs_corr": market_corr,
        "pure_score_abs_corr": pure_corr,
        "corr_stop_threshold": CORR_STOP,
        "leak_warning": bool(leak_warning),
    }


def _ensure_gate_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    if "horse_num" not in out.columns and "horse_number" in out.columns:
        out["horse_num"] = out["horse_number"]
    if "odds" not in out.columns and "ln_market_q" not in out.columns:
        out = attach_odds_from_se_parquet(out)
    if "ln_market_q" not in out.columns:
        out = attach_market_q(out)
    if "cand_score_z" not in out.columns:
        out = attach_candidate_z(out)
    return out


def run_alpha_gate_on_dataframe(
    df: pd.DataFrame,
    *,
    candidate_name: str,
    gamma_bounds: tuple[float, float] = (0.0, 5.0),
    top1_floor: float = TOP1_REGRESSION_FLOOR,
) -> dict[str, Any]:
    """Evaluate a candidate dataframe and return a JSON-serializable report."""
    work = _ensure_gate_inputs(df)
    fit_df, eval_df = split_alpha_gate_cv(work)
    fit_races = build_race_tuples(fit_df, x_col="cand_score_z")
    eval_races = build_race_tuples(eval_df, x_col="cand_score_z")
    if not fit_races:
        raise ValueError("No fit races for alpha gate")
    if not eval_races:
        raise ValueError("No eval races for alpha gate")

    fitted = fit_fusion_mle(fit_races, gamma_bounds=gamma_bounds)
    lrt = gamma_likelihood_ratio_test(fit_races, fitted, gamma_bounds=gamma_bounds)

    h0_alpha = float(lrt["h0_alpha"])
    h0_beta = float(lrt["h0_beta"])
    eval_ll_h1 = mean_logloss(
        eval_df,
        fitted.alpha,
        fitted.beta,
        x_col="cand_score_z",
        gamma=fitted.gamma,
    )
    eval_ll_h0 = mean_logloss(eval_df, h0_alpha, h0_beta, x_col="cand_score_z", gamma=0.0)
    delta_ll = eval_ll_h0 - eval_ll_h1
    eval_top1 = top1_hit_rate(
        eval_df,
        fitted.alpha,
        fitted.beta,
        x_col="cand_score_z",
        gamma=fitted.gamma,
    )
    fit_leak_warnings = evaluate_leak_warnings(fit_df)
    eval_leak_warnings = evaluate_leak_warnings(eval_df)
    leak_warnings = {
        "fit": fit_leak_warnings,
        "eval": eval_leak_warnings,
        "leak_warning": bool(fit_leak_warnings["leak_warning"] or eval_leak_warnings["leak_warning"]),
    }
    gate_dates = pd.concat([pd.to_datetime(fit_df["race_date"]), pd.to_datetime(eval_df["race_date"])])
    used_test = bool((gate_dates >= pd.Timestamp(TEST_START)).any())

    gates = {
        "gamma_lrt_p_lt_0_01": float(lrt["p_value"]) < GAMMA_LRT_ALPHA,
        "delta_ll_per_race_positive": float(delta_ll) > DELTA_LL_GATE,
        "top1_regression_ok": float(eval_top1) >= top1_floor,
        "leak_warning_clear": not leak_warnings["leak_warning"],
    }
    passed = all(gates.values())
    return {
        "candidate": candidate_name,
        "fit_period": str(FIT_YEAR),
        "eval_period": str(EVAL_YEAR),
        "fit_n_races": len(fit_races),
        "eval_n_races": len(eval_races),
        "alpha": fitted.alpha,
        "beta": fitted.beta,
        "gamma": fitted.gamma,
        "fit_nll_h1": fitted.nll,
        "fit_nll_h0_gamma0": lrt["h0_nll"],
        "gamma_lrt_statistic": lrt["lr_statistic"],
        "gamma_lrt_p": lrt["p_value"],
        "eval_logloss_h1": eval_ll_h1,
        "eval_logloss_h0_gamma0": eval_ll_h0,
        "delta_ll_per_race": delta_ll,
        "eval_top1": eval_top1,
        "top1_floor": top1_floor,
        "leak_warnings": leak_warnings,
        "gates": gates,
        "pass": passed,
        "used_test_2025_plus": used_test,
    }


def load_candidate_dataset(
    candidate_path: Path,
    scores_path: Path,
    features_path: Path,
) -> pd.DataFrame:
    """Merge candidate scores with fold2 OOS scores and outcomes."""
    cand = pd.read_parquet(candidate_path)
    scores = pd.read_parquet(scores_path)
    features = pd.read_parquet(features_path, columns=["race_id", "horse_num", "finish_rank", "race_date"])
    for frame in (cand, scores, features):
        frame["race_id"] = frame["race_id"].astype(str)
        if "horse_num" not in frame.columns and "horse_number" in frame.columns:
            frame["horse_num"] = frame["horse_number"]
        frame["horse_num"] = pd.to_numeric(frame["horse_num"], errors="coerce").astype("Int64")
    base_cols = ["race_id", "horse_num", "pure_score_z"]
    merged = scores[base_cols].merge(cand[["race_id", "horse_num", "cand_score"]], on=["race_id", "horse_num"], how="inner")
    merged = merged.merge(features, on=["race_id", "horse_num"], how="inner")
    return _ensure_gate_inputs(merged)


def run_alpha_gate(
    candidate_path: Path,
    *,
    scores_path: Path,
    features_path: Path,
    candidate_name: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    name = candidate_name or candidate_path.stem
    df = load_candidate_dataset(candidate_path, scores_path, features_path)
    report = run_alpha_gate_on_dataframe(df, candidate_name=name)
    out_root = out_dir or (ROOT / "evaluation" / "reports")
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"alpha_gate_{name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alpha-gate candidate evaluation")
    parser.add_argument("--candidate", type=Path, required=True)
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
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    report = run_alpha_gate(
        args.candidate,
        scores_path=args.scores,
        features_path=args.features,
        candidate_name=args.name,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

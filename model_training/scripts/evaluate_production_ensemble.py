"""
evaluate_production_ensemble.py — rank 本番 OOF 評価（ROI/MDD/Sharpe）

train_model() が出力した evaluation_all_non_leak.csv を strategy 層でバックテストする。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_DIR = PROJECT_ROOT / "model_training" / "data" / "03_train"
EVAL_PATH = TRAIN_DIR / "evaluation_all_non_leak.csv"
FOLD_PATH = TRAIN_DIR / "fold_metrics_all_non_leak.csv"
STRATEGY_CFG = PROJECT_ROOT / "strategy" / "config" / "strategy_config.json"
CALIB_PATH = PROJECT_ROOT / "strategy" / "models" / "calibration_isotonic.json"
BASELINE_META = PROJECT_ROOT / "model_training" / "models" / "ensemble_v5" / "ensemble_meta.json"

GATES = {"roi_min": 1.05, "mdd_min": -0.20, "sharpe_min": 0.10, "n_bets_min": 500}


def _load_strategy_config():
    from strategy.src.betting_framework import StrategyConfig

    cfg = json.loads(STRATEGY_CFG.read_text(encoding="utf-8"))
    return StrategyConfig(
        fractional_kelly=float(cfg.get("kelly_fraction", 0.08)),
        min_edge=float(cfg.get("min_edge", 0.02)),
        min_odds=float(cfg.get("min_odds", 1.2)),
        ev_threshold=float(cfg.get("ev_threshold", 1.05)),
    )


def _metrics_for_year(eval_df: pd.DataFrame, year: int | None) -> dict:
    from strategy.src.betting_framework import ProbabilityCalibrator, run_betting_backtest

    df = eval_df.copy()
    if year is not None and "valid_year" in df.columns:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
    if df.empty:
        return {"error": f"no rows for year={year}"}

    calibrator = ProbabilityCalibrator.from_json(CALIB_PATH) if CALIB_PATH.exists() else None
    config = _load_strategy_config()
    _, _, metrics = run_betting_backtest(df, config, calibrator=calibrator)
    roi = float(metrics.get("return_multiple", metrics.get("roi", 0)))
    mdd = float(metrics.get("max_drawdown_rate", metrics.get("mdd", -1)))
    sharpe = float(metrics.get("sharpe", 0))
    n_bets = int(metrics.get("n_bets", 0))
    return {
        "year": year,
        "roi": roi,
        "mdd": mdd,
        "sharpe": sharpe,
        "n_bets": n_bets,
        "raw": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    parser.add_argument("--baseline-meta", type=Path, default=BASELINE_META)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.eval_path.exists():
        print(f"[NG] missing {args.eval_path}")
        return 1

    from strategy.src.betting_framework import load_evaluation

    eval_df = load_evaluation(args.eval_path)
    overall = _metrics_for_year(eval_df, None)
    y2025 = _metrics_for_year(eval_df, 2025)
    y2026 = _metrics_for_year(eval_df, 2026) if "valid_year" in eval_df.columns else {}

    baseline = {}
    if args.baseline_meta.exists():
        baseline = json.loads(args.baseline_meta.read_text(encoding="utf-8"))

    rank1_oof: dict = {}
    if FOLD_PATH.exists():
        fm = pd.read_csv(FOLD_PATH)
        r1 = fm[fm["rank"] == 1]
        for _, row in r1.iterrows():
            yr = int(row["valid_year"])
            rank1_oof[f"top3_overlap_{yr}"] = float(row["top3_overlap"]) if pd.notna(row.get("top3_overlap")) else None

    def _gate(m: dict) -> dict:
        if "error" in m:
            return {"passed": False, "error": m["error"]}
        checks = {
            "roi": m["roi"] >= GATES["roi_min"],
            "mdd": m["mdd"] >= GATES["mdd_min"],
            "sharpe": m["sharpe"] >= GATES["sharpe_min"],
            "n_bets": m["n_bets"] >= GATES["n_bets_min"],
        }
        base_roi = float(baseline.get("backtest_roi") or 1.275)
        checks["roi_vs_baseline"] = m["roi"] >= base_roi - 0.02
        return {"checks": checks, "passed": all(checks.values())}

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_path": str(args.eval_path),
        "baseline_meta": str(args.baseline_meta),
        "baseline_ref": {
            "backtest_roi": baseline.get("backtest_roi"),
            "backtest_sharpe": baseline.get("backtest_sharpe"),
            "backtest_mdd": baseline.get("backtest_mdd"),
            "backtest_n_bets": baseline.get("backtest_n_bets"),
        },
        "overall": overall,
        "fold_2025": y2025,
        "fold_2026": y2026,
        "rank1_oof_top3_overlap": rank1_oof,
        "gates_overall": _gate(overall),
        "gates_2025": _gate(y2025),
    }

    out = args.out or (TRAIN_DIR / "production_ensemble_eval_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nSaved: {out}")

    passed = report["gates_2025"].get("passed") or report["gates_overall"].get("passed")
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())

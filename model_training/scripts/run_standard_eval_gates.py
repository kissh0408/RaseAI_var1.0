"""
run_standard_eval_gates.py — 標準評価パイプラインの合格ゲート検証

本番接続後の calibrator + strategy_config（race_num 含む）で
年別・2025 テストフォールドの ROI/MDD/Sharpe/n_bets を確認する。
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

SPECv2_OOF = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_specv2_oof.csv"
STRATEGY_CFG = PROJECT_ROOT / "strategy" / "config" / "strategy_config.json"
OUT = PROJECT_ROOT / "model_training" / "data" / "03_train" / "standard_eval_gates_report.json"

GATES = {"roi_min": 1.05, "mdd_min": -0.20, "sharpe_min": 0.10, "n_bets_min": 500}


def _gate(m: dict) -> dict:
    return {
        "roi": m.get("roi", 0) >= GATES["roi_min"],
        "mdd": m.get("mdd", -1) >= GATES["mdd_min"],
        "sharpe": m.get("sharpe", 0) >= GATES["sharpe_min"],
        "n_bets": m.get("n_bets", 0) >= GATES["n_bets_min"],
    }


def _eval_year(eval_df: pd.DataFrame, year: int | None) -> dict:
    from main.pipeline.strategy_pipeline import (
        resolve_strategy_calibration_path,
        strategy_config_from_runtime,
        load_strategy_runtime_config,
    )
    from strategy.src.betting_framework import ProbabilityCalibrator, run_betting_backtest

    df = eval_df.copy()
    if year is not None:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()

    runtime = load_strategy_runtime_config(STRATEGY_CFG)
    config = strategy_config_from_runtime(runtime)
    cal_path = resolve_strategy_calibration_path(PROJECT_ROOT, STRATEGY_CFG)
    calibrator = ProbabilityCalibrator.from_json(cal_path) if cal_path.is_file() else None

    _, _, metrics = run_betting_backtest(df, config, calibrator=calibrator)
    out = {
        "year": year,
        "calibration_path": str(cal_path),
        "race_num_min": runtime.get("race_num_min"),
        "race_num_max": runtime.get("race_num_max"),
        "roi": float(metrics.get("return_multiple", 0)),
        "mdd": float(metrics.get("max_drawdown_rate", 0)),
        "sharpe": float(metrics.get("sharpe", 0)),
        "n_bets": int(metrics.get("n_bets", 0)),
        "hit_rate": float(metrics.get("hit_rate", 0)),
    }
    checks = _gate(out)
    out["gates"] = checks
    out["passed"] = all(checks.values())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from strategy.src.betting_framework import load_evaluation

    eval_df = load_evaluation(SPECv2_OOF)
    years = sorted(pd.to_numeric(eval_df["valid_year"], errors="coerce").dropna().astype(int).unique())

    rows = [_eval_year(eval_df, int(y)) for y in years]
    rows.append(_eval_year(eval_df, None))
    y2025 = next(r for r in rows if r.get("year") == 2025)

    all_period = next(r for r in rows if r.get("year") is None)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_csv": str(SPECv2_OOF),
        "strategy_config": str(STRATEGY_CFG),
        "gates": GATES,
        "by_year": rows,
        "primary_test_fold_2025": y2025,
        "all_period": all_period,
        "all_period_mdd_needs_mitigation": all_period["mdd"] < GATES["mdd_min"],
        "all_period_note": (
            "全期間 MDD は参考。本番移行ゲートは CLAUDE.md のテストフォールド（2025）を優先。"
            if all_period["mdd"] < GATES["mdd_min"]
            else "全期間も MDD 合格"
        ),
        "deployment_e2e_eligible": bool(y2025.get("passed")),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {args.out}")
    print(f"calibrator: {y2025['calibration_path']}")
    print(
        f"2025: ROI={y2025['roi']*100:.1f}% MDD={y2025['mdd']*100:.1f}% "
        f"Sharpe={y2025['sharpe']:.3f} n={y2025['n_bets']} passed={y2025['passed']}"
    )
    print(
        f"all:  ROI={all_period['roi']*100:.1f}% MDD={all_period['mdd']*100:.1f}% "
        f"Sharpe={all_period['sharpe']:.3f} n={all_period['n_bets']} passed={all_period['passed']}"
    )
    for r in rows:
        if r.get("year") is None:
            continue
        flag = "OK" if r["passed"] else "NG"
        print(
            f"  {r['year']}: ROI={r['roi']*100:.1f}% MDD={r['mdd']*100:.1f}% "
            f"Sharpe={r['sharpe']:.3f} [{flag}]"
        )
    return 0 if report["deployment_e2e_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

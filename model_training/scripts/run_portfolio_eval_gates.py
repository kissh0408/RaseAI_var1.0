"""
run_portfolio_eval_gates.py — C2 win+wide ポートフォリオ OOF ゲート

北極星 KPI: Floor ROI ≥115%, Target ≥130%, MDD ≤-20%, 単勝 151%±2pp
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPECv2_OOF = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_specv2_oof.csv"
STRATEGY_CFG = PROJECT_ROOT / "strategy" / "config" / "strategy_config.json"
TRAIN_CFG = PROJECT_ROOT / "model_training" / "config" / "train_config.json"
ODDS_DIR = PROJECT_ROOT / "common" / "data" / "output" / "odds"
OUT = PROJECT_ROOT / "model_training" / "data" / "03_train" / "portfolio_eval_gates_report.json"

GATES = {
    "portfolio_roi_floor": 1.15,
    "portfolio_roi_target": 1.30,
    "portfolio_mdd_min": -0.20,
    "win_roi_min": 1.49,
    "win_roi_max": 1.53,
    "n_bets_min": 500,
}

K_GRID = [0.5, 0.65, 0.8]
CAP_RATIO_GRID = [0.85, 0.88, 0.90, 0.92, 0.95]
VALID_YEARS = [2024]  # fold3 valid period (train_config walkforward)
DEFAULT_IND_CAP_RATIO = 0.92


def _ensure_combo_odds(eval_df: pd.DataFrame, years: list[int]) -> None:
    from model_training.scripts.compare_production_ensemble_eval import _ensure_combo_odds as _ensure

    _ensure(eval_df, years)


def _load_eval(year: int | None) -> pd.DataFrame:
    from strategy.src.betting_framework import load_evaluation

    df = load_evaluation(SPECv2_OOF)
    if year is not None:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
    return df


def _run_mode(
    eval_df: pd.DataFrame,
    *,
    sizing_mode: str,
    growth_ratio_min: float = 0.65,
    mc_samples: int = 500,
    ind_cap_ratio: float = DEFAULT_IND_CAP_RATIO,
) -> dict[str, Any]:
    from main.pipeline.strategy_pipeline import (
        load_strategy_runtime_config,
        resolve_strategy_calibration_path,
        strategy_config_from_runtime,
    )
    from strategy.src.betting_framework import ProbabilityCalibrator
    from strategy.src.portfolio_backtest import DEFAULT_IND_CAP_RATIO, run_portfolio_backtest

    runtime = load_strategy_runtime_config(STRATEGY_CFG)
    config = strategy_config_from_runtime(runtime)
    cal_path = resolve_strategy_calibration_path(PROJECT_ROOT, STRATEGY_CFG)
    calibrator = ProbabilityCalibrator.from_json(cal_path) if cal_path.is_file() else None

    years = sorted(pd.to_numeric(eval_df["valid_year"], errors="coerce").dropna().astype(int).unique().tolist())
    _ensure_combo_odds(eval_df, years)

    _, _, metrics = run_portfolio_backtest(
        eval_df,
        config,
        ODDS_DIR,
        calibrator=calibrator,
        runtime=runtime,
        sizing_mode=sizing_mode,
        mc_samples=mc_samples,
        mc_seed=int(config.random_seed),
        growth_ratio_min=growth_ratio_min,
        wide_bets_enabled=bool(runtime.get("wide_bets_enabled", True)),
        ind_cap_ratio=ind_cap_ratio,
    )
    comb = metrics["combined"]
    wide = metrics.get("wide") or {}

    from strategy.src.betting_framework import run_betting_backtest

    _, _, win_only_metrics = run_betting_backtest(eval_df, config, calibrator=calibrator)
    win = win_only_metrics
    return {
        "sizing_mode": sizing_mode,
        "growth_ratio_min": growth_ratio_min if sizing_mode == "portfolio_kelly_fractional" else None,
        "portfolio_roi": float(comb.get("return_multiple", 0)),
        "portfolio_mdd": float(comb.get("max_drawdown_rate", -1)),
        "portfolio_sharpe": float(comb.get("sharpe", 0)),
        "portfolio_n_bets": int(comb.get("n_bets", 0)),
        "portfolio_hit_rate": float(comb.get("hit_rate", 0)),
        "win_roi": float(win.get("return_multiple", 0)) if win else 0.0,
        "win_mdd": float(win.get("max_drawdown_rate", 0)) if win else 0.0,
        "win_n_bets": int(win.get("n_bets", 0)) if win else 0,
        "wide_roi": float(wide.get("return_multiple", 0)) if wide else 0.0,
        "wide_n_bets": int(wide.get("n_bets", 0)) if wide else 0,
    }


def _gate_checks(m: dict[str, Any], *, require_c2: bool) -> dict[str, bool]:
    checks = {
        "portfolio_roi_floor": m.get("portfolio_roi", 0) >= GATES["portfolio_roi_floor"],
        "portfolio_roi_target": m.get("portfolio_roi", 0) >= GATES["portfolio_roi_target"],
        "portfolio_mdd": m.get("portfolio_mdd", -1) >= GATES["portfolio_mdd_min"],
        "win_roi_band": GATES["win_roi_min"] <= m.get("win_roi", 0) <= GATES["win_roi_max"],
        "n_bets": m.get("portfolio_n_bets", 0) >= GATES["n_bets_min"],
    }
    if require_c2:
        checks["c2_floor_pass"] = (
            checks["portfolio_roi_floor"] and checks["portfolio_mdd"] and checks["n_bets"]
        )
    return checks


def _select_hyperparams_on_valid() -> tuple[float, float, dict]:
    """Valid 期間のみ: growth_ratio_min と independent cap ratio を選択。"""
    valid_df = _load_eval(None)
    valid_df = valid_df[pd.to_numeric(valid_df["valid_year"], errors="coerce").isin(VALID_YEARS)].copy()
    grid_results: dict[str, Any] = {}
    best_k = 0.5
    best_ratio = DEFAULT_IND_CAP_RATIO
    best_score = -1e9
    for ratio in CAP_RATIO_GRID:
        for k in K_GRID:
            m = _run_mode(
                valid_df,
                sizing_mode="portfolio_kelly_fractional",
                growth_ratio_min=k,
                ind_cap_ratio=ratio,
            )
            key = f"ratio={ratio}_k={k}"
            grid_results[key] = m
            mdd_ok = m["portfolio_mdd"] >= GATES["portfolio_mdd_min"]
            score = m["portfolio_roi"] if mdd_ok else -1.0
            if score > best_score:
                best_score = score
                best_k = k
                best_ratio = ratio
    return best_k, best_ratio, grid_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "independent", "portfolio_kelly", "portfolio_kelly_fractional", "all"],
        default="all",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--growth-ratio-min", type=float, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not SPECv2_OOF.is_file():
        print(f"ERROR: missing {SPECv2_OOF}")
        return 1

    eval_df = _load_eval(args.year)
    if eval_df.empty:
        print(f"ERROR: empty eval for year={args.year}")
        return 1

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": "C2_portfolio",
        "eval_csv": str(SPECv2_OOF),
        "strategy_config": str(STRATEGY_CFG),
        "year": args.year,
        "gates": GATES,
        "mc_samples": args.mc_samples,
    }

    modes_to_run: list[str]
    if args.mode == "all":
        modes_to_run = ["baseline", "portfolio_kelly", "portfolio_kelly_fractional"]
    else:
        modes_to_run = [args.mode]

    if "portfolio_kelly_fractional" in modes_to_run and args.growth_ratio_min is None:
        best_k, best_ratio, valid_grid = _select_hyperparams_on_valid()
        report["valid_hyperparam_selection"] = {
            "years": VALID_YEARS,
            "grid": valid_grid,
            "selected_k": best_k,
            "selected_ind_cap_ratio": best_ratio,
        }
        growth_k = best_k
        ind_cap_ratio = best_ratio
    else:
        from main.pipeline.strategy_pipeline import load_strategy_runtime_config

        runtime = load_strategy_runtime_config(STRATEGY_CFG)
        growth_k = float(
            args.growth_ratio_min
            if args.growth_ratio_min is not None
            else runtime.get("portfolio_growth_ratio_min", 0.5)
        )
        ind_cap_ratio = float(runtime.get("portfolio_ind_cap_ratio", DEFAULT_IND_CAP_RATIO))

    results: dict[str, Any] = {}
    for mode in modes_to_run:
        k = growth_k if mode == "portfolio_kelly_fractional" else 0.65
        m = _run_mode(
            eval_df,
            sizing_mode=mode,
            growth_ratio_min=k,
            mc_samples=args.mc_samples,
            ind_cap_ratio=ind_cap_ratio if mode != "baseline" else DEFAULT_IND_CAP_RATIO,
        )
        checks = _gate_checks(m, require_c2=(mode != "baseline"))
        m["checks"] = checks
        m["passed_c2_floor"] = all(
            [checks["portfolio_roi_floor"], checks["portfolio_mdd"], checks["n_bets"]]
        ) if mode != "baseline" else None
        m["passed_c2_target"] = checks["portfolio_roi_target"] if mode != "baseline" else None
        results[mode] = m
        print(
            f"{mode}: ROI={m['portfolio_roi']*100:.1f}% MDD={m['portfolio_mdd']*100:.1f}% "
            f"win={m['win_roi']*100:.1f}% n={m['portfolio_n_bets']} "
            f"floor={m.get('passed_c2_floor')}"
        )

    primary = results.get("portfolio_kelly_fractional") or results.get("portfolio_kelly") or results.get("baseline")
    baseline = results.get("baseline")
    if primary and baseline:
        report["mdd_improvement_vs_baseline"] = float(baseline["portfolio_mdd"]) - float(primary["portfolio_mdd"])

    report["results"] = results
    report["primary"] = primary
    report["deployment_eligible"] = bool(
        primary
        and primary.get("passed_c2_floor")
        and primary.get("checks", {}).get("win_roi_band", False)
    )
    report["target_eligible"] = bool(primary and primary.get("passed_c2_target"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")
    print(f"deployment_eligible={report['deployment_eligible']} target_eligible={report['target_eligible']}")

    if args.mode in {"portfolio_kelly", "portfolio_kelly_fractional", "all"}:
        return 0 if report["deployment_eligible"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

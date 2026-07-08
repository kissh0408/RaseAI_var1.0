"""
diagnose_win_mdd.py — 単勝 WIN バックテストの MDD 要因切り分け

標準 OOF（evaluation_specv2_oof.csv）を固定入力とし、
race_num / calibrator / Kelly / dynamic_edge / flat の各条件を 1 変数ずつ変えて比較する。
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPECv2_OOF = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_specv2_oof.csv"
EVAL_FALLBACK = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_all_non_leak.csv"
STRATEGY_CFG_PATH = PROJECT_ROOT / "strategy" / "config" / "strategy_config.json"
LEGACY_CALIB = PROJECT_ROOT / "strategy" / "models" / "calibration_isotonic.json"
OUT_JSON = PROJECT_ROOT / "model_training" / "data" / "03_train" / "mdd_diagnosis_report.json"
OUT_MD = PROJECT_ROOT / "docs" / "analysis" / "mdd_diagnosis_report_20260617.md"


def _load_eval() -> pd.DataFrame:
    from strategy.src.betting_framework import load_evaluation

    path = SPECv2_OOF if SPECv2_OOF.exists() else EVAL_FALLBACK
    return load_evaluation(path)


def _runtime_cfg(**overrides) -> dict:
    cfg = json.loads(STRATEGY_CFG_PATH.read_text(encoding="utf-8"))
    cfg = deepcopy(cfg)
    cfg.update(overrides)
    return cfg


def _build_config(runtime: dict):
    from main.pipeline.strategy_pipeline import strategy_config_from_runtime

    return strategy_config_from_runtime(runtime)


def _fit_oof_calibrator(eval_df: pd.DataFrame, fit_until_year: int) -> IsotonicRegression:
    sub = eval_df[pd.to_numeric(eval_df["valid_year"], errors="coerce") < fit_until_year].copy()
    x = pd.to_numeric(sub["pred_rank1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = (pd.to_numeric(sub["finish_rank"], errors="coerce") == 1).astype(float).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    return iso


def _iso_to_calibrator(iso: IsotonicRegression):
    from strategy.src.betting_framework import ProbabilityCalibrator

    return ProbabilityCalibrator(
        method="isotonic",
        params={
            "method": "isotonic",
            "x_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "interpolation": "linear",
        },
    )


def _run_scenario(
    eval_df: pd.DataFrame,
    *,
    name: str,
    runtime: dict,
    calibrator=None,
    year: int | None = None,
) -> dict[str, Any]:
    from strategy.src.betting_framework import ProbabilityCalibrator, run_betting_backtest

    df = eval_df.copy()
    if year is not None:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
    if df.empty:
        return {"scenario": name, "error": "empty"}

    config = _build_config(runtime)
    _, race_df, metrics = run_betting_backtest(df, config, calibrator=calibrator)
    roi = float(metrics.get("return_multiple", metrics.get("roi", 0)))
    mdd = float(metrics.get("max_drawdown_rate", metrics.get("mdd", 0)))
    sharpe = float(metrics.get("sharpe", 0))
    n_bets = int(metrics.get("n_bets", 0))
    hit = float(metrics.get("hit_rate", 0))

    # 月次 DD 停止シミュレーション（race 単位 profit、valid_year を月 proxy）
    monthly_mdd = None
    if not race_df.empty and "profit" in race_df.columns:
        rd = race_df.copy()
        if "valid_year" not in rd.columns and year is not None:
            rd["valid_year"] = year
        if "valid_year" in df.columns:
            yr_map = df.groupby("race_id")["valid_year"].first()
            rd["valid_year"] = rd["race_id"].map(yr_map)
        rd["race_date"] = pd.to_datetime(
            rd["valid_year"].astype(str) + "-06-15", errors="coerce"
        )
        try:
            from strategy.src.ev_filters import apply_monthly_drawdown_filter

            lim = float(runtime.get("monthly_drawdown_limit", -0.08))
            bankroll = float(runtime.get("initial_bankroll", 100_000))
            filtered = apply_monthly_drawdown_filter(
                rd.rename(columns={"profit": "profit"}),
                monthly_dd_limit=lim,
                bankroll=bankroll,
                date_col="race_date",
                profit_col="profit",
            )
            if len(filtered) < len(rd):
                from strategy.src.betting_framework import compute_metrics

                m2 = compute_metrics(
                    pd.DataFrame(),
                    filtered,
                    initial_bankroll=bankroll,
                )
                monthly_mdd = {
                    "mdd_rate": float(m2.get("max_drawdown_rate", 0)),
                    "roi": float(m2.get("return_multiple", 0)),
                    "n_races_kept": int(len(filtered)),
                    "limit": lim,
                }
        except Exception as exc:
            monthly_mdd = {"error": str(exc)}

    return {
        "scenario": name,
        "year": year,
        "race_num_min": runtime.get("race_num_min"),
        "race_num_max": runtime.get("race_num_max"),
        "kelly_fraction": runtime.get("kelly_fraction"),
        "dynamic_edge_enabled": runtime.get("dynamic_edge_enabled"),
        "calibrator": calibrator is not None,
        "roi": roi,
        "mdd": mdd,
        "sharpe": sharpe,
        "n_bets": n_bets,
        "hit_rate": hit,
        "monthly_dd_sim": monthly_mdd,
    }


def _scenarios(eval_df: pd.DataFrame, year: int | None) -> list[dict]:
    from strategy.src.betting_framework import ProbabilityCalibrator

    legacy = ProbabilityCalibrator.from_json(LEGACY_CALIB) if LEGACY_CALIB.exists() else None
    oof_iso = _fit_oof_calibrator(eval_df, fit_until_year=year if year is not None else 2025)
    oof_cal = _iso_to_calibrator(oof_iso)

    base = _runtime_cfg()
    rows: list[dict] = []

    defs = [
        ("production_baseline", {"race_num_min": 8, "race_num_max": 12}, legacy),
        ("v5_meta_baseline", {"race_num_min": None, "race_num_max": None}, legacy),
        ("no_race_num_filter", {"race_num_min": None, "race_num_max": None}, legacy),
        ("race_num_8_12_only", {"race_num_min": 8, "race_num_max": 12}, legacy),
        ("no_calibrator", {"race_num_min": 8, "race_num_max": 12}, None),
        ("no_calibrator_v5_meta", {"race_num_min": None, "race_num_max": None}, None),
        ("calibrator_oof_fit", {"race_num_min": 8, "race_num_max": 12}, oof_cal),
        ("calibrator_oof_fit_v5_meta", {"race_num_min": None, "race_num_max": None}, oof_cal),
        ("kelly_0.05", {"race_num_min": 8, "race_num_max": 12, "kelly_fraction": 0.05}, legacy),
        ("kelly_0.04", {"race_num_min": 8, "race_num_max": 12, "kelly_fraction": 0.04}, legacy),
        ("flat_stake", {"race_num_min": 8, "race_num_max": 12}, legacy),
        ("no_dynamic_edge", {"race_num_min": 8, "race_num_max": 12, "dynamic_edge_enabled": False}, legacy),
        ("no_dynamic_edge_v5_meta", {"race_num_min": None, "race_num_max": None, "dynamic_edge_enabled": False}, legacy),
    ]

    for name, overrides, cal in defs:
        rt = _runtime_cfg(**overrides)
        row = _run_scenario(eval_df, name=name, runtime=rt, calibrator=cal, year=year)
        if name == "flat_stake":
            from strategy.src.betting_framework import run_betting_backtest

            df = eval_df.copy()
            if year is not None:
                df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
            cfg = _build_config(rt)
            cfg = replace(cfg, force_flat_staking=True, sizing_mode="flat")
            _, _, metrics = run_betting_backtest(df, cfg, calibrator=cal)
            row.update(
                {
                    "roi": float(metrics.get("return_multiple", 0)),
                    "mdd": float(metrics.get("max_drawdown_rate", 0)),
                    "sharpe": float(metrics.get("sharpe", 0)),
                    "n_bets": int(metrics.get("n_bets", 0)),
                    "hit_rate": float(metrics.get("hit_rate", 0)),
                }
            )
        rows.append(row)

    # race_num 効果の差分（同一 calibrator）
    prod = next(r for r in rows if r["scenario"] == "production_baseline")
    meta = next(r for r in rows if r["scenario"] == "v5_meta_baseline")
    rows.append(
        {
            "scenario": "delta_race_num_8_12",
            "year": year,
            "mdd_delta_production_minus_v5meta": prod["mdd"] - meta["mdd"],
            "roi_delta": prod["roi"] - meta["roi"],
            "n_bets_delta": prod["n_bets"] - meta["n_bets"],
            "interpretation": (
                "mdd_delta < 0 → race_num 8-12 が MDD を悪化"
                if prod["mdd"] < meta["mdd"]
                else "mdd_delta >= 0 → race_num 8-12 は MDD 改善または同等"
            ),
        }
    )
    return rows


def _write_md(report: dict) -> None:
    lines = [
        "# MDD 切り分けレポート（specv2 OOF 固定）",
        "",
        f"生成: {report['created_at']}",
        "",
        "## サマリー",
        "",
    ]
    for yblock in report["years"]:
        y = yblock["year"]
        lines.append(f"### {y}")
        delta = next((r for r in yblock["scenarios"] if r["scenario"] == "delta_race_num_8_12"), None)
        if delta:
            lines.append(
                f"- **race_num 8–12 効果**: MDD Δ={delta['mdd_delta_production_minus_v5meta']*100:.1f}pp, "
                f"ROI Δ={delta['roi_delta']*100:.1f}pp, n_bets Δ={delta['n_bets_delta']}"
            )
            lines.append(f"- {delta['interpretation']}")
        lines.append("")
        lines.append("| scenario | ROI | MDD | Sharpe | n_bets | hit |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in yblock["scenarios"]:
            if r["scenario"].startswith("delta_"):
                continue
            if "error" in r:
                continue
            lines.append(
                f"| {r['scenario']} | {r['roi']*100:.1f}% | {r['mdd']*100:.1f}% | "
                f"{r['sharpe']:.3f} | {r['n_bets']} | {r['hit_rate']*100:.1f}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## 結論テンプレート",
            "",
            "1. race_num 8–12 は MDD を **悪化** させているか → delta 行を参照",
            "2. calibrator 無し vs legacy vs OOF fit → 最大 MDD 改善シナリオを特定",
            "3. Kelly 縮小 / flat / dynamic_edge off → 戦略層の改善余地",
            "4. monthly_dd_limit は本番停止用。バックテスト MDD とは別指標（sim 列参照）",
            "",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", default=["2025", "all"])
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    eval_df = _load_eval()
    years: list[int | None] = []
    for y in args.years:
        years.append(None if str(y).lower() == "all" else int(y))

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_path": str(SPECv2_OOF if SPECv2_OOF.exists() else EVAL_FALLBACK),
        "years": [],
    }
    for year in years:
        ykey = "all" if year is None else year
        print(f"[diagnose] year={ykey} ...")
        scenarios = _scenarios(eval_df, year)
        report["years"].append({"year": ykey, "scenarios": scenarios})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(report)
    print(f"Saved: {args.out}")
    print(f"Saved: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

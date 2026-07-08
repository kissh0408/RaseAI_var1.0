"""
run_regime_heatmap.py — Track D-1: 学習期間のみ 年別 ROI/MDD ヒートマップ

OOF eval に馬場列が無いため v1 は年×race_num 帯で可視化。
静的除外ルールは採用しない。
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
OUT = PROJECT_ROOT / "model_training" / "data" / "03_train" / "regime_heatmap_report.json"

TRAIN_YEAR_MAX = 2023
WEAK_YEARS = {2019, 2020, 2022}


def _mdd_from_profits(profits: pd.Series, initial: float = 100_000.0) -> float:
    if profits.empty:
        return 0.0
    equity = initial + profits.cumsum()
    running_max = equity.cummax()
    dd_rate = (equity - running_max) / running_max.clip(lower=1.0)
    return float(dd_rate.min())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from main.pipeline.strategy_pipeline import (
        load_strategy_runtime_config,
        resolve_strategy_calibration_path,
        strategy_config_from_runtime,
    )
    from strategy.src.betting_framework import ProbabilityCalibrator, load_evaluation, run_betting_backtest
    from strategy.src.race_filters import attach_race_num

    eval_df = load_evaluation(SPECv2_OOF)
    if "race_num" not in eval_df.columns:
        eval_df = attach_race_num(eval_df)
    eval_df = eval_df[pd.to_numeric(eval_df["valid_year"], errors="coerce") <= TRAIN_YEAR_MAX].copy()

    runtime = load_strategy_runtime_config(PROJECT_ROOT / "strategy" / "config" / "strategy_config.json")
    config = strategy_config_from_runtime(runtime)
    cal_path = resolve_strategy_calibration_path(PROJECT_ROOT)
    calibrator = ProbabilityCalibrator.from_json(cal_path) if cal_path.is_file() else None

    _, races, _ = run_betting_backtest(eval_df, config, calibrator=calibrator)
    meta = eval_df[["race_id", "valid_year", "race_num"]].drop_duplicates("race_id")
    race_pnl = races.merge(meta, on="race_id", how="left")
    race_pnl["race_num_band"] = pd.cut(
        pd.to_numeric(race_pnl["race_num"], errors="coerce").fillna(0),
        bins=[0, 6, 9, 12, 99],
        labels=["1-6", "7-9", "10-12", "13+"],
    )

    cells = []
    for (year, band), g in race_pnl.groupby(["valid_year", "race_num_band"], dropna=False):
        g = g.sort_values("race_id")
        invest = float(g["invest"].sum())
        ret = float(g["return"].sum())
        bet_mask = g["invest"] > 0
        mdd = _mdd_from_profits(g.loc[bet_mask, "profit"]) if bet_mask.any() else 0.0
        cells.append(
            {
                "year": int(year),
                "race_num_band": str(band),
                "n_races_bet": int(bet_mask.sum()),
                "roi": ret / invest if invest > 0 else 0.0,
                "mdd": mdd,
                "invest": invest,
            }
        )

    year_agg = []
    for year, g in race_pnl.groupby("valid_year"):
        g = g.sort_values("race_id")
        invest = float(g["invest"].sum())
        ret = float(g["return"].sum())
        bet_mask = g["invest"] > 0
        year_agg.append(
            {
                "year": int(year),
                "roi": ret / invest if invest > 0 else 0.0,
                "mdd": _mdd_from_profits(g.loc[bet_mask, "profit"]) if bet_mask.any() else 0.0,
                "n_races_bet": int(bet_mask.sum()),
            }
        )

    weak = [c for c in year_agg if c["year"] in WEAK_YEARS and (c["mdd"] < -0.30 or c["roi"] < 1.0)]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": "D1_regime_heatmap",
        "train_year_max": TRAIN_YEAR_MAX,
        "cells_year_x_race_num_band": cells,
        "year_summary": year_agg,
        "weak_years_train_only": weak,
        "gate_visualized": len(weak) > 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out} cells={len(cells)} weak_years={len(weak)}")
    return 0 if report["gate_visualized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

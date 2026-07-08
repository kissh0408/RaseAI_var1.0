"""CLI: OOS 正式ベッティングバックテスト（Phase 3）— 単勝のみ。

VALID=2024 で EV 閾値を選択し、TEST=2025+ で 1 回だけ判定する。
α,β,λ は evaluation/reports/fusion_oos_fold2.json の formal 値を使用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from betting.src.backtest import (
    fuse_period,
    load_betting_config,
    load_scored_odds_frame,
    simulate_bets,
)
from betting.src.ev_engine import select_ev_threshold_on_valid
from prob_fusion.src.oos_protocol import TEST_START

VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
MIN_FORMAL_BETS = 200


def run_backtest_oos(scores_path: Path, features_path: Path, fusion_report_path: Path) -> dict:
    fusion_report = json.loads(fusion_report_path.read_text(encoding="utf-8"))
    formal = fusion_report["formal"]
    params = {
        "alpha": float(formal["alpha"]),
        "beta": float(formal["beta"]),
        "lam2": float(formal["lam2"]),
        "lam3": float(formal["lam3"]),
    }
    fusion_cfg = fusion_report.get("config", {})
    bet_cfg = load_betting_config()

    df = load_scored_odds_frame(scores_path, features_path)
    dates = pd.to_datetime(df["race_date"])
    valid_df = df.loc[(dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))]
    test_df = df.loc[dates >= pd.Timestamp(TEST_START)]

    valid_fused = fuse_period(valid_df, params, fusion_cfg)
    test_fused = fuse_period(test_df, params, fusion_cfg)

    thr_result = select_ev_threshold_on_valid(
        valid_fused,
        bet_cfg.get("ev_threshold_grid", [1.05]),
        bet_type="win",
        ev_haircut=bet_cfg.get("ev_haircut", 0.95),
    )
    if thr_result.fallback_used:
        report = {
            "version": "benter_betting_oos_fold2",
            "status": "skipped",
            "reason": "VALID threshold tuning failed",
            "ev_threshold_warnings": thr_result.warnings,
        }
    else:
        bets = simulate_bets(
            test_fused, bet_type="win", ev_threshold=thr_result.threshold, cfg=bet_cfg
        )
        n_bets = len(bets)
        total_stake = float(bets["stake"].sum()) if n_bets else 0.0
        total_payout = float(bets["payout"].sum()) if n_bets else 0.0
        roi = total_payout / total_stake * 100.0 if total_stake > 0 else 0.0
        sharpe = 0.0
        if n_bets > 1:
            rets = bets["pnl"].astype(float) / bets["stake"].astype(float).clip(lower=1.0)
            std = rets.std()
            sharpe = float(rets.mean() / std) if std > 1e-9 else 0.0
        report = {
            "version": "benter_betting_oos_fold2",
            "status": "measured",
            "protocol": {
                "valid_period": f"{VALID_START}..{VALID_END}",
                "test_period": f"{TEST_START}..",
                "bet_type": "win",
                "fusion_params_from": str(fusion_report_path.relative_to(ROOT)),
            },
            "ev_threshold": thr_result.threshold,
            "ev_threshold_warnings": thr_result.warnings,
            "valid_n_races": thr_result.valid_n_races,
            "n_bets": n_bets,
            "roi_pct": roi,
            "sharpe": sharpe,
            "gates": {
                "roi_above_100": roi > 100.0,
                "n_bets_gte_200": n_bets >= MIN_FORMAL_BETS,
                "phase3_pass": roi > 100.0 and n_bets >= MIN_FORMAL_BETS,
            },
        }

    out = ROOT / "evaluation" / "reports" / "betting_backtest_oos.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS formal betting backtest (win only)")
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
    parser.add_argument(
        "--fusion-report",
        type=Path,
        default=ROOT / "evaluation" / "reports" / "fusion_oos_fold2.json",
    )
    args = parser.parse_args()
    run_backtest_oos(args.scores, args.features, args.fusion_report)


if __name__ == "__main__":
    main()

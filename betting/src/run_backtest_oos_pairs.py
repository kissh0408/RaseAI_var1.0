"""CLI: OOS 正式ベッティングバックテスト（Phase 3 拡張）— ワイド/馬連。

VALID=2024 で EV 閾値を選択し、TEST=2025+ で 1 回だけ判定する（Rule 3）。
α,β,λ は evaluation/reports/fusion_oos_fold2.json の formal 値を使用。
確率モデルは bet_type ごとに使い分ける（betting/analysis/pair_probability_model_comparison.json
の fold2 OOS 実測に基づく、2026-07-09決定）:
  wide     -> Stern式    (pair_probs.py)
  quinella -> Harville式 (ev_filters.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from betting.src.backtest import fuse_period, load_betting_config, load_scored_odds_frame  # noqa: E402
from betting.src.pair_bets import simulate_pair_bets, tune_pair_ev_threshold_on_valid  # noqa: E402
from betting.src.wide_ev_core import load_wide_odds_lookup  # noqa: E402
from prob_fusion.src.oos_protocol import TEST_START  # noqa: E402

VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
MIN_FORMAL_BETS = 200


def _pair_odds_lookup(bet_type: str, years: list[int]) -> dict:
    odds_type = "Wide" if bet_type == "wide" else "Quinella"
    odds_dir = ROOT / "common" / "data" / "output" / "odds"
    return load_wide_odds_lookup(years, odds_dir, odds_type=odds_type)


def run_backtest_oos_pairs(
    scores_path: Path,
    features_path: Path,
    fusion_report_path: Path,
    bet_type: str,
) -> dict:
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

    years = sorted(set(dates.dt.year.dropna().astype(int).tolist()))
    odds_lookup = _pair_odds_lookup(bet_type, years)

    grid = bet_cfg.get("ev_threshold_grid", [1.05])
    thr_result = tune_pair_ev_threshold_on_valid(
        valid_fused, bet_type=bet_type, odds_lookup=odds_lookup, grid=grid,
        lam2=params["lam2"], lam3=params["lam3"],
    )

    if thr_result["fallback_used"]:
        report = {
            "version": f"benter_betting_oos_fold2_{bet_type}",
            "status": "skipped",
            "reason": "VALID threshold tuning failed",
        }
    else:
        bets = simulate_pair_bets(
            test_fused, bet_type=bet_type, odds_lookup=odds_lookup,
            ev_threshold=thr_result["threshold"], lam2=params["lam2"], lam3=params["lam3"],
        )
        n_bets = len(bets)
        n_hits = int(bets["hit"].sum()) if n_bets else 0
        total_stake = float(bets["stake"].sum()) if n_bets else 0.0
        total_payout = float(bets["payout"].sum()) if n_bets else 0.0
        roi = total_payout / total_stake * 100.0 if total_stake > 0 else 0.0
        top1_payout = float(bets["payout"].max()) if n_bets else 0.0
        top1_payout_share = top1_payout / total_payout if total_payout > 0 else 0.0
        roi_excl_top1 = (total_payout - top1_payout) / total_stake * 100.0 if total_stake > 0 else 0.0

        # argmax EV選択（レース内~100候補中1件）は必然的に稀な超大穴の的中に払戻が
        # 集中しやすい（winner's curse）。2026-07-09にwideで実測: payout の98.65%が
        # 単一の的中（オッズ14,257倍）由来で、それを除くとROIは4.09%まで崩壊した。
        # n_bets>=200 だけでは検出できないため、payout集中度ゲートを追加する。
        payout_not_concentrated = top1_payout_share <= 0.3
        min_hits_ok = n_hits >= 10

        report = {
            "version": f"benter_betting_oos_fold2_{bet_type}",
            "status": "measured",
            "protocol": {
                "valid_period": f"{VALID_START}..{VALID_END}",
                "test_period": f"{TEST_START}..",
                "bet_type": bet_type,
                "prob_model": "stern" if bet_type == "wide" else "harville",
                "fusion_params_from": str(fusion_report_path.relative_to(ROOT)),
            },
            "ev_threshold": thr_result["threshold"],
            "valid_n_races": thr_result["valid_n_races"],
            "n_bets": n_bets,
            "n_hits": n_hits,
            "roi_pct": roi,
            "roi_pct_excl_top1_payout": roi_excl_top1,
            "top1_payout_share": top1_payout_share,
            "gates": {
                "roi_above_100": roi > 100.0,
                "n_bets_gte_200": n_bets >= MIN_FORMAL_BETS,
                "n_hits_gte_10": min_hits_ok,
                "payout_not_concentrated_top1_lte_30pct": payout_not_concentrated,
                "phase3_pass": (
                    roi > 100.0 and n_bets >= MIN_FORMAL_BETS and min_hits_ok and payout_not_concentrated
                ),
            },
        }

    out = ROOT / "evaluation" / "reports" / f"betting_backtest_oos_{bet_type}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS formal betting backtest (wide/quinella)")
    parser.add_argument("--bet-type", choices=["wide", "quinella"], required=True)
    parser.add_argument(
        "--scores", type=Path,
        default=ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet",
    )
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet",
    )
    parser.add_argument(
        "--fusion-report", type=Path,
        default=ROOT / "evaluation" / "reports" / "fusion_oos_fold2.json",
    )
    args = parser.parse_args()
    run_backtest_oos_pairs(args.scores, args.features, args.fusion_report, args.bet_type)


if __name__ == "__main__":
    main()

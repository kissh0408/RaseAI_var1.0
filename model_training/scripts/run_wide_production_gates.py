"""
run_wide_production_gates.py — Track C1 ワイド本番統合の OOF ゲート

specv2 OOF + strategy_config（wide_min_edge / race_num）で
2025 wide anchor bet の ROI / 的中率 / n_bets を検証する。
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
OUT = PROJECT_ROOT / "model_training" / "data" / "03_train" / "wide_production_gates_report.json"

GATES = {
    "roi_min": 1.15,
    "n_bets_min": 200,
    "hit_rate_ref": 0.20,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not SPECv2_OOF.is_file():
        print(f"ERROR: missing {SPECv2_OOF}")
        return 1

    from model_training.scripts.compare_production_ensemble_eval import (
        _load_specv2_eval,
        _wide_anchor_bet_metrics,
    )

    eval_df = _load_specv2_eval()
    metrics = _wide_anchor_bet_metrics(eval_df, "production", args.year)
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return 1

    checks = {
        "roi": metrics.get("roi", 0) >= GATES["roi_min"],
        "n_bets": metrics.get("n_bets", 0) >= GATES["n_bets_min"],
    }
    passed = all(checks.values())

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": "C1_wide_production",
        "eval_csv": str(SPECv2_OOF),
        "strategy_config": str(STRATEGY_CFG),
        "year": args.year,
        "gates": GATES,
        "wide_anchor_bet": metrics,
        "checks": checks,
        "passed": passed,
        "hit_rate_above_ref": metrics.get("hit_rate", 0) >= GATES["hit_rate_ref"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"year={args.year} wide ROI={metrics['roi']:.1%} hit={metrics['hit_rate']:.1%} n={metrics['n_bets']}")
    print(f"checks={checks} passed={passed}")
    print(f"written: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Regenerate evaluation/reports/gate_summary.json from component reports."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "evaluation" / "reports"


def main() -> None:
    fusion_path = REPORTS / "fusion_benter_v1.json"
    betting_path = REPORTS / "betting_backtest.json"
    market_path = REPORTS / "market_baseline.json"

    fusion = json.loads(fusion_path.read_text(encoding="utf-8")) if fusion_path.is_file() else {}
    betting = json.loads(betting_path.read_text(encoding="utf-8")) if betting_path.is_file() else {}
    market = json.loads(market_path.read_text(encoding="utf-8")) if market_path.is_file() else {}

    fold3 = next((f for f in fusion.get("folds", []) if f.get("fold") == 3), {})
    formal_bets = next(
        (r for r in betting.get("formal_results", []) if r.get("fold") == 3 and not r.get("skipped_for_formal_gate")),
        {},
    )
    fav = market.get("favorite_baseline", {})

    fav_rate = fav.get("favorite_top1_rate")
    fav_roi = fav.get("favorite_roi")

    summary = {
        "generated_at": date.today().isoformat(),
        "protocol": {
            "formal_judgment_fold": 3,
            "market_beta_fit_on": "train+valid",
            "l1_fold1_2": "reference_only_l1_contaminated",
            "bet_types": ["win"],
        },
        "phase2_l2_gates": {
            "fold3_test_logloss_beats_market": fold3.get("test_logloss_beats_market"),
            "fold3_test_top1": fold3.get("test_top1"),
            "fold3_top1_gate_33pct": (fold3.get("test_top1") or 0) >= 0.33,
            "note": "Only fold3 counts formally; folds 1-2 are L1 in-sample contaminated reference.",
            "see": str(fusion_path.relative_to(ROOT)),
        },
        "phase3_l3_gates": {
            "fold3_win_roi_pct": formal_bets.get("roi_pct"),
            "fold3_win_roi_above_100": (formal_bets.get("roi_pct") or 0) >= 100.0,
            "ev_threshold_warnings": formal_bets.get("ev_threshold_warnings"),
            "note": "Win-only backtest; place disabled until HR place odds integrated.",
            "see": str(betting_path.relative_to(ROOT)),
        },
        "phase1_market_baseline": {
            "favorite_top1_rate": fav_rate,
            "favorite_roi": fav_roi,
            "favorite_roi_recomputed": fav_roi is not None,
            "gate_32_90pct": (fav_rate or 0) >= 0.329,
            "see": str(market_path.relative_to(ROOT)),
        },
    }

    out = REPORTS / "gate_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

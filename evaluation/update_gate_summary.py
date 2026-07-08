"""Regenerate evaluation/reports/gate_summary.json from component reports."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "evaluation" / "reports"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    fusion_path = REPORTS / "fusion_benter_v1.json"
    betting_path = REPORTS / "betting_backtest.json"
    market_path = REPORTS / "market_baseline.json"
    fusion_oos_path = REPORTS / "fusion_oos_fold2.json"
    betting_oos_path = REPORTS / "betting_backtest_oos.json"

    fusion = _load(fusion_path)
    betting = _load(betting_path)
    market = _load(market_path)
    fusion_oos = _load(fusion_oos_path)
    betting_oos = _load(betting_oos_path)

    fold3 = next((f for f in fusion.get("folds", []) if f.get("fold") == 3), {})
    formal_bets = next(
        (r for r in betting.get("formal_results", []) if r.get("fold") == 3 and not r.get("skipped_for_formal_gate")),
        {},
    )
    fav = market.get("favorite_baseline", {})
    fav_rate = fav.get("favorite_top1_rate")
    fav_roi = fav.get("favorite_roi")

    oos_formal = fusion_oos.get("formal", {})
    oos_gates = oos_formal.get("gates", {})
    oos_measured = bool(oos_formal)

    if oos_measured:
        phase2 = {
            "status": "measured_oos",
            "verdict": "PASS" if oos_gates.get("phase2_pass") else "FAIL",
            "alpha": oos_formal.get("alpha"),
            "beta": oos_formal.get("beta"),
            "lrt_p_value": oos_formal.get("lrt_p_value"),
            "test_logloss_fusion": oos_formal.get("test_logloss_fusion"),
            "test_logloss_market": oos_formal.get("test_logloss_market"),
            "test_top1": oos_formal.get("test_top1"),
            "gates": oos_gates,
            "note": (
                "OOS 正式測定（fold2 スコア、fit=2023-2024、TEST=2025+）。"
                "α=0 は『L1 スコアの情報が市場オッズに完全に織り込まれている』ことを意味する。"
                "z 品質は健全（OOS Top-1 29-31%, Spearman~0.50）であり整列バグではない。"
            ),
            "see": str(fusion_oos_path.relative_to(ROOT)),
        }
    else:
        phase2 = {
            "status": "unmeasured_pending_oos_l1_scores",
            "verdict": None,
            "see": str(fusion_oos_path.relative_to(ROOT)),
        }

    bet_status = betting_oos.get("status")
    if bet_status == "measured":
        bet_gates = betting_oos.get("gates", {})
        phase3 = {
            "status": "measured_oos",
            "verdict": "PASS" if bet_gates.get("phase3_pass") else "FAIL",
            "roi_pct": betting_oos.get("roi_pct"),
            "n_bets": betting_oos.get("n_bets"),
            "ev_threshold": betting_oos.get("ev_threshold"),
            "gates": bet_gates,
            "see": str(betting_oos_path.relative_to(ROOT)),
        }
    elif bet_status == "skipped":
        phase3 = {
            "status": "measured_oos",
            "verdict": "FAIL",
            "reason": betting_oos.get("reason"),
            "ev_threshold_warnings": betting_oos.get("ev_threshold_warnings"),
            "note": "α=0 のため fusion=市場確率となり、正の EV が VALID に存在せずベット不成立。",
            "see": str(betting_oos_path.relative_to(ROOT)),
        }
    else:
        phase3 = {
            "status": "unmeasured_pending_oos_l1_scores",
            "verdict": None,
            "see": str(betting_oos_path.relative_to(ROOT)),
        }

    summary = {
        "generated_at": date.today().isoformat(),
        "protocol": {
            "formal_judgment": "oos_fold2 (fit=2023-2024, TEST=2025+); market β fit on same fit period",
            "caveat": "2023 は fold2 early stopping 年（弱い接触）。感度分析 fit=2024 単独でも α=0。",
            "bet_types": ["win"],
        },
        "phase2_l2_gates": phase2,
        "phase3_l3_gates": phase3,
        "contaminated_reference_runs": {
            "note": "15モデル全fold平均スコアによる旧測定。in-sample 汚染のため合否判定に使用不可。",
            "fold3_test_logloss_fusion": fold3.get("test_logloss_fusion"),
            "fold3_test_logloss_market": fold3.get("test_logloss_market"),
            "fold3_test_top1": fold3.get("test_top1"),
            "fold3_win_roi_pct": formal_bets.get("roi_pct"),
            "see": [str(fusion_path.relative_to(ROOT)), str(betting_path.relative_to(ROOT))],
        },
        "phase1_market_baseline": {
            "status": "complete",
            "favorite_top1_rate": fav_rate,
            "favorite_roi": fav_roi,
            "favorite_roi_recomputed": fav_roi is not None,
            "gate_32_90pct": (fav_rate or 0) >= 0.329,
            "see": str(market_path.relative_to(ROOT)),
        },
        "phase0_cleanup": {
            "status": "complete_except_l4_runtime",
            "note": "main/main.py and strategy_pipeline still import archived model_training/strategy paths.",
        },
    }

    out = REPORTS / "gate_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

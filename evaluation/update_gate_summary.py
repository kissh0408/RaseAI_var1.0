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


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _collect_alpha_gate_reports(reports_dir: Path, root: Path) -> dict:
    reports = []
    for path in sorted(reports_dir.glob("alpha_gate_*.json")):
        payload = _load(path)
        if not payload:
            continue
        reports.append(
            {
                "candidate": payload.get("candidate", path.stem.replace("alpha_gate_", "")),
                "pass": payload.get("pass"),
                "gamma": payload.get("gamma"),
                "gamma_lrt_p": payload.get("gamma_lrt_p"),
                "delta_ll_per_race": payload.get("delta_ll_per_race"),
                "eval_top1": payload.get("eval_top1"),
                "see": _rel(path, root),
            }
        )
    return {
        "status": "measured" if reports else "no_reports",
        "n_reports": len(reports),
        "passing_candidates": [r["candidate"] for r in reports if r.get("pass") is True],
        "reports": reports,
    }


# evaluator formal sign-off date for P2 Track B (procedure verified 2026-07-10;
# spec: docs/specs/2026-07-10-p2-track-b-training-spec.md, all 5 candidates failed
# the primary gamma-LRT gate, retreat criterion B-4 applied).
TRACK_B_EVALUATOR_PASS_DATE = "2026-07-10"


def _track_b_section(track_b_path: Path, root: Path) -> dict:
    """Summarize the isolated track_b_training experiment (P2, JV-Link workout data)."""
    payload = _load(track_b_path)
    if not payload:
        return {"status": "unmeasured", "see": _rel(track_b_path, root)}
    candidates = {
        key: {
            "id": cand.get("id"),
            "name": cand.get("name"),
            "gamma": cand.get("gamma"),
            "gamma_lrt_p": cand.get("gamma_lrt_p"),
            "delta_ll_per_race": cand.get("delta_ll_per_race"),
            "eval_top1": cand.get("eval_top1"),
            "eval_spearman": cand.get("eval_spearman"),
            "primary_pass": cand.get("primary_pass"),
            "leak_stop": cand.get("leak_stop"),
        }
        for key, cand in payload.get("candidates", {}).items()
    }
    return {
        "status": "measured",
        "verdict": payload.get("verdict"),
        "candidates": candidates,
        "candidates_completed": payload.get("candidates_completed"),
        "evaluator_pass": TRACK_B_EVALUATOR_PASS_DATE,
        "note": (
            "P2 Track B primary gamma-LRT gate (fit=2023, eval=2024, TEST untouched). "
            "All 5 pre-registered workout time-series candidates failed p<0.01 & dLL/race>0; "
            "retreat criterion B-4: no market-beating signal in JV-Link training data at current granularity."
        ),
        "see": _rel(track_b_path, root),
    }


def _place_section(place_path: Path, root: Path) -> dict:
    place = _load(place_path)
    if not place:
        return {"status": "unmeasured", "see": _rel(place_path, root)}
    model = place.get("model_top1", {})
    favorite = place.get("favorite", {})
    return {
        "status": place.get("status", "measured"),
        "verdict": place.get("verdict"),
        "gates": place.get("gates"),
        "model_top1_roi_pct": model.get("roi_pct"),
        "favorite_roi_pct": favorite.get("roi_pct"),
        "roi_minus_favorite_pp": model.get("roi_minus_favorite_pp"),
        "model_n_races": model.get("n_races"),
        "favorite_n_races": favorite.get("n_races"),
        "known_limitations": place.get("known_limitations", []),
        "see": _rel(place_path, root),
    }


def build_gate_summary(reports_dir: Path = REPORTS, *, root: Path = ROOT) -> dict:
    fusion_path = reports_dir / "fusion_benter_v1.json"
    betting_path = reports_dir / "betting_backtest.json"
    market_path = reports_dir / "market_baseline.json"
    fusion_oos_path = reports_dir / "fusion_oos_fold2.json"
    betting_oos_path = reports_dir / "betting_backtest_oos.json"
    place_path = reports_dir / "place_baseline_oos.json"
    track_b_path = root / "pure_rank" / "experiments" / "track_b_training" / "results" / "track_b_summary.json"

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
                "OOS formal measurement (fold2 scores, fit=2023-2024, TEST=2025+). "
                "Alpha=0 means the current L1 score adds no conditional logloss signal beyond market odds."
            ),
            "see": _rel(fusion_oos_path, root),
        }
    else:
        phase2 = {
            "status": "unmeasured_pending_oos_l1_scores",
            "verdict": None,
            "see": _rel(fusion_oos_path, root),
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
            "see": _rel(betting_oos_path, root),
        }
    elif bet_status == "skipped":
        phase3 = {
            "status": "measured_oos",
            "verdict": "FAIL",
            "reason": betting_oos.get("reason"),
            "ev_threshold_warnings": betting_oos.get("ev_threshold_warnings"),
            "note": "Alpha=0 makes fusion effectively market-only; no positive EV bets on VALID.",
            "see": _rel(betting_oos_path, root),
        }
    else:
        phase3 = {
            "status": "unmeasured_pending_oos_l1_scores",
            "verdict": None,
            "see": _rel(betting_oos_path, root),
        }

    summary = {
        "generated_at": date.today().isoformat(),
        "protocol": {
            "formal_judgment": "oos_fold2 (fit=2023-2024, TEST=2025+); market beta fit on same fit period",
            "caveat": "2023 is fold2 early-stopping year (weak contact). Sensitivity fit=2024 only also measured alpha=0.",
            "bet_types": ["win"],
        },
        "phase2_l2_gates": phase2,
        "phase3_l3_gates": phase3,
        "alpha_gate": _collect_alpha_gate_reports(reports_dir, root),
        "track_b_training": _track_b_section(track_b_path, root),
        "track_c_place": _place_section(place_path, root),
        "contaminated_reference_runs": {
            "note": "Old measurement with 15-model all-fold average scores; not valid for pass/fail.",
            "fold3_test_logloss_fusion": fold3.get("test_logloss_fusion"),
            "fold3_test_logloss_market": fold3.get("test_logloss_market"),
            "fold3_test_top1": fold3.get("test_top1"),
            "fold3_win_roi_pct": formal_bets.get("roi_pct"),
            "see": [_rel(fusion_path, root), _rel(betting_path, root)],
        },
        "phase1_market_baseline": {
            "status": "complete",
            "favorite_top1_rate": fav_rate,
            "favorite_roi": fav_roi,
            "favorite_roi_recomputed": fav_roi is not None,
            "gate_32_90pct": (fav_rate or 0) >= 0.329,
            "see": _rel(market_path, root),
        },
        "phase0_cleanup": {
            "status": "complete_except_l4_runtime",
            "note": "main/main.py and strategy_pipeline still import archived model_training/strategy paths.",
        },
    }
    return summary


def main() -> None:
    summary = build_gate_summary(REPORTS, root=ROOT)

    out = REPORTS / "gate_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

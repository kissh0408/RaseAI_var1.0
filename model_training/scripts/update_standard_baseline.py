"""
update_standard_baseline.py — 標準評価パイプライン baseline を JSON に記録

入力: compare_v5_specv2_eval.json（specv2 行）+ mdd_diagnosis_report.json
出力: baseline_standard_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARE = PROJECT_ROOT / "model_training" / "data" / "03_train" / "compare_v5_specv2_eval.json"
GATES_REPORT = PROJECT_ROOT / "model_training" / "data" / "03_train" / "standard_eval_gates_report.json"
MDD_REPORT = PROJECT_ROOT / "model_training" / "data" / "03_train" / "mdd_diagnosis_report.json"
OUT = PROJECT_ROOT / "model_training" / "data" / "03_train" / "baseline_standard_eval.json"
GATES = {"roi_min": 1.05, "mdd_min": -0.20, "sharpe_min": 0.10, "n_bets_min": 500}


def _gate(m: dict) -> dict:
    return {
        "roi": m.get("roi", 0) >= GATES["roi_min"],
        "mdd": m.get("mdd", -1) >= GATES["mdd_min"],
        "sharpe": m.get("sharpe", 0) >= GATES["sharpe_min"],
        "n_bets": m.get("n_bets", 0) >= GATES["n_bets_min"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", type=Path, default=COMPARE)
    parser.add_argument("--mdd", type=Path, default=MDD_REPORT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not args.compare.exists():
        print(f"[NG] missing {args.compare}")
        return 1

    compare = json.loads(args.compare.read_text(encoding="utf-8"))
    spec = compare.get("specv2", {})
    mdd_diag = {}
    if args.mdd.exists():
        mdd_diag = json.loads(args.mdd.read_text(encoding="utf-8"))

    gates_report = {}
    if GATES_REPORT.exists():
        gates_report = json.loads(GATES_REPORT.read_text(encoding="utf-8"))

    baseline = {
        "version": "standard_eval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "model": "ensemble_v5_specv2",
            "eval_csv": "evaluation_specv2_oof.csv",
            "strategy": "strategy_config.json + strategy_config_from_runtime()",
            "calibrator": "strategy/models/calibration_isotonic_specv2.json",
            "calibrator_resolver": "main.pipeline.strategy_pipeline.resolve_strategy_calibration_path",
            "walkforward": "2018-2025 OOF",
            "supersedes": {
                "legacy_roi": 1.275,
                "legacy_n_bets": 530,
                "legacy_note": "2fold WF・race_num なし・条件不一致のため参照のみ",
            },
        },
        "profiles": {},
        "mdd_diagnosis_ref": str(args.mdd) if args.mdd.exists() else None,
        "legacy_baseline": compare.get("baseline_meta", {}),
    }

    if gates_report:
        baseline["standard_eval_gates"] = gates_report
        y2025g = gates_report.get("primary_test_fold_2025", {})
        if y2025g:
            baseline["profiles"]["production_live"] = {
                "2025": {
                    "win": {
                        "roi": y2025g.get("roi"),
                        "mdd": y2025g.get("mdd"),
                        "sharpe": y2025g.get("sharpe"),
                        "n_bets": y2025g.get("n_bets"),
                        "hit_rate": y2025g.get("hit_rate"),
                    },
                    "gates_win": y2025g.get("gates", {}),
                    "gates_passed_win": y2025g.get("passed", False),
                }
            }

    for pid, prof in spec.get("profiles", {}).items():
        baseline["profiles"][pid] = {}
        for ykey, ym in prof.get("years", {}).items():
            win = ym.get("win", {})
            wide = ym.get("wide_anchor_bet", {})
            baseline["profiles"][pid][ykey] = {
                "win": win,
                "wide_anchor_bet": wide,
                "gates_win": _gate(win) if win and "error" not in win else {},
                "gates_passed_win": all(_gate(win).values()) if win and "error" not in win else False,
            }

    # MDD 切り分けハイライト（2025）
    if mdd_diag.get("years"):
        y2025 = next((y for y in mdd_diag["years"] if y.get("year") == 2025), None)
        if y2025:
            delta = next(
                (s for s in y2025["scenarios"] if s.get("scenario") == "delta_race_num_8_12"),
                None,
            )
            oof_cal = next(
                (s for s in y2025["scenarios"] if s.get("scenario") == "calibrator_oof_fit"),
                None,
            )
            baseline["mdd_highlights_2025"] = {
                "race_num_delta": delta,
                "calibrator_oof_fit_production": oof_cal,
                "best_mdd_scenario": min(
                    (s for s in y2025["scenarios"] if "mdd" in s and not s["scenario"].startswith("delta")),
                    key=lambda s: s["mdd"],
                    default=None,
                ),
            }
            if oof_cal:
                baseline["recommended_next"] = {
                    "calibrator": "strategy/models/calibration_isotonic_specv2.json",
                    "profile": "production + specv2 calibrator",
                    "expected_2025": {
                        "roi": oof_cal.get("roi"),
                        "mdd": oof_cal.get("mdd"),
                        "sharpe": oof_cal.get("sharpe"),
                        "gates_passed": _gate(oof_cal),
                    },
                }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")
    for pid, years in baseline["profiles"].items():
        for yk, block in years.items():
            g = block.get("gates_passed_win")
            print(f"  {pid}/{yk} gates_passed_win={g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

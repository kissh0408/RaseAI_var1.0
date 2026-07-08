"""v6 馬場改善モデルの本番昇格（ゲート PASS 時のみ）。

Usage:
  python model_training/scripts/run_going_v6_promotion.py --dry-run
  python model_training/scripts/run_going_v6_promotion.py --models-dir ensemble_v6_expC
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "scripts"))

from evaluate_going_experiment_gate import (  # noqa: E402
    _load_meta,
    evaluate_roi_gate,
    evaluate_segment_gate,
    evaluate_sensitivity_gate,
)
from diagnostics_going_sensitivity import run as run_sensitivity  # noqa: E402

TRAIN_CONFIG = ROOT / "model_training" / "config" / "train_config.json"
BASELINE_META = ROOT / "model_training" / "models" / "ensemble_v5" / "ensemble_meta.json"
DEFAULT_V6 = ROOT / "model_training" / "models" / "ensemble_v6_expC"


def check_promotion_gates(models_dir: Path, segment_report: Path | None) -> dict:
    meta = _load_meta(models_dir / "ensemble_meta.json")
    baseline = _load_meta(BASELINE_META)
    sens = run_sensitivity(
        ROOT / "main" / "results" / "today_predictions_with_bets.parquet",
        models_dir,
    )
    roi_gate = evaluate_roi_gate(meta, baseline)
    sens_gate = evaluate_sensitivity_gate(sens)
    seg_gate = evaluate_segment_gate(
        json.loads(segment_report.read_text(encoding="utf-8")).get("segments")
        if segment_report and segment_report.exists()
        else meta.get("segment_stats")
    )
    passed = (
        bool(meta)
        and roi_gate.get("passed") is True
        and sens_gate.get("passed") is True
        and seg_gate.get("passed") is not False
    )
    return {
        "models_dir": str(models_dir),
        "roi_gate": roi_gate,
        "sensitivity_gate": sens_gate,
        "segment_gate": seg_gate,
        "promotion_allowed": passed,
    }


def apply_promotion(models_dir: Path, *, dry_run: bool) -> None:
    output_name = models_dir.name
    train_cfg = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    prod = train_cfg.setdefault("production_training", {})
    prod["output_dir"] = output_name
    exp_spec = None
    for spec in train_cfg.get("going_improvement", {}).get("experiments", {}).values():
        if spec.get("ensemble_output_dir") == output_name:
            exp_spec = spec
            break
    if exp_spec and exp_spec.get("feature_file"):
        prod["feature_file"] = exp_spec["feature_file"]
    prod["description"] = f"Going v6 promotion from {output_name}"

    if dry_run:
        print("[dry-run] would update production_training:")
        print(json.dumps({"output_dir": output_name, "feature_file": prod.get("feature_file")}, indent=2))
        print("[dry-run] next: python model_training/scripts/fit_specv2_calibrator.py")
        print("[dry-run] next: python main/tests/e2e_test.py")
        return

    TRAIN_CONFIG.write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Promoted production_training.output_dir -> {output_name}")
    print("Run: python model_training/scripts/fit_specv2_calibrator.py")
    print("Run: python main/tests/e2e_test.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="v6 本番昇格（ゲート必須）")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_V6)
    parser.add_argument("--segment-report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="ゲート未達でも設定のみ更新（非推奨）")
    args = parser.parse_args()

    models_dir = args.models_dir
    if not models_dir.is_absolute():
        models_dir = ROOT / "model_training" / "models" / models_dir

    report = check_promotion_gates(models_dir, args.segment_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["promotion_allowed"] or args.force:
        apply_promotion(models_dir, dry_run=args.dry_run)
    else:
        print("Promotion blocked: gates not passed. Use --force to override (not recommended).")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""馬場改善実験 A/B/C/D ランナー（train_config の experiments 定義を自動マージ）。

Usage:
  python model_training/scripts/run_going_experiment.py --experiment A --dry-run
  python model_training/scripts/run_going_experiment.py --experiment B --n-trials 25 --fast
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG = ROOT / "model_training" / "config" / "train_config.json"
FEATURES_DIR = ROOT / "model_training" / "data" / "02_features"
TRAIN_ENSEMBLE = ROOT / "model_training" / "src" / "train_ensemble.py"

_EXPERIMENT_KEYS = (
    "interaction_constraints_enabled",
    "monotone_constraints_enabled",
    "rank1_baba_weight_mode",
    "ensemble_output_dir",
    "feature_file",
    "description",
)


def merge_experiment_config(cfg: dict, experiment: str) -> dict:
    exp_id = experiment.upper()
    gi = cfg.setdefault("going_improvement", {})
    experiments = gi.get("experiments") or {}
    if exp_id not in experiments:
        raise KeyError(f"going_improvement.experiments に {exp_id} がありません")
    spec = experiments[exp_id]
    merged = copy.deepcopy(cfg)
    mg = merged.setdefault("going_improvement", {})
    for key in _EXPERIMENT_KEYS:
        if key in spec:
            mg[key] = spec[key]
    mg["active_experiment"] = exp_id
    merged.setdefault("ensemble", {})["output_dir"] = spec["ensemble_output_dir"]
    merged.setdefault("training", {})["feature_file"] = spec["feature_file"]
    return merged, spec


def build_train_command(
    spec: dict,
    *,
    n_trials: int | None,
    fast: bool,
    disable_feature_selection: bool,
) -> list[str]:
    feature_file = spec["feature_file"]
    features_path = FEATURES_DIR / feature_file
    cmd = [
        sys.executable,
        str(TRAIN_ENSEMBLE),
        "--features-path",
        str(features_path),
        "--output-dir",
        spec["ensemble_output_dir"],
    ]
    if n_trials is not None:
        cmd.extend(["--n-trials", str(n_trials)])
    # 実験ランナーでは常に feature selection を無効化する。
    # reuse_optuna_from_first_seed 時に seed42 のみ selection が走り他 seed と feature set が
    # 不一致になるため（seed42=259, seed100/200=323 のような不整合が発生）。
    cmd.append("--disable-feature-selection")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="馬場改善実験ランナー")
    parser.add_argument("--experiment", required=True, choices=["A", "B", "C", "D", "E", "a", "b", "c", "d", "e"])
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="fast_mode 相当（n_trials=20, feature selection off）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-feature-selection", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    exp_id = args.experiment.upper()

    # Exp D は Two-Stage Residual: Stage 1 = Exp C (学習不要), Stage 2 = going correction 学習
    if exp_id == "D":
        spec = cfg["going_improvement"]["experiments"]["D"]
        stage2_script = ROOT / spec.get("stage2_correction_script", "model_training/scripts/train_going_correction.py")
        stage2_n_trials = args.n_trials if args.n_trials is not None else 30
        cmd = [sys.executable, str(stage2_script), "--n-trials", str(stage2_n_trials)]
        print("Experiment: D (Two-Stage Residual)")
        print("Description:", spec.get("description", ""))
        print("Stage 1: ensemble_v6_expC (no retraining)")
        print("Stage 2 command:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)
        return

    merged, spec = merge_experiment_config(cfg, args.experiment)
    n_trials = args.n_trials if args.n_trials is not None else (20 if args.fast else None)
    cmd = build_train_command(
        spec,
        n_trials=n_trials,
        fast=args.fast,
        disable_feature_selection=args.disable_feature_selection,
    )

    print("Experiment:", exp_id)
    print("Description:", spec.get("description", ""))
    print("Merged going_improvement flags:")
    print(json.dumps({k: merged["going_improvement"].get(k) for k in _EXPERIMENT_KEYS if k in spec}, indent=2))
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    backup = TRAIN_CONFIG.with_suffix(".json.bak_going_exp")
    shutil.copy2(TRAIN_CONFIG, backup)
    try:
        TRAIN_CONFIG.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    finally:
        shutil.copy2(backup, TRAIN_CONFIG)
        backup.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

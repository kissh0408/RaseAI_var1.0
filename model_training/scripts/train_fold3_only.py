"""Fold 3 binary モデルのみ再学習（P-43 / P0 ablation 等の単一変数実験用）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from pipeline_common import MODELS_DIR, load_config  # noqa: E402
from train import (  # noqa: E402
    _load_binary_training_features,
    get_feature_cols,
    train_fold,
)

SEEDS = [42, 43, 44, 45, 46]
FOLD = 3
VAR1_COL = "var1_pure_score_z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold 3 binary のみ再学習")
    parser.add_argument(
        "--p0-var1-ablation",
        action="store_true",
        help="P0: var1_pure_score_z を特徴量から完全除外して学習",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    t_cfg = cfg["training"]
    fold_cfg = next(f for f in t_cfg["walkforward_folds"] if f["fold"] == FOLD)

    print(f"min_child_samples={t_cfg.get('backtest_conservative_params', {}).get('min_child_samples')}")
    print(f"reg_lambda={t_cfg.get('backtest_conservative_params', {}).get('reg_lambda')}")
    bt_mono = t_cfg.get("backtest_monotone_constraints", {})
    print(f"monotone_enabled={bt_mono.get('enabled')}, plus_patterns={bt_mono.get('plus_patterns')}")

    df = _load_binary_training_features()
    feature_cols = get_feature_cols(cfg)
    if args.p0_var1_ablation:
        if VAR1_COL not in feature_cols:
            print(f"WARNING: {VAR1_COL} not in feature_cols — ablation may be no-op")
        feature_cols = [c for c in feature_cols if c != VAR1_COL]
        print(f"P0 var1 ablation: excluded {VAR1_COL} ({len(feature_cols)} features remain)")
    experiment = (
        "P0_var1_ablation_fold3_only" if args.p0_var1_ablation else "fold3_retrain"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f"\n=== Fold {FOLD} seed {seed} ===")
        t_cfg["seed"] = seed
        model, meta = train_fold(df, fold_cfg, feature_cols, cfg)
        meta["seed"] = seed
        meta["experiment"] = experiment
        if args.p0_var1_ablation:
            meta["var1_ablation"] = True
        model_path = MODELS_DIR / f"lgbm_binary_fold{FOLD}_seed{seed}.txt"
        meta_path = MODELS_DIR / f"lgbm_binary_fold{FOLD}_seed{seed}_meta.json"
        model.save_model(str(model_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  saved: {model_path.name} (best_iteration={meta['best_iteration']})")


if __name__ == "__main__":
    main()

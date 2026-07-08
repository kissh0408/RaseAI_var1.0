"""Phase 1a-A2: 指定 Fold を var1 init_score (beta*) で 5-seed 再学習。"""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1a-A fold retrain with var1 init_score")
    p.add_argument("--folds", type=int, nargs="+", required=True, help="Fold IDs e.g. 1 2")
    return p.parse_args()


def train_fold_ensemble(fold: int, df, cfg: dict, feature_cols: list[str]) -> list[dict]:
    t_cfg = cfg["training"]
    fold_cfg = next(f for f in t_cfg["walkforward_folds"] if f["fold"] == fold)
    vis = t_cfg.get("var1_init_score", {})
    beta = float(vis.get("beta", 0.15))
    print(f"\n{'='*50}\nFold {fold}: beta={beta}, enabled={vis.get('enabled')}\n{'='*50}")

    metas = []
    for seed in SEEDS:
        print(f"\n=== Fold {fold} seed {seed} ===")
        t_cfg["seed"] = seed
        model, meta = train_fold(df, fold_cfg, feature_cols, cfg)
        meta["seed"] = seed
        meta["beta_var1_init_score"] = beta
        meta["experiment"] = f"Phase1aA2_var1_init_score_fold{fold}"
        path = MODELS_DIR / f"lgbm_binary_fold{fold}_seed{seed}.txt"
        meta_path = MODELS_DIR / f"lgbm_binary_fold{fold}_seed{seed}_meta.json"
        model.save_model(str(path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        metas.append(meta)
        print(f"  saved {path.name} iter={meta['best_iteration']} top10={meta.get('top10_concentration', 0):.3f}")
    return metas


def main() -> None:
    args = parse_args()
    cfg = load_config()
    df = _load_binary_training_features()
    feature_cols = get_feature_cols(cfg)
    print(f"features={len(feature_cols)} (var1 excluded)")

    all_metas: dict[int, list[dict]] = {}
    for fold in args.folds:
        all_metas[fold] = train_fold_ensemble(fold, df, cfg, feature_cols)

    summary = {}
    for fold, metas in all_metas.items():
        iters = [m["best_iteration"] for m in metas]
        top10 = [m.get("top10_concentration", 0) for m in metas]
        summary[fold] = {
            "best_iterations": iters,
            "min_iteration": min(iters),
            "top10_concentrations": top10,
            "max_top10": max(top10),
            "learning_gate_iter50": all(i >= 50 for i in iters),
            "learning_gate_top10": all(t < 0.50 for t in top10),
        }

    out_path = MODELS_DIR / "phase1aA2_folds_training_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"folds": args.folds, "summary": summary, "metas": all_metas}, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {out_path}")
    for fold, s in summary.items():
        print(
            f"  Fold {fold}: iter min={s['min_iteration']}, "
            f"top10 max={s['max_top10']:.3f}, "
            f"gate_iter50={s['learning_gate_iter50']}, gate_top10={s['learning_gate_top10']}"
        )


if __name__ == "__main__":
    main()

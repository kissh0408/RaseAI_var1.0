"""Phase 1a-A: Fold 3 のみ var1 init_score beta sweep（seed 42 で VALID logloss 選択）。"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
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

FOLD = 3
SWEEP_SEED = 42
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]


def select_beta(results: list[dict], delta: float) -> dict:
    """VALID logloss 最小。tie-break: 0 除く最小 beta（オッカムの剃刀）。"""
    best_ll = min(r["valid_logloss"] for r in results if r["valid_logloss"] is not None)
    candidates = [r for r in results if r["valid_logloss"] is not None and r["valid_logloss"] <= best_ll + delta]
    positive = [r for r in candidates if r["beta"] > 0]
    if positive:
        return min(positive, key=lambda r: r["beta"])
    return min(candidates, key=lambda r: r["beta"])


def set_var1_init_score(t_cfg: dict, beta: float) -> None:
    vis = t_cfg.setdefault("var1_init_score", {})
    vis["enabled"] = beta > 0
    vis["beta"] = beta
    vis.setdefault("z_col", "var1_pure_score_z")
    vis.setdefault("market_col", "market_log_odds")


def train_fold3_seed42(df, fold_cfg, feature_cols, cfg, beta: float) -> dict:
    t_cfg = cfg["training"]
    set_var1_init_score(t_cfg, beta)
    t_cfg["seed"] = SWEEP_SEED
    model, meta = train_fold(df, fold_cfg, feature_cols, cfg)
    meta["beta_var1_init_score"] = beta
    return meta


def train_fold3_ensemble(df, fold_cfg, feature_cols, cfg, beta: float) -> list[dict]:
    t_cfg = cfg["training"]
    set_var1_init_score(t_cfg, beta)
    metas = []
    for seed in ENSEMBLE_SEEDS:
        print(f"\n=== Fold {FOLD} seed {seed} beta={beta} ===")
        t_cfg["seed"] = seed
        model, meta = train_fold(df, fold_cfg, feature_cols, cfg)
        meta["seed"] = seed
        meta["beta_var1_init_score"] = beta
        meta["experiment"] = "Phase1aA_var1_init_score_fold3"
        path = MODELS_DIR / f"lgbm_binary_fold{FOLD}_seed{seed}.txt"
        meta_path = MODELS_DIR / f"lgbm_binary_fold{FOLD}_seed{seed}_meta.json"
        model.save_model(str(path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        metas.append(meta)
        print(f"  saved {path.name} iter={meta['best_iteration']}")
    return metas


def main() -> None:
    cfg = load_config()
    t_cfg = cfg["training"]
    fold_cfg = next(f for f in t_cfg["walkforward_folds"] if f["fold"] == FOLD)
    vis_cfg = t_cfg.get("var1_init_score", {})
    betas = vis_cfg.get("beta_sweep_values", [0.0, 0.15, 0.25, 0.35, 0.50])
    delta = float(vis_cfg.get("tie_break_logloss_delta", 0.001))

    df = _load_binary_training_features()
    feature_cols = get_feature_cols(cfg)
    print(f"features={len(feature_cols)} (var1 excluded from feature_cols)")

    sweep_results = []
    for beta in betas:
        print(f"\n{'='*50}\nBeta sweep: beta={beta}\n{'='*50}")
        meta = train_fold3_seed42(df, fold_cfg, feature_cols, deepcopy(cfg), beta)
        row = {
            "beta": beta,
            "valid_logloss": meta.get("valid_logloss"),
            "best_iteration": meta.get("best_iteration"),
            "top10_concentration": meta.get("top10_concentration"),
        }
        sweep_results.append(row)
        print(f"  -> valid_logloss={row['valid_logloss']}, iter={row['best_iteration']}")

    winner = select_beta(sweep_results, delta)
    beta_star = winner["beta"]
    print(f"\n*** beta* = {beta_star} (valid_logloss={winner['valid_logloss']}) ***")

    out = {
        "experiment": "Phase1aA_fold3_beta_sweep",
        "sweep_seed": SWEEP_SEED,
        "tie_break_logloss_delta": delta,
        "sweep_results": sweep_results,
        "beta_star": beta_star,
        "winner": winner,
    }
    sweep_path = MODELS_DIR / "phase1aA_fold3_beta_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSweep saved: {sweep_path}")

    print(f"\nTraining 5-seed ensemble with beta*={beta_star}...")
    ensemble_metas = train_fold3_ensemble(df, fold_cfg, feature_cols, cfg, beta_star)
    out["ensemble_metas"] = [
        {
            "seed": m["seed"],
            "best_iteration": m["best_iteration"],
            "valid_logloss": m.get("valid_logloss"),
            "top10_concentration": m.get("top10_concentration"),
        }
        for m in ensemble_metas
    ]
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Persist winning beta to config (enabled flag)
    set_var1_init_score(t_cfg, beta_star)
    print(f"Config updated: var1_init_score.enabled={beta_star > 0}, beta={beta_star}")


if __name__ == "__main__":
    main()

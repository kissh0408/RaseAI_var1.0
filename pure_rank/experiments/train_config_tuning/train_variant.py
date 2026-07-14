"""train_config_tuning: variant別に fold2 モデルを学習する。

使用法:
    python train_variant.py --variant truncation
    python train_variant.py --variant bagging
    python train_variant.py --variant seeds10
"""
from __future__ import annotations

import argparse
import json
import time

from lib import (
    EXP_DIR, FOLD, load_config, load_fold2_train_valid,
    train_lambdarank_ext, get_group_sizes,
)

VARIANTS = {
    "truncation": {
        "extra_params": {"lambdarank_truncation_level": 3},
        "seeds": None,  # config既定の5シード
    },
    "truncation1": {
        "extra_params": {"lambdarank_truncation_level": 1},
        "seeds": None,
    },
    "bagging": {
        "extra_params": {
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        },
        "seeds": None,
    },
    "seeds10": {
        "extra_params": {},
        "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    seeds = spec["seeds"] or training_cfg["seeds"]

    print(f"=== variant={args.variant} extra_params={spec['extra_params']} seeds={seeds} ===")

    train_df, valid_df, feature_cols, cat_features = load_fold2_train_valid(cfg)
    print(f"Train: {len(train_df):,} rows / Valid: {len(valid_df):,} rows / features: {len(feature_cols)}")

    y_train = train_df[cfg["features"]["lr_label"]]
    y_valid = valid_df[cfg["features"]["lr_label"]]
    group_train = get_group_sizes(train_df)
    group_valid = get_group_sizes(valid_df)

    models_dir = EXP_DIR / "models" / args.variant
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for seed in seeds:
        print(f"\n--- {args.variant} / seed {seed} ---")
        model = train_lambdarank_ext(
            X_train=train_df, y_train=y_train, group_train=group_train,
            X_valid=valid_df, y_valid=y_valid, group_valid=group_valid,
            feature_cols=feature_cols, cat_features=cat_features,
            params_cfg=params_cfg, training_cfg=training_cfg,
            seed=seed, extra_params=spec["extra_params"],
        )
        model_path = models_dir / f"lambdarank_fold{FOLD}_seed{seed}.txt"
        model.save_model(str(model_path))
        print(f"  Saved: {model_path} (best_iteration={model.best_iteration})")

    elapsed = time.time() - t0
    print(f"\n=== {args.variant} done in {elapsed:.1f}s ===")

    meta_path = models_dir / "meta.json"
    meta_path.write_text(
        json.dumps({"variant": args.variant, "extra_params": spec["extra_params"],
                     "seeds": seeds, "elapsed_sec": elapsed}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

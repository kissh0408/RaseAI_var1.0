"""
regenerate_v5_ensemble_oof.py — ensemble_v5 相当の WF OOF を再生成

3シード walk-forward OOF を算術平均し evaluation_v5_oof.csv に保存する。
specv2 の evaluation_all_non_leak.csv は退避・復元して上書きしない。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_DIR = PROJECT_ROOT / "model_training" / "data" / "03_train"
EVAL_CSV = TRAIN_DIR / "evaluation_all_non_leak.csv"
SPECv2_BACKUP = TRAIN_DIR / "evaluation_specv2_oof.csv"
V5_OOF_CSV = TRAIN_DIR / "evaluation_v5_oof.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 100, 200],
        help="アンサンブル OOF 平均に使うシード",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存 evaluation_v5_oof.csv があっても再生成する",
    )
    args = parser.parse_args()

    if V5_OOF_CSV.exists() and not args.force:
        print(f"[skip] 既存 OOF: {V5_OOF_CSV}")
        return 0

    from model_training.src.train import train_model, _eval_path
    from model_training.src.train_ensemble import load_production_training_kwargs

    prod = load_production_training_kwargs()
    wf_start = prod.get("walkforward_start_year")
    wf_end = prod.get("walkforward_end_year")
    feature_set = str(prod.get("feature_set", "all_non_leak"))

    if not SPECv2_BACKUP.exists() and EVAL_CSV.exists():
        shutil.copy2(EVAL_CSV, SPECv2_BACKUP)
        print(f"[backup] specv2 OOF -> {SPECv2_BACKUP}")
    elif SPECv2_BACKUP.exists():
        print(f"[backup] specv2 OOF 既存: {SPECv2_BACKUP}")

    seed_dfs: list[pd.DataFrame] = []
    eval_path = _eval_path(feature_set)

    for i, seed in enumerate(args.seeds):
        print(f"\n[OOF] seed={seed} ({i + 1}/{len(args.seeds)}) walk-forward {wf_start}-{wf_end}")
        train_model(
            feature_set=feature_set,
            n_trials=0,
            walkforward_start_year=wf_start,
            walkforward_end_year=wf_end,
            show_progress=True,
            seed=seed,
            enable_feature_selection=(seed == args.seeds[0]),
            features_path=prod.get("features_path"),
            optuna_params_dir=PROJECT_ROOT / "model_training" / "models" / "ensemble_v5",
            final_max_rounds=prod.get("final_max_rounds"),
            optuna_max_rounds=prod.get("optuna_max_rounds"),
        )
        if not eval_path.exists():
            print(f"[NG] missing {eval_path}")
            return 1
        seed_dfs.append(pd.read_csv(eval_path))
        print(f"  rows={len(seed_dfs[-1]):,}")

    base = seed_dfs[0].copy()
    keys = ["race_id", "horse_num"]
    for rank in (1, 2, 3):
        col = f"pred_rank{rank}"
        stacked = np.stack([d[col].to_numpy(dtype=float) for d in seed_dfs], axis=0)
        base[col] = np.mean(stacked, axis=0)

    V5_OOF_CSV.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(V5_OOF_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[done] v5 ensemble OOF -> {V5_OOF_CSV} ({len(base):,} rows)")

    if SPECv2_BACKUP.exists():
        shutil.copy2(SPECv2_BACKUP, EVAL_CSV)
        print(f"[restore] specv2 OOF -> {EVAL_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

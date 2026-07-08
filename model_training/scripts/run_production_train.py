"""
run_production_train.py — 本番 rank アンサンブル学習

train_config.json の production_training セクションを唯一の仕様ソースとして読み込み、
train_ensemble() を実行する。

Usage:
    python model_training/scripts/run_production_train.py
    python model_training/scripts/run_production_train.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.src.train_ensemble import (
    load_production_training_kwargs,
    train_ensemble,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="本番 rank アンサンブル学習")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="設定のみ表示し、学習は実行しない",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="保存済み seed モデルがあっても再学習する（skip_existing=False）",
    )
    args = parser.parse_args()

    kwargs = load_production_training_kwargs()
    if args.force:
        kwargs["skip_existing"] = False

    print("=== 本番 rank 学習 ===")
    print(json.dumps(kwargs, ensure_ascii=False, indent=2, default=str))

    if args.dry_run:
        print("[dry-run] 学習はスキップしました")
        return 0

    meta_path = train_ensemble(**kwargs)
    print(f"[DONE] ensemble_meta: {meta_path}")
    print("[NEXT] @backtest-evaluator で ROI/MDD/Sharpe を検証してください")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

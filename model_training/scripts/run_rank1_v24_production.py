"""
run_rank1_v24_production.py — v24 test 合格後の production アンサンブル学習

experiment report の pass_gate=true の場合のみ実行する。
ensemble_v4 は上書きせず ensemble_v5 に出力する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.src.train_ensemble import load_production_training_kwargs, train_ensemble

FEATURES_DIR = PROJECT_ROOT / "model_training" / "data" / "02_features"
TRAIN_DIR = PROJECT_ROOT / "model_training" / "data" / "03_train"
EXP_REPORT = TRAIN_DIR / "rank1_v24_experiment_report.json"
V24_PATH = FEATURES_DIR / "features_past_v24_rank1.parquet"
OUTPUT_DIR = "ensemble_v5"


def main() -> int:
    if not EXP_REPORT.exists():
        print(f"[NG] experiment report missing: {EXP_REPORT}")
        return 1

    report = json.loads(EXP_REPORT.read_text(encoding="utf-8"))
    if not report.get("comparison", {}).get("pass_gate"):
        print("[SKIP] pass_gate=false - production training skipped")
        return 2

    if not V24_PATH.exists():
        print(f"[NG] missing {V24_PATH}")
        return 1

    print("[INFO] v24 test passed — starting production ensemble training")
    prod_kwargs = load_production_training_kwargs(
        overrides={
            "features_path": str(V24_PATH),
            "output_dir": OUTPUT_DIR,
        }
    )
    meta_path = train_ensemble(**prod_kwargs)
    print(f"[OK] ensemble meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

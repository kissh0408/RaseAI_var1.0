"""P1 going_track: features_v6 + going_v1 列 → features_v6_going_v1.parquet

実行:
    python model_training/src/build_features_v6_going_v1.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_training.src.features_going_v1 import (  # noqa: E402
    add_going_v1_features,
    going_v1_column_names,
)
from model_training.src.pipeline_common import FEATURES_DIR  # noqa: E402

INPUT_PATH = FEATURES_DIR / "features_v6.parquet"
OUTPUT_PATH = FEATURES_DIR / "features_v6_going_v1.parquet"
MANIFEST_PATH = FEATURES_DIR / "features_v6_going_v1_manifest.json"


def build_features_v6_going_v1(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    print(f"[INFO] 読み込み: {input_path}")
    df = pd.read_parquet(input_path)
    new_cols = going_v1_column_names()
    drop = [c for c in new_cols if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    df = add_going_v1_features(df)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] 保存: {output_path} ({len(df):,} rows × {df.shape[1]} cols)")

    nan_report = {}
    for col in new_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        nan_report[col] = {
            "nan_rate": round(float(s.isna().mean()), 4),
            "mean": round(float(s.mean()), 4) if s.notna().any() else None,
        }
        print(f"  {col}: nan={nan_report[col]['nan_rate']:.1%} mean={nan_report[col]['mean']}")

    manifest = {
        "name": "features_v6_going_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path.name),
        "rows": len(df),
        "columns_added": new_cols,
        "nan_report": nan_report,
        "spec": "docs/specs/domain_planner_spec_going_v1.md",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] manifest: {manifest_path}")
    return df


if __name__ == "__main__":
    build_features_v6_going_v1()

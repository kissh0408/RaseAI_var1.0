"""going_v1 ベース parquet に単一特徴量セットを追加（1 実験 = 1 セット）。

実行:
    python model_training/scripts/build_going_plus_features.py form_momentum
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from model_training.src.features_form_momentum import (  # noqa: E402
    add_form_momentum_features,
    form_momentum_column_names,
)
from model_training.src.features_rival_strength import (  # noqa: E402
    add_rival_strength_features,
    rival_strength_column_names,
)
from model_training.src.features_track_cond_streak import (  # noqa: E402
    add_track_cond_streak_features,
    track_cond_streak_column_names,
)
from model_training.src.pipeline_common import FEATURES_DIR  # noqa: E402

GOING_BASE = FEATURES_DIR / "features_v6_going_v1.parquet"

SETS = {
    "form_momentum": {
        "suffix": "going_form_momentum_v1",
        "cols_fn": form_momentum_column_names,
        "add_fn": add_form_momentum_features,
    },
    "track_cond_streak": {
        "suffix": "going_track_cond_streak_v1",
        "cols_fn": track_cond_streak_column_names,
        "add_fn": add_track_cond_streak_features,
    },
    "rival_strength": {
        "suffix": "going_rival_strength_v1",
        "cols_fn": rival_strength_column_names,
        "add_fn": add_rival_strength_features,
    },
}


def build(set_name: str, base_path: Path = GOING_BASE) -> Path:
    if set_name not in SETS:
        raise ValueError(f"Unknown set: {set_name}")
    meta = SETS[set_name]
    out_path = FEATURES_DIR / f"features_v6_{meta['suffix']}.parquet"
    manifest_path = FEATURES_DIR / f"features_v6_{meta['suffix']}_manifest.json"
    new_cols = meta["cols_fn"]()

    print(f"[INFO] base: {base_path}")
    df = pd.read_parquet(base_path)
    drop = [c for c in new_cols if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    df = meta["add_fn"](df)
    df.to_parquet(out_path, index=False)
    print(f"[INFO] saved: {out_path} ({len(df):,} rows)")

    nan_report = {}
    for col in new_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        nan_report[col] = {"nan_rate": round(float(s.isna().mean()), 4)}
        print(f"  {col}: nan={nan_report[col]['nan_rate']:.1%}")

    manifest = {
        "name": out_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": base_path.name,
        "feature_set": set_name,
        "columns_added": new_cols,
        "nan_report": nan_report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("set", choices=list(SETS))
    args = parser.parse_args()
    build(args.set)


if __name__ == "__main__":
    main()

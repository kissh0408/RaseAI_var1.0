"""v25 特徴量 Parquet から v26/v27 派生ファイルを生成する（既存 v25 非破壊）。

Usage:
  python model_training/scripts/build_going_feature_parquets.py
  python model_training/scripts/build_going_feature_parquets.py --source features_past_v25_odds.parquet
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

from model_training.src.builders.going_delta import add_going_delta_features, going_delta_feature_names
from model_training.src.builders.track_variant import add_track_variant_features

FEATURES_DIR = ROOT / "model_training" / "data" / "02_features"


def _save_manifest(path: Path, df: pd.DataFrame, extra: dict) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "columns": len(df.columns),
        **extra,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build(source_name: str) -> tuple[Path, Path]:
    src = FEATURES_DIR / source_name
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    print(f"Loaded {src.name}: {len(df):,} rows, {len(df.columns)} cols")

    v26 = add_going_delta_features(df)
    out26 = FEATURES_DIR / "features_past_v26_going_delta.parquet"
    v26.to_parquet(out26, index=False)
    _save_manifest(
        out26.with_name("features_past_v26_going_delta_manifest.json"),
        v26,
        {"source": source_name, "added_features": list(going_delta_feature_names())},
    )
    print(f"Saved {out26.name} (+{len(going_delta_feature_names())} cols)")

    v27 = add_track_variant_features(v26)
    out27 = FEATURES_DIR / "features_past_v27_track_variant.parquet"
    v27.to_parquet(out27, index=False)
    _save_manifest(
        out27.with_name("features_past_v27_track_variant_manifest.json"),
        v27,
        {
            "source": "features_past_v26_going_delta.parquet",
            "added_features": ["daily_track_variant", "tm_score_surface_adj"],
        },
    )
    print(f"Saved {out27.name}")
    return out26, out27


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="features_past_v25_odds.parquet")
    args = parser.parse_args()
    build(args.source)


if __name__ == "__main__":
    main()

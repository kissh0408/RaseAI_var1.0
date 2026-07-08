"""
build_features_v25_odds.py — v23 に市場オッズ乖離特徴量（v25 Exp-3）を追加

入力  : features_past_v23.parquet（読み取り専用）
出力  : features_past_v25_odds.parquet + features_past_v25_odds_manifest.json

実行:
    python model_training/src/build_features_v25_odds.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.src.features_odds_divergence import (
    add_odds_divergence_features,
    v25_odds_column_names,
)

FEATURES_DIR = PROJECT_ROOT / "model_training" / "data" / "02_features"
INPUT_PATH = FEATURES_DIR / "features_past_v23.parquet"
OUTPUT_PATH = FEATURES_DIR / "features_past_v25_odds.parquet"
MANIFEST_PATH = FEATURES_DIR / "features_past_v25_odds_manifest.json"

NEW_COLUMNS = v25_odds_column_names()


def _quality_check(df: pd.DataFrame) -> dict:
    report: dict = {"nan_rates": {}, "value_stats": {}}
    for col in NEW_COLUMNS:
        if col not in df.columns:
            report["nan_rates"][col] = {"status": "MISSING"}
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_rate = float(s.isna().mean())
        report["nan_rates"][col] = {
            "nan_rate": round(nan_rate, 4),
            "min": round(float(s.min()), 4) if s.notna().any() else None,
            "max": round(float(s.max()), 4) if s.notna().any() else None,
            "mean": round(float(s.mean()), 4) if s.notna().any() else None,
            "status": "OK" if nan_rate <= 0.20 else "WARN",
        }
    return report


def build_features_v25_odds(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    print(f"[INFO] 読み込み中: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[INFO] 入力 shape: {df.shape}")

    # 冪等性: 出力列が既存の場合は削除して再生成
    existing = [c for c in NEW_COLUMNS if c in df.columns]
    if existing:
        print(f"[INFO] 既存列を削除: {existing}")
        df = df.drop(columns=existing)

    print("[INFO] 市場オッズ乖離特徴量を追加 ...")
    df = add_odds_divergence_features(df)
    print(f"[INFO] v25_odds shape: {df.shape}")

    qc = _quality_check(df)
    print("[INFO] 品質チェック:")
    for col, info in qc["nan_rates"].items():
        nan_r = info.get("nan_rate")
        status = info.get("status")
        mean = info.get("mean")
        if nan_r is not None:
            print(f"  {col}: nan_rate={nan_r:.4f}  mean={mean:.3f}  status={status}")
        else:
            print(f"  {col}: status={status}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] 保存完了: {output_path}")

    # race_date / date のどちらかから日付範囲を取得
    date_col = "race_date" if "race_date" in df.columns else ("date" if "date" in df.columns else None)
    manifest = {
        "name": "features_past_v25_odds",
        "base": str(input_path.name),
        "added_columns": NEW_COLUMNS,
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": [
            str(df[date_col].min()) if date_col else None,
            str(df[date_col].max()) if date_col else None,
        ],
        "quality_check": qc,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] マニフェスト保存: {manifest_path}")

    return df


if __name__ == "__main__":
    build_features_v25_odds()

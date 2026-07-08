"""Champion 特徴量 parquet のビルドオーケストレータ。

実行:
  python model_training/scripts/build_champion_features.py
  python model_training/scripts/build_champion_features.py --full   # v2→v6 全再生成（時間がかかる）

出力: features_v6_going_v1_top3.parquet (+ manifest)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from champion_features import (  # noqa: E402
    CHAMPION_PARQUET,
    LEGACY_CHAMPION_PARQUET,
    TOP3_PAST_COLS,
    build_champion_parquet,
    validate_champion_columns,
)
from pipeline_common import FEATURES_DIR  # noqa: E402
from train import get_feature_cols, load_train_config, TRAIN_CONFIG_PATH  # noqa: E402

CHAIN = [
    ROOT / "model_training/src/create_features_v2.py",
    ROOT / "model_training/src/create_features_v3.py",
    ROOT / "model_training/src/create_features_v4.py",
    ROOT / "model_training/src/create_features_v6.py",
    ROOT / "model_training/src/build_features_v6_going_v1.py",
]


def run_full_chain() -> None:
    for script in CHAIN:
        print(f"\n=== {script.name} ===")
        subprocess.run([sys.executable, str(script)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="v2→v6→going_v1 をフル再生成してから top3 付与",
    )
    parser.add_argument(
        "--legacy-alias",
        action="store_true",
        help=f"互換用に {LEGACY_CHAMPION_PARQUET} も書き出す",
    )
    args = parser.parse_args()

    if args.full:
        run_full_chain()
    elif not (FEATURES_DIR / "features_v6_going_v1.parquet").exists():
        print("[ERROR] features_v6_going_v1.parquet がありません。--full を指定してください。")
        return 1

    df = build_champion_parquet(output_name=CHAMPION_PARQUET)

    cfg = load_train_config(TRAIN_CONFIG_PATH)
    required = get_feature_cols(cfg)
    missing = validate_champion_columns(df, required)
    if missing:
        print(f"[WARN] champion parquet に不足列: {missing[:10]}{'...' if len(missing)>10 else ''}")
        return 1

    for col in TOP3_PAST_COLS:
        s = df[col]
        print(f"  {col}: nan={s.isna().mean():.1%} mean={s.mean():.4f}")

    if args.legacy_alias:
        legacy_path = FEATURES_DIR / LEGACY_CHAMPION_PARQUET
        df.to_parquet(legacy_path, index=False)
        print(f"[champion] legacy alias: {legacy_path.name}")

    print(f"[OK] champion ready: {CHAMPION_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

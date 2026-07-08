"""[DEPRECATED] patch_features_v6_top3 → build_champion_features.py を使用。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from champion_features import LEGACY_CHAMPION_PARQUET, build_champion_parquet  # noqa: E402


def main() -> None:
    print("[DEPRECATED] build_champion_features.py へ移行しました。")
    build_champion_parquet(output_name=LEGACY_CHAMPION_PARQUET)


if __name__ == "__main__":
    main()

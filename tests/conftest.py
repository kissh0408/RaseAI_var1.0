"""pytest 共通: プロジェクトルートを import パスに追加する。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MT_SRC = ROOT / "model_training" / "src"
if str(MT_SRC) not in sys.path:
    sys.path.insert(0, str(MT_SRC))

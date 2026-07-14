"""Item 9 (spec section 8): source-text guard against market-derived columns
in the candidate-generation layer (training_lib.py, build_candidates.py)."""

from __future__ import annotations

import re
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]

FORBIDDEN_PATTERN = re.compile(
    r"odds|popularity|ninki|market_log_odds|init_score|market_q|ln_market",
    re.IGNORECASE,
)


def test_training_lib_has_no_market_references():
    text = (EXP_DIR / "training_lib.py").read_text(encoding="utf-8")
    hits = FORBIDDEN_PATTERN.findall(text)
    assert not hits, f"market-derived references found in training_lib.py: {hits}"


def test_build_candidates_has_no_market_references():
    text = (EXP_DIR / "build_candidates.py").read_text(encoding="utf-8")
    hits = FORBIDDEN_PATTERN.findall(text)
    assert not hits, f"market-derived references found in build_candidates.py: {hits}"

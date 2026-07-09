"""get_feature_cols の市場情報混入ガード（完全一致リスト + 正規表現の第二防波堤）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _SRC)
from common import get_feature_cols

sys.path.remove(_SRC)
sys.modules.pop("common", None)


def _cfg() -> dict:
    return {"features": {"id_cols": ["race_id", "ketto_num", "race_date", "finish_rank", "is_win", "lr_label"]}}


def test_normal_features_pass_through():
    df = pd.DataFrame({
        "race_id": ["r1"], "ketto_num": [1], "race_date": pd.to_datetime(["2024-01-01"]),
        "finish_rank": [1], "is_win": [1], "lr_label": [1],
        "hist_win_rate": [0.3], "wakuban": [1],
    })
    cols = get_feature_cols(df, _cfg())
    assert set(cols) == {"hist_win_rate", "wakuban"}


def test_exact_match_blocklist_still_excludes_odds():
    df = pd.DataFrame({
        "race_id": ["r1"], "ketto_num": [1], "race_date": pd.to_datetime(["2024-01-01"]),
        "finish_rank": [1], "is_win": [1], "lr_label": [1],
        "hist_win_rate": [0.3], "odds": [3.5], "popularity": [1],
    })
    cols = get_feature_cols(df, _cfg())
    assert "odds" not in cols
    assert "popularity" not in cols


@pytest.mark.parametrize("suspicious_col", [
    "implied_prob", "book_pct", "field_strength_market", "win_prob_est", "ninki_diff",
])
def test_fuzzy_market_name_raises(suspicious_col):
    df = pd.DataFrame({
        "race_id": ["r1"], "ketto_num": [1], "race_date": pd.to_datetime(["2024-01-01"]),
        "finish_rank": [1], "is_win": [1], "lr_label": [1],
        "hist_win_rate": [0.3], suspicious_col: [0.5],
    })
    with pytest.raises(ValueError, match="市場情報混入"):
        get_feature_cols(df, _cfg())

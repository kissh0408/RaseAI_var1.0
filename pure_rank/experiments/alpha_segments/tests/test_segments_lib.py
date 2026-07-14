"""TDD tests for segments_lib.py (spec section 9, items 1/2/3/7/8/9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import segments_lib as sl  # noqa: E402


def _cfg() -> dict:
    return sl.load_config()


# ─── item 1: segment filter correctness ────────────────────────────────────


def test_s1_debut_ratio_boundaries():
    # Race A: nan ratio 0 (0/4), Race B: 0.4 (2/5), Race C: 0.5 (2/4), Race D: 1.0 (3/3)
    rows = []
    rows += [{"race_id": "A", "hist_last_rank": v} for v in [1.0, 2.0, 3.0, 4.0]]
    rows += [{"race_id": "B", "hist_last_rank": v} for v in [1.0, 2.0, np.nan, np.nan, 5.0]]
    rows += [{"race_id": "C", "hist_last_rank": v} for v in [1.0, 2.0, np.nan, np.nan]]
    rows += [{"race_id": "D", "hist_last_rank": v} for v in [np.nan, np.nan, np.nan]]
    df = pd.DataFrame(rows)
    flag = sl.flag_debut_ratio(df, threshold=0.5)
    per_race = flag.groupby(df["race_id"]).first()
    assert per_race["A"] == False  # noqa: E712
    assert per_race["B"] == False  # noqa: E712
    assert per_race["C"] == True  # boundary 0.5 inclusive  # noqa: E712
    assert per_race["D"] == True  # noqa: E712


def test_s2_small_field_boundary():
    df = pd.DataFrame(
        {
            "race_id": ["A"] * 8 + ["B"] * 9,
            "horse_count": [8] * 8 + [9] * 9,
        }
    )
    flag = sl.flag_lte(df, col="horse_count", threshold=8)
    per_race = flag.groupby(df["race_id"]).first()
    assert per_race["A"] == True  # noqa: E712
    assert per_race["B"] == False  # noqa: E712


def test_s3_heavy_track():
    df = pd.DataFrame(
        {
            "race_id": ["A", "B", "C", "D"],
            "track_condition_code": [0, 1, 2, 3],
        }
    )
    flag = sl.flag_in(df, col="track_condition_code", values=[3, 4])
    assert flag.tolist() == [False, False, False, True]

    df2 = pd.DataFrame({"race_id": ["E"], "track_condition_code": [4]})
    assert sl.flag_in(df2, col="track_condition_code", values=[3, 4]).iloc[0]


def test_s4_local_venue():
    df = pd.DataFrame(
        {
            "race_id": list("ABCDEFGHIJ"),
            "course_code": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    flag = sl.flag_in(df, col="course_code", values=[1, 2, 3, 4, 7, 10])
    expected = [True, True, True, True, False, False, True, False, False, True]
    assert flag.tolist() == expected


def test_s5_low_class():
    df = pd.DataFrame(
        {
            "race_id": list("ABCDEF"),
            "race_condition_code": [701, 703, 5, 10, 16, 999],
        }
    )
    flag = sl.flag_in(df, col="race_condition_code", values=[703, 5])
    assert flag.tolist() == [False, True, True, False, False, False]


def test_race_level_assignment_is_uniform():
    """All rows of the same race_id get the same flag."""
    df = pd.DataFrame(
        {
            "race_id": ["A", "A", "A", "B", "B"],
            "horse_count": [8, 8, 8, 9, 9],
        }
    )
    flag = sl.flag_lte(df, col="horse_count", threshold=8)
    assert flag[df["race_id"] == "A"].nunique() == 1
    assert flag[df["race_id"] == "B"].nunique() == 1


def test_add_all_segment_flags_uses_config():
    cfg = _cfg()
    df = pd.DataFrame(
        {
            "race_id": ["A", "A"],
            "hist_last_rank": [np.nan, np.nan],
            "horse_count": [5, 5],
            "track_condition_code": [4, 4],
            "course_code": [1, 1],
            "race_condition_code": [703, 703],
        }
    )
    out = sl.add_all_segment_flags(df, cfg)
    for seg_id in cfg["segment_order"]:
        assert f"seg_{seg_id}" in out.columns
        assert out[f"seg_{seg_id}"].all()


# ─── item 2: n<300 exclusion logic ──────────────────────────────────────────


def test_confirm_segments_n_min_boundary():
    counts = {
        "S1": {"n_2023": 300, "n_2024": 299},
        "S2": {"n_2023": 300, "n_2024": 300},
    }
    out = sl.confirm_segments(counts, n_min=300)
    assert out["S1"]["confirmed"] is False
    assert out["S2"]["confirmed"] is True


# ─── item 3: Bonferroni threshold ───────────────────────────────────────────


def test_bonferroni_threshold():
    assert sl.bonferroni_threshold(4) == pytest.approx(0.0025)
    assert sl.bonferroni_threshold(1) == pytest.approx(0.01)
    assert sl.bonferroni_threshold(0) == 0.0


# ─── item 7: leak-stop flag ─────────────────────────────────────────────────


def test_leak_stop_triggers():
    assert sl.leak_stop(top1=0.41, spearman=0.10) is True
    assert sl.leak_stop(top1=0.10, spearman=0.61) is True
    assert sl.leak_stop(top1=0.30, spearman=0.40) is False
    # boundary: exactly at threshold does not trigger (spec uses strict >)
    assert sl.leak_stop(top1=0.40, spearman=0.60) is False


def test_primary_pass_excludes_leak_stopped_segments():
    # Would otherwise pass (p well below threshold, positive deltaLL) but leak-stopped.
    assert sl.primary_pass(p_value=0.0001, delta_ll_per_race=0.05, bonferroni_threshold_value=0.0025, leak=True) is False
    assert sl.primary_pass(p_value=0.0001, delta_ll_per_race=0.05, bonferroni_threshold_value=0.0025, leak=False) is True
    assert sl.primary_pass(p_value=0.01, delta_ll_per_race=0.05, bonferroni_threshold_value=0.0025, leak=False) is False
    assert sl.primary_pass(p_value=0.0001, delta_ll_per_race=-0.01, bonferroni_threshold_value=0.0025, leak=False) is False


# ─── item 8: market column guard ────────────────────────────────────────────


def test_segment_columns_whitelist_matches_spec():
    expected = {
        "hist_last_rank",
        "horse_count",
        "track_condition_code",
        "course_code",
        "race_condition_code",
    }
    assert sl.SEGMENT_COLUMNS == frozenset(expected)


def test_segment_columns_not_forbidden_or_suspicious():
    from pure_rank.src.common import FORBIDDEN_MARKET_COLS, SUSPICIOUS_MARKET_NAME_PATTERN

    for col in sl.SEGMENT_COLUMNS:
        assert col not in FORBIDDEN_MARKET_COLS
        assert not SUSPICIOUS_MARKET_NAME_PATTERN.search(col), f"{col} matches suspicious market pattern"


def test_apply_segment_flag_rejects_unregistered_column():
    cfg = _cfg()
    bad_cfg = {
        "segments": {
            "SX": {"column": "odds", "rule": "lte", "threshold": 1.0},
        }
    }
    df = pd.DataFrame({"race_id": ["A"], "odds": [1.5]})
    with pytest.raises(ValueError):
        sl.apply_segment_flag(df, "SX", bad_cfg)


# ─── item 9: reproducibility (config load is deterministic) ────────────────


def test_config_load_is_deterministic():
    cfg1 = _cfg()
    cfg2 = _cfg()
    assert cfg1 == cfg2
    assert cfg1["thresholds"]["n_min_eval_2024"] == 300
    assert cfg1["thresholds"]["lrt_alpha"] == 0.01

"""BASE_LEAK_COLS と assert_no_leak_columns の単体テスト（BT-2）。"""
from __future__ import annotations

import pytest

from pipeline_common import BASE_LEAK_COLS, assert_no_leak_columns


def test_assert_no_leak_columns_passes_clean_feature_list() -> None:
    features = ["horse_age", "distance", "past_speed_index_mean", "rpr_score"]
    assert_no_leak_columns(features, BASE_LEAK_COLS)


def test_assert_no_leak_columns_raises_on_overlap() -> None:
    features = ["horse_age", "speed_index", "pci_past_mean"]
    with pytest.raises(SystemExit, match="Leak columns detected"):
        assert_no_leak_columns(features, BASE_LEAK_COLS)


@pytest.mark.parametrize(
    "col",
    [
        "agari_z_race",
        "lap_time_std",
        "early_pace_ratio",
        "speed_index",
        "base_time_cond_zscore",
    ],
)
def test_phase4_leak_cols_registered(col: str) -> None:
    assert col in BASE_LEAK_COLS

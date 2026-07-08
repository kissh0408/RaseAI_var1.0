"""Tests for evaluation.splits."""

from __future__ import annotations

import pandas as pd

from evaluation.splits import TRAIN_END, VALID_END, assign_split, get_walkforward_folds


def test_assign_split_boundaries():
    df = pd.DataFrame(
        {
            "race_date": ["2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01"],
        }
    )
    splits = assign_split(df["race_date"], train_end=TRAIN_END, valid_end=VALID_END)
    assert splits.tolist() == ["train", "valid", "valid", "test"]


def test_walkforward_folds_count():
    folds = get_walkforward_folds()
    assert len(folds) == 3
    assert folds[2]["test_start"] == "2025-01-01"

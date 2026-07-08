"""Single source of truth for TRAIN / VALID / TEST date splits."""

from __future__ import annotations

from typing import Literal

import pandas as pd

SplitName = Literal["train", "valid", "test"]

TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"

FORMAL_JUDGMENT_FOLD = 3
L1_CONTAMINATED_REFERENCE_FOLDS = (1, 2)


def assign_split(
    race_dates: pd.Series,
    *,
    train_end: str = TRAIN_END,
    valid_end: str = VALID_END,
) -> pd.Series:
    """Return split label per row based on race_date."""
    dates = pd.to_datetime(race_dates)
    train_cut = pd.Timestamp(train_end)
    valid_cut = pd.Timestamp(valid_end)
    out = pd.Series("test", index=dates.index, dtype="object")
    out.loc[dates <= train_cut] = "train"
    out.loc[(dates > train_cut) & (dates <= valid_cut)] = "valid"
    return out


def filter_by_split(
    df: pd.DataFrame,
    split: SplitName,
    *,
    race_date_col: str = "race_date",
    train_end: str = TRAIN_END,
    valid_end: str = VALID_END,
) -> pd.DataFrame:
    """Filter dataframe to one split."""
    labels = assign_split(df[race_date_col], train_end=train_end, valid_end=valid_end)
    return df.loc[labels == split].copy()


def get_walkforward_folds() -> list[dict]:
    """Walk-forward fold definitions for L2/L3 (Benter rebuild)."""
    return [
        {
            "fold": 1,
            "train_end": "2021-12-31",
            "valid_start": "2022-01-01",
            "valid_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-12-31",
        },
        {
            "fold": 2,
            "train_end": "2022-12-31",
            "valid_start": "2023-01-01",
            "valid_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2024-12-31",
        },
        {
            "fold": 3,
            "train_end": "2023-12-31",
            "valid_start": "2024-01-01",
            "valid_end": "2024-12-31",
            "test_start": "2025-01-01",
            "test_end": "2026-12-31",
        },
    ]


def filter_fold(df: pd.DataFrame, fold: dict, period: Literal["train", "valid", "test"]) -> pd.DataFrame:
    """Filter dataframe to a walk-forward fold period."""
    dates = pd.to_datetime(df["race_date"])
    if period == "train":
        end = pd.Timestamp(fold["train_end"])
        return df.loc[dates <= end].copy()
    if period == "valid":
        start = pd.Timestamp(fold["valid_start"])
        end = pd.Timestamp(fold["valid_end"])
        return df.loc[(dates >= start) & (dates <= end)].copy()
    start = pd.Timestamp(fold["test_start"])
    end = pd.Timestamp(fold["test_end"])
    return df.loc[(dates >= start) & (dates <= end)].copy()

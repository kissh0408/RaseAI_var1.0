"""OOS 正式測定プロトコル（fold2 スコア: fit=2023-2024, test=2025+）のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prob_fusion.src.oos_protocol import evaluate_oos_gates, split_oos_periods


def _frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": [f"r{i}" for i in range(len(dates))],
            "race_date": pd.to_datetime(dates),
        }
    )


def test_split_oos_periods_boundaries() -> None:
    df = _frame(["2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01", "2025-06-01"])
    fit_df, test_df = split_oos_periods(
        df, fit_start="2023-01-01", fit_end="2024-12-31", test_start="2025-01-01"
    )
    assert list(fit_df["race_id"]) == ["r1", "r2"]  # 2022以前は捨てる
    assert list(test_df["race_id"]) == ["r3", "r4"]


def test_gates_pass_when_beats_market_and_top1() -> None:
    gates = evaluate_oos_gates(
        {"test_logloss_fusion": 1.90, "test_logloss_market": 1.93, "test_top1": 0.335}
    )
    assert gates["logloss_beats_market"] is True
    assert gates["top1_gate_33pct"] is True
    assert gates["phase2_pass"] is True


def test_gates_fail_on_logloss() -> None:
    gates = evaluate_oos_gates(
        {"test_logloss_fusion": 1.95, "test_logloss_market": 1.93, "test_top1": 0.34}
    )
    assert gates["logloss_beats_market"] is False
    assert gates["phase2_pass"] is False


def test_gates_fail_on_top1() -> None:
    gates = evaluate_oos_gates(
        {"test_logloss_fusion": 1.90, "test_logloss_market": 1.93, "test_top1": 0.31}
    )
    assert gates["top1_gate_33pct"] is False
    assert gates["phase2_pass"] is False

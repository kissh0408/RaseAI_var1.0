"""OOS 正式測定プロトコル（Phase 2 再測定）。

fold2 モデル（train <2023、2023 は early stopping のみ、2024/2025 完全未使用）の
OOS スコアを用いる:

    fit   = 2023-01-01 〜 2024-12-31  （α, β, 市場β, Stern λ の推定）
    TEST  = 2025-01-01 〜             （判定は 1 回のみ）

注意: 2023 は fold2 の early stopping に使われており弱い汚染がある。
manifest / レポートに caveat として記録する。感度分析として 2024 単独 fit も
併記するが、正式判定は fit=2023+2024 とする（本プロトコルは TEST 実行前に確定済み）。
"""

from __future__ import annotations

import pandas as pd

FIT_START = "2023-01-01"
FIT_END = "2024-12-31"
TEST_START = "2025-01-01"
TOP1_GATE = 0.33


def split_oos_periods(
    df: pd.DataFrame,
    *,
    fit_start: str = FIT_START,
    fit_end: str = FIT_END,
    test_start: str = TEST_START,
    race_date_col: str = "race_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fit_df, test_df) に分割。fit 開始前の行は捨てる。"""
    dates = pd.to_datetime(df[race_date_col])
    fit_mask = (dates >= pd.Timestamp(fit_start)) & (dates <= pd.Timestamp(fit_end))
    test_mask = dates >= pd.Timestamp(test_start)
    return df.loc[fit_mask].copy(), df.loc[test_mask].copy()


def evaluate_oos_gates(metrics: dict) -> dict:
    """Phase 2 合格ゲート判定（logloss 市場超え AND Top-1 ≥ 33%）。"""
    beats = bool(metrics["test_logloss_fusion"] < metrics["test_logloss_market"])
    top1_ok = bool(metrics["test_top1"] >= TOP1_GATE)
    return {
        "logloss_beats_market": beats,
        "top1_gate_33pct": top1_ok,
        "phase2_pass": beats and top1_ok,
    }

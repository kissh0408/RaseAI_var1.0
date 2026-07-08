"""inference_common 純粋関数の characterization テスト（数値は現行実装の観測値）。"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MT_SRC = ROOT / "model_training" / "src"
if str(MT_SRC) not in sys.path:
    sys.path.insert(0, str(MT_SRC))

# train.compute_base_margin が train.py から参照不能な環境でも import できるようスタブ
if "train" not in sys.modules:
    _train_stub = types.ModuleType("train")

    def _compute_base_margin(df: pd.DataFrame, col: str = "market_log_odds") -> np.ndarray:
        if col not in df.columns or df[col].isna().all():
            n = df["horse_count"].fillna(10) if "horse_count" in df.columns else 10
            p = (1.0 / n).clip(1e-6, 1 - 1e-6)
            return np.log(p / (1 - p)).values
        base = df[col].copy()
        missing_mask = base.isna()
        if missing_mask.any():
            n = df.loc[missing_mask, "horse_count"].fillna(10) if "horse_count" in df.columns else 10
            p = (1.0 / n).clip(1e-6, 1 - 1e-6)
            base.loc[missing_mask] = np.log(p / (1 - p))
        return base.values

    _train_stub.compute_base_margin = _compute_base_margin
    sys.modules["train"] = _train_stub

from strategy.src.inference_common import (  # noqa: E402
    apply_condition_overrides,
    apply_race_budget_cap,
    compute_market_log_odds,
    normalize_within_race,
)


class TestNormalizeWithinRace:
    def test_sums_to_one_per_race(self):
        df = pd.DataFrame({"race_id": ["A", "A", "B"], "x": [1, 2, 3]})
        probs = np.array([0.3, 0.5, 0.8])
        out = normalize_within_race(probs, df)
        assert np.isclose(out[:2].sum(), 1.0)
        assert np.isclose(out[2], 1.0)

    def test_degenerate_all_zero_normalizes_uniform(self):
        df = pd.DataFrame({"race_id": ["A", "A"], "x": [1, 2]})
        probs = np.array([0.0, 0.0])
        out = normalize_within_race(probs, df)
        assert np.allclose(out, [0.5, 0.5])


class TestComputeMarketLogOdds:
    def test_known_odds_produce_finite_log_odds(self):
        df = pd.DataFrame({"race_id": ["A", "A"], "odds": [3.0, 5.0]})
        result = compute_market_log_odds(df)
        assert "market_log_odds" in result.columns
        assert result["market_log_odds"].notna().all()
        # 3.0 / 5.0 → レース内正規化後の log-odds（観測値固定）
        expected = [0.5108256237659907, -0.5108256237659907]
        assert np.allclose(result["market_log_odds"].tolist(), expected, rtol=1e-9)


class TestApplyConditionOverrides:
    def test_no_overrides_returns_base_mask(self):
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "surface_code": [1, 1],
                "ev_rate": [1.10, 1.04],
            }
        )
        base = pd.Series([True, True], index=df.index)
        out = apply_condition_overrides(df, base, [], default_ev_threshold=1.05)
        assert out.tolist() == [True, True]

    def test_surface_override_raises_threshold(self):
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "surface_code": [2, 2],
                "ev_rate": [1.10, 1.04],
            }
        )
        base = pd.Series([True, True], index=df.index)
        overrides = [{"surface_code": 2, "min_ev": 1.08}]
        out = apply_condition_overrides(df, base, overrides, default_ev_threshold=1.05)
        assert out.tolist() == [True, False]


class TestApplyRaceBudgetCap:
    def test_scales_down_when_race_total_exceeds_cap(self):
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "kelly_ratio": [0.04, 0.04],
            }
        )
        mask = pd.Series([True, True], index=df.index)
        out = apply_race_budget_cap(df, mask, max_bet_ratio=0.05, bankroll=100_000.0)
        assert np.isclose(out.loc[mask, "kelly_ratio"].sum(), 0.05, rtol=1e-9)
        assert (out.loc[mask, "kelly_bet_yen"] % 100 == 0).all()

    def test_no_scaling_when_under_cap(self):
        df = pd.DataFrame({"race_id": ["R1"], "kelly_ratio": [0.03]})
        mask = pd.Series([True], index=df.index)
        out = apply_race_budget_cap(df, mask, max_bet_ratio=0.05, bankroll=100_000.0)
        assert out["kelly_ratio"].tolist() == [0.03]
        assert out["kelly_bet_yen"].tolist() == [3000.0]


class TestComputeMarketLogOddsEdgeCases:
    def test_zero_odds_row_is_nan_market_log_odds(self):
        df = pd.DataFrame({"race_id": ["A", "A"], "odds": [0.0, 3.0]})
        result = compute_market_log_odds(df)
        assert np.isnan(result["market_log_odds"].iloc[0])
        assert np.isclose(result["market_log_odds"].iloc[1], 13.815509557935018, rtol=1e-9)

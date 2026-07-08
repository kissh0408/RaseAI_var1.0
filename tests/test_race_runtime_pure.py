"""race_runtime 純粋関数の characterization テスト。"""
from __future__ import annotations

import pandas as pd

from main.race_runtime import filter_scratched


class TestFilterScratched:
    def test_odds_zero_excluded(self):
        recs = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "horse_id": ["H1", "H2", "H3"],
                "odds": [3.0, 0.0, 5.0],
                "win_prob_est": [0.4, 0.3, 0.3],
                "expected_return": [1.2, 0.0, 1.5],
            }
        )
        out = filter_scratched(recs)
        assert len(out) == 2
        assert set(out["horse_id"]) == {"H1", "H3"}

    def test_odds_nan_excluded_and_renormalized(self):
        recs = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "horse_id": ["H1", "H2", "H3"],
                "odds": [3.0, float("nan"), 5.0],
                "win_prob_est": [0.4, 0.3, 0.3],
                "expected_return": [1.2, float("nan"), 1.5],
            }
        )
        out = filter_scratched(recs)
        assert len(out) == 2
        assert abs(out["win_prob_est"].sum() - 1.0) < 1e-6
        assert abs(out.loc[out["horse_id"] == "H1", "expected_return"].iloc[0] - (3.0 * 0.4 / 0.7)) < 1e-6

    def test_explicit_scratch_list(self):
        recs = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "horse_id": ["H1", "H2"],
                "odds": [3.0, 5.0],
            }
        )
        out = filter_scratched(recs, scratched_horses=["H2"])
        assert len(out) == 1
        assert out["horse_id"].iloc[0] == "H1"

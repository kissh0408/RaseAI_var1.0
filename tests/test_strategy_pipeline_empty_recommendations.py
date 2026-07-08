import tempfile
import unittest
from pathlib import Path

import pandas as pd

from main.pipeline.strategy_pipeline import (
    build_empty_recommendation_notice,
    persist_recommendations,
)


class EmptyRecommendationPipelineTests(unittest.TestCase):
    def test_persist_recommendations_writes_empty_outputs(self):
        rec_df = pd.DataFrame(columns=["ticket_type", "race_id", "expected_value"])

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "today_recommendations.csv"

            persist_recommendations(rec_df, out)

            self.assertTrue(out.exists())
            self.assertTrue(out.with_suffix(".parquet").exists())
            self.assertEqual(list(pd.read_csv(out).columns), list(rec_df.columns))
            self.assertEqual(len(pd.read_parquet(out.with_suffix(".parquet"))), 0)

    def test_build_empty_recommendation_notice_reports_missing_today_odds(self):
        pred_df = pd.DataFrame(
            {
                "month_day": [614, 614, 614],
                "odds": [pd.NA, pd.NA, pd.NA],
                "race_id": ["2026061405010101", "2026061405010101", "2026061405010102"],
            }
        )

        message = build_empty_recommendation_notice(pred_df)

        self.assertIn("推奨馬券なし", message)
        self.assertIn("6月14日", message)
        self.assertIn("単勝オッズ未取得", message)


if __name__ == "__main__":
    unittest.main()

"""Fetch historical WideOdds / QuinellaOdds CSV (Step 1-1).

Usage:
  python model_training/scripts/fetch_wide_odds_yearly.py --start-year 2024 --end-year 2026

Requires JV-Link / accumulated RACE data. Output:
  common/data/output/odds/WideOdds_YYYY.csv
  common/data/output/odds/QuinellaOdds_YYYY.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common" / "data" / "src"))

from legacy_get_data_impl import fetch_pairwide_odds_0b31_yearly  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch yearly Wide/Quinella odds CSV")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    odds_dir = ROOT / "common" / "data" / "output" / "odds"
    odds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching pair/wide odds {args.start_year}..{args.end_year} -> {odds_dir}")
    results = fetch_pairwide_odds_0b31_yearly(
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )
    for year, info in sorted(results.items()):
        wide = info.get("wide", {})
        quin = info.get("quinella", {})
        print(
            f"  {year}: races={info.get('races', 0)} "
            f"wide_added={wide.get('added', 0)} quin_added={quin.get('added', 0)}"
        )


if __name__ == "__main__":
    main()

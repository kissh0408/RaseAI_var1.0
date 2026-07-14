"""
時系列オッズ（0B41 単複枠 / 0B42 馬連）取得の CLI エントリポイント。

保持期間が1年間しかない（docs/JV-Data.md 2438-2439行）ため、他のオッズ取得より
優先してこのフェッチャを動かし蓄積を開始する。実体は
legacy_get_data_impl.fetch_odds_ts_yearly / fetch_odds_ts_0b41_0b42_for_main_races。

使い方:
    # 初回一括取得（当年分。過去の年を指定しても1年より前は自動的に0件になる）
    python common/data/src/fetch_odds_ts_cli.py init --start-year 2025 --end-year 2026

    # 週次差分取得（main/data/race/race_ra.csv に載っている直近レース分）
    python common/data/src/fetch_odds_ts_cli.py weekly --start-date 20260706 --end-date 20260712

JV-Link は 32bit Python が必要な環境が多い。64bit から実行してエラーになる場合は
`py -3-32 common/data/src/fetch_odds_ts_cli.py ...` のように 32bit ランチャー経由
で実行すること（common/data/src/jv_subprocess.py の run_with_32bit_python も利用可）。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

try:
    from .legacy_get_data_impl import (
        fetch_odds_ts_0b41_0b42_for_main_races,
        fetch_odds_ts_yearly,
    )
except ImportError:
    from legacy_get_data_impl import (
        fetch_odds_ts_0b41_0b42_for_main_races,
        fetch_odds_ts_yearly,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="0B41(O1時系列単複枠) / 0B42(O2時系列馬連) フェッチャ"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    init_p = sub.add_parser(
        "init", help="初回一括取得（common/data/output/race_ra/*.csv 起点）"
    )
    init_p.add_argument("--start-year", type=int, default=datetime.now().year)
    init_p.add_argument("--end-year", type=int, default=None)
    init_p.add_argument("--source-output-dir", default=None)
    init_p.add_argument("--overwrite", action="store_true")

    weekly_p = sub.add_parser(
        "weekly", help="週次差分取得（main/data/race/race_ra.csv 起点）"
    )
    weekly_p.add_argument("--start-date", default=None, help="YYYYMMDD")
    weekly_p.add_argument("--end-date", default=None, help="YYYYMMDD")
    weekly_p.add_argument("--output-dir", default=None)

    args = ap.parse_args()

    if args.mode == "init":
        result = fetch_odds_ts_yearly(
            start_year=args.start_year,
            end_year=args.end_year,
            source_output_dir=args.source_output_dir,
            overwrite=args.overwrite,
        )
    else:
        result = fetch_odds_ts_0b41_0b42_for_main_races(
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            output_dir=args.output_dir,
        )

    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

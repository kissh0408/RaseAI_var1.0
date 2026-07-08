"""Compare strategy output for two race_num_min settings on saved pred_df (no config mutation)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.main import (  # noqa: E402
    O2_ODDS_PATH,
    O3_ODDS_PATH,
    PREDICTION_OUTPUT_PATH,
    STRATEGY_CALIBRATION_PATH,
    STRATEGY_CONFIG_PATH,
)
from main.pipeline.strategy_pipeline import run_today_strategy_pipeline  # noqa: E402
from strategy.src.betting_framework import load_today_prediction_frame  # noqa: E402


def _summarize(rec_df: pd.DataFrame, pred_races: int) -> dict:
    exec_df = rec_df[rec_df.get("is_executable", True) != False].copy()
    by_type = (
        exec_df.groupby("ticket_type")
        .agg(n=("suggested_stake", "count"), invest=("suggested_stake", "sum"))
        if not exec_df.empty
        else pd.DataFrame()
    )
    n_races = int(exec_df["race_id"].nunique()) if not exec_df.empty else 0
    both = 0
    if not exec_df.empty and "race_id" in exec_df.columns:
        both = int((exec_df.groupby("race_id")["ticket_type"].nunique() >= 2).sum())
    win_n = int(by_type.loc["単勝", "n"]) if not by_type.empty and "単勝" in by_type.index else 0
    wide_n = int(by_type.loc["ワイド", "n"]) if not by_type.empty and "ワイド" in by_type.index else 0
    invest = int(exec_df["suggested_stake"].sum()) if not exec_df.empty else 0
    return {
        "pred_races_after_filter": pred_races,
        "rec_races": n_races,
        "win_n": win_n,
        "wide_n": wide_n,
        "total_invest": invest,
        "win_wide_both_races": both,
        "by_type": by_type,
    }


def _run_with_rmin(pred_df: pd.DataFrame, rmin: int, label: str) -> tuple[pd.DataFrame, dict]:
    base = json.loads(STRATEGY_CONFIG_PATH.read_text(encoding="utf-8"))
    base["race_num_min"] = rmin
    tmp_cfg = ROOT / "model_training" / "data" / "03_train" / f"_tmp_strategy_r{rmin}.json"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    out = ROOT / "model_training" / "data" / "03_train" / f"_tmp_rec_r{rmin}.csv"
    rec = run_today_strategy_pipeline(
        pred_df,
        strategy_config_path=tmp_cfg,
        strategy_calibration_path=STRATEGY_CALIBRATION_PATH,
        recommendation_output_path=out,
        o2_odds_path=O2_ODDS_PATH,
        o3_odds_path=O3_ODDS_PATH,
    )
    from strategy.src.race_filters import filter_df_by_race_num

    filtered = filter_df_by_race_num(
        pred_df.copy(),
        race_id_col="race_id",
        race_num_min=rmin,
        race_num_max=int(base.get("race_num_max", 12)),
    )
    summary = _summarize(rec, int(filtered["race_id"].nunique()))
    summary["label"] = label
    return rec, summary


def main() -> None:
    pred_df = load_today_prediction_frame(
        PREDICTION_OUTPUT_PATH,
        parquet_path=PREDICTION_OUTPUT_PATH.with_suffix(".parquet"),
        prefer_parquet=True,
    )
    print(f"pred_df: {len(pred_df)} rows, {pred_df['race_id'].nunique()} races")
    if "month_day" in pred_df.columns:
        md = pred_df["month_day"].dropna().iloc[0]
        print(f"month_day: {md}")

    results = []
    recs = {}
    for rmin, label in [(8, "8-12R (本番)"), (1, "1-12R (試行)")]:
        rec, s = _run_with_rmin(pred_df, rmin, label)
        recs[rmin] = rec
        results.append(s)
        print(f"\n=== {label} ===")
        print(f"  対象レース数: {s['pred_races_after_filter']}")
        print(f"  推奨レース数: {s['rec_races']}")
        print(f"  単勝: {s['win_n']}件 / ワイド: {s['wide_n']}件 / 合計投資: {s['total_invest']:,}円")
        print(f"  単勝+ワイド同居: {s['win_wide_both_races']}レース")
        if not s["by_type"].empty:
            print(s["by_type"].to_string())

    only_1_12 = recs[1].copy()
    if not only_1_12.empty:
        only_1_12_rids = set(recs[1]["race_id"].astype(str))
        only_8_12_rids = set(recs[8]["race_id"].astype(str)) if not recs[8].empty else set()
        extra_rids = only_1_12_rids - only_8_12_rids
        extra = only_1_12[only_1_12["race_id"].astype(str).isin(extra_rids)]
        if not extra.empty:
            print("\n=== 1-12R のみに追加された推奨（1-7R 等） ===")
            cols = [
                c
                for c in [
                    "race_id",
                    "race_num",
                    "ticket_type",
                    "ticket",
                    "expected_value",
                    "edge",
                    "suggested_stake",
                    "phase",
                ]
                if c in extra.columns
            ]
            if "race_num" not in extra.columns and "race_id" in extra.columns:
                extra = extra.copy()
            print(extra[cols].to_string(index=False) if cols else extra.head(20).to_string())


if __name__ == "__main__":
    main()

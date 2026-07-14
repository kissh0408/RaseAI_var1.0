"""variable_sizing: build_dataset.py

scores（fold2 OOS）+ features + オッズ → `select_top1_bets` 適用済みの
ベット候補データセット（margin・tier 列付き）を betting/experiments/variable_sizing/data/
に生成する。

既存モジュールの import 再利用（コピー禁止。仕様書§2・§10-3）:
- betting.src.backtest::load_scored_odds_frame（scores+features+オッズ結合）
- betting.src.flat_top1::select_top1_bets（モデル1位馬の選定・オッズ除外）
- confidence_tiers.tiers_lib::compute_race_margin / assign_tier_batch（margin計算・
  階層割当。凍結境界の再利用。仕様書§2）

全期間を対象にビルドする（TEST行を含めてよい。仕様書§4 Stage V0手順）。
Stage V0/V1/V2 の各スクリプトが io 直後に自身の対象期間でフィルタし、他期間の行に
一切触れない（本スクリプト自体は期間フィルタしない）。

除外フィルタ（grade_code/abnormal_code/horse_count/finish_rank）は
features_v39_course_slim.parquet 生成時点で既に適用済みのため、本スクリプトでは
再適用しない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from betting.src.backtest import load_scored_odds_frame  # noqa: E402
from betting.src.flat_top1 import select_top1_bets  # noqa: E402

import sizing_lib as sl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_DIR = EXP_DIR / "data"
OUT_PATH = DATA_DIR / "bets_dataset.parquet"
LOG_PATH = DATA_DIR / "build_log.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _select_cfg(cfg: dict) -> dict:
    return {
        "min_odds": cfg["odds_filter"]["min_odds"],
        "max_odds": cfg["odds_filter"]["max_odds"],
        "loss_min": {"score_col": cfg["score_col"]},
    }


def _attach_margin_and_tier(df: pd.DataFrame, score_col: str, boundaries: list[float]) -> pd.DataFrame:
    """レースごとに compute_race_margin を適用し margin・tier 列を付与する
    （confidence_tiers の凍結境界を再利用。新たな境界導出は行わない。仕様書§2）。
    """
    margins: dict = {}
    for race_id, grp in df.groupby("race_id", sort=False):
        margins[race_id] = sl.compute_race_margin(
            grp[score_col].to_numpy(), grp["horse_num"].to_numpy()
        )
    out = df.copy()
    out["margin"] = out["race_id"].map(margins)
    out["tier"] = sl.assign_tier_batch(out["margin"].to_numpy(dtype=float), boundaries)
    return out


def build_variable_sizing_dataset() -> dict:
    cfg = _load_config()
    score_col = cfg["score_col"]
    boundaries = [
        float(cfg["boundaries"]["b1"]),
        float(cfg["boundaries"]["b2"]),
        float(cfg["boundaries"]["b3"]),
    ]

    scores_path = ROOT / cfg["data_sources"]["scores_path"]
    features_path = ROOT / cfg["data_sources"]["features_path"]

    df = load_scored_odds_frame(scores_path, features_path)
    n_rows_loaded = int(len(df))
    n_races_loaded = int(df["race_id"].nunique())
    n_missing_odds = int(pd.to_numeric(df["odds"], errors="coerce").isna().sum())
    odds_attach_success_rate = 1.0 - (n_missing_odds / n_rows_loaded if n_rows_loaded else 0.0)
    print(
        f"loaded: rows={n_rows_loaded:,}, races={n_races_loaded:,}, "
        f"odds_attach_success_rate={odds_attach_success_rate:.4%} "
        f"(missing_odds_rows={n_missing_odds:,})"
    )

    df = _attach_margin_and_tier(df, score_col, boundaries)

    select_cfg = _select_cfg(cfg)
    picks, skipped = select_top1_bets(df, cfg=select_cfg)

    skip_breakdown = skipped["reason"].value_counts().to_dict() if len(skipped) else {}
    print(f"select_top1_bets: n_bets={len(picks):,}, n_skipped={len(skipped):,}, breakdown={skip_breakdown}")

    out_cols = [
        "race_id",
        "race_date",
        "horse_num",
        score_col,
        "margin",
        "tier",
        "odds",
        "finish_rank",
    ]
    dataset = picks[out_cols].copy()
    dataset["race_date"] = pd.to_datetime(dataset["race_date"])
    dataset = dataset.sort_values(["race_date", "race_id"], kind="mergesort").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUT_PATH, index=False, compression="snappy")

    n_missing_margin = int(dataset["margin"].isna().sum())
    tier_counts = dataset["tier"].value_counts().sort_index().to_dict()

    build_log = {
        "score_col": score_col,
        "odds_filter": cfg["odds_filter"],
        "boundaries_used": {"b1": boundaries[0], "b2": boundaries[1], "b3": boundaries[2]},
        "boundaries_source": cfg["boundaries"]["source"],
        "n_rows_loaded": n_rows_loaded,
        "n_races_loaded": n_races_loaded,
        "odds_attach_success_rate": float(odds_attach_success_rate),
        "n_missing_odds_rows": n_missing_odds,
        "n_bets_selected": int(len(picks)),
        "n_races_skipped": int(len(skipped)),
        "skip_reason_breakdown": {str(k): int(v) for k, v in skip_breakdown.items()},
        "n_missing_margin_in_output": n_missing_margin,
        "tier_counts_all_periods": {str(k): int(v) for k, v in tier_counts.items()},
        "output_path": str(OUT_PATH.relative_to(ROOT)),
        "output_rows": int(len(dataset)),
        "output_races": int(dataset["race_id"].nunique()),
        "date_range": {
            "min": str(dataset["race_date"].min().date()) if len(dataset) else None,
            "max": str(dataset["race_date"].max().date()) if len(dataset) else None,
        },
    }
    LOG_PATH.write_text(json.dumps(build_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(build_log, indent=2, ensure_ascii=False))
    return build_log


if __name__ == "__main__":
    build_variable_sizing_dataset()

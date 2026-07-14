"""confidence_tiers: build_dataset.py

scores（fold2 OOS）+ features + オッズ → `select_top1_bets` 適用済みの
ベット候補データセット（margin 列付き）を betting/experiments/confidence_tiers/data/
に生成する。

既存モジュールの import 再利用（コピー禁止。仕様書§10・§11項目11）:
- betting.src.backtest::load_scored_odds_frame（scores+features+オッズ結合）
- betting.src.flat_top1::select_top1_bets（モデル1位馬の選定・オッズ除外）

全期間を対象にビルドする（TEST行を含めてよい。仕様書§8 Stage1手順）。
Stage 1/2/3 の各スクリプトが io 直後に自身の対象期間でフィルタし、他期間の行には
一切触れない（本スクリプト自体は期間フィルタしない）。

除外フィルタ（grade_code/abnormal_code/horse_count/finish_rank）は
features_v39_course_slim.parquet 生成時点で既に適用済みのため、本スクリプトでは
再適用しない（L1 特徴量生成スクリプト create_features.py 参照。本ファイルは
そのスクリプトを import も参照もしない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from betting.src.backtest import load_scored_odds_frame  # noqa: E402
from betting.src.flat_top1 import select_top1_bets  # noqa: E402

import tiers_lib as tl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_DIR = EXP_DIR / "data"
OUT_PATH = DATA_DIR / "bets_dataset.parquet"
LOG_PATH = DATA_DIR / "build_log.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _select_cfg(cfg: dict) -> dict:
    """select_top1_bets が期待する cfg 形状（min_odds/max_odds/loss_min.score_col）。"""
    return {
        "min_odds": cfg["odds_filter"]["min_odds"],
        "max_odds": cfg["odds_filter"]["max_odds"],
        "loss_min": {"score_col": cfg["score_col"]},
    }


def _attach_margin(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """レースごとに tiers_lib.compute_race_margin を適用し margin 列を付与する。"""
    margins: dict = {}
    for race_id, grp in df.groupby("race_id", sort=False):
        margins[race_id] = tl.compute_race_margin(
            grp[score_col].to_numpy(), grp["horse_num"].to_numpy()
        )
    out = df.copy()
    out["margin"] = out["race_id"].map(margins)
    return out


def _attach_favorite(df: pd.DataFrame) -> pd.DataFrame:
    """レース内の1番人気（オッズ最小、同オッズは馬番昇順で決定的タイブレーク）を
    特定し favorite_horse_num/favorite_odds/favorite_finish_rank 列を付与する。
    モデル側の選定（select_top1_bets）とは独立に、常に full race group から求める。
    """
    fav_rows = []
    for race_id, grp in df.groupby("race_id", sort=False):
        odds = pd.to_numeric(grp["odds"], errors="coerce")
        valid = grp.loc[odds.notna()].copy()
        if valid.empty:
            fav_rows.append(
                {
                    "race_id": race_id,
                    "favorite_horse_num": None,
                    "favorite_odds": None,
                    "favorite_finish_rank": None,
                }
            )
            continue
        valid["_odds"] = pd.to_numeric(valid["odds"], errors="coerce")
        valid["_horse_num"] = pd.to_numeric(valid["horse_num"], errors="coerce")
        valid = valid.sort_values(["_odds", "_horse_num"], ascending=[True, True], kind="mergesort")
        top = valid.iloc[0]
        fav_rows.append(
            {
                "race_id": race_id,
                "favorite_horse_num": int(top["_horse_num"]),
                "favorite_odds": float(top["_odds"]),
                "favorite_finish_rank": int(top["finish_rank"]),
            }
        )
    fav_df = pd.DataFrame(fav_rows)
    return df.merge(fav_df, on="race_id", how="left")


def build_confidence_tiers_dataset() -> dict:
    cfg = _load_config()
    score_col = cfg["score_col"]

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

    df = _attach_margin(df, score_col)
    df = _attach_favorite(df)

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
        "odds",
        "finish_rank",
        "favorite_horse_num",
        "favorite_odds",
        "favorite_finish_rank",
    ]
    dataset = picks[out_cols].copy()
    dataset["race_date"] = pd.to_datetime(dataset["race_date"])
    dataset = dataset.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUT_PATH, index=False, compression="snappy")

    n_missing_margin = int(dataset["margin"].isna().sum())

    build_log = {
        "score_col": score_col,
        "odds_filter": cfg["odds_filter"],
        "n_rows_loaded": n_rows_loaded,
        "n_races_loaded": n_races_loaded,
        "odds_attach_success_rate": float(odds_attach_success_rate),
        "n_missing_odds_rows": n_missing_odds,
        "n_bets_selected": int(len(picks)),
        "n_races_skipped": int(len(skipped)),
        "skip_reason_breakdown": {str(k): int(v) for k, v in skip_breakdown.items()},
        "n_missing_margin_in_output": n_missing_margin,
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
    build_confidence_tiers_dataset()

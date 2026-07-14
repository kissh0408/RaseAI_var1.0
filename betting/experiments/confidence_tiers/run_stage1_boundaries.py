"""confidence_tiers: run_stage1_boundaries.py

Stage 1（境界決定）: 2023-01-01〜2023-12-31 の「実際にベット対象となったレース」
（bets_dataset.parquet = build_dataset.py が select_top1_bets 適用済みで生成した
候補データセット）の margin の 25/50/75 パーセンタイル（numpy.quantile,
method="linear"）を境界 [b1, b2, b3] として results/stage1_boundaries.json に出力する。

**outcome-blind 厳守（仕様書§4.1・§8 Stage1手順）**: 本スクリプトは
finish_rank・odds・payout・ROI・的中率を一切読み込まず・計算せず・出力しない
（bets_dataset.parquet から race_id/race_date/margin 列のみを読み込む）。

Rule 3（期間規律）: io 直後に race_date で 2023 年のみへフィルタし、2024/2025 年の
行には一切触れない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import tiers_lib as tl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_PATH = EXP_DIR / "data" / "bets_dataset.parquet"
OUT_PATH = EXP_DIR / "results" / "stage1_boundaries.json"

STAGE1_START = "2023-01-01"
STAGE1_END = "2023-12-31"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def run_stage1_boundaries() -> dict:
    cfg = _load_config()
    period = cfg["protocol"]["stage1_boundary_period"]
    start, end = period["start"], period["end"]
    assert start == STAGE1_START and end == STAGE1_END, "config.json の Stage1期間と定数が不一致"

    # outcome-blind 担保: race_id/race_date/margin のみを読み込む
    # （finish_rank/odds/favorite_* 列は本ステージでは一切読まない）。
    df = pd.read_parquet(DATA_PATH, columns=["race_id", "race_date", "margin"])

    dates = pd.to_datetime(df["race_date"])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    df_2023 = df.loc[mask].copy()

    n_2023 = int(len(df_2023))
    n_races_2023 = int(df_2023["race_id"].nunique())
    margins = df_2023["margin"].dropna().to_numpy(dtype=float)

    b1, b2, b3 = tl.compute_quartile_boundaries(margins)

    result = {
        "disclaimer": (
            "本測定は境界決定(margin四分位算出)のみを行うoutcome-blindな統計処理であり、"
            "着順・払戻・ROI・的中率は本ステージで一切計算・出力していない。"
            "黒字化を示唆・保証するものではない。"
        ),
        "protocol": {
            "stage": "stage1_boundary_decision",
            "period": f"{start}..{end}",
            "outcome_blind": True,
            "outcome_blind_note": (
                "本スクリプトは finish_rank/odds/payout/ROI/的中率を一切読み込まず・"
                "計算せず・出力しない（読み込み列は race_id/race_date/margin のみ）。"
            ),
            "source_dataset": str(DATA_PATH.relative_to(ROOT)),
            "quantile_method": "linear",
            "score_col": cfg["score_col"],
        },
        "n_2023": n_2023,
        "n_races_2023": n_races_2023,
        "n_margin_values": int(len(margins)),
        "boundaries": {"b1": b1, "b2": b2, "b3": b3},
        "margin_distribution": {
            "min": float(margins.min()) if len(margins) else None,
            "max": float(margins.max()) if len(margins) else None,
            "mean": float(margins.mean()) if len(margins) else None,
            "std": float(margins.std()) if len(margins) else None,
        },
        "caveat": (
            "fold2 OOSスコアの2023年はearly-stoppingに使われた弱汚染年"
            "（evaluation/reports/fusion_oos_fold2.json の protocol.caveat 参照）。"
            "ただし本ステージで得るのはmargin分布の四分位のみ（outcome-blind）であり、"
            "汚染がもたらしうるのはスコア分布のわずかな楽観化に留まる。判定は2024/2025の"
            "完全OOS年（Stage2/Stage3）で行うため、判定へのリークにはならない"
            "（仕様書§4.2）。"
        ),
        "next_step": (
            "本結果の [b1,b2,b3] と n_2023 を "
            "docs/specs/2026-07-11-confidence-tiers-spec.md §13 に追記し凍結してから "
            "Stage 2（run_stage2_valid.py）に進む。2024/2025年の結果は本ステージでは"
            "一切参照していない。"
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_stage1_boundaries()

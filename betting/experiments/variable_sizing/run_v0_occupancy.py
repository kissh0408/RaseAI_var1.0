"""variable_sizing: run_v0_occupancy.py

Stage V0（占有率・リスク予算保存則検証。VALID = 2024-01-01〜2024-12-31 のみ）:
凍結境界で割当済みの tier 列から階層占有率 w_t を算出し、占有率加重平均倍率
M̄ = Σ w_t*m_t が [0.95, 1.05] に入るか（仕様書§3.2-1・§6.1 budget_preserved）を
検証し results/occupancy_valid.json に出力する。

**outcome-blind 厳守（仕様書§4 Stage V0手順・§9-3）**: 本スクリプトは
finish_rank・odds・payout・ROI・的中率を一切読み込まず・計算せず・出力しない
（bets_dataset.parquet から race_id/race_date/tier 列のみを読み込む）。

Rule 3（期間規律）: io 直後に race_date で 2024 年のみへフィルタし、2023/2025+ の
行には一切触れない。保存則不成立の場合はここで停止し planner へ差し戻す
（倍率は調整しない。仕様書§3.2-1）。
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

import sizing_lib as sl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_PATH = EXP_DIR / "data" / "bets_dataset.parquet"
OUT_PATH = EXP_DIR / "results" / "occupancy_valid.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def run_v0_occupancy() -> dict:
    cfg = _load_config()
    period = cfg["protocol"]["valid_period"]
    start, end = period["start"], period["end"]
    multipliers = [cfg["multipliers"]["m1"], cfg["multipliers"]["m2"], cfg["multipliers"]["m3"], cfg["multipliers"]["m4"]]
    tol_low = float(cfg["budget_preservation"]["tolerance_low"])
    tol_high = float(cfg["budget_preservation"]["tolerance_high"])

    # outcome-blind 担保: race_id/race_date/tier のみを読み込む
    # （finish_rank/odds/margin 列は本ステージでは一切読まない）。
    df = pd.read_parquet(DATA_PATH, columns=["race_id", "race_date", "tier"])

    dates = pd.to_datetime(df["race_date"])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    valid_df = df.loc[mask].copy()

    n_bets_valid = int(len(valid_df))
    n_races_valid = int(valid_df["race_id"].nunique())

    occupancy = sl.compute_tier_occupancy(valid_df["tier"].to_numpy(dtype=int))
    m_bar = sl.weighted_mean_multiplier(occupancy, multipliers)
    preserved = sl.budget_preserved(m_bar, tol_low=tol_low, tol_high=tol_high)

    tier_counts = valid_df["tier"].value_counts().sort_index().to_dict()

    payload = {
        "protocol": {
            "stage": "stage_v0_occupancy",
            "period": f"{start}..{end}",
            "outcome_blind": True,
            "outcome_blind_note": (
                "本スクリプトは finish_rank/odds/payout/ROI/的中率を一切読み込まず・"
                "計算せず・出力しない（読み込み列は race_id/race_date/tier のみ）。"
            ),
            "source_dataset": str(DATA_PATH.relative_to(ROOT)),
            "multipliers": {"m1": multipliers[0], "m2": multipliers[1], "m3": multipliers[2], "m4": multipliers[3]},
            "boundaries_source": cfg["boundaries"]["source"],
        },
        "n_bets_valid": n_bets_valid,
        "n_races_valid": n_races_valid,
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "tier_occupancy": {str(k): v for k, v in occupancy.items()},
        "weighted_mean_multiplier": m_bar,
        "budget_tolerance": [tol_low, tol_high],
        "budget_preserved": preserved,
        "next_step": (
            "budget_preserved=true の場合のみ Stage V1（run_v1_risk_valid.py）へ進む。"
            "false の場合は倍率を調整せず planner へ差し戻す（仕様書§3.2-1）。"
        ),
        "caveats": [
            "本測定は階層占有率のみを扱うoutcome-blindな統計処理であり、着順・払戻・ROI・"
            "的中率は本ステージで一切計算・出力していない。",
        ],
    }
    result = sl.build_result_envelope(payload)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(sl.DISCLAIMER)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not preserved:
        print(
            "STOP: budget_preserved=False. Per spec §3.2-1, multipliers must NOT be "
            "adjusted to force this to pass. Escalate to planner."
        )
    return result


if __name__ == "__main__":
    run_v0_occupancy()

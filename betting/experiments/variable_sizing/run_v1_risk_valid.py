"""variable_sizing: run_v1_risk_valid.py

Stage V1（リスク再検証と f_var の機械的導出。VALID = 2024-01-01〜2024-12-31 のみ）:
可変stake系列（§3の倍率適用済み）に対し derive_flat_fraction.py と同一セマンティクスの
月次MDD・最繁忙日エクスポージャ分析を実行し、決定規則（v2の可変版。仕様書§4 Stage V1-3）
を機械適用して f_var を導出する。

グリッド {0.001, 0.0005, 0.00025} の各候補 f について、base_stake が常に厳密400円と
なるよう bankroll_f = min_bankroll_variable(f) = 4*100/f を用いる（f=0.001なら
400,000円、f=0.0005なら800,000円）。これにより全候補で base_stake=400円固定・
月次DDはfに比例するという線形スケーリング前提が保たれる（仕様書§3.3）。

flat比較系列（同一bankroll_f・base_stake=400円固定・倍率=1固定）も同時に算出する
（仕様書§4 Stage V1-5「flat（同一bankroll・base 400円）との月次DD比較」）。

Rule 3（期間規律）: io 直後に race_date で 2024 年のみへフィルタし、2025+ の行には
一切触れない。決定規則で導出した f_var を config.json に凍結追記してから Stage V2 へ
進む（この時点まで TEST(2025+) を一切読まない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from betting.src.flat_top1 import settle_win_bets  # noqa: E402

import sizing_lib as sl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_PATH = EXP_DIR / "data" / "bets_dataset.parquet"
OUT_PATH = EXP_DIR / "results" / "risk_valid.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _grid_entry(df: pd.DataFrame, f: float, multipliers: list[float]) -> dict[str, Any]:
    bankroll_f = sl.min_bankroll_variable(f)
    base_stake = sl.compute_base_stake(bankroll_f, f)  # always 400.0 by construction

    # Variable series
    var_sized = sl.apply_variable_stake(df, base_stake=base_stake, multipliers=multipliers)
    var_settled = settle_win_bets(var_sized)
    var_worst, var_by_month = sl.monthly_max_drawdown(var_settled["race_date"], var_settled["pnl"], bankroll_f)
    # Busiest-day exposure for the variable series uses per-bet stake (Σstake(day)/bankroll),
    # not the flat f-only approximation that sl.busiest_day_exposure computes for a constant
    # stake_fraction (spec §4 Stage V1-4: "busiest_day_exposure@f_var の可変stake版").
    day = pd.to_datetime(var_settled["race_date"]).dt.date
    stake_by_day = var_settled.assign(_day=day).groupby("_day")["stake"].sum()
    var_busiest_day_exact = str(stake_by_day.idxmax()) if not stake_by_day.empty else ""
    var_busiest_exposure_exact = float(stake_by_day.max() / bankroll_f) if not stake_by_day.empty else 0.0

    # Flat comparison series: same bankroll_f, same base_stake=400, multiplier=1 fixed.
    flat_sized = df.copy()
    flat_sized["stake"] = base_stake
    flat_settled = settle_win_bets(flat_sized)
    flat_worst, flat_by_month = sl.monthly_max_drawdown(flat_settled["race_date"], flat_settled["pnl"], bankroll_f)

    return {
        "f": f,
        "bankroll": bankroll_f,
        "base_stake": base_stake,
        "variable": {
            "worst_month_dd": var_worst,
            "monthly_dd_by_month": var_by_month,
            "busiest_day": var_busiest_day_exact,
            "busiest_day_exposure": var_busiest_exposure_exact,
        },
        "flat_same_bankroll": {
            "worst_month_dd": flat_worst,
            "monthly_dd_by_month": flat_by_month,
        },
    }


def run_v1_risk_valid() -> dict:
    cfg = _load_config()
    period = cfg["protocol"]["valid_period"]
    start, end = period["start"], period["end"]
    multipliers = [
        cfg["multipliers"]["m1"],
        cfg["multipliers"]["m2"],
        cfg["multipliers"]["m3"],
        cfg["multipliers"]["m4"],
    ]
    grid = list(cfg["f_var"]["grid"])
    f0 = float(cfg["f_var"]["f0_for_scaling"])
    monthly_mdd_limit = float(cfg["f_var"]["monthly_mdd_limit"])
    safety_factor = float(cfg["f_var"]["safety_factor_k"])
    headroom_limit = float(cfg["busiest_day_exposure_headroom_limit"])

    df = pd.read_parquet(DATA_PATH)
    dates = pd.to_datetime(df["race_date"])
    valid_df = df.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()

    grid_results = {f: _grid_entry(valid_df, f, multipliers) for f in grid}

    worst_month_dd_at_f0 = grid_results[f0]["variable"]["worst_month_dd"]
    derivation = sl.derive_f_var(
        worst_month_dd_at_f0, f0, grid=grid, monthly_mdd_limit=monthly_mdd_limit, safety_factor=safety_factor
    )
    adopted_f_var = derivation["adopted_f_var"]

    # Busiest-day exposure headroom check at the adopted f_var; step down (grid is
    # already sorted descending by construction: [0.001, 0.0005, 0.00025]) if it fails.
    exposure_check_log = []
    candidates_desc = sorted([f for f in grid if f <= derivation["f_capped"]], reverse=True)
    final_f_var = None
    for f in candidates_desc:
        exp_val = grid_results[f]["variable"]["busiest_day_exposure"]
        ok = exp_val <= headroom_limit
        exposure_check_log.append({"f": f, "busiest_day_exposure": exp_val, "headroom_ok": ok})
        if ok:
            final_f_var = f
            break

    danger_flags = {
        f: {
            "danger_roi_gt_100_reference": None,  # ROI not computed in V1 (risk-only stage); see V2
        }
        for f in grid
    }

    payload = {
        "protocol": {
            "stage": "stage_v1_risk_valid",
            "period": f"{start}..{end}",
            "grid": grid,
            "f0_for_scaling": f0,
            "monthly_mdd_limit": monthly_mdd_limit,
            "safety_factor_k": safety_factor,
            "busiest_day_exposure_headroom_limit": headroom_limit,
            "multipliers": multipliers,
            "decision_rule": (
                "v2 variable-sizing version: f_scale = monthly_mdd_limit / "
                f"(worst_month_dd_var@f0 / f0), f0={f0}; f_capped = {safety_factor} * f_scale; "
                f"adopt largest value in grid {grid} (downward-extension only) with f <= f_capped, "
                f"then step down within eligible candidates until busiest_day_exposure <= {headroom_limit}. "
                "base_stake is held exactly at 400 yen for every grid candidate via "
                "bankroll_f = min_bankroll_variable(f) = 4*stake_rounding_yen/f (spec §3.3), so "
                "worst_month_dd scales linearly in f by construction."
            ),
        },
        "worst_month_dd_var_at_f0": worst_month_dd_at_f0,
        "f_scale": derivation["f_scale"],
        "f_capped": derivation["f_capped"],
        "grid_results": grid_results,
        "exposure_check_log": exposure_check_log,
        "adopted_f_var_before_exposure_check": adopted_f_var,
        "adopted_f_var": final_f_var,
        "adopted_bankroll": sl.min_bankroll_variable(final_f_var) if final_f_var is not None else None,
        "danger_flags_reference": danger_flags,
        "caveats": [
            "本ステージはリスク（月次MDD・最繁忙日エクスポージャ）の測定・機械的導出のみを"
            "行い、着順・payoutは決済（win/pnl）計算にのみ使用する。倍率・f_varの選択に"
            "ROI・的中率は使用していない（outcome-blind設計。仕様書§0.5）。",
            "flat_same_bankroll系列との月次DD比較は記述的な比較であり、可変系列のROI優劣を"
            "意味しない。",
        ],
    }
    result = sl.build_result_envelope(payload)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(sl.DISCLAIMER)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if final_f_var is None:
        print(
            "STOP: no grid candidate satisfies both f_capped and busiest_day_exposure "
            "headroom. Per spec, do not extend the grid upward or relax thresholds; "
            "escalate to planner."
        )
        return result

    # --- freeze adopted f_var into config.json (spec §4 Stage V1-6) ---
    cfg["f_var"]["status"] = "frozen_2026-07-11"
    cfg["f_var"]["f_scale"] = derivation["f_scale"]
    cfg["f_var"]["f_capped"] = derivation["f_capped"]
    cfg["f_var"]["adopted_f_var"] = final_f_var
    cfg["f_var"]["adopted_bankroll"] = sl.min_bankroll_variable(final_f_var)
    cfg["f_var"]["worst_month_dd_var_at_f0"] = worst_month_dd_at_f0
    cfg["f_var"]["frozen_note"] = (
        "run_v1_risk_valid.py により VALID(2024)実測から機械的に導出・凍結（下方拡張のみ）。"
        "以後変更禁止。TEST(2025+)は本導出に一切使用していない。"
        f"出典: {OUT_PATH.relative_to(ROOT)}"
    )
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"config.json frozen: f_var.adopted_f_var={final_f_var}")

    return result


if __name__ == "__main__":
    run_v1_risk_valid()

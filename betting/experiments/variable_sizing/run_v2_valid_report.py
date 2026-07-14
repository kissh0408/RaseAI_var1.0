"""variable_sizing: run_v2_valid_report.py

Stage V2（VALID記述的報告。判定に使わない参考測定。仕様書§4 Stage V2）:
凍結済み f_var・bankroll・倍率で可変系列とflat系列（同一bankroll・base_stake=400円
固定・倍率=1固定）の ROI・的中率・階層別stake内訳を VALID(2024)で記述的に算出する。

**倍率・f_var の再調整には一切使わない**（Stage V1で凍結済み。使えば仕様書§0.5違反）。
ROI差はいかなる符号・大きさでも性能の優劣を示す記述は行わない（roi_note は中立文言
テンプレート固定。sizing_lib.ROI_NOTE_TEMPLATE）。

Rule 3（期間規律）: io 直後に race_date で 2024 年のみへフィルタし、2025+ の行には
一切触れない。
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

from betting.src.flat_top1 import settle_win_bets  # noqa: E402

import sizing_lib as sl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_PATH = EXP_DIR / "data" / "bets_dataset.parquet"
OUT_PATH = EXP_DIR / "results" / "valid_report.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _tier_breakdown(settled: pd.DataFrame) -> list[dict]:
    rows = []
    for t in (1, 2, 3, 4):
        sub = settled.loc[settled["tier"] == t]
        n = int(len(sub))
        stake_sum = float(sub["stake"].sum()) if n else 0.0
        payout_sum = float(sub["payout"].sum()) if n else 0.0
        roi = (payout_sum / stake_sum) if stake_sum > 0 else float("nan")
        hit_rate = float(sub["win"].mean()) if n else float("nan")
        rows.append(
            {
                "tier": t,
                "n_bets": n,
                "stake_sum": stake_sum,
                "payout_sum": payout_sum,
                "roi_pct": roi * 100.0 if np.isfinite(roi) else None,
                "hit_rate": hit_rate,
                "n_hits": int(sub["win"].sum()) if n else 0,
            }
        )
    return rows


def run_v2_valid_report() -> dict:
    cfg = _load_config()
    assert str(cfg["f_var"]["status"]).startswith("frozen"), (
        "f_var が凍結されていません（Stage V1 → run_v1_risk_valid.py を先に完了すること）"
    )
    period = cfg["protocol"]["valid_period"]
    start, end = period["start"], period["end"]
    multipliers = [
        cfg["multipliers"]["m1"],
        cfg["multipliers"]["m2"],
        cfg["multipliers"]["m3"],
        cfg["multipliers"]["m4"],
    ]
    f_var = float(cfg["f_var"]["adopted_f_var"])
    bankroll = float(cfg["f_var"]["adopted_bankroll"])
    base_stake = sl.compute_base_stake(bankroll, f_var)

    df = pd.read_parquet(DATA_PATH)
    dates = pd.to_datetime(df["race_date"])
    valid_df = df.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()

    # --- variable series ---
    var_sized = sl.apply_variable_stake(valid_df, base_stake=base_stake, multipliers=multipliers)
    var_settled = settle_win_bets(var_sized)
    n_var = int(len(var_settled))
    var_stake_total = float(var_settled["stake"].sum())
    var_payout_total = float(var_settled["payout"].sum())
    var_roi_pct = var_payout_total / var_stake_total * 100.0 if var_stake_total > 0 else None
    var_hit_rate = float(var_settled["win"].mean()) if n_var else None

    # --- flat comparison series: same bankroll, base_stake=400 fixed, multiplier=1 fixed ---
    flat_sized = valid_df.copy()
    flat_sized["stake"] = base_stake
    flat_settled = settle_win_bets(flat_sized)
    n_flat = int(len(flat_settled))
    flat_stake_total = float(flat_settled["stake"].sum())
    flat_payout_total = float(flat_settled["payout"].sum())
    flat_roi_pct = flat_payout_total / flat_stake_total * 100.0 if flat_stake_total > 0 else None
    flat_hit_rate = float(flat_settled["win"].mean()) if n_flat else None

    n_bets_match = n_var == n_flat
    race_set_match = set(var_settled["race_id"]) == set(flat_settled["race_id"])
    win_flags_match = list(var_settled["win"]) == list(flat_settled["win"])

    roi_diff_pp = (var_roi_pct - flat_roi_pct) if (var_roi_pct is not None and flat_roi_pct is not None) else None

    # --- danger signals (§7) ---
    danger_var_roi = sl.danger_roi_gt_100((var_roi_pct or 0.0) / 100.0)
    danger_flat_roi = sl.danger_roi_gt_100((flat_roi_pct or 0.0) / 100.0)
    tier_breakdown = _tier_breakdown(var_settled)
    danger_leak_by_tier = {
        int(row["tier"]): sl.leak_review_flag(row["hit_rate"]) for row in tier_breakdown if row["hit_rate"] is not None
    }
    any_danger = bool(danger_var_roi or danger_flat_roi or any(danger_leak_by_tier.values()))

    # --- effective multiplier detector (spec §7-4 / §9-5) ---
    eff = sl.effective_multiplier(var_settled["stake"], base_stake)
    design_multipliers = np.asarray([multipliers[t - 1] for t in var_settled["tier"]])
    effective_multiplier_matches_design = bool(np.allclose(eff, design_multipliers, atol=1e-9))

    payload = {
        "protocol": {
            "stage": "stage_v2_valid_report",
            "period": f"{start}..{end}",
            "purpose": "記述的報告のみ。倍率・f_varの再調整には使用しない（仕様書§0.5・§4 Stage V2）。",
            "f_var": f_var,
            "bankroll": bankroll,
            "base_stake": base_stake,
            "multipliers": multipliers,
        },
        "variable_series": {
            "n_bets": n_var,
            "stake_total": var_stake_total,
            "payout_total": var_payout_total,
            "roi_pct": var_roi_pct,
            "hit_rate": var_hit_rate,
        },
        "flat_series_same_bankroll": {
            "n_bets": n_flat,
            "stake_total": flat_stake_total,
            "payout_total": flat_payout_total,
            "roi_pct": flat_roi_pct,
            "hit_rate": flat_hit_rate,
        },
        "roi_diff_pp_variable_minus_flat": roi_diff_pp,
        "roi_note": sl.ROI_NOTE_TEMPLATE,
        "n_bets_match": n_bets_match,
        "race_set_match": race_set_match,
        "win_flags_match": win_flags_match,
        "tier_breakdown_variable_series": tier_breakdown,
        "effective_multiplier_matches_design": effective_multiplier_matches_design,
        "danger_signals": {
            "danger_roi_gt_100_variable": danger_var_roi,
            "danger_roi_gt_100_flat": danger_flat_roi,
            "leak_review_required_by_tier": danger_leak_by_tier,
            "any_danger": any_danger,
        },
        "caveats": [
            "本測定はVALID(2024)の記述的報告であり、判定には使用しない（仕様書§6.2）。"
            "ROI差はいかなる符号・大きさでも性能の優劣を意味しない。",
            "flat_series_same_bankrollは倍率=1固定・base_stake=400円固定の比較系列であり、"
            "本番の flat_top1 運用（bankroll=100,000, stake_fraction=0.001）とは規模が"
            "異なる（仕様書§3.3: stake規模を揃えるための同一bankroll比較）。",
        ],
    }
    result = sl.build_result_envelope(payload)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(sl.DISCLAIMER)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not (n_bets_match and race_set_match and win_flags_match):
        print(
            "STOP: variable and flat series diverge in bet set (n_bets/race_set/win_flags). "
            "Per spec §7-3, this indicates a selection-logic bug; escalate to evaluator."
        )
    if not effective_multiplier_matches_design:
        print(
            "STOP: effective stake/base_stake ratio does not exactly match the design "
            "multipliers. Per spec §7-4, this indicates a rounding-degradation bug."
        )
    if any_danger:
        print("WARNING: danger signal(s) flagged; see danger_signals in the output JSON.")

    return result


if __name__ == "__main__":
    run_v2_valid_report()

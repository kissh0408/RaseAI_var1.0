"""variable_sizing: run_v3_test.py

Stage V3（TEST(2025+)確認。1回のみ実行。仕様書§4 Stage V3）:
凍結済み f_var・bankroll・倍率をそのまま TEST 期間へ適用し、以下を算出する。

1. リスク指標（合否の確認対象はこれのみ）:
   - worst月次DD（可変系列）≤ monthly_mdd_limit = 0.15
   - 最繁忙日エクスポージャ ≤ max_daily_exposure = 0.25
   - flat系列（同一bankroll）とのDD特性比較
2. 再現性アンカー: 100円均等flat換算の合算 n_bets・ROI が
   evaluation/reports/betting_backtest_oos_flat.json（n=3,758、ROI_model 83.34752527940394%）
   と一致（±0.1pp / nは完全一致）。
3. 記述的報告（判定・優位性主張に使わない。事前登録）: 可変系列ROI・flat系列ROI・
   その差、階層別内訳。

単一実行ガード: results/test_risk.json が既に存在する場合は例外を送出し、バグ修正に
よる再実行のみレポートに理由明記のうえ許可する（仕様書§4 Stage V3）。
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

from betting.src.flat_top1 import settle_win_bets  # noqa: E402

import sizing_lib as sl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_PATH = EXP_DIR / "data" / "bets_dataset.parquet"
OUT_PATH = EXP_DIR / "results" / "test_risk.json"
ANCHOR_PATH = ROOT / "evaluation" / "reports" / "betting_backtest_oos_flat.json"


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
                "roi_pct": roi * 100.0 if pd.notna(roi) else None,
                "hit_rate": hit_rate,
                "n_hits": int(sub["win"].sum()) if n else 0,
            }
        )
    return rows


def run_v3_test() -> dict:
    if OUT_PATH.exists():
        raise RuntimeError(
            f"{OUT_PATH} already exists. Stage V3 is a one-time TEST execution (spec §4 Stage V3); "
            "re-running requires an explicit bug-fix rationale documented in the spec, not a silent "
            "overwrite. Remove the file only after recording the reason."
        )

    cfg = _load_config()
    assert str(cfg["f_var"]["status"]).startswith("frozen"), (
        "f_var が凍結されていません（Stage V1 → run_v1_risk_valid.py を先に完了すること）"
    )
    period = cfg["protocol"]["test_period"]
    start = period["start"]
    multipliers = [
        cfg["multipliers"]["m1"],
        cfg["multipliers"]["m2"],
        cfg["multipliers"]["m3"],
        cfg["multipliers"]["m4"],
    ]
    f_var = float(cfg["f_var"]["adopted_f_var"])
    bankroll = float(cfg["f_var"]["adopted_bankroll"])
    base_stake = sl.compute_base_stake(bankroll, f_var)
    monthly_mdd_limit = float(cfg["f_var"]["monthly_mdd_limit"])
    max_daily_exposure = float(cfg["max_daily_exposure"])

    df = pd.read_parquet(DATA_PATH)
    dates = pd.to_datetime(df["race_date"])
    test_df = df.loc[dates >= pd.Timestamp(start)].copy()

    # --- variable series ---
    var_sized = sl.apply_variable_stake(test_df, base_stake=base_stake, multipliers=multipliers)
    var_settled = settle_win_bets(var_sized)
    n_var = int(len(var_settled))
    var_stake_total = float(var_settled["stake"].sum())
    var_payout_total = float(var_settled["payout"].sum())
    var_roi_pct = var_payout_total / var_stake_total * 100.0 if var_stake_total > 0 else None
    var_hit_rate = float(var_settled["win"].mean()) if n_var else None

    var_worst_dd, var_by_month = sl.monthly_max_drawdown(
        var_settled["race_date"], var_settled["pnl"], bankroll
    )
    day = pd.to_datetime(var_settled["race_date"]).dt.date
    stake_by_day = var_settled.assign(_day=day).groupby("_day")["stake"].sum()
    var_busiest_day = str(stake_by_day.idxmax()) if not stake_by_day.empty else ""
    var_busiest_exposure = float(stake_by_day.max() / bankroll) if not stake_by_day.empty else 0.0

    # --- flat comparison series: same bankroll, base_stake=400 fixed, multiplier=1 fixed ---
    flat_sized = test_df.copy()
    flat_sized["stake"] = base_stake
    flat_settled = settle_win_bets(flat_sized)
    n_flat = int(len(flat_settled))
    flat_stake_total = float(flat_settled["stake"].sum())
    flat_payout_total = float(flat_settled["payout"].sum())
    flat_roi_pct = flat_payout_total / flat_stake_total * 100.0 if flat_stake_total > 0 else None
    flat_hit_rate = float(flat_settled["win"].mean()) if n_flat else None

    flat_worst_dd, flat_by_month = sl.monthly_max_drawdown(
        flat_settled["race_date"], flat_settled["pnl"], bankroll
    )

    n_bets_match = n_var == n_flat
    race_set_match = set(var_settled["race_id"]) == set(flat_settled["race_id"])
    win_flags_match = list(var_settled["win"]) == list(flat_settled["win"])

    roi_diff_pp = (
        (var_roi_pct - flat_roi_pct) if (var_roi_pct is not None and flat_roi_pct is not None) else None
    )

    # --- risk gates (§4 Stage V3-1; the only pass/fail criteria) ---
    worst_month_dd_ok = bool(var_worst_dd <= monthly_mdd_limit)
    busiest_day_exposure_ok = bool(var_busiest_exposure <= max_daily_exposure)

    # --- reproduction anchor (§4 Stage V3-2): 100-yen flat-equivalent aggregate ---
    anchor_flat100_sized = test_df.copy()
    anchor_flat100_sized["stake"] = 100.0
    anchor_flat100_settled = settle_win_bets(anchor_flat100_sized)
    anchor_n_bets = int(len(anchor_flat100_settled))
    anchor_stake_total = float(anchor_flat100_settled["stake"].sum())
    anchor_payout_total = float(anchor_flat100_settled["payout"].sum())
    anchor_roi_pct = (
        anchor_payout_total / anchor_stake_total * 100.0 if anchor_stake_total > 0 else None
    )

    anchor_ref = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    anchor_ref_n_bets = int(anchor_ref["production"]["n_bets"])
    anchor_ref_roi_pct = float(anchor_ref["production"]["roi_pct"])
    reproduction_n_match = anchor_n_bets == anchor_ref_n_bets
    reproduction_roi_match = (
        anchor_roi_pct is not None and abs(anchor_roi_pct - anchor_ref_roi_pct) <= 0.1
    )
    reproduction_ok = bool(reproduction_n_match and reproduction_roi_match)

    # --- danger signals (§7) ---
    danger_var_roi = sl.danger_roi_gt_100((var_roi_pct or 0.0) / 100.0)
    danger_flat_roi = sl.danger_roi_gt_100((flat_roi_pct or 0.0) / 100.0)
    tier_breakdown = _tier_breakdown(var_settled)
    danger_leak_by_tier = {
        int(row["tier"]): sl.leak_review_flag(row["hit_rate"])
        for row in tier_breakdown
        if row["hit_rate"] is not None
    }
    any_danger = bool(danger_var_roi or danger_flat_roi or any(danger_leak_by_tier.values()))

    # --- effective multiplier detector (spec §7-4 / §9-5) ---
    import numpy as np

    eff = sl.effective_multiplier(var_settled["stake"], base_stake)
    design_multipliers = np.asarray([multipliers[t - 1] for t in var_settled["tier"]])
    effective_multiplier_matches_design = bool(np.allclose(eff, design_multipliers, atol=1e-9))

    payload = {
        "protocol": {
            "stage": "stage_v3_test",
            "period": f"{start}..",
            "purpose": (
                "リスク指標の確認（唯一の合否対象）と再現性アンカー照合。ROI差は記述的報告のみで、"
                "判定・優位性主張には使わない（仕様書§0.5・§4 Stage V3・§6.2）。1回のみ実行。"
            ),
            "f_var": f_var,
            "bankroll": bankroll,
            "base_stake": base_stake,
            "multipliers": multipliers,
            "monthly_mdd_limit": monthly_mdd_limit,
            "max_daily_exposure": max_daily_exposure,
        },
        "risk_metrics": {
            "variable_series": {
                "worst_month_dd": var_worst_dd,
                "monthly_dd_by_month": var_by_month,
                "busiest_day": var_busiest_day,
                "busiest_day_exposure": var_busiest_exposure,
            },
            "flat_series_same_bankroll": {
                "worst_month_dd": flat_worst_dd,
                "monthly_dd_by_month": flat_by_month,
            },
            "worst_month_dd_ok": worst_month_dd_ok,
            "busiest_day_exposure_ok": busiest_day_exposure_ok,
            "risk_gates_pass": bool(worst_month_dd_ok and busiest_day_exposure_ok),
        },
        "reproduction_anchor": {
            "measured": {
                "n_bets": anchor_n_bets,
                "stake_total": anchor_stake_total,
                "payout_total": anchor_payout_total,
                "roi_pct": anchor_roi_pct,
            },
            "reference": {
                "source": str(ANCHOR_PATH.relative_to(ROOT)),
                "n_bets": anchor_ref_n_bets,
                "roi_pct": anchor_ref_roi_pct,
            },
            "tolerance_pp": 0.1,
            "tolerance_n": 0,
            "n_match": reproduction_n_match,
            "roi_match_within_tolerance": reproduction_roi_match,
            "reproduction_ok": reproduction_ok,
        },
        "descriptive_report": {
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
            "tier_breakdown_variable_series": tier_breakdown,
        },
        "n_bets_match": n_bets_match,
        "race_set_match": race_set_match,
        "win_flags_match": win_flags_match,
        "effective_multiplier_matches_design": effective_multiplier_matches_design,
        "danger_signals": {
            "danger_roi_gt_100_variable": danger_var_roi,
            "danger_roi_gt_100_flat": danger_flat_roi,
            "leak_review_required_by_tier": danger_leak_by_tier,
            "any_danger": any_danger,
        },
        "caveats": [
            "本測定は合否をリスク指標（worst月次DD・最繁忙日エクスポージャ）のみで判定する"
            "（仕様書§4 Stage V3-1）。ROI差はいかなる符号・大きさでも性能の優劣を意味しない"
            "（descriptive_report は事前登録された参考測定）。",
            "flat_series_same_bankrollは倍率=1固定・base_stake=400円固定の比較系列であり、"
            "本番の flat_top1 運用（bankroll=100,000, stake_fraction=0.001）とは規模が"
            "異なる（仕様書§3.3: stake規模を揃えるための同一bankroll比較）。",
            "先行検証（confidence-tiers §15、verdict=confidence_does_not_predict_market_edge）で "
            "margin（自信度）は対1番人気ROI優位を予測しないことが確定している。本Stageの目的は"
            "リスク配分の設計選好の妥当性確認であり、性能改善の検証ではない。",
        ],
    }
    result = sl.build_result_envelope(payload)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(sl.DISCLAIMER)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not reproduction_ok:
        print(
            "STOP: reproduction anchor mismatch vs evaluation/reports/betting_backtest_oos_flat.json. "
            "Per spec §4 Stage V3-2, this indicates a selection/join bug; escalate to evaluator."
        )
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
    if not (worst_month_dd_ok and busiest_day_exposure_ok):
        print(
            "WARNING: risk gate(s) failed (worst_month_dd_ok="
            f"{worst_month_dd_ok}, busiest_day_exposure_ok={busiest_day_exposure_ok}). "
            "See risk_metrics in the output JSON."
        )
    if any_danger:
        print("WARNING: danger signal(s) flagged; see danger_signals in the output JSON.")

    return result


if __name__ == "__main__":
    run_v3_test()

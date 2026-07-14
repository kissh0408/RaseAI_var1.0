"""VALID (2024) 専用: 固定比率 f の凍結スクリプト（Rule 3）。

docs/specs/2026-07-10-loss-minimization-implementation-spec.md §1.2/§2 の
決定規則 v2（R2改訂）を機械的に適用する:

    1. f_scale = monthly_mdd_limit / (worst_month_dd@f0 / f0)
       （f0 = 0.0025 実測ベース。ドローダウンは f に厳密に線形という
       VALID実測の性質を利用した線形スケーリング）
    2. 安全係数 k = 0.5 を掛ける: f_capped = k * f_scale
       （f_scale をそのまま採用するとヘッドルームゼロになるため禁止。
       k はfを小さくする方向にのみ働く固定値で、VALID結果を見て緩める余地はない）
    3. グリッドを下方にのみ拡張した {0.001, 0.0025, 0.005, 0.01} のうち
       f_capped 以下となる最大値を採用する（上方拡張は禁止）。
    4. 併せて採用 f での busiest_day_exposure <= 0.5 * max_daily_exposure を確認する。

この規則はVALID実測（f0=0.0025 の worst_month_dd=0.1596）から機械的に
f=0.001 を導出することが事前に期待されている（R2の設計時点の計算）。
実行結果が一致しない場合はスクリプト内でグリッド・係数を変更せず、
plannerへ差し戻すこと。

TEST期間（2025-01-01以降）のデータはこのスクリプト内では一切参照しない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting.src.backtest import load_betting_config, load_scored_odds_frame
from betting.src.flat_top1 import DISCLAIMER, apply_flat_sizing, select_top1_bets, settle_win_bets

VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
# R2: グリッドを下方にのみ拡張。0.0025 は f_scale 導出の基準点として維持する。
STAKE_FRACTION_GRID = [0.001, 0.0025, 0.005, 0.01]
BASE_F_FOR_SCALING = 0.0025
SAFETY_FACTOR = 0.5
MONTHLY_MDD_LIMIT = 0.15
MAX_DAILY_EXPOSURE_LIMIT = 0.25
BUSIEST_DAY_EXPOSURE_HEADROOM_LIMIT = 0.5 * MAX_DAILY_EXPOSURE_LIMIT

DEFAULT_SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"
DEFAULT_FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
OUT_PATH = ROOT / "evaluation" / "reports" / "flat_fraction_valid_2024.json"


def _max_losing_streak(win_flags: pd.Series) -> int:
    streak = 0
    best = 0
    for w in win_flags.tolist():
        if not w:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _monthly_max_drawdown(dates: pd.Series, pnl: pd.Series, bankroll: float) -> tuple[float, dict[str, float]]:
    """Worst single-calendar-month cumulative loss as a fraction of initial bankroll.

    Matches the production semantics in main/pipeline/monthly_dd_tracker.py
    (archived; check_monthly_dd_limit): each calendar month's P&L is tracked
    independently against initial_bankroll and resets at month start — this is
    NOT a continuously-compounding running-peak drawdown, since a fixed
    (non-compounding) flat stake repeated over hundreds of bets in a year would
    trivially exceed 100% cumulative loss and make the metric meaningless.
    """
    frame = pd.DataFrame({"race_date": pd.to_datetime(dates), "pnl": pnl.to_numpy(dtype=float)})
    frame["month"] = frame["race_date"].dt.to_period("M").astype(str)
    monthly_pnl = frame.groupby("month")["pnl"].sum()
    monthly_loss_ratio = (-monthly_pnl / bankroll).clip(lower=0.0)
    overall = float(monthly_loss_ratio.max()) if not monthly_loss_ratio.empty else 0.0
    return overall, {k: float(v) for k, v in monthly_loss_ratio.items()}


def _busiest_day_exposure(dates: pd.Series, stake_fraction: float) -> tuple[float, str, int]:
    day = pd.to_datetime(dates).dt.date
    counts = day.value_counts()
    if counts.empty:
        return 0.0, "", 0
    busiest_day = counts.idxmax()
    n = int(counts.max())
    return float(n * stake_fraction), str(busiest_day), n


def derive_flat_fraction(
    scores_path: Path = DEFAULT_SCORES_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
) -> dict[str, Any]:
    bet_cfg = load_betting_config()
    bankroll = float(bet_cfg.get("bankroll", 100_000))

    df = load_scored_odds_frame(scores_path, features_path)
    dates = pd.to_datetime(df["race_date"])
    valid_df = df.loc[(dates >= pd.Timestamp(VALID_START)) & (dates <= pd.Timestamp(VALID_END))].copy()

    picks, skipped = select_top1_bets(valid_df, cfg=bet_cfg)
    n_races_total = int(valid_df["race_id"].nunique())
    n_bets = len(picks)
    hit_rate = float((picks["finish_rank"].astype(int) == 1).mean()) if n_bets else None

    # Reference ROI at flat 100-yen/unit (comparable to fold2 OOS known value scale).
    ref_stake = 100.0
    ref_picks = picks.copy()
    ref_picks["stake"] = ref_stake
    ref_settled = settle_win_bets(ref_picks)
    valid_roi_pct = (
        float(ref_settled["payout"].sum() / ref_settled["stake"].sum() * 100.0) if n_bets else None
    )
    max_losing_streak = _max_losing_streak(ref_settled["win"]) if n_bets else 0

    # Actual (simulated, not scaled) metrics for every grid candidate — reported for
    # transparency / cross-check against the linear-scaling decision rule below.
    grid_results = []
    grid_by_f: dict[float, dict[str, Any]] = {}
    for f in STAKE_FRACTION_GRID:
        sized = apply_flat_sizing(picks, bankroll=bankroll, stake_fraction=f)
        settled = settle_win_bets(sized)
        monthly_mdd, monthly_dd_by_month = _monthly_max_drawdown(settled["race_date"], settled["pnl"], bankroll)
        busiest_exposure, busiest_day, busiest_n = _busiest_day_exposure(settled["race_date"], f)
        meets_mdd = monthly_mdd <= MONTHLY_MDD_LIMIT
        meets_exposure = busiest_exposure <= MAX_DAILY_EXPOSURE_LIMIT
        entry = {
            "stake_fraction": f,
            "monthly_max_drawdown": monthly_mdd,
            "monthly_max_drawdown_by_month": monthly_dd_by_month,
            "busiest_day": busiest_day,
            "busiest_day_n_races": busiest_n,
            "busiest_day_exposure": busiest_exposure,
            "meets_monthly_mdd_limit": meets_mdd,
            "meets_daily_exposure_limit": meets_exposure,
        }
        grid_results.append(entry)
        grid_by_f[f] = entry

    # --- Decision rule v2 (R2): linear scaling from the f0=0.0025 observed worst month,
    # then a fixed safety factor, then pick the largest grid value not exceeding the cap.
    # This is a *derivation*, not a per-candidate threshold scan: it does not use the
    # simulated monthly_max_drawdown of the other grid points to decide eligibility
    # (those are reported above only for cross-checking the linearity assumption).
    base_entry = grid_by_f[BASE_F_FOR_SCALING]
    worst_month_dd_at_f0 = base_entry["monthly_max_drawdown"]
    f_scale = MONTHLY_MDD_LIMIT / (worst_month_dd_at_f0 / BASE_F_FOR_SCALING)
    f_capped = SAFETY_FACTOR * f_scale

    eligible = [f for f in STAKE_FRACTION_GRID if f <= f_capped]
    adopted_f = max(eligible) if eligible else None

    adopted_busiest_exposure_headroom_ok = None
    if adopted_f is not None:
        adopted_busiest_exposure_headroom_ok = bool(
            grid_by_f[adopted_f]["busiest_day_exposure"] <= BUSIEST_DAY_EXPOSURE_HEADROOM_LIMIT
        )

    # Mark each grid entry's relationship to the v2 rule (informational; does not
    # itself decide adoption — adoption is f <= f_capped, computed above).
    for entry in grid_results:
        entry["within_f_capped"] = bool(entry["stake_fraction"] <= f_capped)
        entry["is_adopted"] = bool(adopted_f is not None and entry["stake_fraction"] == adopted_f)

    report = {
        "protocol": "VALID-only (2024-01-01..2024-12-31); TEST (2025+) not read by this script",
        "disclaimer": DISCLAIMER,
        "decision_rule": (
            "v2 (R2): f_scale = monthly_mdd_limit / (worst_month_dd@f0 / f0), f0="
            f"{BASE_F_FOR_SCALING}; f_capped = {SAFETY_FACTOR} * f_scale; adopt the largest "
            f"value in grid {STAKE_FRACTION_GRID} (downward-extension only) with f <= f_capped. "
            f"monthly_mdd_limit={MONTHLY_MDD_LIMIT}, busiest_day_exposure headroom check "
            f"<= {BUSIEST_DAY_EXPOSURE_HEADROOM_LIMIT} at adopted f."
        ),
        "decision_rule_v1_superseded": (
            "grid {0.0025, 0.005, 0.01}: adopt the largest f such that "
            f"monthly_max_drawdown <= {MONTHLY_MDD_LIMIT} and busiest_day_exposure <= {MAX_DAILY_EXPOSURE_LIMIT} "
            "(all candidates failed on VALID 2024 data; see spec R2 changelog)"
        ),
        "n_races_total_valid": n_races_total,
        "n_bets_valid": n_bets,
        "n_skipped_valid": len(skipped),
        "hit_rate_valid": hit_rate,
        "reference_roi_pct_100yen_flat": valid_roi_pct,
        "max_losing_streak_valid": max_losing_streak,
        "base_f_for_scaling": BASE_F_FOR_SCALING,
        "worst_month_dd_at_base_f": worst_month_dd_at_f0,
        "f_scale": f_scale,
        "safety_factor_k": SAFETY_FACTOR,
        "f_capped": f_capped,
        "grid": grid_results,
        "adopted_stake_fraction": adopted_f,
        "adopted_busiest_day_exposure_headroom_ok": adopted_busiest_exposure_headroom_ok,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    derive_flat_fraction()

"""CLI: flat top-1 損失最小化運用の OOS 正式バックテスト（1回限り実行、fold2 OOS）。

docs/specs/2026-07-10-loss-minimization-implementation-spec.md §4 準拠。

**実行条件（Rule 3。全て揃うまで実行しない）**:
  1. betting/tests/test_flat_top1.py が全て合格
  2. betting/src/derive_flat_fraction.py の VALID(2024) 専用凍結が完了し、
     betting_config.json の loss_min.stake_fraction に反映・コミット済み
  3. main/unified_pipeline.py の E2E が成功

1回の実行で以下を全て算出する（事前登録済み。複数回実行の口実を作らない）:
  1. 本番設定: flat top-1、オッズ除外あり、f凍結値
  2. 再現性確認用: flat top-1、オッズ除外なし（100円均等）→ 既知実測 81.89% と比較
  3. ベースライン: 同一レース集合で1番人気（オッズ最小）への同額flat bet（除外あり/なし）
  4. ペアドブートストラップ（レース単位、B=10,000, seed=42）ROI差 95%CI
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
from evaluation.market_baseline import build_win_odds_lookup_from_df, compute_favorite_baseline
from prob_fusion.src.oos_protocol import TEST_START

MIN_FORMAL_BETS = 200
KNOWN_REPRODUCTION_ROI_PCT = 81.89
REPRODUCTION_TOLERANCE_PP = 0.1
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42

DEFAULT_SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"
DEFAULT_FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
OUT_PATH = ROOT / "evaluation" / "reports" / "betting_backtest_oos_flat.json"


def _model_flat_roi(
    df: pd.DataFrame, *, cfg: dict[str, Any], bankroll: float, stake_fraction: float, apply_odds_exclusion: bool
) -> dict[str, Any]:
    """Flat top-1 (model rank-1) bets, optionally with min/max odds exclusion."""
    work_cfg = dict(cfg)
    if not apply_odds_exclusion:
        work_cfg["min_odds"] = 0.0
        work_cfg["max_odds"] = float("inf")

    picks, skipped = select_top1_bets(df, cfg=work_cfg)
    sized = apply_flat_sizing(picks, bankroll=bankroll, stake_fraction=stake_fraction)
    settled = settle_win_bets(sized)

    n_bets = len(settled)
    total_stake = float(settled["stake"].sum()) if n_bets else 0.0
    total_payout = float(settled["payout"].sum()) if n_bets else 0.0
    roi_pct = (total_payout / total_stake * 100.0) if total_stake > 0 else None
    hit_rate = float(settled["win"].mean()) if n_bets else None
    return {
        "n_bets": n_bets,
        "n_skipped": len(skipped),
        "hit_rate": hit_rate,
        "total_stake": total_stake,
        "total_payout": total_payout,
        "total_expected_loss": total_stake - total_payout,
        "roi_pct": roi_pct,
        "_settled": settled,
    }


def _favorite_flat_roi(
    df: pd.DataFrame, *, bankroll: float, stake_fraction: float, apply_odds_exclusion: bool, cfg: dict[str, Any]
) -> dict[str, Any]:
    """1番人気（オッズ最小）への同額 flat bet。evaluation.market_baseline と同一定義。"""
    work = df.copy()
    if apply_odds_exclusion:
        odds = pd.to_numeric(work["odds"], errors="coerce")
        work = work.loc[
            odds.notna() & (odds >= float(cfg.get("min_odds", 2.0))) & (odds <= float(cfg.get("max_odds", 50.0)))
        ]
        # Odds-range exclusion for the favorite baseline is applied at race level: if the
        # favorite of a race would be excluded, the whole race is skipped (mirrors §1.1's
        # no-fallback-to-rank-2 rule so the comparison is apples-to-apples).
        favorite_race_ids = []
        for rid, grp in df.groupby("race_id"):
            odds_g = pd.to_numeric(grp["odds"], errors="coerce")
            if odds_g.notna().sum() == 0:
                continue
            min_odds_val = odds_g.min()
            if float(cfg.get("min_odds", 2.0)) <= min_odds_val <= float(cfg.get("max_odds", 50.0)):
                favorite_race_ids.append(rid)
        work = df.loc[df["race_id"].isin(favorite_race_ids)]

    lookup = build_win_odds_lookup_from_df(work if not work.empty else df)
    stake = float(np.floor(bankroll * stake_fraction / 100.0) * 100.0)

    # HR win-payout lookup keyed the same way as odds (decimal odds -> yen payout per 100).
    hr_lookup: dict[str, dict[int, int]] = {}
    for rid, grp in work.groupby("race_id"):
        for _, row in grp.iterrows():
            if int(row["finish_rank"]) == 1:
                hr_lookup.setdefault(str(rid), {})[int(row["horse_num"])] = int(round(float(row["odds"]) * 100))

    fav = compute_favorite_baseline(work, lookup, hr_lookup, stake=stake)
    return {
        "n_bets": fav.get("favorite_roi_n_races", 0),
        "hit_rate": fav.get("favorite_top1_rate"),
        "roi_pct": fav.get("favorite_roi"),
        "coverage_rate": fav.get("coverage_rate"),
    }


def _paired_bootstrap_ci(
    model_settled: pd.DataFrame,
    favorite_settled: pd.DataFrame,
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Race-level paired bootstrap for ROI(model) - ROI(favorite), both odds-excluded."""
    merged = model_settled[["race_id", "stake", "payout"]].merge(
        favorite_settled[["race_id", "stake", "payout"]],
        on="race_id",
        suffixes=("_model", "_favorite"),
        how="inner",
    )
    n = len(merged)
    if n == 0:
        return {"n_races_paired": 0, "roi_diff_pp": None, "ci_low_pp": None, "ci_high_pp": None}

    rng = np.random.default_rng(seed)
    stake_m = merged["stake_model"].to_numpy(dtype=float)
    payout_m = merged["payout_model"].to_numpy(dtype=float)
    stake_f = merged["stake_favorite"].to_numpy(dtype=float)
    payout_f = merged["payout_favorite"].to_numpy(dtype=float)

    diffs = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        roi_m = payout_m[idx].sum() / stake_m[idx].sum() * 100.0 if stake_m[idx].sum() > 0 else 0.0
        roi_f = payout_f[idx].sum() / stake_f[idx].sum() * 100.0 if stake_f[idx].sum() > 0 else 0.0
        diffs[i] = roi_m - roi_f

    point_diff = (payout_m.sum() / stake_m.sum() * 100.0) - (payout_f.sum() / stake_f.sum() * 100.0)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_races_paired": n,
        "roi_diff_pp_point": float(point_diff),
        "ci_low_pp": float(ci_low),
        "ci_high_pp": float(ci_high),
        "bootstrap_b": b,
        "bootstrap_seed": seed,
    }


def run_backtest_oos_flat(
    scores_path: Path = DEFAULT_SCORES_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
) -> dict[str, Any]:
    bet_cfg = load_betting_config()
    loss_min_cfg = bet_cfg.get("loss_min", {})
    stake_fraction = float(loss_min_cfg.get("stake_fraction"))
    bankroll = float(bet_cfg.get("bankroll", 100_000))

    df = load_scored_odds_frame(scores_path, features_path)
    dates = pd.to_datetime(df["race_date"])
    test_df = df.loc[dates >= pd.Timestamp(TEST_START)].copy()

    prod = _model_flat_roi(
        test_df, cfg=bet_cfg, bankroll=bankroll, stake_fraction=stake_fraction, apply_odds_exclusion=True
    )
    repro = _model_flat_roi(
        test_df, cfg=bet_cfg, bankroll=bankroll, stake_fraction=stake_fraction, apply_odds_exclusion=False
    )
    fav_excl = _favorite_flat_roi(
        test_df, bankroll=bankroll, stake_fraction=stake_fraction, apply_odds_exclusion=True, cfg=bet_cfg
    )
    fav_no_excl = _favorite_flat_roi(
        test_df, bankroll=bankroll, stake_fraction=stake_fraction, apply_odds_exclusion=False, cfg=bet_cfg
    )

    reproduction_ok = (
        repro["roi_pct"] is not None
        and abs(repro["roi_pct"] - KNOWN_REPRODUCTION_ROI_PCT) <= REPRODUCTION_TOLERANCE_PP
    )

    # Paired bootstrap needs per-race settled bets for both model (odds-excluded) and
    # favorite (odds-excluded) on the SAME race set (inner join on race_id already applied
    # inside _paired_bootstrap_ci via merge). Favorite settled frame must be rebuilt with
    # per-bet rows (compute_favorite_baseline only returns aggregates).
    fav_picks_rows = []
    for rid, grp in test_df.groupby("race_id"):
        odds_g = pd.to_numeric(grp["odds"], errors="coerce")
        if odds_g.notna().sum() == 0:
            continue
        idx_min = odds_g.idxmin()
        row = grp.loc[idx_min]
        if not (float(bet_cfg.get("min_odds", 2.0)) <= float(row["odds"]) <= float(bet_cfg.get("max_odds", 50.0))):
            continue
        fav_picks_rows.append(row)
    fav_picks = pd.DataFrame(fav_picks_rows)
    fav_sized = apply_flat_sizing(fav_picks, bankroll=bankroll, stake_fraction=stake_fraction) if len(fav_picks) else fav_picks
    fav_settled = settle_win_bets(fav_sized) if len(fav_sized) else fav_sized

    bootstrap = (
        _paired_bootstrap_ci(prod["_settled"], fav_settled)
        if len(fav_settled) and prod["n_bets"]
        else {"n_races_paired": 0, "roi_diff_pp_point": None, "ci_low_pp": None, "ci_high_pp": None}
    )

    gates = {
        "n_bets_gte_200": prod["n_bets"] >= MIN_FORMAL_BETS,
        "reproduction_ok": bool(reproduction_ok),
        "roi_above_market_point": (
            prod["roi_pct"] is not None and fav_excl["roi_pct"] is not None and prod["roi_pct"] > fav_excl["roi_pct"]
        ),
        "roi_above_market_ci95": bootstrap.get("ci_low_pp") is not None and bootstrap["ci_low_pp"] > 0,
    }
    verdict = (
        "pass"
        if all(gates.values())
        else "pass_point_only"
        if gates["n_bets_gte_200"] and gates["reproduction_ok"] and gates["roi_above_market_point"]
        else "fail"
    )

    report = {
        "disclaimer": DISCLAIMER,
        "protocol": {
            "test_period": f"{TEST_START}..",
            "bet_type": "win",
            "stake_fraction_frozen_valid_2024": stake_fraction,
            "bankroll": bankroll,
            "score_col": loss_min_cfg.get("score_col", "pure_score_z"),
        },
        "production": {k: v for k, v in prod.items() if k != "_settled"},
        "reproduction_no_odds_exclusion": {k: v for k, v in repro.items() if k != "_settled"},
        "known_reproduction_roi_pct": KNOWN_REPRODUCTION_ROI_PCT,
        "favorite_baseline_odds_excluded": fav_excl,
        "favorite_baseline_no_odds_exclusion": fav_no_excl,
        "paired_bootstrap": bootstrap,
        "gates": gates,
        "verdict": verdict,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run_backtest_oos_flat()

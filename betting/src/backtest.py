"""Walk-forward backtest using on-the-fly fusion (valid+test, fair market beta)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from betting.src.ev_engine import EvThresholdResult, apply_ev_filters, enrich_predictions, select_ev_threshold_on_valid
from betting.src.kelly_sizer import apply_kelly_sizing, apply_mutually_exclusive_decay
from evaluation.odds_loader import attach_odds_from_se_parquet
from evaluation.splits import FORMAL_JUDGMENT_FOLD, L1_CONTAMINATED_REFERENCE_FOLDS, filter_fold, get_walkforward_folds
from prob_fusion.src.market_prob import attach_market_q
from prob_fusion.src.predict_fusion import fuse_dataframe, load_fusion_config


def load_betting_config(path: Path | None = None) -> dict[str, Any]:
    p = path or ROOT / "betting" / "config" / "betting_config.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_fusion_params(path: Path | None = None) -> dict[str, Any]:
    p = path or ROOT / "prob_fusion" / "data" / "fusion_params.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_scored_odds_frame(scores_path: Path, features_path: Path) -> pd.DataFrame:
    """Scores + outcomes + odds for full timeline (not test-only probs parquet)."""
    scores = pd.read_parquet(scores_path)
    scores["race_id"] = scores["race_id"].astype(str)
    if "horse_number" not in scores.columns:
        scores["horse_number"] = scores["horse_num"]
    scores["horse_num"] = scores["horse_number"].astype(int)

    feat = pd.read_parquet(features_path, columns=["race_id", "horse_num", "finish_rank", "race_date"])
    feat["race_id"] = feat["race_id"].astype(str)
    for col in ("finish_rank", "race_date"):
        if col in scores.columns:
            scores = scores.drop(columns=[col])
    df = scores.merge(feat, on=["race_id", "horse_num"], how="inner")
    if "race_date" not in df.columns and "race_date_x" in df.columns:
        df["race_date"] = df["race_date_y"].combine_first(df["race_date_x"])

    if "odds" not in df.columns:
        df = attach_odds_from_se_parquet(df)
    return df


def _fold_params(fusion_report: dict, fold_num: int) -> dict[str, float]:
    for f in fusion_report.get("folds", []):
        if f.get("fold") == fold_num:
            return {
                "alpha": float(f["alpha"]),
                "beta": float(f["beta"]),
                "lam2": float(f.get("lam2", 1.0)),
                "lam3": float(f.get("lam3", 1.0)),
            }
    raise KeyError(f"fold {fold_num} not in fusion_params")


def fuse_period(df: pd.DataFrame, params: dict[str, float], cfg: dict) -> pd.DataFrame:
    """Attach fused p_win/p_place to a period dataframe."""
    if df.empty:
        return df
    work = attach_market_q(df, method=cfg.get("q_method", "proportional"), power=cfg.get("q_power", 0.81))
    fused = fuse_dataframe(
        work,
        alpha=params["alpha"],
        beta=params["beta"],
        lam2=params["lam2"],
        lam3=params["lam3"],
        q_method=cfg.get("q_method", "proportional"),
        q_power=cfg.get("q_power", 0.81),
    )
    return work.merge(
        fused[["race_id", "horse_number", "p_win", "p_place"]],
        left_on=["race_id", "horse_num"],
        right_on=["race_id", "horse_number"],
        how="left",
    )


def simulate_bets(
    df: pd.DataFrame,
    *,
    bet_type: str = "win",
    ev_threshold: float = 1.05,
    cfg: dict[str, Any],
    place_odds_col: str | None = None,
) -> pd.DataFrame:
    if bet_type == "place":
        raise ValueError("place bets disabled until place odds data is available")

    work = df.copy()
    work["model_prob"] = work["p_win"]
    work = enrich_predictions(work, ev_haircut=cfg.get("ev_haircut", 0.95))
    mask = apply_ev_filters(
        work,
        ev_col="ev_adjusted",
        ev_threshold=ev_threshold,
        min_odds=cfg.get("min_odds", 2.0),
        max_odds=cfg.get("max_odds", 50.0),
        min_model_prob=cfg.get("min_model_prob", 0.05),
    )
    picks = work.loc[mask].copy()
    if picks.empty:
        return picks

    picks = apply_kelly_sizing(
        picks,
        bankroll=cfg.get("bankroll", 100000),
        kelly_frac=cfg.get("kelly_fraction", 0.08),
        max_bet_ratio=cfg.get("max_bet_ratio", 0.05),
    )
    picks = apply_mutually_exclusive_decay(picks)
    picks = picks.sort_values("ev_adjusted", ascending=False)
    picks = picks.groupby("race_id", sort=False).head(cfg.get("max_picks_per_race", 2))

    stake = picks["kelly_bet_yen"].astype(float)
    win = picks["finish_rank"].astype(int) == 1
    payout = np.where(win, stake * picks["odds"].astype(float), 0.0)

    picks["stake"] = stake
    picks["payout"] = payout
    picks["pnl"] = payout - stake
    picks["bet_type"] = bet_type
    picks["ev"] = picks["ev_adjusted"]
    picks["kelly_fraction"] = picks["kelly_ratio"]
    return picks


def evaluate_fold(
    scored_df: pd.DataFrame,
    fold: dict,
    fold_params: dict[str, float],
    fusion_cfg: dict,
    bet_cfg: dict[str, Any],
) -> dict[str, Any]:
    fold_num = fold["fold"]
    tier = "formal" if fold_num == FORMAL_JUDGMENT_FOLD else "reference_l1_contaminated"

    valid_df = filter_fold(scored_df, fold, "valid")
    test_df = filter_fold(scored_df, fold, "test")
    valid_fused = fuse_period(valid_df, fold_params, fusion_cfg)
    test_fused = fuse_period(test_df, fold_params, fusion_cfg)

    thr_result: EvThresholdResult = select_ev_threshold_on_valid(
        valid_fused,
        bet_cfg.get("ev_threshold_grid", [1.05]),
        bet_type="win",
        ev_haircut=bet_cfg.get("ev_haircut", 0.95),
    )

    skipped = tier != "formal" and fold_num in L1_CONTAMINATED_REFERENCE_FOLDS
    if skipped:
        return {
            "fold": fold_num,
            "judgment_tier": tier,
            "skipped_for_formal_gate": True,
            "reason": "L1 in-sample contamination; reference only",
            "valid_n_races": thr_result.valid_n_races,
            "ev_threshold_warnings": thr_result.warnings,
        }

    if thr_result.fallback_used and fold_num == FORMAL_JUDGMENT_FOLD:
        return {
            "fold": fold_num,
            "judgment_tier": tier,
            "skipped_for_formal_gate": True,
            "reason": "VALID threshold tuning failed",
            "ev_threshold": thr_result.threshold,
            "ev_threshold_warnings": thr_result.warnings,
            "valid_n_races": thr_result.valid_n_races,
        }

    bets = simulate_bets(test_fused, bet_type="win", ev_threshold=thr_result.threshold, cfg=bet_cfg)
    n_bets = len(bets)
    total_stake = float(bets["stake"].sum()) if n_bets else 0.0
    total_payout = float(bets["payout"].sum()) if n_bets else 0.0
    roi = total_payout / total_stake * 100.0 if total_stake > 0 else 0.0
    sharpe = 0.0
    if n_bets > 1:
        rets = bets["pnl"].astype(float) / bets["stake"].astype(float).clip(lower=1.0)
        std = rets.std()
        sharpe = float(rets.mean() / std) if std > 1e-9 else 0.0

    return {
        "fold": fold_num,
        "judgment_tier": tier,
        "bet_type": "win",
        "ev_threshold": thr_result.threshold,
        "ev_threshold_fallback": thr_result.fallback_used,
        "ev_threshold_warnings": thr_result.warnings,
        "valid_n_races": thr_result.valid_n_races,
        "n_bets": n_bets,
        "roi_pct": roi,
        "sharpe": sharpe,
        "skipped_for_formal_gate": False,
    }


def run_walkforward_backtest(
    scored_df: pd.DataFrame,
    *,
    bet_cfg: dict[str, Any] | None = None,
    fusion_report: dict | None = None,
) -> dict[str, Any]:
    bet_cfg = bet_cfg or load_betting_config()
    fusion_cfg = load_fusion_config()
    fusion_report = fusion_report or load_fusion_params()

    bet_types = bet_cfg.get("bet_types", ["win"])
    if "place" in bet_types:
        bet_types = [t for t in bet_types if t != "place"]

    results = []
    for fold in get_walkforward_folds():
        params = _fold_params(fusion_report, fold["fold"])
        for bet_type in bet_types:
            if bet_type != "win":
                continue
            results.append(evaluate_fold(scored_df, fold, params, fusion_cfg, bet_cfg))

    formal = [r for r in results if r.get("fold") == FORMAL_JUDGMENT_FOLD and not r.get("skipped_for_formal_gate")]
    report = {
        "results": results,
        "config": bet_cfg,
        "formal_judgment_fold": FORMAL_JUDGMENT_FOLD,
        "formal_results": formal,
        "note": "Only fold 3 (2025+ TEST) counts for formal L3 gate; folds 1-2 are L1-contaminated reference.",
    }
    out = ROOT / "evaluation" / "reports" / "betting_backtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

"""binary残差学習アンサンブル → EV → Kelly → 推奨リスト（v4本番パス）。

main/build_today_features.py や e2e_test 向け。Notebook の lambdarank 系は
strategy_engine.recommend_today（旧 online パス）を使用する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from ev_calculator import apply_ev_filters, enrich_predictions
from inference_common import (
    apply_condition_overrides,
    apply_max_picks_per_race,
    apply_race_budget_cap,
    apply_var1_market_blend_probs,
    compute_market_log_odds,
    load_ensemble_models,
    predict_model_probs,
)
from kelly_sizer import apply_kelly_sizing
from pipeline_common import load_config
from train import get_feature_cols

RESULT_DIR = ROOT / "main" / "results"


def generate_recommendations(
    df_with_probs: pd.DataFrame,
    cfg: dict,
    bankroll: float = 100_000,
    *,
    ev_threshold: float | None = None,
    max_picks: int | None = None,
) -> pd.DataFrame:
    """model_prob 付きDataFrameからEV・Kelly・推奨フラグを計算する。"""
    t_cfg = cfg["training"]
    ev_thr = float(ev_threshold if ev_threshold is not None else t_cfg["ev_threshold"])
    max_pick = int(max_picks if max_picks is not None else t_cfg.get("max_picks_per_race", 2))

    blend_cfg = t_cfg.get("var1_market_blend", {})
    prob_col = "model_prob"
    df = df_with_probs.copy()
    if blend_cfg.get("enabled", False) and "var1_pure_score_z" in df.columns:
        df["ev_prob"] = apply_var1_market_blend_probs(
            df, beta=float(blend_cfg.get("beta", 0.30))
        )
        prob_col = "ev_prob"

    df = enrich_predictions(df, model_prob_col=prob_col, odds_col="odds")
    SELECTION_BANKROLL = 10_000_000
    df = apply_kelly_sizing(
        df,
        bankroll=SELECTION_BANKROLL,
        kelly_frac=t_cfg["kelly_fraction"],
        max_bet_ratio=t_cfg["max_bet_ratio"],
    )
    base_mask = apply_ev_filters(
        df,
        ev_threshold=ev_thr,
        min_odds=t_cfg["min_odds"],
        max_odds=t_cfg["max_odds"],
        min_model_prob=t_cfg["min_model_prob"],
        model_prob_col=prob_col,
    ) & (df["kelly_bet_yen"] > 0)
    base_mask = apply_condition_overrides(
        df, base_mask, cfg.get("condition_ev_overrides", []), ev_thr
    )
    if max_pick > 0:
        base_mask = apply_max_picks_per_race(df, base_mask, max_pick)
    unc_cfg = t_cfg.get("uncertainty_skip", {})
    if unc_cfg.get("enabled", False) and "pred_uncertainty" in df.columns:
        thr = unc_cfg.get("threshold")
        if thr is None:
            try:
                import json as _json

                from pipeline_common import MODELS_DIR as _MD

                with open(_MD / "backtest_results.json", encoding="utf-8") as _f:
                    _res = _json.load(_f)
                thr = next(
                    (r.get("uncertainty_threshold") for r in _res if r.get("fold") == 3),
                    None,
                )
            except Exception:
                thr = None
        if thr is not None:
            base_mask = base_mask & (df["pred_uncertainty"] <= float(thr))
    df["is_recommended"] = base_mask
    df = apply_race_budget_cap(df, base_mask, t_cfg["max_bet_ratio"], bankroll=bankroll)
    import numpy as np

    df["kelly_bet_yen"] = np.floor(bankroll * df["kelly_ratio"] / 100.0) * 100.0
    return df


def save_recommendations(df: pd.DataFrame, date_str: str = "") -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{date_str}" if date_str else ""
    df.to_parquet(RESULT_DIR / f"today_recommendations{suffix}.parquet", index=False)
    df.to_csv(
        RESULT_DIR / f"today_recommendations{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    n_rec = df["is_recommended"].sum() if "is_recommended" in df.columns else 0
    print(f"推奨リスト保存完了: {len(df)}件中 {n_rec}件が推奨 → {RESULT_DIR}")


def run_strategy(
    df_features: pd.DataFrame,
    odds_df: pd.DataFrame,
    bankroll: float = 100_000,
    fold: int = 3,
    date_str: str = "",
    rank_preds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """特徴量 + オッズ → 推奨リストを生成して返す（バックテストと同一推論パス）。"""
    cfg = load_config()
    t_cfg = cfg["training"]
    feature_cols = get_feature_cols(cfg)

    df = df_features.merge(
        odds_df[["race_id", "horse_id", "odds"]],
        on=["race_id", "horse_id"],
        how="left",
        suffixes=("_stale", ""),
    )
    if "odds_stale" in df.columns:
        df["odds"] = df["odds"].combine_first(df["odds_stale"])
        df = df.drop(columns=["odds_stale"])

    df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
    if len(df) == 0:
        print("Warning: オッズのある出走馬がいないため推奨リストは空です。")
        return df

    base_margin_col = t_cfg.get("base_margin_col")
    if base_margin_col == "market_log_odds":
        df = compute_market_log_odds(df, odds_col="odds")

    if "var1_pure_score_z" not in df.columns:
        sys.path.insert(0, str(ROOT / "model_training" / "scripts"))
        from merge_var1_pure_scores import (  # noqa: E402
            attach_var1_score_z,
            attach_var1_z_from_rank_preds,
        )

        if rank_preds is not None:
            df = attach_var1_z_from_rank_preds(df, rank_preds)
        else:
            scores_path = t_cfg.get("var1_scores_path")
            if scores_path:
                sp = Path(scores_path)
                if not sp.is_absolute():
                    sp = ROOT / sp
                if sp.exists():
                    df = attach_var1_score_z(df, sp)

    models = load_ensemble_models(fold)
    unc_cfg = t_cfg.get("uncertainty_skip", {})
    if unc_cfg.get("enabled", False):
        from inference_common import predict_with_uncertainty

        probs, unc = predict_with_uncertainty(models, df, feature_cols, base_margin_col, t_cfg=t_cfg)
        df["model_prob"] = probs
        df["pred_uncertainty"] = unc / probs.clip(lower=1e-6)
    else:
        df["model_prob"] = predict_model_probs(
            models, df, feature_cols, base_margin_col, t_cfg=t_cfg
        )

    df_recs = generate_recommendations(df, cfg, bankroll)
    if date_str:
        save_recommendations(df_recs, date_str)
    return df_recs

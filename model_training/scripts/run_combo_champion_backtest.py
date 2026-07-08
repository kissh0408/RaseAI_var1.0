"""Phase4 binary champion の馬連・ワイド BT（1位軸×2・3位）を walk-forward 3 fold で実行。"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from backtest import load_features_with_odds  # noqa: E402
from combo_backtest import run_combo_backtest  # noqa: E402
from inference_common import load_ensemble_models, predict_model_probs  # noqa: E402
from pipeline_common import load_config  # noqa: E402


def _build_eval_df(test_df: pd.DataFrame, models, cols: list[str], temperature: float) -> pd.DataFrame:
    probs = predict_model_probs(
        models,
        test_df,
        cols,
        base_margin_col="market_log_odds",
        temperature=temperature,
    )
    out = test_df[["race_id", "horse_num", "finish_rank", "race_date"]].copy()
    out["pred_rank1"] = probs
    out["pred_rank2"] = probs
    out["valid_year"] = pd.to_datetime(out["race_date"]).dt.year
    out["race_num"] = 0
    return out


def main() -> int:
    cfg = load_config()
    t_cfg = cfg["training"]
    odds_dir = ROOT / "common" / "data" / "output" / "odds"
    df_all = load_features_with_odds()
    temperature = float(t_cfg.get("calibration", {}).get("temperature", 1.0))

    combo_cfg = {
        "use_fixed_stake": False,
        "min_edge": 0.05,
        "wide_min_edge": 0.05,
        "max_expected_value": 5.0,
        "fractional_kelly": t_cfg["kelly_fraction"],
        "max_stake_per_bet": 3000,
        "max_invest_per_race": 50000,
        "bet_unit": 100,
        "base_slippage": 0.01,
        "initial_bankroll": 100_000,
        "wide_min_edge": 0.08,
        "wide_max_stake_per_bet": 3000,
    }

    results: list[dict] = []
    for fold_cfg in t_cfg["walkforward_folds"]:
        fold = int(fold_cfg["fold"])
        ts = pd.Timestamp(fold_cfg["test_start"])
        te = pd.Timestamp(fold_cfg["test_end"])
        test_df = df_all[(df_all["race_date"] >= ts) & (df_all["race_date"] <= te)].copy()
        excl_surface = t_cfg.get("exclude_surface_codes", [3])
        if excl_surface and "surface_code" in test_df.columns:
            test_df = test_df[~test_df["surface_code"].isin(excl_surface)]
        excl_abnormal = t_cfg.get("exclude_abnormal_codes", [])
        if excl_abnormal and "abnormal_code" in test_df.columns:
            test_df = test_df[~test_df["abnormal_code"].isin(excl_abnormal)]
        models = load_ensemble_models(fold)
        cols = list(models[0].feature_name())
        eval_df = _build_eval_df(test_df, models, cols, temperature)

        _, summary = run_combo_backtest(
            eval_df,
            combo_cfg,
            odds_dir,
            pair_top_n=2,
            wide_top_n=2,
            rank2_blend=0.35,
        )
        summary["fold"] = fold
        summary["period"] = "test"
        results.append(summary)
        print(
            f"Fold {fold}: wide hit={summary.get('hit_rate_wide', 0):.1%} "
            f"ROI={summary.get('roi_wide', 0):.1%} n={summary.get('n_bets_wide', 0)} | "
            f"quinella hit={summary.get('hit_rate_quinella', 0):.1%} "
            f"ROI={summary.get('roi_quinella', 0):.1%} n={summary.get('n_bets_quinella', 0)}"
        )

    out_path = ROOT / "model_training" / "models" / "backtest_results_cycle0_combo_baseline.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": combo_cfg,
        "pair_top_n": 2,
        "wide_top_n": 2,
        "rank2_blend": 0.35,
        "folds": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

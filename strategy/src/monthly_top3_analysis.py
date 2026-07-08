"""月別スコア上位3頭の的中率・回収率分析。

EV フィルタなし。各レースでモデルスコア上位3頭を選び、
月別に単勝的中率と回収率を集計する。

実行:
    python strategy/src/monthly_top3_analysis.py
    python strategy/src/monthly_top3_analysis.py --no-market
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from backtest import load_features_with_odds
from inference_common import load_ensemble_models, predict_model_probs
from pipeline_common import load_config
from train import get_feature_cols


def _run_fold_top3(
    fold: int,
    df_all: pd.DataFrame,
    fold_cfg: dict,
    cfg: dict,
    no_market: bool = False,
) -> pd.DataFrame:
    """fold のテスト期間でスコア上位3頭を返す。"""
    t_cfg = cfg["training"]
    nm_cfg = cfg.get("no_market_experiment", {})

    test_df = df_all[
        (df_all["race_date"] >= fold_cfg["test_start"])
        & (df_all["race_date"] <= fold_cfg["test_end"])
    ].copy()

    if len(test_df) == 0:
        return pd.DataFrame()

    model_prefix = (
        nm_cfg.get("binary_model_prefix", "lgbm_binary_no_market")
        if no_market else "lgbm_binary"
    )
    models = load_ensemble_models(fold, model_prefix=model_prefix)
    model_feature_cols = list(models[0].feature_name()) if hasattr(models[0], "feature_name") else []
    resolved_cols = model_feature_cols or get_feature_cols(cfg)
    available = [c for c in resolved_cols if c in test_df.columns]

    base_margin_col = (
        nm_cfg.get("base_margin_col", "jra_tm_log_odds")
        if no_market
        else t_cfg.get("base_margin_col")
    )

    test_df["model_prob"] = predict_model_probs(
        models, test_df, available, base_margin_col, temperature=1.0
    )

    # レース内スコアランク (1=最高スコア)
    test_df["score_rank"] = test_df.groupby("race_id")["model_prob"].rank(
        method="first", ascending=False
    ).astype(int)

    # 上位3頭のみ
    top3 = test_df[test_df["score_rank"] <= 3].copy()
    top3["is_win"] = (top3["finish_rank"] == 1).astype(int)
    top3["fold"] = fold
    return top3


def compute_monthly_stats(bets: pd.DataFrame) -> pd.DataFrame:
    """month ごとに的中率・回収率を集計する。

    Returns:
        月別DataFrame (year_month, n_bets, hit_rate, roi, n_wins)
    """
    bets = bets.copy()
    bets["year_month"] = pd.to_datetime(bets["race_date"]).dt.to_period("M")

    rows = []
    for ym, grp in bets.groupby("year_month"):
        n = len(grp)
        n_wins = grp["is_win"].sum()
        hit_rate = n_wins / n if n > 0 else float("nan")
        # 単純均等ベット100円想定: 当たりは odds × 100、外れは 0
        # odds が tan-waku オッズ（払戻）の場合は win_return = odds * 100
        total_bet = n * 100
        total_return = (grp.loc[grp["is_win"] == 1, "odds"] * 100).sum()
        roi = total_return / total_bet if total_bet > 0 else float("nan")
        rows.append(
            dict(year_month=str(ym), n_bets=n, n_wins=int(n_wins),
                 hit_rate=hit_rate, roi=roi)
        )

    return pd.DataFrame(rows)


def main(no_market: bool = False) -> None:
    cfg = load_config()
    mode_label = "[no-market]" if no_market else "[baseline]"
    print(f"\nスコア上位3頭 月別分析 {mode_label}")
    print("=" * 60)

    df_all = load_features_with_odds(no_market=no_market)

    all_top3: list[pd.DataFrame] = []
    for fold_cfg in cfg["training"]["walkforward_folds"]:
        fold = fold_cfg["fold"]
        print(f"Fold {fold} 推論中 ({fold_cfg['test_start']} - {fold_cfg['test_end']})...")
        top3 = _run_fold_top3(fold, df_all, fold_cfg, cfg, no_market=no_market)
        if len(top3) > 0:
            all_top3.append(top3)

    if not all_top3:
        print("データなし")
        return

    bets = pd.concat(all_top3, ignore_index=True)
    monthly = compute_monthly_stats(bets)

    # 表示
    print(f"\n{'月':>8}  {'ベット数':>7}  {'的中数':>5}  {'的中率':>7}  {'回収率':>7}")
    print("-" * 50)
    for _, row in monthly.iterrows():
        print(
            f"{row['year_month']:>8}  {row['n_bets']:>7d}  {int(row['n_wins']):>5d}  "
            f"{row['hit_rate']:>7.1%}  {row['roi']:>7.1%}"
        )

    # 全体集計
    total_bets = monthly["n_bets"].sum()
    total_wins = monthly["n_wins"].sum()
    overall_hit = total_wins / total_bets if total_bets > 0 else float("nan")
    # ROIは集計しなおす（月平均では重みが違う）
    overall_roi = (
        (bets.loc[bets["is_win"] == 1, "odds"] * 100).sum()
        / (len(bets) * 100)
    )
    print("-" * 50)
    print(
        f"{'合計':>8}  {total_bets:>7d}  {int(total_wins):>5d}  "
        f"{overall_hit:>7.1%}  {overall_roi:>7.1%}"
    )

    # フォールド別集計
    print(f"\n{'フォールド別':>12}  {'ベット数':>7}  {'的中率':>7}  {'回収率':>7}")
    print("-" * 40)
    for fold_n, grp in bets.groupby("fold"):
        fh = grp["is_win"].mean()
        fr = (grp.loc[grp["is_win"] == 1, "odds"] * 100).sum() / (len(grp) * 100)
        print(f"  Fold {fold_n:>8}  {len(grp):>7d}  {fh:>7.1%}  {fr:>7.1%}")

    # CSV 保存
    from pathlib import Path
    from pipeline_common import MODELS_DIR
    tag = "no_market" if no_market else "baseline"
    out_csv = MODELS_DIR / f"monthly_top3_{tag}.csv"
    monthly.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] CSV 保存: {out_csv}")


if __name__ == "__main__":
    no_market = "--no-market" in sys.argv
    main(no_market=no_market)

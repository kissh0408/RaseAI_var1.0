"""馬連・ワイド向けの順位的中率を binary champion の test 期間で集計。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from backtest import load_features_with_odds  # noqa: E402
from inference_common import (  # noqa: E402
    load_ensemble_models,
    load_lambdarank_top3_ensemble_models,
    load_top3_ensemble_models,
    predict_lambdarank_scores,
    predict_model_probs,
    predict_top3_probs,
)
from pipeline_common import load_config  # noqa: E402


def _in_top3(fr: float) -> bool:
    return 1 <= fr <= 3


def _in_top2(fr: float) -> bool:
    return 1 <= fr <= 2


def race_metrics(g: pd.DataFrame, prob_col: str = "model_prob") -> dict:
    g = g.copy()
    g["_fr"] = pd.to_numeric(g["finish_rank"], errors="coerce")
    ranked = g.nlargest(3, prob_col)
    if len(ranked) < 3:
        return {}
    fr1 = float(ranked.iloc[0]["_fr"])
    fr2 = float(ranked.iloc[1]["_fr"])
    fr3 = float(ranked.iloc[2]["_fr"])

    top3 = ranked
    top2 = ranked.iloc[:2]
    fr3_arr = top3["_fr"].values
    fr2_arr = top2["_fr"].values
    in3_3 = ((fr3_arr >= 1) & (fr3_arr <= 3)).astype(int)
    in3_2 = ((fr2_arr >= 1) & (fr2_arr <= 3)).astype(int)
    in12_2 = ((fr2_arr >= 1) & (fr2_arr <= 2)).astype(int)
    n_in3_top3 = int(in3_3.sum())

    # 1位軸 × 2位・3位相手（本番 pair_top_n=2 / wide_top_n=2 と同型）
    q_12 = _in_top2(fr1) and _in_top2(fr2)
    q_13 = _in_top2(fr1) and _in_top2(fr3)
    w_12 = _in_top3(fr1) and _in_top3(fr2)
    w_13 = _in_top3(fr1) and _in_top3(fr3)

    return {
        "quinella_top2": bool(in12_2.all()),
        "wide_top2": bool(in3_2.all()),
        "wide_box3_top3": n_in3_top3 >= 2,
        "trifecta_box3": n_in3_top3 == 3,
        "overlap_frac": n_in3_top3 / 3.0,
        "top1_win": fr1 == 1,
        "top1_in_top3": _in_top3(fr1),
        "quinella_anchor_12": q_12,
        "quinella_anchor_13": q_13,
        "quinella_anchor_any": q_12 or q_13,
        "wide_anchor_12": w_12,
        "wide_anchor_13": w_13,
        "wide_anchor_any": w_12 or w_13,
        "n_in_top3": n_in3_top3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prob-source",
        choices=["win", "top3", "blend", "lambdarank"],
        default="win",
        help="win=単勝champion / top3=3着以内モデル / blend=線形混合 / lambdarank=NDCG順位",
    )
    parser.add_argument("--blend-alpha", type=float, default=0.5, help="blend時の top3 重み")
    args = parser.parse_args()

    cfg = load_config()
    t_cfg = cfg["training"]
    df_all = load_features_with_odds()
    rows: list[dict] = []

    for fold_cfg in t_cfg["walkforward_folds"]:
        fold = int(fold_cfg["fold"])
        ts = pd.Timestamp(fold_cfg["test_start"])
        te = pd.Timestamp(fold_cfg["test_end"])
        test_df = df_all[(df_all["race_date"] >= ts) & (df_all["race_date"] <= te)].copy()
        win_models = load_ensemble_models(fold)
        cols = list(win_models[0].feature_name())
        temperature = float(t_cfg.get("calibration", {}).get("temperature", 1.0))
        win_prob = predict_model_probs(
            win_models,
            test_df,
            cols,
            base_margin_col="market_log_odds",
            temperature=temperature,
        )
        if args.prob_source == "win":
            test_df["model_prob"] = win_prob
        elif args.prob_source == "top3":
            top3_models = load_top3_ensemble_models(fold)
            test_df["model_prob"] = predict_top3_probs(top3_models, test_df, cols, temperature)
        elif args.prob_source == "lambdarank":
            lr_models = load_lambdarank_top3_ensemble_models(fold)
            test_df["model_prob"] = predict_lambdarank_scores(lr_models, test_df, cols)
        else:
            top3_models = load_top3_ensemble_models(fold)
            top3_prob = predict_top3_probs(top3_models, test_df, cols, temperature)
            a = float(args.blend_alpha)
            test_df["model_prob"] = (1.0 - a) * win_prob + a * top3_prob
        excl_surface = t_cfg.get("exclude_surface_codes", [3])
        if excl_surface and "surface_code" in test_df.columns:
            test_df = test_df[~test_df["surface_code"].isin(excl_surface)]
        excl_abnormal = t_cfg.get("exclude_abnormal_codes", [])
        if excl_abnormal and "abnormal_code" in test_df.columns:
            test_df = test_df[~test_df["abnormal_code"].isin(excl_abnormal)]

        for _, g in test_df.groupby("race_id", sort=False):
            if len(g) < 3:
                continue
            m = race_metrics(g)
            m["fold"] = fold
            rows.append(m)

    df = pd.DataFrame(rows)
    print(f"=== prob_source={args.prob_source} (test period) ===")
    print(f"n_races: {len(df)}")
    labels = [
        ("top1_win", "1位予測が単勝"),
        ("top1_in_top3", "1位予測が3着以内（軸の複勝相当）"),
        ("quinella_anchor_12", "馬連: 1位軸×2位相手（1点）"),
        ("quinella_anchor_13", "馬連: 1位軸×3位相手（1点）"),
        ("quinella_anchor_any", "馬連: 1位軸×2・3位（2点）いずれか的中"),
        ("wide_anchor_12", "ワイド: 1位軸×2位相手（1点）"),
        ("wide_anchor_13", "ワイド: 1位軸×3位相手（1点）"),
        ("wide_anchor_any", "ワイド: 1位軸×2・3位（2点）いずれか的中"),
        ("quinella_top2", "馬連: 上位2頭が1-2着（軸なし）"),
        ("wide_top2", "ワイド: 上位2頭が両方3着以内（軸なし）"),
        ("wide_box3_top3", "ワイド: 上位3頭BOX(3点)いずれか的中"),
        ("trifecta_box3", "参考: 上位3頭すべて3着以内"),
    ]
    for col, label in labels:
        print(f"  {label}: {df[col].mean():.1%}")

    print("\n--- fold別 ---")
    for f in sorted(df["fold"].unique()):
        sub = df[df["fold"] == f]
        print(
            f"  F{f} n={len(sub)}: "
            f"軸ワイド2点={sub['wide_anchor_any'].mean():.1%} "
            f"軸馬連2点={sub['quinella_anchor_any'].mean():.1%} "
            f"軸3着内={sub['top1_in_top3'].mean():.1%}"
        )

    print("\n--- 上位3頭の3着以内頭数分布 (全test) ---")
    cnt = df["n_in_top3"].astype(int)
    for k in range(4):
        print(f"  {k}頭: {(cnt == k).mean():.1%}")
    print(f"  => 2頭以上 (ワイドBOX的中条件): {(cnt >= 2).mean():.1%}")


if __name__ == "__main__":
    main()

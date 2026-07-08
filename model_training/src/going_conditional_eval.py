"""馬場状態別 top3 順位 KPI 診断（保存済みモデル・再学習不要）。

実行:
    python model_training/src/going_conditional_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from evaluation import calculate_ranking_metrics
from inference_common import _load_booster_crlf_safe, compute_market_log_odds, predict_model_probs
from pipeline_common import FEATURES_DIR, MODELS_DIR, load_config

RESULTS_DIR = MODELS_DIR / "ablation_latent"
OUT_PATH = RESULTS_DIR / "going_conditional_ranking.json"

CONDITION_LABELS = {1: "良", 2: "稍重", 3: "重", 4: "不良"}
GROUPS = {
    "all": lambda tc: pd.Series(True, index=tc.index),
    "soft": lambda tc: tc == 2,
    "heavy_plus": lambda tc: tc >= 3,
}


def _load_models(model_dir: Path, fold: int) -> list:
    paths = sorted(model_dir.glob(f"lgbm_binary_fold{fold}_seed*.txt"))
    if not paths:
        p = model_dir / f"lgbm_binary_fold{fold}.txt"
        paths = [p] if p.exists() else []
    return [_load_booster_crlf_safe(p) for p in paths]


def _prepare_df(parquet: str) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_DIR / parquet)
    df["race_date"] = pd.to_datetime(df.get("race_date", df.get("date")))
    if "market_log_odds" not in df.columns and "odds" in df.columns:
        df = compute_market_log_odds(df, odds_col="odds")
    return df


def _eval_variant(
    vname: str,
    parquet: str,
    model_dir: Path,
    fold_cfg: dict,
    base_margin: str | None,
) -> list[dict]:
    if not model_dir.exists():
        return []
    df_all = _prepare_df(parquet)
    fold = fold_cfg["fold"]
    test_df = df_all[
        (df_all["race_date"] >= pd.Timestamp(fold_cfg["test_start"]))
        & (df_all["race_date"] <= pd.Timestamp(fold_cfg["test_end"]))
    ].copy()
    models = _load_models(model_dir, fold)
    if not models:
        return []

    feat_cols = list(models[0].feature_name())
    test_df["model_prob"] = predict_model_probs(
        models, test_df, feat_cols, base_margin or None
    ).values
    tc = pd.to_numeric(test_df["track_condition_code"], errors="coerce")

    rows: list[dict] = []
    for group_name, mask_fn in GROUPS.items():
        for code in sorted(tc.dropna().unique()):
            code = int(code)
            sub = test_df[mask_fn(tc) & (tc == code)]
            if len(sub) < 100:
                continue
            m = calculate_ranking_metrics(sub)
            rows.append({
                "variant": vname,
                "fold": fold,
                "group": group_name,
                "track_condition_code": code,
                "condition_label": CONDITION_LABELS.get(code, str(code)),
                **m,
            })
        # グループ集計（稍重のみ / 重+不良）
        if group_name != "all":
            sub_g = test_df[mask_fn(tc)]
            if len(sub_g) >= 100:
                m = calculate_ranking_metrics(sub_g)
                rows.append({
                    "variant": vname,
                    "fold": fold,
                    "group": group_name,
                    "track_condition_code": None,
                    "condition_label": group_name,
                    **m,
                })
    return rows


def run_conditional_eval() -> dict:
    cfg = load_config()
    folds = cfg["training"]["walkforward_folds"]
    base_margin = cfg["training"].get("base_margin_col", "market_log_odds")

    variants = [
        ("baseline_v6", "features_v6.parquet", MODELS_DIR / "backup_baseline_v6", base_margin),
        ("v6_going_v1", "features_v6_going_v1.parquet", RESULTS_DIR / "models_v6_going_v1", base_margin),
        ("v6_going_v2", "features_v6_going_v2.parquet", RESULTS_DIR / "models_v6_going_v2", base_margin),
        ("v6_going_v1_v2", "features_v6_going_v1_v2.parquet", RESULTS_DIR / "models_v6_going_v1_v2", base_margin),
        ("v6_going_v3", "features_v6_going_v3.parquet", RESULTS_DIR / "models_v6_going_v3", base_margin),
        ("v6_no_market_going_v1", "features_v6_going_v1.parquet", RESULTS_DIR / "models_v6_no_market_going_v1", ""),
        ("v6_no_market_going_v2", "features_v6_going_v2.parquet", RESULTS_DIR / "models_v6_no_market_going_v2", ""),
        (
            "v6_no_market_going_v1_v2",
            "features_v6_going_v1_v2.parquet",
            RESULTS_DIR / "models_v6_no_market_going_v1_v2",
            "",
        ),
    ]

    all_rows: list[dict] = []
    for vname, parquet, model_dir, bm in variants:
        if not (FEATURES_DIR / parquet).exists():
            print(f"  [SKIP] {vname}: parquet missing")
            continue
        for fold_cfg in folds:
            if fold_cfg["fold"] != 1:
                continue  # F1 test のみ（診断用）
            rows = _eval_variant(vname, parquet, model_dir, fold_cfg, bm)
            all_rows.extend(rows)
            for r in rows:
                if r["track_condition_code"] is not None:
                    print(
                        f"  {vname} cond={r['condition_label']}: "
                        f"top3={r['top3_overlap_rate']:.1%} n={r['n_races']}"
                    )

    # baseline vs going の差分（F1・code別）
    summary: list[dict] = []
    base_f1 = [r for r in all_rows if r["variant"] == "baseline_v6" and r["fold"] == 1]
    for vname in (
        "v6_going_v1",
        "v6_going_v2",
        "v6_going_v1_v2",
        "v6_going_v3",
        "v6_no_market_going_v1",
        "v6_no_market_going_v2",
        "v6_no_market_going_v1_v2",
    ):
        var_f1 = [r for r in all_rows if r["variant"] == vname and r["fold"] == 1]
        for b in base_f1:
            if b["track_condition_code"] is None:
                continue
            v = next(
                (
                    x
                    for x in var_f1
                    if x["track_condition_code"] == b["track_condition_code"]
                    and x["group"] == b["group"]
                ),
                None,
            )
            if not v:
                continue
            summary.append({
                "condition": b["condition_label"],
                "code": b["track_condition_code"],
                "baseline_top3": b["top3_overlap_rate"],
                "variant": vname,
                "variant_top3": v["top3_overlap_rate"],
                "delta_top3": v["top3_overlap_rate"] - b["top3_overlap_rate"],
                "n_races": v["n_races"],
            })

    out = {"by_condition": all_rows, "delta_vs_baseline_f1": summary}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {OUT_PATH}")
    if summary:
        print("\n--- F1 top3 delta vs baseline (by condition) ---")
        for s in sorted(summary, key=lambda x: x["delta_top3"], reverse=True):
            print(
                f"  {s['variant']} {s['condition']}: "
                f"{s['baseline_top3']:.1%} -> {s['variant_top3']:.1%} "
                f"({s['delta_top3']:+.1%}) n={s['n_races']}"
            )
    return out


if __name__ == "__main__":
    run_conditional_eval()

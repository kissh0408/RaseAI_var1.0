"""features_v6 特徴量監査: 重要度・単変量信号・カテゴリ別ギャップ分析。

実行:
    python model_training/src/feature_audit.py
    python model_training/src/feature_audit.py --output model_training/models/feature_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from inference_common import _load_booster_crlf_safe
from pipeline_common import FEATURES_DIR, MODELS_DIR, load_config
from train import get_feature_cols

MODELS_BACKUP = MODELS_DIR / "backup_baseline_v6"
DEFAULT_OUT = MODELS_DIR / "feature_audit"

# ドメイン上「効くはず」のカテゴリ（強化・新規追加候補）
DOMAIN_PRIORITY_CATEGORIES = {
    "going_track",
    "pedigree",
    "course_geometry",
    "training",
    "moisture_cushion",
    "pace_pci",
}

FEATURE_CATEGORY: dict[str, str] = {
    "market_prob": "market",
    "jra_tm_score": "jra_tm",
    "jra_tm_rank": "jra_tm",
    "jra_tm_orthogonalized": "jra_tm",
    "jra_dm_rank": "jra_tm",
    "jra_dm_gap_to_best": "jra_tm",
    "jra_dm_uncertainty": "jra_tm",
    "sire_surface_win_rate": "pedigree",
    "bms_win_rate": "pedigree",
    "jockey_30d_win_rate": "human",
    "trainer_30d_win_rate": "human",
    "jockey_course_win_rate": "human",
    "last3_agari3f_mean": "horse_form",
    "last5_rank_mean": "horse_form",
    "last5_time_diff_mean": "horse_form",
    "career_win_rate": "horse_form",
    "ema_rank": "horse_form",
    "ema_time_diff": "horse_form",
    "rpr_score": "horse_form",
    "past_speed_index_mean": "horse_form",
    "agari_z_score": "horse_form",
    "distance_win_rate": "course_distance",
    "course_win_rate": "course_distance",
    "surface_win_rate": "course_distance",
    "gate_surface_cross": "course_distance",
    "course_last_straight_m": "course_geometry",
    "course_n_corners": "course_geometry",
    "draw_bias_score": "course_geometry",
    "surface_code": "course_distance",
    "distance": "course_distance",
    "track_condition_code": "going_track",
    "condition_win_rate": "going_track",
    "surface_cond_code": "going_track",
    "pci_past_mean": "pace_pci",
    "race_pci": "pace_pci",
    "corner_pos_mean": "pace_style",
    "corner_pos_last": "pace_style",
    "running_style_mode": "pace_style",
    "front_rate": "pace_style",
    "race_front_count": "pace_style",
    "race_front_ratio": "pace_style",
    "race_style_entropy": "pace_style",
    "weight_diff": "weight",
    "weight_diff_trend": "weight",
    "weight_relative_z": "weight",
    "carry_weight": "weight",
    "horse_age": "basic_race",
    "horse_sex_code": "basic_race",
    "horse_count": "basic_race",
    "gate_number": "basic_race",
    "class_change": "basic_race",
    "days_since_last_race": "basic_race",
    "summer_mare_sin": "seasonal",
    "summer_mare_cos": "seasonal",
}

CATEGORY_LABELS = {
    "market": "市場確率",
    "jra_tm": "JRA TM/DM",
    "pedigree": "血統",
    "human": "騎手・調教師",
    "horse_form": "馬フォーム",
    "course_distance": "コース・距離適性",
    "course_geometry": "コース形状",
    "going_track": "馬場状態",
    "pace_pci": "ペース(PCI)",
    "pace_style": "脚質・ペース構造",
    "weight": "馬体重",
    "basic_race": "レース基本",
    "seasonal": "季節",
    "other": "その他",
}

# v6 に未収載だが domain 上期待が高い列（潜在変量実験・仕様書より）
MISSING_HIGH_POTENTIAL = [
    {"name": "moisture_pct_goal", "category": "moisture_cushion", "reason": "含水率(ダート/芝)。R2でF1 ROI152%だが順位±0"},
    {"name": "moisture_available_flag", "category": "moisture_cushion", "reason": "高NaN列の利用可否フラグ"},
    {"name": "training_days_before_race", "category": "training", "reason": "調教間隔。単独でもgap悪化、信号は弱い"},
    {"name": "training_last1f_z", "category": "training", "reason": "施設Z。過学習要因の疑い"},
    {"name": "sire_dirt_win_rate", "category": "pedigree", "reason": "configでNaN97%除外。ダート血統適性"},
    {"name": "dam_sire_dirt_win_rate", "category": "pedigree", "reason": "configでNaN97%除外"},
    {"name": "horse_turf_heavy_win_rate", "category": "going_track", "reason": "v4系に存在。重馬場適性"},
    {"name": "horse_dirt_heavy_win_rate", "category": "going_track", "reason": "v4系に存在。ダート重適性"},
    {"name": "pci_past_mean_ortho_residual", "category": "pace_pci", "reason": "pci冗長性解消版。順位改善なし"},
]


def _categorize(feature: str) -> str:
    if feature in FEATURE_CATEGORY:
        return FEATURE_CATEGORY[feature]
    if feature.startswith("jra_"):
        return "jra_tm"
    if "pci" in feature:
        return "pace_pci"
    if "sire" in feature or "bms" in feature or "dam" in feature:
        return "pedigree"
    if "jockey" in feature or "trainer" in feature:
        return "human"
    if "weight" in feature or feature == "carry_weight":
        return "weight"
    if "course" in feature or feature in ("distance", "surface_code", "gate_surface_cross"):
        return "course_distance"
    if "condition" in feature or "track" in feature or "going" in feature:
        return "going_track"
    return "other"


def _load_ensemble_importance(models_dir: Path) -> pd.DataFrame:
    """全 seed × fold モデルの gain/split 重要度を平均。"""
    rows: list[dict] = []
    for path in sorted(models_dir.glob("lgbm_binary_fold*_seed*.txt")):
        stem = path.stem  # lgbm_binary_fold1_seed42
        parts = stem.replace("lgbm_binary_fold", "").split("_seed")
        fold = int(parts[0])
        seed = int(parts[1])
        booster = _load_booster_crlf_safe(path)
        names = list(booster.feature_name())
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        for feat, g, s in zip(names, gain, split):
            rows.append({"fold": fold, "seed": seed, "feature": feat, "gain": float(g), "split": float(s)})
    if not rows:
        raise FileNotFoundError(f"No models in {models_dir}")
    df = pd.DataFrame(rows)
    agg = df.groupby("feature", as_index=False).agg(
        gain_mean=("gain", "mean"),
        gain_std=("gain", "std"),
        split_mean=("split", "mean"),
        n_models=("gain", "count"),
    )
    total_gain = agg["gain_mean"].sum()
    agg["gain_pct"] = agg["gain_mean"] / total_gain if total_gain > 0 else 0.0
    agg["gain_rank"] = agg["gain_mean"].rank(ascending=False, method="min").astype(int)
    agg["category"] = agg["feature"].map(_categorize)
    return agg.sort_values("gain_mean", ascending=False).reset_index(drop=True)


def _univariate_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """学習期間のみ: NaN率・is_win/is_top3 との点相関（リーク防止）。"""
    is_win = (df["finish_rank"] == 1).astype(float)
    is_top3 = ((df["finish_rank"] >= 1) & (df["finish_rank"] <= 3)).astype(float)
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_rate = float(s.isna().mean())
        valid = s.notna()
        corr_win = float(s[valid].corr(is_win[valid])) if valid.sum() > 100 else float("nan")
        corr_top3 = float(s[valid].corr(is_top3[valid])) if valid.sum() > 100 else float("nan")
        rows.append({
            "feature": col,
            "nan_rate": nan_rate,
            "corr_is_win": corr_win,
            "corr_is_top3": corr_top3,
            "abs_corr_win": abs(corr_win) if np.isfinite(corr_win) else float("nan"),
            "abs_corr_top3": abs(corr_top3) if np.isfinite(corr_top3) else float("nan"),
        })
    return pd.DataFrame(rows)


def _category_summary(feat_df: pd.DataFrame) -> pd.DataFrame:
    g = feat_df.groupby("category", as_index=False).agg(
        n_features=("feature", "count"),
        gain_pct_sum=("gain_pct", "sum"),
        gain_pct_mean=("gain_pct", "mean"),
        abs_corr_win_mean=("abs_corr_win", "mean"),
    )
    g["category_label"] = g["category"].map(CATEGORY_LABELS).fillna(g["category"])
    g["domain_priority"] = g["category"].isin(DOMAIN_PRIORITY_CATEGORIES)
    return g.sort_values("gain_pct_sum", ascending=False)


def _find_underutilized(feat_df: pd.DataFrame) -> list[dict]:
    """単変量信号はあるがモデル重要度が低い / ドメイン優先カテゴリが弱い。"""
    median_gain = feat_df["gain_pct"].median()
    findings: list[dict] = []

    for _, row in feat_df.iterrows():
        cat = row["category"]
        gain_pct = row["gain_pct"]
        abs_corr = row.get("abs_corr_win", float("nan"))
        if not np.isfinite(abs_corr):
            continue

        # 信号あり & 重要度低 → 交互作用不足・init_score に飲まれている疑い
        if abs_corr >= 0.04 and gain_pct < median_gain * 0.5:
            findings.append({
                "type": "signal_not_used",
                "feature": row["feature"],
                "category": cat,
                "gain_pct": round(float(gain_pct), 4),
                "abs_corr_win": round(float(abs_corr), 4),
                "note": "単変量相関は中程度だが木の gain が低い。交互作用列 or 正規化不足の疑い",
            })

        # ドメイン優先カテゴリで gain 下位
        if cat in DOMAIN_PRIORITY_CATEGORIES and gain_pct < 0.008 and row["gain_rank"] > len(feat_df) * 0.6:
            findings.append({
                "type": "domain_underweight",
                "feature": row["feature"],
                "category": cat,
                "gain_pct": round(float(gain_pct), 4),
                "gain_rank": int(row["gain_rank"]),
                "note": "馬場/血統/コース等で効くはずだが重要度下位",
            })

    # 冗長・低信号候補（削除/pruning 候補）
    for _, row in feat_df.iterrows():
        if row["gain_pct"] < 0.005 and row.get("abs_corr_win", 0) < 0.02:
            findings.append({
                "type": "prune_candidate",
                "feature": row["feature"],
                "category": row["category"],
                "gain_pct": round(float(row["gain_pct"]), 4),
                "abs_corr_win": round(float(row.get("abs_corr_win", float("nan"))), 4),
                "note": "重要度・相関とも低い。v6 pruning 候補",
            })

    return findings


def _recommend_actions(feat_df: pd.DataFrame, cat_df: pd.DataFrame) -> list[dict]:
    actions: list[dict] = []
    going = cat_df[cat_df["category"] == "going_track"]
    pedigree = cat_df[cat_df["category"] == "pedigree"]
    pace = cat_df[cat_df["category"] == "pace_pci"]

    if len(going) and going.iloc[0]["gain_pct_sum"] < 0.05:
        actions.append({
            "priority": 1,
            "action": "going_track強化",
            "detail": "track_condition_code/condition_win_rate の gain 合計が低い。"
            "condition_win_rate の Bayes 平滑・馬場別 win_rate 列の追加を data-generator で検討",
            "target_features": ["condition_win_rate", "surface_cond_code", "track_condition_code"],
        })

    if len(pedigree) and pedigree.iloc[0]["gain_pct_sum"] < 0.03:
        actions.append({
            "priority": 2,
            "action": "血統列の再設計",
            "detail": "sire_surface_win_rate/bms_win_rate のみで signal 弱。"
            "高NaN除外列(sire_dirt等)の impute 版 or 距離帯別 sire 適性",
            "target_features": ["sire_surface_win_rate", "bms_win_rate"],
        })

    pci_feats = feat_df[feat_df["category"] == "pace_pci"]
    if len(pci_feats) >= 2:
        top_pci = pci_feats.iloc[0]["feature"]
        actions.append({
            "priority": 3,
            "action": "PCI冗長整理",
            "detail": f"pci_past_mean と race_pci が共存。{top_pci} 以外の pruning または直交化1列化",
            "target_features": pci_feats["feature"].tolist(),
        })

    low_human = feat_df[(feat_df["category"] == "human") & (feat_df["gain_pct"] < 0.01)]
    if len(low_human) >= 2:
        actions.append({
            "priority": 4,
            "action": "騎手・調教師列の統合",
            "detail": "jockey/trainer 3列の gain 分散。30d + course を1列に集約するか交互作用のみ残す",
            "target_features": low_human["feature"].tolist(),
        })

    top10 = feat_df.head(10)["feature"].tolist()
    actions.append({
        "priority": 0,
        "action": "現状の主軸を維持",
        "detail": f"上位10列: {', '.join(top10)}。市場・フォーム・TM が主信号",
        "target_features": top10,
    })

    return sorted(actions, key=lambda x: x["priority"])


def run_audit(output_dir: Path) -> dict:
    cfg = load_config()
    feature_cols = get_feature_cols(cfg)
    parquet = cfg["training"].get("backtest_feature_file", "features_v6.parquet")
    df = pd.read_parquet(FEATURES_DIR / parquet)

    if "race_date" not in df.columns:
        df["race_date"] = pd.to_datetime(df["date"])
    else:
        df["race_date"] = pd.to_datetime(df["race_date"])

    # 学習期間のみで単変量統計（Fold1 train_end 以前）
    train_end = pd.Timestamp(cfg["training"]["walkforward_folds"][0]["train_end"])
    train_df = df[df["race_date"] <= train_end].copy()

    print("=== features_v6 特徴量監査 ===")
    print(f"  parquet: {parquet}, features: {len(feature_cols)}, train rows: {len(train_df)}")

    imp = _load_ensemble_importance(MODELS_BACKUP)
    uni = _univariate_stats(train_df, feature_cols)
    feat_df = imp.merge(uni, on="feature", how="left")
    feat_df["category"] = feat_df["feature"].map(_categorize)
    cat_df = _category_summary(feat_df)
    findings = _find_underutilized(feat_df)
    actions = _recommend_actions(feat_df, cat_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_dir / "feature_importance_detail.csv", index=False, encoding="utf-8-sig")
    cat_df.to_csv(output_dir / "category_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "parquet": parquet,
        "n_features": len(feature_cols),
        "n_models": int(imp["n_models"].iloc[0]) if len(imp) else 0,
        "top10_features": feat_df.head(10)[["feature", "gain_pct", "category"]].to_dict("records"),
        "category_summary": cat_df.to_dict("records"),
        "underutilized_findings": findings,
        "missing_high_potential": MISSING_HIGH_POTENTIAL,
        "recommended_actions": actions,
    }
    (output_dir / "feature_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n--- Top 15 重要度 (gain) ---")
    for _, row in feat_df.head(15).iterrows():
        print(
            f"  {int(row['gain_rank']):2d}. {row['feature']:<28} "
            f"gain={row['gain_pct']:.1%} corr_win={row.get('abs_corr_win', float('nan')):.3f} "
            f"[{CATEGORY_LABELS.get(row['category'], row['category'])}]"
        )

    print("\n--- カテゴリ別 gain 合計 ---")
    for _, row in cat_df.iterrows():
        pri = "*" if row["domain_priority"] else " "
        print(
            f"  {pri} {row['category_label']:<12} "
            f"cols={int(row['n_features']):2d} gain={row['gain_pct_sum']:.1%} "
            f"avg_corr={row['abs_corr_win_mean']:.3f}"
        )

    print("\n--- 効いていないはずの信号 (domain_underweight / signal_not_used) ---")
    for f in findings:
        if f["type"] in ("domain_underweight", "signal_not_used"):
            print(f"  [{f['type']}] {f['feature']}: {f['note']}")

    print("\n--- pruning 候補 (gain<0.5% & |corr|<0.02) ---")
    prune = [f for f in findings if f["type"] == "prune_candidate"]
    print(f"  {len(prune)} cols: {', '.join(p['feature'] for p in prune[:12])}{'...' if len(prune)>12 else ''}")

    print("\n--- 推奨アクション ---")
    for a in actions:
        if a["priority"] == 0:
            continue
        print(f"  P{a['priority']}: {a['action']} — {a['detail'][:80]}")

    print(f"\n保存: {output_dir}")
    return report


PRUNE_V1_COLS = [
    "class_change",
    "surface_code",
    "course_n_corners",
    "track_condition_code",
    "race_front_count",
    "surface_cond_code",
    "race_pci",
]


def build_prune_v1_parquet(output_name: str = "features_v6_prune_v1.parquet") -> pd.DataFrame:
    """監査で特定した低 gain 7列を除外した v6。"""
    base = pd.read_parquet(FEATURES_DIR / "features_v6.parquet")
    out = base.drop(columns=[c for c in PRUNE_V1_COLS if c in base.columns], errors="ignore")
    path = FEATURES_DIR / output_name
    out.to_parquet(path, index=False)
    print(f"  {output_name}: {base.shape[1]} -> {out.shape[1]} cols, dropped {PRUNE_V1_COLS}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="features_v6 特徴量監査")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_audit(args.output)


if __name__ == "__main__":
    main()

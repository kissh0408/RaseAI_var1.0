"""Phase B (v30_relative) 後退原因の定量分析。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT_ROOT, get_feature_cols, load_config
from evaluate import ensemble_predict

FEAT_DIR = PROJECT_ROOT / "pure_rank/data/02_features"
V29_PATH = FEAT_DIR / "features_v29_fixed.parquet"
if not V29_PATH.exists():
    V29_PATH = FEAT_DIR / "features_v29_fixed_backup.parquet"
V30_PATH = FEAT_DIR / "features_v30_relative.parquet"

NEW_COLS = [
    "field_z_time_dev", "field_z_prize", "field_z_last3f", "field_z_win_rate",
    "field_z_speed_idx", "field_z_place_rate", "field_front_runner_density",
    "relative_post_position", "hist_front_running_pref",
]

PAIRS = [
    ("field_z_win_rate", "win_rate_vs_field"),
    ("field_z_win_rate", "hist_win_rate"),
    ("field_z_prize", "prize_vs_field"),
    ("field_z_prize", "hist_total_prize"),
    ("field_z_time_dev", "hist_last_time_dev"),
    ("field_z_speed_idx", "hist_speed_idx_avg3"),
    ("field_z_place_rate", "hist_place_rate"),
    ("field_z_last3f", "hist_last_last3f"),
    ("relative_post_position", "wakuban"),
    ("relative_post_position", "wakuban_surface"),
    ("field_z_win_rate", "trn_hc_zscore_3f"),
]


def top_gain(model_dir: Path, n: int = 15) -> tuple[list, float, dict]:
    imp: dict[str, float] = {}
    for p in sorted(model_dir.glob("lambdarank_fold*_seed*.txt")):
        m = lgb.Booster(model_file=str(p))
        for nm, g in zip(m.feature_name(), m.feature_importance(importance_type="gain")):
            imp[nm] = imp.get(nm, 0) + g
    total = sum(imp.values())
    top = sorted(imp.items(), key=lambda x: -x[1])[:n]
    return top, total, imp


def main() -> None:
    print("=== 1. Feature redundancy: correlation analysis ===")
    df29 = pd.read_parquet(V29_PATH)
    df30 = pd.read_parquet(V30_PATH)
    valid_end = pd.Timestamp("2024-12-31")
    test30 = df30[df30["race_date"] > valid_end].copy()
    test29 = df29[df29["race_date"] > valid_end].copy()

    print("Pairwise |r| (test set):")
    for a, b in PAIRS:
        if a in test30.columns and b in test30.columns:
            sub = test30[[a, b]].dropna()
            r = sub[a].corr(sub[b])
            print(f"  {a:30s} vs {b:25s}: r={r:+.4f}  n={len(sub):,}")

    race_std = test30.groupby("race_id")["hist_win_rate"].transform("std")
    reconstructed = test30["win_rate_vs_field"] / (race_std + 1e-6)
    mask = test30["field_z_win_rate"].notna() & reconstructed.notna()
    if mask.sum() > 0:
        r_recon = test30.loc[mask, "field_z_win_rate"].corr(reconstructed[mask])
        print(f"  field_z_win_rate vs win_rate_vs_field/std: r={r_recon:+.4f} (math duplicate)")

    print("\n=== 2. Within-race rank correlation (500 races) ===")
    sample_races = test30["race_id"].drop_duplicates().sample(500, random_state=42)
    acc: dict[str, list[float]] = defaultdict(list)
    rank_specs = [
        ("field_z_win_rate", "hist_win_rate", False),
        ("field_z_win_rate", "win_rate_vs_field", False),
        ("field_z_time_dev", "hist_last_time_dev", True),
        ("field_z_prize", "prize_vs_field", False),
        ("field_z_speed_idx", "hist_speed_idx_avg3", False),
    ]
    for rid in sample_races:
        g = test30[test30["race_id"] == rid]
        if len(g) < 5:
            continue
        for z, base, asc_base in rank_specs:
            if z not in g.columns or base not in g.columns:
                continue
            a = g[z].rank(ascending=False)
            b = g[base].rank(ascending=asc_base)
            if a.notna().sum() >= 3 and b.notna().sum() >= 3:
                r, _ = spearmanr(a, b, nan_policy="omit")
                if not np.isnan(r):
                    acc[f"{z} vs {base}"].append(r)
    for k, v in sorted(acc.items()):
        print(f"  {k:45s}: {np.mean(v):+.4f}")

    print("\n=== 3. Feature importance shift ===")
    v29_dir = PROJECT_ROOT / "pure_rank/models_backup_v29_fixed_label63"
    v30_dir = PROJECT_ROOT / "pure_rank/models"
    top29, tot29, imp29 = top_gain(v29_dir)
    top30, tot30, imp30 = top_gain(v30_dir)

    print("v29 top-10:")
    for nm, g in top29[:10]:
        print(f"  {nm:35s} {g / tot29 * 100:5.2f}%")
    print("v30 top-10:")
    for nm, g in top30[:10]:
        print(f"  {nm:35s} {g / tot30 * 100:5.2f}%")

    new_gain = sum(imp30.get(c, 0) for c in NEW_COLS)
    print(f"\nNew v30 features gain share: {new_gain / tot30 * 100:.2f}%")
    print("v29 top features importance change:")
    for nm, g in top29[:8]:
        s29 = g / tot29 * 100
        s30 = imp30.get(nm, 0) / tot30 * 100
        print(f"  {nm:35s} {s29:5.2f}% -> {s30:5.2f}%  ({s30 - s29:+.2f}pp)")

    print("\n=== 4. Paired prediction analysis ===")
    cfg = load_config()
    cfg29 = json.loads(json.dumps(cfg))
    cfg29["data"]["features_version"] = "v29_fixed"
    fc29 = get_feature_cols(test29, cfg29)
    fc30 = get_feature_cols(test30, cfg)

    m29 = [lgb.Booster(model_file=str(p)) for p in sorted(v29_dir.glob("*.txt"))]
    m30 = [lgb.Booster(model_file=str(p)) for p in sorted(v30_dir.glob("*.txt"))]
    test29 = test29.copy()
    test30 = test30.copy()
    test29["pred"] = ensemble_predict(m29, test29[fc29])
    test30["pred"] = ensemble_predict(m30, test30[fc30])

    top1_29 = top1_30 = disagree = v29_only = v30_only = 0
    for rid in test29["race_id"].unique():
        g29 = test29[test29["race_id"] == rid]
        g30 = test30[test30["race_id"] == rid]
        i29 = g29["pred"].idxmax()
        i30 = g30["pred"].idxmax()
        hit29 = int(g29.loc[i29, "finish_rank"] == 1)
        hit30 = int(g30.loc[i30, "finish_rank"] == 1)
        top1_29 += hit29
        top1_30 += hit30
        if i29 != i30:
            disagree += 1
            if hit29 and not hit30:
                v29_only += 1
            elif hit30 and not hit29:
                v30_only += 1

    n = test29["race_id"].nunique()
    print(f"  Races: {n}")
    print(f"  Top-1 v29: {top1_29 / n * 100:.2f}%  v30: {top1_30 / n * 100:.2f}%")
    print(f"  Pred disagreements: {disagree} ({disagree / n * 100:.1f}%)")
    print(f"  v29-only hits (discordant): {v29_only}")
    print(f"  v30-only hits (discordant): {v30_only}")
    if v29_only + v30_only > 0:
        p = binomtest(min(v29_only, v30_only), v29_only + v30_only, 0.5).pvalue
        print(f"  McNemar p-value: {p:.4f}  (>{0.05} = not significant)")

    print("\n=== 5. z-score instability ===")
    race_std_wr = test30.groupby("race_id")["hist_win_rate"].std()
    low_var = (race_std_wr < 0.01).sum()
    print(f"  Races with hist_win_rate std<0.01: {low_var} ({low_var / len(race_std_wr) * 100:.1f}%)")
    print(f"  field_z_win_rate NaN rate: {test30['field_z_win_rate'].isna().mean() * 100:.1f}%")

    print("\n=== 6. Feature count / tree budget dilution ===")
    print(f"  v29 feature cols: {len(fc29)}")
    print(f"  v30 feature cols: {len(fc30)}")
    print(f"  Added: {len(fc30) - len(fc29)} features")


if __name__ == "__main__":
    main()

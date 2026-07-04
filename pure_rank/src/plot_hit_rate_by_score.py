"""
plot_hit_rate_by_score.py — 予測スコア区間ごとの的中率・回収率グラフ

単勝: 馬ごと p_win に 100円ずつ購入
ワイド: レースごと Harville 最大 P_wide 組に 100円
馬連: レースごと予測1-2位に 100円
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT_ROOT, get_feature_cols, load_config
from evaluate import ensemble_predict, load_models
from predict import _best_wide_pair, compute_race_probabilities

N_BINS = 10
STAKE = 100.0
OUT_DIR = PROJECT_ROOT / "pure_rank" / "data" / "02_features" / "charts"


def _normalize_pair(h1: int, h2: int) -> tuple[int, int]:
    return min(h1, h2), max(h1, h2)


def _build_win_lookup(hr_df: pd.DataFrame) -> dict[str, dict[int, int]]:
    lookup: dict[str, dict[int, int]] = {}
    sub = hr_df[hr_df["bet_type"] == "win"]
    for _, row in sub.iterrows():
        rid = str(row["race_id"])
        h = int(row["horse_num_1"])
        lookup.setdefault(rid, {})[h] = int(row["payout"])
    return lookup


def _build_pair_lookup(hr_df: pd.DataFrame, bet_type: str) -> dict[str, dict[tuple[int, int], int]]:
    lookup: dict[str, dict[tuple[int, int], int]] = {}
    sub = hr_df[hr_df["bet_type"] == bet_type]
    for _, row in sub.iterrows():
        rid = str(row["race_id"])
        key = _normalize_pair(int(row["horse_num_1"]), int(row["horse_num_2"]))
        lookup.setdefault(rid, {})[key] = int(row["payout"])
    return lookup


def _bin_stats(
    values: np.ndarray,
    hits: np.ndarray,
    payouts: np.ndarray,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """等頻度ビンで的中率・回収率を集計する。"""
    df = pd.DataFrame({"score": values, "hit": hits.astype(float), "payout": payouts.astype(float)})
    df = df[np.isfinite(df["score"])]
    if len(df) == 0:
        return pd.DataFrame()
    df["bin"] = pd.qcut(df["score"], q=n_bins, duplicates="drop")
    agg = df.groupby("bin", observed=True).agg(
        score_mid=("score", "mean"),
        hit_rate=("hit", "mean"),
        count=("hit", "count"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        payout_sum=("payout", "sum"),
    ).reset_index()
    agg["return_rate"] = agg["payout_sum"] / (agg["count"] * STAKE)
    agg["hits"] = (agg["hit_rate"] * agg["count"]).round().astype(int)
    return agg


def _plot_calibration(
    agg: pd.DataFrame,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = agg["score_mid"].values
    hit_pct = agg["hit_rate"].values * 100
    ret_pct = agg["return_rate"].values * 100
    counts = agg["count"].values
    n = len(x)
    width = 0.38
    idx = np.arange(n)

    ax1.bar(idx - width / 2, hit_pct, width, alpha=0.8, color="#2563eb", label="的中率")
    ax1.bar(idx + width / 2, ret_pct, width, alpha=0.8, color="#16a34a", label="回収率")
    ax1.plot(idx, x * 100, "r--", linewidth=1.2, label="完全キャリブレーション")
    ax1.axhline(100, color="#94a3b8", linestyle=":", linewidth=1, label="回収100%")
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"{v:.3f}" for v in x], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("率 (%)")
    ax1.set_title(title)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(idx, counts, "o-", color="#64748b", linewidth=1, markersize=4, label="票数")
    ax2.set_ylabel("票数", color="#64748b")
    ax2.tick_params(axis="y", labelcolor="#64748b")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    for i in range(n):
        ax1.text(
            i - width / 2, hit_pct[i] + 1,
            f"{hit_pct[i]:.1f}%\n({int(agg['hits'].iloc[i])}票)",
            ha="center", va="bottom", fontsize=6, color="#1e40af",
        )
        ax1.text(
            i + width / 2, ret_pct[i] + 1,
            f"{ret_pct[i]:.0f}%",
            ha="center", va="bottom", fontsize=6, color="#15803d",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def collect_win_records(
    df_test: pd.DataFrame,
    preds: np.ndarray,
    T: float,
    win_lookup: dict[str, dict[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """馬単位: p_win / 1着的中 / 単勝払戻（100円購入）。"""
    df = df_test.copy()
    df["pred_score"] = preds
    scores_list: list[float] = []
    hits_list: list[float] = []
    payouts_list: list[float] = []

    for race_id, grp in df.groupby("race_id"):
        rid = str(race_id)
        g = grp.sort_values("pred_score", ascending=False)
        raw = g["pred_score"].values
        probs = compute_race_probabilities(raw, T)
        p_win = probs["p_win"]
        horses = g["horse_num"].astype(int).values
        ranks = g["finish_rank"].values
        race_win = win_lookup.get(rid, {})
        for i, horse in enumerate(horses):
            hit = float(ranks[i] == 1)
            payout = float(race_win.get(int(horse), 0)) if hit else 0.0
            scores_list.append(p_win[i])
            hits_list.append(hit)
            payouts_list.append(payout)

    return np.array(scores_list), np.array(hits_list), np.array(payouts_list)


def collect_wide_records(
    df_test: pd.DataFrame,
    preds: np.ndarray,
    T: float,
    wide_lookup: dict[str, dict[tuple[int, int], int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """レース単位: max P_wide / ワイド的中 / 払戻。"""
    df = df_test.copy()
    df["pred_score"] = preds
    scores_list: list[float] = []
    hits_list: list[float] = []
    payouts_list: list[float] = []

    for race_id, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        g = grp.sort_values("pred_score", ascending=False).reset_index(drop=True)
        raw = g["pred_score"].values
        horses = g["horse_num"].astype(int).values
        probs = compute_race_probabilities(raw, T)
        hi, hj = _best_wide_pair(probs["wide_matrix"])
        p_wide = probs["wide_matrix"][hi, hj]
        key = _normalize_pair(int(horses[hi]), int(horses[hj]))
        payout = float(wide_lookup.get(rid, {}).get(key, 0))
        hit = float(payout > 0)
        scores_list.append(p_wide)
        hits_list.append(hit)
        payouts_list.append(payout)

    return np.array(scores_list), np.array(hits_list), np.array(payouts_list)


def collect_quinella_records(
    df_test: pd.DataFrame,
    preds: np.ndarray,
    T: float,
    quin_lookup: dict[str, dict[tuple[int, int], int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """レース単位: pred1-2 P_quinella / 馬連的中 / 払戻。"""
    df = df_test.copy()
    df["pred_score"] = preds
    scores_list: list[float] = []
    hits_list: list[float] = []
    payouts_list: list[float] = []

    for race_id, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        g = grp.sort_values("pred_score", ascending=False).reset_index(drop=True)
        raw = g["pred_score"].values
        horses = g["horse_num"].astype(int).values
        probs = compute_race_probabilities(raw, T)
        p_quin = probs["quinella_matrix"][0, 1]
        key = _normalize_pair(int(horses[0]), int(horses[1]))
        payout = float(quin_lookup.get(rid, {}).get(key, 0))
        hit = float(payout > 0)
        scores_list.append(p_quin)
        hits_list.append(hit)
        payouts_list.append(payout)

    return np.array(scores_list), np.array(hits_list), np.array(payouts_list)


def main() -> None:
    cfg = load_config()
    T = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))
    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    hr_path = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "HR_preprocessed.parquet"
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]

    if not hr_path.exists():
        raise FileNotFoundError(f"HR_preprocessed.parquet がありません: {hr_path}")

    print(f"Loading {feat_path} ...")
    df = pd.read_parquet(feat_path)
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])
    df_test = df[df["race_date"] > valid_end].copy()
    print(f"Test: {df_test['race_id'].nunique():,} races, {len(df_test):,} rows, T={T}")

    hr_df = pd.read_parquet(hr_path)
    test_race_ids = set(df_test["race_id"].astype(str))
    hr_df = hr_df[hr_df["race_id"].astype(str).isin(test_race_ids)]
    print(f"HR rows (test period): {len(hr_df):,}")

    win_lookup = _build_win_lookup(hr_df)
    wide_lookup = _build_pair_lookup(hr_df, "wide")
    quin_lookup = _build_pair_lookup(hr_df, "quinella")

    feature_cols = get_feature_cols(df_test, cfg)
    models = load_models(models_dir)
    preds = ensemble_predict(models, df_test[feature_cols])

    print("Collecting records...")
    specs = [
        (*collect_win_records(df_test, preds, T, win_lookup),
         "単勝（馬ごと p_win × 100円）", "予測勝率 p_win", "hit_rate_win.png"),
        (*collect_wide_records(df_test, preds, T, wide_lookup),
         "ワイド（max P_wide × 100円/レース）", "予測ワイド確率 P_wide", "hit_rate_wide.png"),
        (*collect_quinella_records(df_test, preds, T, quin_lookup),
         "馬連（pred1-2 × 100円/レース）", "予測馬連確率 P_quinella", "hit_rate_quinella.png"),
    ]

    summary: dict = {
        "T_opt": T,
        "n_bins": N_BINS,
        "stake_yen": STAKE,
        "test_races": int(df_test["race_id"].nunique()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scores, hits, payouts, title, xlabel, fname in specs:
        agg = _bin_stats(scores, hits, payouts)
        if agg.empty:
            print(f"[WARN] No data for {title}")
            continue
        _plot_calibration(agg, title, xlabel, OUT_DIR / fname)
        key = fname.replace(".png", "")
        summary[key] = [
            {
                "score_mid": float(r["score_mid"]),
                "score_min": float(r["score_min"]),
                "score_max": float(r["score_max"]),
                "hit_rate": float(r["hit_rate"]),
                "return_rate": float(r["return_rate"]),
                "count": int(r["count"]),
                "hits": int(r["hits"]),
                "payout_sum": float(r["payout_sum"]),
            }
            for _, r in agg.iterrows()
        ]
        total_ret = agg["payout_sum"].sum() / (agg["count"].sum() * STAKE)
        print(f"  [{key}] overall return_rate={total_ret*100:.1f}%")

    summary_path = OUT_DIR / "hit_rate_by_score_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()

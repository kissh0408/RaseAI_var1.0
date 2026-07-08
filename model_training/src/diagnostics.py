"""全条件診断スクリプト: 馬場・距離・競馬場・オッズ帯・月別・グレード別のROI弱点を洗い出す。

注意: condition_ev_overrides は意図的に適用しない。
このスクリプトの目的はオーバーライド対象となる弱点条件の発見であり、
適用すると既知の弱点が分析から隠れてしまうため。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from backtest import load_ensemble_models
from ev_calculator import apply_ev_filters, enrich_predictions
from kelly_sizer import apply_kelly_sizing
from pipeline_common import FEATURES_DIR, get_db_connection, load_config
from train import compute_base_margin, get_feature_cols


def build_predictions() -> pd.DataFrame:
    cfg = load_config()
    t_cfg = cfg["training"]

    # backtest.py と同一の特徴量解決順を採用する。
    # 診断は binary 検証系（57 feature_cols）の弱点を見るものなので、
    # rank/本番が掴む feature_file（features_past_v25_odds, race_date 列なし）ではなく
    # backtest_feature_file（features_v6, race_date を素で持つ）を最優先で読む。
    # これを揃えないと race_date が無く KeyError でクラッシュする。
    feat_path = None
    backtest_feature_file = t_cfg.get("backtest_feature_file")
    if backtest_feature_file:
        p = FEATURES_DIR / backtest_feature_file
        if p.exists():
            feat_path = p
    if feat_path is None:
        feature_file = t_cfg.get("feature_file")
        if feature_file:
            p = FEATURES_DIR / feature_file
            if p.exists():
                feat_path = p
    if feat_path is None:
        for ver in ["v6", "v4", "v3"]:
            p = FEATURES_DIR / f"features_{ver}.parquet"
            if p.exists():
                feat_path = p
                break
    if feat_path is None:
        raise FileNotFoundError(
            "feature parquet が見つかりません。train_config.json の "
            "training.backtest_feature_file / feature_file を確認してください。"
        )
    print(f"  {feat_path.name} 読み込み中...")
    df = pd.read_parquet(feat_path)

    # スキーマ整合（backtest.py と同一ロジック）。features_v6 は race_date を素で持つが、
    # 後方互換で active feature_file に落ちた場合でも date / year+month_day から導出する。
    if "race_date" not in df.columns:
        if "date" in df.columns:
            df["race_date"] = pd.to_datetime(df["date"])
        elif "year" in df.columns and "month_day" in df.columns:
            # month_day は MDD 形式の int（例: 104 -> 1月4日）。year と合成して日付化
            md = df["month_day"].astype(int)
            df["race_date"] = pd.to_datetime(
                {
                    "year": df["year"].astype(int),
                    "month": md // 100,
                    "day": md % 100,
                },
                errors="coerce",
            )
        else:
            raise KeyError("race_date を導出できません（date / year+month_day が無い）")
    else:
        df["race_date"] = pd.to_datetime(df["race_date"])

    conn = get_db_connection()
    odds_df = pd.read_sql_query(
        "SELECT race_id, horse_id, odds FROM SE WHERE finish_rank >= 0", conn
    )
    ra_df = pd.read_sql_query(
        "SELECT race_id, course_code, grade_code FROM RA", conn
    )
    conn.close()

    df = df.merge(odds_df, on=["race_id", "horse_id"], how="left", suffixes=("_feat", ""))
    if "odds_feat" in df.columns:
        df["odds"] = df["odds"].combine_first(df["odds_feat"])
        df = df.drop(columns=["odds_feat"], errors="ignore")
    df = df.merge(ra_df, on="race_id", how="left", suffixes=("", "_ra"))
    for col in ["course_code", "grade_code"]:
        if f"{col}_ra" in df.columns:
            df[col] = df[col].combine_first(df[f"{col}_ra"])
            df = df.drop(columns=[f"{col}_ra"], errors="ignore")

    df["is_win"] = (df["finish_rank"] == 1).astype(int)
    feature_cols = get_feature_cols(cfg)
    base_margin_col = t_cfg.get("base_margin_col")

    # フォールドのテスト期間は config に従う（backtest.py と同一の分割）
    folds = [
        (fc["fold"], fc["test_start"], fc["test_end"])
        for fc in t_cfg["walkforward_folds"]
    ]

    all_parts = []
    for fold, ts, te in folds:
        fd = df[(df["race_date"] >= ts) & (df["race_date"] <= te)].copy().reset_index(drop=True)
        # backtest.py と同じシードアンサンブルで予測する（本番構成との一致が診断の前提）
        models = load_ensemble_models(fold)
        avail = [c for c in feature_cols if c in fd.columns]

        X = fd[avail].values
        raw = np.mean(
            [m.predict(X, num_iteration=m.best_iteration, raw_score=True) for m in models],
            axis=0,
        )
        margin = compute_base_margin(fd, base_margin_col)
        final = 1.0 / (1.0 + np.exp(-np.clip(raw + margin, -20, 20)))

        res = np.empty(len(fd))
        for _, grp in fd.groupby("race_id"):
            idx = grp.index.tolist()
            p = final[idx].clip(1e-7, 1.0)
            res[idx] = p / p.sum()

        fd["model_prob"] = res
        fd = enrich_predictions(fd, model_prob_col="model_prob", odds_col="odds")
        fd = apply_kelly_sizing(
            fd, bankroll=10_000_000,
            kelly_frac=t_cfg["kelly_fraction"],
            max_bet_ratio=t_cfg["max_bet_ratio"],
        )
        # backtest.py の base_mask と同一基準（ev_rate >= 閾値）でベット選択する
        fd["is_recommended"] = apply_ev_filters(
            fd,
            ev_threshold=t_cfg["ev_threshold"],
            min_odds=t_cfg["min_odds"],
            max_odds=t_cfg["max_odds"],
            min_model_prob=t_cfg["min_model_prob"],
        ) & (fd["kelly_bet_yen"] > 0)
        fd["fold"] = fold
        all_parts.append(fd)

    return pd.concat(all_parts, ignore_index=True)


def roi_stats(sub: pd.DataFrame, label: str) -> dict | None:
    n = len(sub)
    if n == 0:
        return None
    hits = sub["is_win"].sum()
    pay = (sub["is_win"] * sub["odds"] * sub["kelly_bet_yen"]).sum()
    cost = sub["kelly_bet_yen"].sum()
    roi = pay / cost if cost > 0 else 0.0
    return {"label": label, "n": n, "hits": int(hits), "hit_rate": hits / n, "roi": roi}


def print_section(title: str, rows: list[dict], min_n: int = 10) -> None:
    rows = [r for r in rows if r and r["n"] >= min_n]
    if not rows:
        return
    print(f"--- {title} ---")
    for r in sorted(rows, key=lambda x: x["roi"]):
        flag = "X" if r["roi"] < 0.95 else ("!" if r["roi"] < 1.05 else "o")
        print(
            f"  [{flag}] {r['label']:<25} n={r['n']:>4}  "
            f"的中率={r['hit_rate']:.1%}  ROI={r['roi']:.1%}"
        )
    print()


def run_diagnostics() -> None:
    print("予測データ構築中...")
    all_df = build_predictions()
    bets = all_df[all_df["is_recommended"]].copy()
    bets["month"] = bets["race_date"].dt.month

    total_cost = bets["kelly_bet_yen"].sum()
    total_pay = (bets["is_win"] * bets["odds"] * bets["kelly_bet_yen"]).sum()
    overall_roi = total_pay / total_cost if total_cost > 0 else 0

    print(f"\n{'='*65}")
    print(f"全条件診断レポート（ev>{load_config()['training']['ev_threshold']}, 3フォールド合算）")
    print(f"{'='*65}")
    print(f"総ベット数: {len(bets)}件  全体ROI: {overall_roi:.1%}")
    print()

    # 1. フォールド別
    print_section("フォールド別", [
        roi_stats(bets[bets["fold"] == f], f"Fold{f}") for f in sorted(bets["fold"].unique())
    ], min_n=1)

    # 2. 馬場状態
    cond_map = {1: "良", 2: "稍重", 3: "重", 4: "不良"}
    print_section("馬場状態", [
        roi_stats(bets[bets["track_condition_code"] == c], f"馬場:{n}") for c, n in cond_map.items()
    ])

    # 3. コース種別
    surface_map = {1: "芝", 2: "ダート"}
    print_section("コース種別", [
        roi_stats(bets[bets["surface_code"] == c], f"コース:{n}") for c, n in surface_map.items()
    ])

    # 4. 芝×馬場状態
    rows = []
    for sc, sn in surface_map.items():
        for cc, cn in cond_map.items():
            sub = bets[(bets["surface_code"] == sc) & (bets["track_condition_code"] == cc)]
            rows.append(roi_stats(sub, f"{sn}×{cn}"))
    print_section("コース×馬場状態", rows)

    # 5. 距離帯
    dist_bins = [
        (0, 1200, "短距離 -1200m"),
        (1201, 1600, "マイル 1201-1600m"),
        (1601, 2000, "中距離 1601-2000m"),
        (2001, 9999, "長距離 2001m+"),
    ]
    print_section("距離帯", [
        roi_stats(bets[(bets["distance"] >= lo) & (bets["distance"] <= hi)], name)
        for lo, hi, name in dist_bins
    ])

    # 6. 頭数帯
    horse_bins = [
        (5, 8, "少頭数 5-8頭"),
        (9, 12, "中頭数 9-12頭"),
        (13, 16, "多頭数 13-16頭"),
        (17, 99, "フルゲート 17+頭"),
    ]
    print_section("頭数帯", [
        roi_stats(bets[(bets["horse_count"] >= lo) & (bets["horse_count"] <= hi)], name)
        for lo, hi, name in horse_bins
    ])

    # 7. オッズ帯
    odds_bins = [
        (2, 4, "低オッズ 2-4倍"),
        (4, 8, "中低 4-8倍"),
        (8, 15, "中 8-15倍"),
        (15, 30, "中高 15-30倍"),
        (30, 50, "高 30-50倍"),
    ]
    print_section("オッズ帯", [
        roi_stats(bets[(bets["odds"] >= lo) & (bets["odds"] < hi)], name)
        for lo, hi, name in odds_bins
    ])

    # 8. 月別
    print_section("月別", [
        roi_stats(bets[bets["month"] == m], f"{m}月") for m in range(1, 13)
    ])

    # 9. 競馬場別
    venue_map = {
        1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
        6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
    }
    print_section("競馬場別", [
        roi_stats(bets[bets["course_code"] == c], f"{n}") for c, n in venue_map.items()
    ], min_n=15)

    # 10. グレード別
    grade_map = {
        1: "G1", 2: "G2", 3: "G3", 4: "重賞以外OP",
        5: "3勝クラス", 6: "2勝クラス", 7: "1勝クラス", 8: "新馬/未勝利",
    }
    print_section("グレード別", [
        roi_stats(bets[bets["grade_code"] == c], f"{n}") for c, n in grade_map.items()
    ])

    # 11. Kelly比率帯（ベットサイズ分布）
    bets["kelly_pct"] = bets["kelly_ratio"] * 100
    kelly_bins = [
        (0, 0.5, "極小 0-0.5%"),
        (0.5, 1.0, "小 0.5-1.0%"),
        (1.0, 2.0, "中 1.0-2.0%"),
        (2.0, 5.0, "大 2.0-5.0%"),
    ]
    print_section("Kelly比率帯（ベットサイズ）", [
        roi_stats(bets[(bets["kelly_pct"] >= lo) & (bets["kelly_pct"] < hi)], name)
        for lo, hi, name in kelly_bins
    ])

    # 12. EV帯別品質
    ev_bins = [
        (1.03, 1.05, "EV 1.03-1.05"),
        (1.05, 1.10, "EV 1.05-1.10"),
        (1.10, 1.20, "EV 1.10-1.20"),
        (1.20, 9.9, "EV 1.20+"),
    ]
    print_section("EV帯別", [
        roi_stats(bets[(bets["ev_rate"] >= lo) & (bets["ev_rate"] < hi)], name)
        for lo, hi, name in ev_bins
    ])

    # 13. 人気順位との関係（model_prob順位 vs market順位）
    all_df["market_rank"] = all_df.groupby("race_id")["odds"].rank(ascending=True, method="first").astype(int)
    bets_with_mrank = all_df[all_df["is_recommended"]].copy()
    mrank_bins = [(1, 1, "1番人気"), (2, 3, "2-3番人気"), (4, 6, "4-6番人気"), (7, 99, "7番人気以下")]
    print_section("市場人気順位別", [
        roi_stats(
            bets_with_mrank[(bets_with_mrank["market_rank"] >= lo) & (bets_with_mrank["market_rank"] <= hi)],
            name
        )
        for lo, hi, name in mrank_bins
    ])


if __name__ == "__main__":
    run_diagnostics()

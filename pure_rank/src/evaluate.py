"""
evaluate.py — RaceAI_var1.0 精度評価スクリプト

計算する指標:
- Top-1 的中率（予測1位が実際の1着か）
- Top-3 的中率
- NDCG@3
- Spearman 相関
- テストセットのレース数・サンプル数

リーク停止閾値:
    Top-1 > 40% または Spearman > 0.6 → 即座に評価停止
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ─── パス解決 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── 特徴量列の取得（train.py と同一ロジック） ──────────────────────────────────

def get_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    id_cols = set(cfg["features"]["id_cols"])
    forbidden = {
        # 市場情報（絶対禁止）
        "odds", "popularity", "win_odds", "place_odds",
        "quinella_odds", "market_prob", "market_log_odds",
        "init_score", "ninki",
        "_time_dev",
        "year", "month_day", "kai", "nichi", "race_num",
        "horse_num", "registered_count", "finish_count",
        "race_type_code", "weight_type", "race_condition_code",
        "race_level", "race_age_type", "course_kubun",
        "track_code",
        "obstacle_mile_time_sec",
        "dead_heat_flag", "dead_heat_count",
        "breed_code", "region_code",
        "sire_id", "bms_id",
        # レース後にしか判明しない後出し情報
        "racetime", "time_3f_after",
        "corner_1", "corner_2", "corner_3", "corner_4",
        "running_style_code",
        "abnormal_code",
        "hon_shokin", "fuka_shokin",
        # 生ラベル
        "finish_rank", "is_win", "is_place", "lr_label",
    }
    exclude = id_cols | forbidden
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype not in ["object", "string"]
    ]


# ─── 予測値生成 ────────────────────────────────────────────────────────────────

def load_models(models_dir: Path) -> list[lgb.Booster]:
    """保存済みモデルを全て読み込む。"""
    model_files = sorted(models_dir.glob("lambdarank_fold*_seed*.txt"))
    if not model_files:
        raise FileNotFoundError(
            f"モデルファイルが見つかりません: {models_dir}\n"
            f"先に train.py を実行してください。"
        )
    models = []
    for path in model_files:
        m = lgb.Booster(model_file=str(path))
        models.append(m)
        print(f"  Loaded: {path.name}")
    return models


def ensemble_predict(models: list[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    """全モデルの予測を平均してアンサンブルスコアを返す。"""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)


# ─── 評価指標計算 ──────────────────────────────────────────────────────────────

def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """DCG@k を計算する（降順ソート済みの relevance 配列を想定）。"""
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    gains = 2.0 ** relevances - 1.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """NDCG@k を計算する。

    Parameters
    ----------
    y_true : 実際のラベル（lr_label: 高いほど良い）
    y_pred : モデルスコア（高いほど上位に推薦）
    k : 評価位置
    """
    # pred が高い順に並べた時の true label 順序
    pred_sorted_idx = np.argsort(-y_pred)
    true_sorted_by_pred = y_true[pred_sorted_idx]

    # 理想の順序
    ideal_sorted = np.sort(y_true)[::-1]

    dcg = _dcg_at_k(true_sorted_by_pred, k)
    idcg = _dcg_at_k(ideal_sorted, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_metrics(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
) -> dict:
    """テストセット全体の評価指標を計算する。

    Parameters
    ----------
    df_test : テスト用 DataFrame（race_id, finish_rank, lr_label を含む）
    predictions : アンサンブル予測スコア（高いほど上位予測）

    Returns
    -------
    dict: 各評価指標の値
    """
    df_eval = df_test.copy()
    df_eval["pred_score"] = predictions

    race_ids = df_eval["race_id"].unique()
    n_races = len(race_ids)

    top1_hits = 0
    top3_hits = 0
    ndcg3_list = []
    spearman_list = []

    for race_id in race_ids:
        race = df_eval[df_eval["race_id"] == race_id].copy()
        if len(race) < 2:
            continue

        # 予測スコアでソート（降順）
        race_sorted = race.sort_values("pred_score", ascending=False)

        actual_ranks = race["finish_rank"].values
        pred_order = race_sorted["finish_rank"].values

        # Top-1 的中: 予測1位の馬が実際の1着か
        top1_hits += int(pred_order[0] == 1)

        # Top-3 的中: 予測1〜3位の中に実際の1着が含まれるか
        top3_hits += int(1 in pred_order[:3])

        # NDCG@3
        y_true = race["lr_label"].values
        y_pred = race["pred_score"].values
        ndcg3_list.append(ndcg_at_k(y_true, y_pred, k=3))

        # Spearman 相関（予測スコアと実際の着順の相関）
        # 着順は小さいほど良いので、-finish_rank を使う
        if len(race) >= 3:
            corr, _ = spearmanr(race["pred_score"].values, -race["finish_rank"].values)
            if not np.isnan(corr):
                spearman_list.append(corr)

    top1_rate = top1_hits / n_races if n_races > 0 else 0.0
    top3_rate = top3_hits / n_races if n_races > 0 else 0.0
    ndcg3 = float(np.mean(ndcg3_list)) if ndcg3_list else 0.0
    spearman = float(np.mean(spearman_list)) if spearman_list else 0.0

    return {
        "top1_rate": top1_rate,
        "top3_rate": top3_rate,
        "ndcg_at_3": ndcg3,
        "spearman": spearman,
        "n_races": n_races,
        "n_samples": len(df_test),
    }


def check_leakage_threshold(metrics: dict) -> None:
    """リーク停止閾値チェック。

    Top-1 > 40% または Spearman > 0.6 の場合、データリークの疑いがあるため
    ValueError を raise する。
    """
    if metrics["top1_rate"] > 0.40:
        raise ValueError(
            f"[DATA LEAK ALERT] Top-1={metrics['top1_rate']:.3f} > 0.40\n"
            f"データリークが強く疑われます。shift(1) の適用を確認してください。\n"
            f"evaluator に報告して実装を停止してください。"
        )
    if metrics["spearman"] > 0.60:
        raise ValueError(
            f"[DATA LEAK ALERT] Spearman={metrics['spearman']:.3f} > 0.60\n"
            f"データリークが強く疑われます。shift(1) の適用を確認してください。\n"
            f"evaluator に報告して実装を停止してください。"
        )


def compute_supplementary_metrics(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    T_opt: float | None = None,
) -> dict:
    """予測1位馬の着順分布・鼻差ミス・Top-2 coverage・ペア指標を計算する。

    Parameters
    ----------
    df_test : テスト用 DataFrame（race_id, finish_rank, racetime を含む）
    predictions : アンサンブル予測スコア（高いほど上位予測）
    T_opt : Softmax 温度（None の場合は config から読み込み、未設定なら 1.0）

    Returns
    -------
    dict:
        pred_top1_avg_actual_rank : 予測1位馬の平均実際着順
        near_miss_rate_narrow     : ミスのうち 2着かつタイム差≤0.3秒の割合
        top2_coverage             : 予測1位が実際1着か2着だった割合
        top3_coverage_rate        : 予測1位が実際1〜3着だった割合
        wide_pair_coverage_rate   : 予測1位・2位が共に1〜3着
        quinella_pair_coverage_rate : 予測1位・2位が{1,2}着を占める
        wide_harville_coverage_rate : Harville最大P_wideペアが共に1〜3着
    """
    from predict import compute_pair_coverage_metrics

    if T_opt is None:
        cfg = load_config()
        T_opt = cfg.get("plackett_luce", {}).get("T_opt", 1.0)

    df_eval = df_test.copy()
    df_eval["pred_score"] = predictions

    pred_top1_actual_ranks: list[int] = []
    pred_top1_time_diffs: list[float] = []

    for _, grp in df_eval.groupby("race_id"):
        pred_best_idx = grp["pred_score"].idxmax()  # 予測1位
        actual_rank = int(grp.loc[pred_best_idx, "finish_rank"])
        pred_top1_actual_ranks.append(actual_rank)

        # 勝ち馬とのタイム差（2着以下のみ意味がある）
        winner_rows = grp[grp["finish_rank"] == 1]
        horse_time = float(grp.loc[pred_best_idx, "racetime"])
        if len(winner_rows) > 0 and actual_rank > 1:
            winner_time = float(winner_rows["racetime"].values[0])
            pred_top1_time_diffs.append(horse_time - winner_time)
        else:
            # 予測1位が実際1着の場合はタイム差=0.0、勝ち馬不在は NaN
            pred_top1_time_diffs.append(0.0 if actual_rank == 1 else float("nan"))

    # 指標1: 予測1位馬の平均実際着順（小さいほど良い）
    pred_top1_avg_rank = float(np.mean(pred_top1_actual_ranks))

    # 指標2: ミスのうち 2着かつタイム差≤0.3秒の割合（鼻差ミス率）
    misses = [
        (r, d)
        for r, d in zip(pred_top1_actual_ranks, pred_top1_time_diffs)
        if r != 1
    ]
    if len(misses) > 0:
        near_miss_rate_narrow = sum(
            1 for r, d in misses
            if r == 2 and not np.isnan(d) and d <= 0.3
        ) / len(misses)
    else:
        near_miss_rate_narrow = 0.0

    # 指標3: 予測1位が実際1着か2着だった割合（Top-2 coverage）
    top2_coverage = sum(
        1 for r in pred_top1_actual_ranks if r <= 2
    ) / max(len(pred_top1_actual_ranks), 1)

    pair_metrics = compute_pair_coverage_metrics(df_test, predictions, float(T_opt))

    return {
        "pred_top1_avg_actual_rank": pred_top1_avg_rank,
        "near_miss_rate_narrow": near_miss_rate_narrow,
        "top2_coverage": top2_coverage,
        **pair_metrics,
    }


def print_report(metrics: dict) -> None:
    """評価結果を読みやすい形式で表示する。"""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  RaceAI_var1.0 Phase 1 評価レポート")
    print(sep)
    print(f"  テストセット: {metrics['n_races']:,} レース / {metrics['n_samples']:,} サンプル")
    print(f"  Top-1 的中率: {metrics['top1_rate']:.3f}  ({metrics['top1_rate']*100:.1f}%)")
    print(f"  Top-3 的中率: {metrics['top3_rate']:.3f}  ({metrics['top3_rate']*100:.1f}%)")
    print(f"  NDCG@3:      {metrics['ndcg_at_3']:.4f}")
    print(f"  Spearman:    {metrics['spearman']:.4f}")
    print(sep)

    # 合否判定（CLAUDE.md の評価基準より）
    print("\n  合否判定:")
    thresholds = [
        ("Top-1", metrics["top1_rate"], 0.30, 0.28),
        ("Top-3", metrics["top3_rate"], 0.55, 0.52),
        ("NDCG@3", metrics["ndcg_at_3"], 0.52, 0.50),
        ("Spearman", metrics["spearman"], 0.50, 0.47),
    ]
    for name, val, pass_th, fail_th in thresholds:
        if val >= pass_th:
            status = "[合格]"
        elif val >= fail_th:
            status = "[要改善]"
        else:
            status = "[不合格]"
        print(f"    {name}: {val:.4f} {status}")

    if metrics["n_races"] < 500:
        print(f"\n  [警告] テストレース数 {metrics['n_races']} < 500 — 判定保留")
    print()


def print_supplementary_report(sup: dict) -> None:
    """Supplementary metrics: predicted-top1 rank distribution, near-miss, top2 coverage."""
    sep = "-" * 60
    print(f"{sep}")
    print("  Supplementary metrics (pred top-1 diagnosis)")
    print(sep)
    print(f"  pred_top1_avg_actual_rank : {sup['pred_top1_avg_actual_rank']:.3f}")
    print(
        f"  top2_coverage (actual 1st or 2nd): "
        f"{sup['top2_coverage']:.3f}  ({sup['top2_coverage']*100:.1f}%)"
    )
    print(
        f"  near_miss_rate_narrow (2nd and time_diff<=0.3s / all misses): "
        f"{sup['near_miss_rate_narrow']:.3f}  ({sup['near_miss_rate_narrow']*100:.1f}%)"
    )
    print(f"  top3_coverage_rate:          {sup['top3_coverage_rate']:.3f}  ({sup['top3_coverage_rate']*100:.1f}%)")
    print(
        f"  wide_pair_coverage_rate:     {sup['wide_pair_coverage_rate']:.3f}  "
        f"({sup['wide_pair_coverage_rate']*100:.1f}%)"
    )
    print(
        f"  quinella_pair_coverage_rate: {sup['quinella_pair_coverage_rate']:.3f}  "
        f"({sup['quinella_pair_coverage_rate']*100:.1f}%)"
    )
    print(
        f"  wide_harville_coverage_rate: {sup['wide_harville_coverage_rate']:.3f}  "
        f"({sup['wide_harville_coverage_rate']*100:.1f}%)"
    )
    print(sep)
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RaceAI_var1.0 Evaluation")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="モデルディレクトリのパス（省略時は train_config.json の models_dir を使用）",
    )
    args = parser.parse_args()

    cfg = load_config()

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    if args.models_dir:
        models_dir = PROJECT_ROOT / args.models_dir
    else:
        models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]

    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)

    # テストデータ抽出（2023-01-01 以降）
    valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
    df_test = df[df["race_date"] > valid_end_ts].copy()
    print(f"Test set: {len(df_test):,} rows, {df_test['race_id'].nunique():,} races")
    print(f"  Date range: {df_test['race_date'].min().date()} - {df_test['race_date'].max().date()}")

    if len(df_test) == 0:
        print("[ERROR] テストデータが空です。")
        return

    # 特徴量列
    feature_cols = get_feature_cols(df_test, cfg)
    print(f"  Feature cols: {len(feature_cols)}")

    # モデル読み込み
    print(f"\nLoading models from: {models_dir}")
    try:
        models = load_models(models_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    print(f"  {len(models)} models loaded")

    # アンサンブル予測
    print("\nPredicting...")
    X_test = df_test[feature_cols]
    preds = ensemble_predict(models, X_test)

    # 評価指標計算
    print("Computing metrics...")
    metrics = compute_metrics(df_test, preds)

    # リーク停止閾値チェック（評価表示の前に実行）
    check_leakage_threshold(metrics)

    # 結果表示
    print_report(metrics)

    # 補助指標計算・表示（訓練不要）
    print("Computing supplementary metrics...")
    sup_metrics = compute_supplementary_metrics(df_test, preds)
    print_supplementary_report(sup_metrics)

    # Phase 7 ベースラインとの比較
    baseline = {"top1_rate": 0.285, "top3_rate": None, "ndcg_at_3": 0.497, "spearman": 0.489}
    print("  Phase 7 ベースライン比較:")
    print(f"    Top-1:  {metrics['top1_rate']:.3f} vs {baseline['top1_rate']:.3f} "
          f"({'↑' if metrics['top1_rate'] > baseline['top1_rate'] else '↓'})")
    print(f"    NDCG@3: {metrics['ndcg_at_3']:.4f} vs {baseline['ndcg_at_3']:.4f} "
          f"({'↑' if metrics['ndcg_at_3'] > baseline['ndcg_at_3'] else '↓'})")
    print(f"    Spearman: {metrics['spearman']:.4f} vs {baseline['spearman']:.4f} "
          f"({'↑' if metrics['spearman'] > baseline['spearman'] else '↓'})")

    # 結果を JSON に保存
    result_path = PROJECT_ROOT / cfg["data"]["features_dir"] / "eval_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": {k: float(v) for k, v in metrics.items()},
                "supplementary": {k: float(v) for k, v in sup_metrics.items()},
                "baseline": baseline,
                "model_count": len(models),
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  Results saved: {result_path}")
    print("\n[evaluate] Done.")


if __name__ == "__main__":
    main()

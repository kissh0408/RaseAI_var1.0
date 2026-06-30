"""
predict.py — Plackett-Luce / Harville 確率変換モジュール

LambdaRank スコアをレース内勝率・ワイド/馬連確率に変換する。
モデル・特徴量は変更せず、既存アンサンブルスコアを入力とする。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# evaluate.py の共通関数を再利用（コード重複禁止）
from evaluate import (
    PROJECT_ROOT,
    CONFIG_PATH,
    ensemble_predict,
    get_feature_cols,
    load_config,
    load_models,
)

_DENOM_EPS = 1e-8


def softmax_with_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    """scores を温度 T でスケーリングして Softmax を適用する。"""
    if T <= 0:
        raise ValueError(f"Temperature T must be positive, got {T}")
    s = np.asarray(scores, dtype=float)
    s = s - s.max()
    exp_s = np.exp(s / T)
    total = exp_s.sum()
    if total <= 0:
        n = len(s)
        return np.full(n, 1.0 / n)
    return exp_s / total


def _log_loss_for_temperature(
    df_valid: pd.DataFrame,
    predictions: np.ndarray,
    T: float,
) -> float:
    """バリデーションセット全体の log-loss（勝ち馬の -log(p) 平均）。"""
    df = df_valid.copy()
    df["pred_score"] = predictions
    losses: list[float] = []
    for _, grp in df.groupby("race_id"):
        scores = grp["pred_score"].values
        p = softmax_with_temperature(scores, T)
        winner_mask = grp["is_win"].values == 1
        if not winner_mask.any():
            continue
        winner_idx = int(np.argmax(winner_mask))
        p_win = max(p[winner_idx], 1e-15)
        losses.append(-np.log(p_win))
    return float(np.mean(losses)) if losses else float("inf")


def calibrate_temperature(
    df_valid: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    T_range: np.ndarray,
) -> float:
    """バリデーションセットで log-loss を最小化する T を返す。"""
    X = df_valid[feature_cols]
    predictions = ensemble_predict(models, X)

    best_T = float(T_range[0])
    best_loss = float("inf")
    for T in T_range:
        loss = _log_loss_for_temperature(df_valid, predictions, float(T))
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T


def harville_place_probs(p_win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harville 公式で 2着・3着確率を計算する。"""
    p = np.asarray(p_win, dtype=float)
    n = len(p)
    p2 = np.zeros(n, dtype=float)
    p3 = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            denom2 = 1.0 - p[j]
            if denom2 < _DENOM_EPS:
                continue
            p2[i] += p[j] * p[i] / denom2

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            denom2 = 1.0 - p[j]
            if denom2 < _DENOM_EPS:
                continue
            p_k_given_j1 = p / denom2
            p_k_given_j1[j] = 0.0
            for k in range(n):
                if k == i or k == j:
                    continue
                denom3 = 1.0 - p[j] - p[k]
                if denom3 < _DENOM_EPS:
                    continue
                p3[i] += p[j] * (p[k] / denom2) * p[i] / denom3

    return p2, p3


def _prob_order_123(p: np.ndarray, a: int, b: int, c: int) -> float:
    """P(a=1着, b=2着, c=3着) を Harville 展開で計算。"""
    denom2 = 1.0 - p[a]
    if denom2 < _DENOM_EPS:
        return 0.0
    denom3 = 1.0 - p[a] - p[b]
    if denom3 < _DENOM_EPS:
        return 0.0
    return p[a] * p[b] / denom2 * p[c] / denom3


def compute_race_probabilities(race_scores: np.ndarray, T: float) -> dict:
    """1レース分のスコアを受け取り、全確率を返す。"""
    p_win = softmax_with_temperature(race_scores, T)
    p2, p3 = harville_place_probs(p_win)
    p_top3 = p_win + p2 + p3
    n = len(p_win)

    quinella_matrix = np.zeros((n, n), dtype=float)
    wide_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            denom_i = 1.0 - p_win[i]
            denom_j = 1.0 - p_win[j]
            q_ij = 0.0
            if denom_i >= _DENOM_EPS:
                q_ij += p_win[i] * p_win[j] / denom_i
            if denom_j >= _DENOM_EPS:
                q_ij += p_win[j] * p_win[i] / denom_j
            quinella_matrix[i, j] = q_ij
            quinella_matrix[j, i] = q_ij

            w_ij = q_ij
            for k in range(n):
                if k == i or k == j:
                    continue
                w_ij += _prob_order_123(p_win, i, k, j)
                w_ij += _prob_order_123(p_win, j, k, i)
                w_ij += _prob_order_123(p_win, k, i, j)
                w_ij += _prob_order_123(p_win, k, j, i)
            wide_matrix[i, j] = w_ij
            wide_matrix[j, i] = w_ij

    return {
        "p_win": p_win,
        "p2": p2,
        "p3": p3,
        "p_top3": p_top3,
        "wide_matrix": wide_matrix,
        "quinella_matrix": quinella_matrix,
    }


def _best_wide_pair(wide_matrix: np.ndarray) -> tuple[int, int]:
    """wide_matrix から最大 P_wide の (i, j) を返す（i != j）。"""
    n = wide_matrix.shape[0]
    best_i, best_j = 0, 1 if n > 1 else 0
    best_p = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if wide_matrix[i, j] > best_p:
                best_p = wide_matrix[i, j]
                best_i, best_j = i, j
    return best_i, best_j


def compute_pair_coverage_metrics(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    T_opt: float,
) -> dict:
    """テストセット全体でペア指標を計算する。"""
    df = df_test.copy()
    df["pred_score"] = predictions

    top3_hits = 0
    wide_pair_hits = 0
    quinella_pair_hits = 0
    wide_harville_hits = 0
    n_races = 0

    for _, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        n_races += 1
        grp = grp.sort_values("pred_score", ascending=False)
        ranks = grp["finish_rank"].values
        scores = grp["pred_score"].values

        # top3_coverage: pred-1st が 1〜3 着
        if ranks[0] <= 3:
            top3_hits += 1

        # wide_pair: score 1位・2位が共に 1〜3 着
        if ranks[0] <= 3 and ranks[1] <= 3:
            wide_pair_hits += 1

        # quinella_pair: score 1位・2位が {1, 2} を占める
        if set(ranks[:2]) == {1, 2}:
            quinella_pair_hits += 1

        # wide_harville: Harville 最大 P_wide ペア
        probs = compute_race_probabilities(scores, T_opt)
        hi, hj = _best_wide_pair(probs["wide_matrix"])
        if ranks[hi] <= 3 and ranks[hj] <= 3:
            wide_harville_hits += 1

    denom = max(n_races, 1)
    return {
        "top3_coverage_rate": top3_hits / denom,
        "wide_pair_coverage_rate": wide_pair_hits / denom,
        "quinella_pair_coverage_rate": quinella_pair_hits / denom,
        "wide_harville_coverage_rate": wide_harville_hits / denom,
    }


def _search_temperature(
    df_valid: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    cfg: dict,
) -> float:
    """粗探索 → 細探索で T_opt を決定する。"""
    pl_cfg = cfg.get("plackett_luce", {})
    coarse = pl_cfg.get(
        "T_search_coarse",
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    )
    fine_step = pl_cfg.get("T_search_fine_step", 0.02)

    print(f"  Coarse search over {len(coarse)} values...")
    T_coarse = calibrate_temperature(
        df_valid, models, feature_cols, np.array(coarse, dtype=float)
    )
    print(f"  Best coarse T: {T_coarse:.2f}")

    fine_low = max(T_coarse - 0.1, 0.01)
    fine_high = T_coarse + 0.1
    fine_range = np.arange(fine_low, fine_high + fine_step / 2, fine_step)
    print(f"  Fine search: [{fine_low:.2f}, {fine_high:.2f}] step={fine_step}")
    T_opt = calibrate_temperature(df_valid, models, feature_cols, fine_range)
    print(f"  T_opt: {T_opt:.2f}")
    return T_opt


def _save_T_opt(cfg: dict, T_opt: float) -> None:
    """train_config.json の plackett_luce.T_opt を更新する。"""
    if "plackett_luce" not in cfg:
        cfg["plackett_luce"] = {
            "T_opt": None,
            "calibration_valid_year": "2024",
            "T_search_coarse": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "T_search_fine_step": 0.02,
        }
    cfg["plackett_luce"]["T_opt"] = round(T_opt, 2)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Saved T_opt={T_opt:.2f} to {CONFIG_PATH}")


def _load_data_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """features parquet から valid (2024) と test (2025+) を返す。"""
    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    df = pd.read_parquet(feat_path)

    valid_year = cfg.get("plackett_luce", {}).get("calibration_valid_year", "2024")
    valid_start = pd.Timestamp(f"{valid_year}-01-01")
    valid_end = pd.Timestamp(f"{valid_year}-12-31")
    test_start = pd.Timestamp(cfg["training"]["valid_end"]) + pd.Timedelta(days=1)

    df_valid = df[(df["race_date"] >= valid_start) & (df["race_date"] <= valid_end)].copy()
    df_test = df[df["race_date"] >= test_start].copy()
    return df, df_valid, df_test


def run_calibrate(cfg: dict | None = None) -> float:
    """温度キャリブレーションを実行し T_opt を config に保存する。"""
    if cfg is None:
        cfg = load_config()

    _, df_valid, _ = _load_data_splits(cfg)
    print(f"Validation set (2024): {len(df_valid):,} rows, {df_valid['race_id'].nunique():,} races")

    feature_cols = get_feature_cols(df_valid, cfg)
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    print(f"\nLoading models from: {models_dir}")
    models = load_models(models_dir)

    T_opt = _search_temperature(df_valid, models, feature_cols, cfg)
    _save_T_opt(cfg, T_opt)
    return T_opt


def run_eval(cfg: dict | None = None) -> dict:
    """ペア coverage 指標をテストセットで評価する。"""
    if cfg is None:
        cfg = load_config()

    T_opt = cfg.get("plackett_luce", {}).get("T_opt")
    if T_opt is None:
        print("[WARN] T_opt not set. Running calibration first...")
        T_opt = run_calibrate(cfg)
        cfg = load_config()

    _, _, df_test = _load_data_splits(cfg)
    print(f"Test set: {len(df_test):,} rows, {df_test['race_id'].nunique():,} races")

    feature_cols = get_feature_cols(df_test, cfg)
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    models = load_models(models_dir)
    preds = ensemble_predict(models, df_test[feature_cols])

    metrics = compute_pair_coverage_metrics(df_test, preds, float(T_opt))
    print("\n--- Pair Coverage Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}  ({v * 100:.1f}%)")
    return metrics


def run_race_detail(race_id: str, cfg: dict | None = None) -> None:
    """単一レースの確率を出力する。"""
    if cfg is None:
        cfg = load_config()

    T_opt = cfg.get("plackett_luce", {}).get("T_opt", 1.0)
    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    df = pd.read_parquet(feat_path)
    race = df[df["race_id"] == race_id].copy()
    if race.empty:
        print(f"[ERROR] race_id={race_id} not found.")
        return

    feature_cols = get_feature_cols(race, cfg)
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    models = load_models(models_dir)
    race["pred_score"] = ensemble_predict(models, race[feature_cols])
    race = race.sort_values("pred_score", ascending=False)

    scores = race["pred_score"].values
    probs = compute_race_probabilities(scores, float(T_opt))
    hi, hj = _best_wide_pair(probs["wide_matrix"])

    print(f"\nRace: {race_id}  (T={T_opt})")
    print(f"{'horse_num':>10} {'score':>8} {'p_win':>8} {'p_top3':>8} {'finish':>8}")
    for idx, row in race.iterrows():
        loc = race.index.get_loc(idx)
        print(
            f"{int(row['horse_num']):>10} "
            f"{row['pred_score']:>8.3f} "
            f"{probs['p_win'][loc]:>8.4f} "
            f"{probs['p_top3'][loc]:>8.4f} "
            f"{int(row['finish_rank']):>8}"
        )

    h1 = int(race.iloc[hi]["horse_num"])
    h2 = int(race.iloc[hj]["horse_num"])
    print(f"\nRecommended wide pair: {h1}-{h2}  P_wide={probs['wide_matrix'][hi, hj]:.4f}")
    print(
        f"Recommended quinella (score top-2): "
        f"{int(race.iloc[0]['horse_num'])}-{int(race.iloc[1]['horse_num'])}  "
        f"P_quinella={probs['quinella_matrix'][0, 1]:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plackett-Luce / Harville probability conversion")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate temperature T on validation set")
    parser.add_argument("--eval", action="store_true", help="Evaluate pair coverage metrics on test set")
    parser.add_argument("--race-id", type=str, default=None, help="Print probabilities for a single race")
    args = parser.parse_args()

    if not any([args.calibrate, args.eval, args.race_id]):
        parser.print_help()
        sys.exit(1)

    cfg = load_config()
    if args.calibrate:
        run_calibrate(cfg)
    if args.eval:
        run_eval(cfg)
    if args.race_id:
        run_race_detail(args.race_id, cfg)


if __name__ == "__main__":
    main()

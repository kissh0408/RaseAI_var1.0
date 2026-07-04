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

# 共通関数は common.py / evaluate.py から再利用（コード重複禁止）
from common import CONFIG_PATH, PROJECT_ROOT, get_feature_cols, load_config
from evaluate import (
    ensemble_predict,
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


# ─── Stern型（べき乗割引）確率変換 ────────────────────────────────────────────
#
# Harville 公式は「1着争いの確率比がそのまま2着以降にも保存される」と仮定するが、
# 実際は下位着順ほど番狂わせが増えるため、Harville は穴馬の連対・複勝確率を
# 系統的に過大評価する（ev_diagnosis_v2.json: 全 bin で predicted > actual、
# 最大 16.6pp）。Stern型は「着順が下がるごとの割引率 λ」を1〜2個だけ導入し、
# λ=1 のとき Harville に厳密に一致する（下記 stern_second_prob / stern_third_prob
# を参照。p_j / sum_{k!=i} p_k = p_j / (1-p_i) は Harville の条件付き確率そのもの）。
# 既存の Harville 実装（harville_place_probs / compute_race_probabilities）は
# 削除せず併存させ、--prob-method で比較可能にする（simulate_ev.py 側）。

def stern_second_prob(p: np.ndarray, winner_idx: int, lam: float) -> np.ndarray:
    """
    Stern型: 1着が winner_idx に確定した後の2着確率分布を返す。

    q_j = p_j^lam / sum_{k != winner_idx} p_k^lam  (winner_idx 自身は 0)

    lam=1 のとき q_j = p_j / (1 - p[winner_idx]) となり、Harville の
    P(j 2着 | winner_idx 1着) と厳密に一致する。lam<1 で下位争いの番狂わせ
    （＝上位人気馬同士の差が縮む）を表現する。
    """
    q = np.asarray(p, dtype=float) ** lam
    q[winner_idx] = 0.0
    total = q.sum()
    n = len(p)
    if total < _DENOM_EPS:
        out = np.zeros(n, dtype=float)
        remaining = [k for k in range(n) if k != winner_idx]
        if remaining:
            out[remaining] = 1.0 / len(remaining)
        return out
    return q / total


def stern_third_prob(p: np.ndarray, first_idx: int, second_idx: int, lam: float) -> np.ndarray:
    """
    Stern型: 1着・2着が確定した後の3着確率分布を返す。

    q_k = p_k^lam / sum_{m not in {first_idx, second_idx}} p_m^lam

    lam=1 のとき Harville の P(k 3着 | first_idx 1着, second_idx 2着) と一致する。
    2着争い（lam2）とは独立に3着争い専用の割引率 lam3 を持つ。
    """
    q = np.asarray(p, dtype=float) ** lam
    q[first_idx] = 0.0
    q[second_idx] = 0.0
    total = q.sum()
    n = len(p)
    if total < _DENOM_EPS:
        out = np.zeros(n, dtype=float)
        remaining = [k for k in range(n) if k not in (first_idx, second_idx)]
        if remaining:
            out[remaining] = 1.0 / len(remaining)
        return out
    return q / total


def stern_place_probs(p_win: np.ndarray, lam2: float, lam3: float) -> tuple[np.ndarray, np.ndarray]:
    """Stern型で 2着・3着確率を計算する（harville_place_probs の一般化。lam2=lam3=1 で一致）。"""
    p = np.asarray(p_win, dtype=float)
    n = len(p)
    p2 = np.zeros(n, dtype=float)
    p3 = np.zeros(n, dtype=float)

    for i in range(n):
        second_probs = stern_second_prob(p, i, lam2)
        for j in range(n):
            if j == i:
                continue
            sp_j = second_probs[j]
            if sp_j <= 0:
                continue
            p2[j] += p[i] * sp_j

            third_probs = stern_third_prob(p, i, j, lam3)
            for k in range(n):
                if k == i or k == j:
                    continue
                tp_k = third_probs[k]
                if tp_k <= 0:
                    continue
                p3[k] += p[i] * sp_j * tp_k

    return p2, p3


def _prob_order_123_stern(p: np.ndarray, a: int, b: int, c: int, lam2: float, lam3: float) -> float:
    """P(a=1着, b=2着, c=3着) を Stern 型展開で計算（_prob_order_123 の一般化）。"""
    sp = stern_second_prob(p, a, lam2)[b]
    if sp <= 0:
        return 0.0
    tp = stern_third_prob(p, a, b, lam3)[c]
    if tp <= 0:
        return 0.0
    return p[a] * sp * tp


def compute_race_probabilities_stern(race_scores: np.ndarray, T: float, lam2: float, lam3: float) -> dict:
    """1レース分のスコアを受け取り、Stern型で全確率を返す（compute_race_probabilities の一般化）。

    lam2=lam3=1.0 を渡すと Harville と数値的に一致する。
    """
    p_win = softmax_with_temperature(race_scores, T)
    p2, p3 = stern_place_probs(p_win, lam2, lam3)
    p_top3 = p_win + p2 + p3
    n = len(p_win)

    quinella_matrix = np.zeros((n, n), dtype=float)
    wide_matrix = np.zeros((n, n), dtype=float)

    # 2着確率分布はペアごとに使い回すため i, j それぞれで1回だけ計算する
    second_probs_cache = [stern_second_prob(p_win, i, lam2) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            q_ij = p_win[i] * second_probs_cache[i][j] + p_win[j] * second_probs_cache[j][i]
            quinella_matrix[i, j] = q_ij
            quinella_matrix[j, i] = q_ij

            w_ij = q_ij
            for k in range(n):
                if k == i or k == j:
                    continue
                w_ij += _prob_order_123_stern(p_win, i, k, j, lam2, lam3)
                w_ij += _prob_order_123_stern(p_win, j, k, i, lam2, lam3)
                w_ij += _prob_order_123_stern(p_win, k, i, j, lam2, lam3)
                w_ij += _prob_order_123_stern(p_win, k, j, i, lam2, lam3)
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


def compute_race_probabilities_stern_from_p_win(
    p_win: np.ndarray, lam2: float, lam3: float
) -> dict:
    """キャリブレーション済み p_win から Stern 型の全確率を計算する（R-6 市場ブレンド用）。"""
    p = np.asarray(p_win, dtype=float)
    total = p.sum()
    if total <= _DENOM_EPS:
        n = len(p)
        p = np.full(n, 1.0 / n)
    else:
        p = p / total

    p2, p3 = stern_place_probs(p, lam2, lam3)
    p_top3 = p + p2 + p3
    n = len(p)

    quinella_matrix = np.zeros((n, n), dtype=float)
    wide_matrix = np.zeros((n, n), dtype=float)
    second_probs_cache = [stern_second_prob(p, i, lam2) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            q_ij = p[i] * second_probs_cache[i][j] + p[j] * second_probs_cache[j][i]
            quinella_matrix[i, j] = q_ij
            quinella_matrix[j, i] = q_ij

            w_ij = q_ij
            for k in range(n):
                if k == i or k == j:
                    continue
                w_ij += _prob_order_123_stern(p, i, k, j, lam2, lam3)
                w_ij += _prob_order_123_stern(p, j, k, i, lam2, lam3)
                w_ij += _prob_order_123_stern(p, k, i, j, lam2, lam3)
                w_ij += _prob_order_123_stern(p, k, j, i, lam2, lam3)
            wide_matrix[i, j] = w_ij
            wide_matrix[j, i] = w_ij

    return {
        "p_win": p,
        "p2": p2,
        "p3": p3,
        "p_top3": p_top3,
        "wide_matrix": wide_matrix,
        "quinella_matrix": quinella_matrix,
    }


# ─── 市場残差ブレンド（R-6 ベッティングレイヤー。特徴量には不使用） ─────────

def _logit_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def market_win_probs_from_odds(odds: np.ndarray) -> np.ndarray | None:
    """単勝オッズ（decimal）からレース内正規化した市場勝率を返す。無効なら None。"""
    odds = np.asarray(odds, dtype=float)
    if len(odds) < 2:
        return None
    valid = odds > 0
    if valid.sum() < 2:
        return None
    raw = np.where(valid, 1.0 / np.clip(odds, 1.01, None), 0.0)
    total = raw.sum()
    if total <= _DENOM_EPS:
        return None
    return raw / total


def standardize_scores_within_race(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    sd = s.std()
    if sd < 1e-8:
        return np.zeros_like(s)
    return (s - s.mean()) / sd


def blend_market_residual_probs(
    p_market: np.ndarray, z_scores: np.ndarray, beta: float
) -> np.ndarray:
    """logit(p) = logit(p_market) + beta * z（var2 残差学習のベッティングレイヤー近似）。"""
    logits = _logit_prob(p_market) + float(beta) * np.asarray(z_scores, dtype=float)
    raw = 1.0 / (1.0 + np.exp(-logits))
    total = raw.sum()
    if total <= _DENOM_EPS:
        n = len(raw)
        return np.full(n, 1.0 / n)
    return raw / total


def build_win_odds_lookup_from_se_parquet(se_path: Path) -> dict[str, dict[int, float]]:
    """SE_preprocessed.parquet から race_id -> {horse_num: odds} を構築（評価専用）。"""
    se = pd.read_parquet(se_path, columns=["race_id", "horse_num", "odds"])
    se = se[se["odds"] > 0]
    lookup: dict[str, dict[int, float]] = {}
    for rid, grp in se.groupby("race_id"):
        lookup[str(rid)] = dict(
            zip(grp["horse_num"].astype(int), grp["odds"].astype(float))
        )
    return lookup


def _extract_race_data_for_market_blend(
    df: pd.DataFrame,
    predictions: np.ndarray,
    win_odds_lookup: dict[str, dict[int, float]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(scores, is_win, odds) のレース単位リスト。オッズ欠損レースは除外。"""
    dfc = df.copy()
    dfc["pred_score"] = predictions
    races: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for race_id, grp in dfc.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        odds_map = win_odds_lookup.get(rid)
        if not odds_map:
            continue
        horse_nums = grp["horse_num"].astype(int).values
        odds_arr = np.array([odds_map.get(int(h), np.nan) for h in horse_nums], dtype=float)
        if np.isnan(odds_arr).any() or (odds_arr <= 0).any():
            continue
        races.append((
            grp["pred_score"].values,
            grp["is_win"].values.astype(int),
            odds_arr,
        ))
    return races


def _market_blend_avg_log_loss(
    races: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    beta: float,
) -> float:
    losses: list[float] = []
    for scores, is_win, odds_arr in races:
        p_market = market_win_probs_from_odds(odds_arr)
        if p_market is None:
            continue
        z = standardize_scores_within_race(scores)
        p_blend = blend_market_residual_probs(p_market, z, beta)
        winner_mask = is_win == 1
        if not winner_mask.any():
            continue
        winner_idx = int(np.argmax(winner_mask))
        p_w = max(float(p_blend[winner_idx]), 1e-15)
        losses.append(-np.log(p_w))
    return float(np.mean(losses)) if losses else float("inf")


def _search_market_blend_beta(
    races: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    coarse_grid: list[float],
    fine_step: float,
) -> tuple[float, float]:
    best_beta = float(coarse_grid[0])
    best_loss = float("inf")
    for beta in coarse_grid:
        loss = _market_blend_avg_log_loss(races, float(beta))
        if loss < best_loss:
            best_loss = loss
            best_beta = float(beta)

    half = max(fine_step * 5, 0.05)
    fine_start = max(0.0, best_beta - half)
    fine_end = best_beta + half
    fine_grid = np.arange(fine_start, fine_end + fine_step * 0.5, fine_step)
    for beta in fine_grid:
        loss = _market_blend_avg_log_loss(races, float(beta))
        if loss < best_loss:
            best_loss = loss
            best_beta = float(beta)
    return best_beta, best_loss


def _save_market_blend_beta(cfg: dict, beta: float, log_loss: float, n_races: int) -> None:
    cfg.setdefault("plackett_luce", {})
    cfg["plackett_luce"]["market_blend_beta_opt"] = round(float(beta), 4)
    cfg["plackett_luce"]["market_blend_fit_train_valid_end"] = cfg["training"]["valid_end"]
    cfg["plackett_luce"]["market_blend_fit_n_races"] = int(n_races)
    cfg["plackett_luce"]["market_blend_log_loss"] = round(float(log_loss), 6)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Saved market_blend_beta_opt={beta:.4f} to {CONFIG_PATH}")


def run_fit_market_blend(cfg: dict | None = None) -> float:
    """TRAIN+VALID（<= valid_end）のみで市場残差ブレンド係数 beta をフィットする（Rule 3）。"""
    if cfg is None:
        cfg = load_config()

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)

    valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
    df_fit = df[df["race_date"] <= valid_end_ts].copy()
    print(
        f"TRAIN+VALID (race_date <= {valid_end_ts.date()}): "
        f"{len(df_fit):,} rows, {df_fit['race_id'].nunique():,} races"
    )

    se_path = Path(cfg["data"]["src_parquet_dir"]) / "SE_preprocessed.parquet"
    if not se_path.exists():
        raise FileNotFoundError(
            f"SE_preprocessed.parquet が見つかりません: {se_path}\n"
            "単勝オッズはベッティングレイヤー評価専用。特徴量には混入しません。"
        )
    print(f"\nLoading win odds from SE (betting layer only): {se_path}")
    win_odds_lookup = build_win_odds_lookup_from_se_parquet(se_path)
    print(f"  Win odds races: {len(win_odds_lookup):,}")

    feature_cols = get_feature_cols(df_fit, cfg)
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    models = load_models(models_dir)
    predictions = ensemble_predict(models, df_fit[feature_cols])

    races = _extract_race_data_for_market_blend(df_fit, predictions, win_odds_lookup)
    print(f"  Usable races with win odds: {len(races):,}")
    if len(races) < 100:
        raise ValueError("市場ブレンド fit 用レース数が不足しています（100未満）")

    pl_cfg = cfg.get("plackett_luce", {})
    beta_coarse = pl_cfg.get(
        "market_blend_beta_search_coarse",
        [round(float(x), 2) for x in np.arange(0.0, 2.01, 0.1)],
    )
    beta_fine_step = pl_cfg.get("market_blend_beta_search_fine_step", 0.02)

    print("\n[market_blend] Fitting beta on winner log-loss (TRAIN+VALID only)...")
    beta_opt, ll_opt = _search_market_blend_beta(races, beta_coarse, float(beta_fine_step))
    print(f"  beta_opt={beta_opt:.4f}, avg_log_loss={ll_opt:.6f}, n_races={len(races):,}")

    _save_market_blend_beta(cfg, beta_opt, ll_opt, len(races))
    return beta_opt


# ─── λ2・λ3 のフィット（TRAIN+VALID のみ。TEST は使わない） ──────────────────

def _extract_race_data(df: pd.DataFrame, predictions: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """race_id ごとの (scores, finish_ranks) タプルのリストを返す（λ探索の高速化用キャッシュ）。"""
    dfc = df.copy()
    dfc["pred_score"] = predictions
    races: list[tuple[np.ndarray, np.ndarray]] = []
    for _, grp in dfc.groupby("race_id"):
        if len(grp) < 2:
            continue
        races.append((grp["pred_score"].values, grp["finish_rank"].values))
    return races


def _lambda2_avg_log_likelihood(
    races: list[tuple[np.ndarray, np.ndarray]], T: float, lam2: float
) -> float:
    """実際の2着馬に対する Stern型 log-likelihood の平均を返す。"""
    losses: list[float] = []
    for scores, ranks in races:
        w_pos = np.where(ranks == 1)[0]
        s_pos = np.where(ranks == 2)[0]
        if len(w_pos) != 1 or len(s_pos) != 1:
            continue
        w_idx, s_idx = int(w_pos[0]), int(s_pos[0])
        p_win = softmax_with_temperature(scores, T)
        q2 = stern_second_prob(p_win, w_idx, lam2)
        losses.append(float(np.log(max(q2[s_idx], 1e-15))))
    return float(np.mean(losses)) if losses else float("-inf")


def _lambda3_avg_log_likelihood(
    races: list[tuple[np.ndarray, np.ndarray]], T: float, lam3: float
) -> float:
    """実際の1着・2着が確定した条件下での、実際の3着馬に対する log-likelihood 平均を返す。

    2着争い（lam2）とは独立に、実着順（正解）を条件として3着争いのみを評価する
    （計画書「2着が決まった後の3着争いに対して独立にλ3を推定する」に対応）。
    """
    losses: list[float] = []
    for scores, ranks in races:
        w_pos = np.where(ranks == 1)[0]
        s_pos = np.where(ranks == 2)[0]
        t_pos = np.where(ranks == 3)[0]
        if len(w_pos) != 1 or len(s_pos) != 1 or len(t_pos) != 1:
            continue
        w_idx, s_idx, t_idx = int(w_pos[0]), int(s_pos[0]), int(t_pos[0])
        p_win = softmax_with_temperature(scores, T)
        q3 = stern_third_prob(p_win, w_idx, s_idx, lam3)
        losses.append(float(np.log(max(q3[t_idx], 1e-15))))
    return float(np.mean(losses)) if losses else float("-inf")


def _search_lambda(
    races: list[tuple[np.ndarray, np.ndarray]],
    T: float,
    ll_func,
    coarse: list[float],
    fine_step: float,
    lam_bounds: tuple[float, float] = (0.5, 1.0),
) -> tuple[float, float]:
    """粗探索 → 細探索で λ を決定する（_search_temperature と同型のパターン）。

    Returns
    -------
    tuple[float, float]: (lam_opt, best_log_likelihood)
    """
    print(f"    Coarse search over {len(coarse)} values...")
    best_lam = float(coarse[0])
    best_ll = float("-inf")
    for lam in coarse:
        ll = ll_func(races, T, float(lam))
        if ll > best_ll:
            best_ll = ll
            best_lam = float(lam)
    print(f"    Best coarse lam: {best_lam:.2f} (log-likelihood={best_ll:.4f})")

    lo, hi = lam_bounds
    fine_low = max(best_lam - 0.05, lo)
    fine_high = min(best_lam + 0.05, hi)
    fine_range = np.arange(fine_low, fine_high + fine_step / 2, fine_step)
    print(f"    Fine search: [{fine_low:.2f}, {fine_high:.2f}] step={fine_step}")

    best_lam_fine = best_lam
    best_ll_fine = best_ll
    for lam in fine_range:
        ll = ll_func(races, T, float(lam))
        if ll > best_ll_fine:
            best_ll_fine = ll
            best_lam_fine = float(lam)
    print(f"    lam_opt: {best_lam_fine:.2f} (log-likelihood={best_ll_fine:.4f})")
    return best_lam_fine, best_ll_fine


def _save_lambda_opt(
    cfg: dict, lam2_opt: float, lam3_opt: float, ll2: float, ll3: float, n_races: int
) -> None:
    """train_config.json の plackett_luce.lam2_opt / lam3_opt を更新する（T_opt と同じ管理方式）。"""
    if "plackett_luce" not in cfg:
        cfg["plackett_luce"] = {}
    cfg["plackett_luce"]["lam2_opt"] = round(lam2_opt, 2)
    cfg["plackett_luce"]["lam3_opt"] = round(lam3_opt, 2)
    cfg["plackett_luce"]["lam_fit_train_valid_end"] = cfg["training"]["valid_end"]
    cfg["plackett_luce"]["lam_fit_n_races"] = n_races
    cfg["plackett_luce"]["lam2_log_likelihood"] = round(ll2, 6)
    cfg["plackett_luce"]["lam3_log_likelihood"] = round(ll3, 6)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Saved lam2_opt={lam2_opt:.2f}, lam3_opt={lam3_opt:.2f} to {CONFIG_PATH}")


def run_fit_lambda(cfg: dict | None = None) -> tuple[float, float]:
    """
    TRAIN+VALID（race_date <= training.valid_end、2024-12-31）のみで λ2・λ3 をフィットする。

    TEST（2025+）は一切参照しない（Rule 3: 後出しじゃんけん禁止）。
    T は既存の T_opt（VALID 2024 でキャリブレーション済み）をそのまま使う。
    """
    if cfg is None:
        cfg = load_config()

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)

    valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
    df_fit = df[df["race_date"] <= valid_end_ts].copy()
    print(
        f"TRAIN+VALID (race_date <= {valid_end_ts.date()}): "
        f"{len(df_fit):,} rows, {df_fit['race_id'].nunique():,} races"
    )

    T_opt = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))
    print(f"Using T_opt={T_opt} (already calibrated on VALID 2024, unchanged by this step)")

    feature_cols = get_feature_cols(df_fit, cfg)
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    print(f"\nLoading models from: {models_dir}")
    models = load_models(models_dir)
    predictions = ensemble_predict(models, df_fit[feature_cols])

    races = _extract_race_data(df_fit, predictions)
    print(f"  Usable races (>=2 horses): {len(races):,}")

    pl_cfg = cfg.get("plackett_luce", {})
    lam_coarse = pl_cfg.get(
        "lam_search_coarse", [round(float(x), 2) for x in np.arange(0.5, 1.001, 0.05)]
    )
    lam_fine_step = pl_cfg.get("lam_search_fine_step", 0.01)

    print("\n[lam2] Fitting on actual 2nd-place horses (given actual 1st)...")
    lam2_opt, ll2 = _search_lambda(races, T_opt, _lambda2_avg_log_likelihood, lam_coarse, lam_fine_step)

    print("\n[lam3] Fitting on actual 3rd-place horses (given actual 1st/2nd, independent of lam2)...")
    lam3_opt, ll3 = _search_lambda(races, T_opt, _lambda3_avg_log_likelihood, lam_coarse, lam_fine_step)

    _save_lambda_opt(cfg, lam2_opt, lam3_opt, ll2, ll3, len(races))

    if 0.95 <= lam2_opt <= 1.05 and 0.95 <= lam3_opt <= 1.05:
        print(
            "\n  [NOTE] lam2/lam3 are both within [0.95, 1.05] -- Harville was already close to "
            "optimal on this objective. This matches the roadmap's failure criterion "
            "(docs/specs/2026-07-04-roi-improvement-roadmap.md Section 2)."
        )

    return lam2_opt, lam3_opt


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


# ─── キャリブレーション共通ヘルパー ──────────────────────────────────────────

def _norm_pair(h1: int, h2: int) -> tuple[int, int]:
    """horse_num ペアを (min, max) 順に正規化する（HR lookup キーと一致）。"""
    return (min(h1, h2), max(h1, h2))


def _build_wide_lookup(hr_df: pd.DataFrame) -> dict[str, dict[tuple[int, int], int]]:
    """race_id → {(h1, h2): payout} の wide 払戻辞書を構築する。"""
    sub = hr_df[hr_df["bet_type"] == "wide"]
    lookup: dict[str, dict[tuple[int, int], int]] = {}
    for _, row in sub.iterrows():
        rid = str(row["race_id"])
        key = _norm_pair(int(row["horse_num_1"]), int(row["horse_num_2"]))
        lookup.setdefault(rid, {})[key] = int(row["payout"])
    return lookup


def compute_race_probabilities_from_p_win(p_win: np.ndarray) -> dict:
    """
    キャリブレーション済み p_win から全確率を計算する。

    Platt スケーリング等で事前調整済みの p_win を受け取り、
    compute_race_probabilities と同じ構造の dict を返す。
    """
    p = np.asarray(p_win, dtype=float)
    p2, p3 = harville_place_probs(p)
    p_top3 = p + p2 + p3
    n = len(p)

    quinella_matrix = np.zeros((n, n), dtype=float)
    wide_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            denom_i = 1.0 - p[i]
            denom_j = 1.0 - p[j]
            q_ij = 0.0
            if denom_i >= _DENOM_EPS:
                q_ij += p[i] * p[j] / denom_i
            if denom_j >= _DENOM_EPS:
                q_ij += p[j] * p[i] / denom_j
            quinella_matrix[i, j] = q_ij
            quinella_matrix[j, i] = q_ij

            w_ij = q_ij
            for k in range(n):
                if k == i or k == j:
                    continue
                w_ij += _prob_order_123(p, i, k, j)
                w_ij += _prob_order_123(p, j, k, i)
                w_ij += _prob_order_123(p, k, i, j)
                w_ij += _prob_order_123(p, k, j, i)
            wide_matrix[i, j] = w_ij
            wide_matrix[j, i] = w_ij

    return {
        "p_win": p,
        "p2": p2,
        "p3": p3,
        "p_top3": p_top3,
        "wide_matrix": wide_matrix,
        "quinella_matrix": quinella_matrix,
    }


def apply_platt_to_p_win(p_win: np.ndarray, platt_scaler) -> np.ndarray:
    """
    Platt スケーラーを p_win に適用し、レース内で正規化する。

    1. LogisticRegression.predict_proba で P_calibrated(win) を計算
    2. 合計が 1 になるよう正規化
    """
    X = p_win.reshape(-1, 1)
    p_cal = platt_scaler.predict_proba(X)[:, 1]
    total = p_cal.sum()
    if total <= _DENOM_EPS:
        n = len(p_cal)
        return np.full(n, 1.0 / n)
    return p_cal / total


def _collect_wide_pair_data(
    df: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    hr_df: pd.DataFrame,
    T: float,
) -> pd.DataFrame:
    """
    各レースの全ペア (i, j) について Harville p_wide と is_wide_hit を返す。

    IsotonicRegression の学習データ生成に使用。
    is_wide_hit = HR 払戻 > 0（双方が 3 着以内）。
    """
    wide_lookup = _build_wide_lookup(hr_df)
    X_feat = df[feature_cols]
    preds = ensemble_predict(models, X_feat)
    dfc = df.copy()
    dfc["pred_score"] = preds

    rows: list[dict] = []
    for race_id, grp in dfc.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        grp_r = grp.reset_index(drop=True)
        horse_nums = grp_r["horse_num"].astype(int).values
        scores = grp_r["pred_score"].values
        probs = compute_race_probabilities(scores, float(T))
        n = len(grp_r)
        for i in range(n):
            for j in range(i + 1, n):
                p_w = float(probs["wide_matrix"][i, j])
                key = _norm_pair(int(horse_nums[i]), int(horse_nums[j]))
                payout = wide_lookup.get(rid, {}).get(key, 0)
                rows.append({"p_wide": p_w, "hit": int(payout > 0)})

    return pd.DataFrame(rows)


# ─── 手法1: Platt スケーリング ────────────────────────────────────────────────

def fit_platt_scaler(
    df_valid: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    T_opt: float,
):
    """
    バリデーションセット（2024年）で Platt スケーリングを学習する。

    p_win (Softmax 後) を入力、is_win を正解として LogisticRegression を学習。
    補正後 sum=1 に正規化してから Harville に渡す。

    Returns
    -------
    sklearn.linear_model.LogisticRegression
    """
    from sklearn.linear_model import LogisticRegression

    X_feat = df_valid[feature_cols]
    preds = ensemble_predict(models, X_feat)
    dfc = df_valid.copy()
    dfc["pred_score"] = preds

    p_win_list: list[float] = []
    y_list: list[int] = []

    for _, grp in dfc.groupby("race_id"):
        scores = grp["pred_score"].values
        p = softmax_with_temperature(scores, T_opt)
        p_win_list.extend(p.tolist())
        y_list.extend(grp["is_win"].astype(int).values.tolist())

    X_platt = np.array(p_win_list).reshape(-1, 1)
    y_platt = np.array(y_list)

    # C=1.0: 適度な正則化でバリデーション過学習を防ぐ
    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(X_platt, y_platt)
    print(
        f"  Platt scaler fitted: coef={platt.coef_[0][0]:.4f}, "
        f"intercept={platt.intercept_[0]:.4f}"
    )
    return platt


# ─── 手法2: ROI 最大化 T 探索 ────────────────────────────────────────────────

def find_T_for_roi(
    df_valid: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    hr_df_valid: pd.DataFrame,
    T_search: list[float] | None = None,
    ev_threshold: float = 1.0,
) -> float:
    """
    バリデーションセットで EV >= ev_threshold フィルタ後の wide ROI を最大化する T を返す。

    log-loss 最小化（calibrate_temperature）ではなく ROI を直接最大化する。
    見つかった T_roi は config に保存する（呼び出し側で save_calibration_models を使う）。

    Parameters
    ----------
    T_search : 探索する T 値リスト（None の場合はデフォルト範囲）
    ev_threshold : EV フィルタの閾値
    """
    if T_search is None:
        T_search = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

    wide_lookup = _build_wide_lookup(hr_df_valid)
    # バリデーションセットの wide 平均払戻（miss 時の参照値）
    wide_ref = float(hr_df_valid[hr_df_valid["bet_type"] == "wide"]["payout"].mean())

    X_feat = df_valid[feature_cols]
    preds = ensemble_predict(models, X_feat)
    dfc = df_valid.copy()
    dfc["pred_score"] = preds

    best_T = float(T_search[0])
    best_roi = float("-inf")

    print(f"  ROI-T search ({len(T_search)} values, EV >= {ev_threshold}):")
    for T in T_search:
        total_payout = 0.0
        total_stake = 0.0
        for race_id, grp in dfc.groupby("race_id"):
            if len(grp) < 2:
                continue
            rid = str(race_id)
            grp_r = grp.reset_index(drop=True)
            horse_nums = grp_r["horse_num"].astype(int).values
            scores = grp_r["pred_score"].values
            probs = compute_race_probabilities(scores, float(T))
            wi, wj = _best_wide_pair(probs["wide_matrix"])
            p_w = float(probs["wide_matrix"][wi, wj])
            key = _norm_pair(int(horse_nums[wi]), int(horse_nums[wj]))
            payout = wide_lookup.get(rid, {}).get(key, 0)
            # NOTE: ref_w は HR レコードの結果払戻であり、事前オッズではない。
            # 真のEV計算には OR レコード（事前ワイドオッズ）が必要。
            # 現在の実装は参照値として使用しているが、厳密なEV評価ではない。
            ref_w = payout if payout > 0 else wide_ref
            ev = p_w * ref_w / 100.0
            if ev >= ev_threshold:
                total_payout += float(payout)
                total_stake += 100.0
        roi = (total_payout / total_stake) if total_stake > 0 else 0.0
        n_bets = int(total_stake / 100.0)
        print(f"    T={T:.2f}: n_bets={n_bets:4d}, ROI={roi * 100:.2f}%")
        if roi > best_roi:
            best_roi = roi
            best_T = float(T)

    print(f"  Best T for ROI: {best_T:.2f} (valid ROI={best_roi * 100:.2f}%)")
    return best_T


# ─── 手法3: Isotonic 回帰（wide ペア直接キャリブレーション）─────────────────

def fit_isotonic_wide(
    df_valid: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    hr_df_valid: pd.DataFrame,
    T_opt: float,
):
    """
    バリデーションセットで Harville wide 確率に Isotonic 回帰を学習する。

    全ペア (i, j) の p_wide_harville を X、is_wide_hit を y として単調増加回帰。
    推論時は Harville p_wide を calibrated_p_wide に変換してから EV 計算に使う。

    Returns
    -------
    sklearn.isotonic.IsotonicRegression
    """
    from sklearn.isotonic import IsotonicRegression

    print("  Collecting all wide pairs from validation set...")
    df_pairs = _collect_wide_pair_data(df_valid, models, feature_cols, hr_df_valid, T_opt)
    X_iso = df_pairs["p_wide"].values
    y_iso = df_pairs["hit"].values

    # increasing=True: 高 p_wide ほど高ヒット率（単調増加制約）
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(X_iso, y_iso)

    n_pairs = len(df_pairs)
    hit_rate = float(y_iso.mean())
    print(
        f"  Isotonic fitted: {n_pairs:,} pairs, "
        f"overall hit_rate={hit_rate:.4f}, "
        f"p_wide range=[{X_iso.min():.4f}, {X_iso.max():.4f}]"
    )
    return iso


# ─── キャリブレーションモデルの保存・読み込み ────────────────────────────────

def save_calibration_models(
    models_dir: Path,
    platt=None,
    isotonic=None,
    T_roi: float | None = None,
) -> None:
    """
    キャリブレーションモデルを models/calibration/ に joblib で保存する。

    Parameters
    ----------
    models_dir : pure_rank/models/ への Path
    platt      : LogisticRegression（手法1）
    isotonic   : IsotonicRegression（手法3）
    T_roi      : ROI 最適化 T（手法2）、calibration_meta.json に保存
    """
    import joblib

    calib_dir = models_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    # joblib でモデルを直列化する。保存先は自プロジェクトの models/calibration/ のみ。
    # 外部ファイルや未検証入力をロードする経路はなく、任意コード実行リスクはない。
    if platt is not None:
        p = calib_dir / "platt_scaler.joblib"
        joblib.dump(platt, p)
        print(f"  Saved: {p}")

    if isotonic is not None:
        p = calib_dir / "isotonic_wide.joblib"
        joblib.dump(isotonic, p)
        print(f"  Saved: {p}")

    if T_roi is not None:
        meta_path = calib_dir / "calibration_meta.json"
        meta: dict = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        meta["T_roi"] = round(float(T_roi), 4)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  Saved: {meta_path}")


def load_calibration_models(models_dir: Path) -> dict:
    """
    models/calibration/ から保存済みキャリブレーションモデルを読み込む。

    Returns
    -------
    dict with keys: "platt", "isotonic", "T_roi" (存在するもののみ)
    """
    import joblib

    calib_dir = models_dir / "calibration"
    result: dict = {}

    # 自プロジェクトが保存した models/calibration/ 配下のファイルのみを読む。安全。
    p = calib_dir / "platt_scaler.joblib"
    if p.exists():
        result["platt"] = joblib.load(p)
        print(f"  Loaded: {p}")

    p = calib_dir / "isotonic_wide.joblib"
    if p.exists():
        result["isotonic"] = joblib.load(p)
        print(f"  Loaded: {p}")

    p = calib_dir / "calibration_meta.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        if "T_roi" in meta:
            result["T_roi"] = float(meta["T_roi"])
            print(f"  T_roi={result['T_roi']:.4f} from {p}")

    return result


# ─── キャリブレーション一括学習エントリポイント ──────────────────────────────

def run_fit_calibration(cfg: dict | None = None) -> dict:
    """
    3手法のキャリブレーションを一括学習し、models/calibration/ に保存する。

    1. Platt スケーリング（勝率の事後補正）
    2. ROI 最大化 T 探索
    3. Isotonic 回帰（wide ペア直接キャリブレーション）

    Returns
    -------
    dict: {"platt": ..., "T_roi": ..., "isotonic": ...}
    """
    if cfg is None:
        cfg = load_config()

    T_opt = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))
    valid_year = cfg.get("plackett_luce", {}).get("calibration_valid_year", "2024")

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    hr_path = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "HR_preprocessed.parquet"
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]

    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)
    valid_start = pd.Timestamp(f"{valid_year}-01-01")
    valid_end = pd.Timestamp(f"{valid_year}-12-31")
    df_valid = df[(df["race_date"] >= valid_start) & (df["race_date"] <= valid_end)].copy()
    print(f"Validation ({valid_year}): {len(df_valid):,} rows, {df_valid['race_id'].nunique():,} races")

    print(f"Loading HR payouts: {hr_path}")
    hr_all = pd.read_parquet(hr_path)
    valid_race_ids = set(df_valid["race_id"].astype(str).unique())
    hr_valid = hr_all[hr_all["race_id"].astype(str).isin(valid_race_ids)].copy()
    print(f"  HR (valid): {len(hr_valid):,} rows")

    feature_cols = get_feature_cols(df_valid, cfg)
    print(f"\nLoading models from: {models_dir}")
    models = load_models(models_dir)

    print(f"\n[手法1] Platt スケーリング (T_opt={T_opt})")
    platt = fit_platt_scaler(df_valid, models, feature_cols, T_opt)

    T_search = cfg.get("plackett_luce", {}).get(
        "roi_T_search", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    )
    print(f"\n[手法2] ROI 最大化 T 探索")
    T_roi = find_T_for_roi(df_valid, models, feature_cols, hr_valid, T_search)

    print(f"\n[手法3] Isotonic wide キャリブレーション (T_opt={T_opt})")
    isotonic = fit_isotonic_wide(df_valid, models, feature_cols, hr_valid, T_opt)

    print(f"\nSaving calibration models...")
    save_calibration_models(models_dir, platt=platt, isotonic=isotonic, T_roi=T_roi)

    return {"platt": platt, "T_roi": T_roi, "isotonic": isotonic}


# ─── オッズ帯別 Isotonic キャリブレーション ──────────────────────────────────────

WIDE_ODDS_BRACKETS: list[tuple[float, float]] = [
    (0.0, 3.0),
    (3.0, 8.0),
    (8.0, 20.0),
    (20.0, float("inf")),
]


def _build_wide_odds_lookup(
    years: list[int],
    odds_dir: Path,
) -> dict[str, dict[tuple[int, int], float]]:
    """
    WideOdds_{year}.csv を複数年読み込み、race_id -> {(h1,h2): odds} を返す。

    simulate_ev.py の _build_odds_lookup() と同等のロジック。
    predict.py 内で直接利用するために定義（循環インポート回避）。
    """
    lookup: dict[str, dict[tuple[int, int], float]] = {}
    for year in years:
        path = odds_dir / f"WideOdds_{year}.csv"
        if not path.exists():
            print(f"  [warn] WideOdds_{year}.csv not found, skipping")
            continue
        df_odds = pd.read_csv(path)
        df_odds = df_odds[(df_odds["odds_status"] == "ok") & df_odds["odds"].notna()].copy()
        df_odds["race_id_str"] = df_odds["race_id"].apply(lambda x: str(int(x)))
        df_odds["h_min"] = df_odds[["horse_num_1", "horse_num_2"]].min(axis=1).astype(int)
        df_odds["h_max"] = df_odds[["horse_num_1", "horse_num_2"]].max(axis=1).astype(int)
        df_odds["pair_key"] = list(zip(df_odds["h_min"], df_odds["h_max"]))
        for rid, grp in df_odds.groupby("race_id_str"):
            lookup[rid] = dict(zip(grp["pair_key"], grp["odds"].astype(float)))
    print(f"  WideOdds loaded: {len(lookup):,} races across {years}")
    return lookup


def assign_odds_bracket(odds: float) -> int:
    """
    WideOdds の decimal multiplier からブラケット番号を返す（0-indexed）。

    Parameters
    ----------
    odds : float（WideOdds decimal multiplier。NaN の場合は -1 を返す）

    Returns
    -------
    int: 0〜3（-1 = 未分類 / NaN）
    """
    if odds != odds:  # NaN check
        return -1
    if odds < 3.0:
        return 0
    elif odds < 8.0:
        return 1
    elif odds < 20.0:
        return 2
    else:
        return 3


def collect_wide_pair_data_with_odds(
    df: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    hr_df: pd.DataFrame,
    wide_odds_lookup: dict[str, dict[tuple[int, int], float]],
    T: float,
) -> pd.DataFrame:
    """
    各レースの全ペア (i, j) について Harville p_wide・WideOdds・is_wide_hit を返す。

    _collect_wide_pair_data() を拡張し、WideOdds ブラケット情報を追加する。

    Returns
    -------
    pd.DataFrame:
        p_wide_harville: float（Harville 生確率）
        prior_odds     : float（WideOdds decimal multiplier。NaN = 未取得）
        hit            : int（0 or 1）
        odds_bracket   : int（0〜3、prior_odds が NaN の場合は -1）
    """
    wide_lookup = _build_wide_lookup(hr_df)
    X_feat = df[feature_cols]
    preds = ensemble_predict(models, X_feat)
    dfc = df.copy()
    dfc["pred_score"] = preds

    rows: list[dict] = []
    for race_id, grp in dfc.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        grp_r = grp.reset_index(drop=True)
        horse_nums = grp_r["horse_num"].astype(int).values
        scores = grp_r["pred_score"].values
        probs = compute_race_probabilities(scores, float(T))
        n = len(grp_r)

        for i in range(n):
            for j in range(i + 1, n):
                p_w = float(probs["wide_matrix"][i, j])
                key = _norm_pair(int(horse_nums[i]), int(horse_nums[j]))
                payout = wide_lookup.get(rid, {}).get(key, 0)
                prior_odds_raw = wide_odds_lookup.get(rid, {}).get(key, None)

                if prior_odds_raw is not None:
                    bracket = assign_odds_bracket(float(prior_odds_raw))
                    prior_odds_val = float(prior_odds_raw)
                else:
                    bracket = -1
                    prior_odds_val = float("nan")

                rows.append({
                    "p_wide_harville": p_w,
                    "prior_odds": prior_odds_val,
                    "hit": int(payout > 0),
                    "odds_bracket": bracket,
                })

    return pd.DataFrame(rows)


def fit_bracket_isotonic(
    df_pairs: pd.DataFrame,
    min_samples: int = 100,
) -> dict[int, object]:
    """
    ブラケット別に Isotonic 回帰を学習する。

    Parameters
    ----------
    df_pairs    : collect_wide_pair_data_with_odds() の出力
    min_samples : 最小サンプル数（これを下回る帯は学習をスキップ）

    Returns
    -------
    dict[int, IsotonicRegression]: {bracket_id: fitted_model}
        bracket=-1 は除外。サンプル不足の帯は辞書に含まない。
    """
    from sklearn.isotonic import IsotonicRegression

    models_dict: dict[int, object] = {}
    for bracket in [0, 1, 2, 3]:
        subset = df_pairs[df_pairs["odds_bracket"] == bracket]
        n = len(subset)
        if n < min_samples:
            print(f"  [bracket {bracket}] samples={n} < {min_samples}, skip")
            continue
        X = subset["p_wide_harville"].values
        y = subset["hit"].values
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(X, y)
        models_dict[bracket] = iso
        hit_rate = float(y.mean())
        print(
            f"  [bracket {bracket}] n={n:,} "
            f"hit_rate={hit_rate:.4f} "
            f"p_wide range=[{X.min():.4f}, {X.max():.4f}]"
        )
    return models_dict


def apply_bracket_isotonic(
    p_wide_harville: float,
    prior_odds: float,
    bracket_models: dict[int, object],
) -> float:
    """
    ペアの Harville p_wide をブラケット別 Isotonic で補正する。

    フォールバック優先順位:
    1. 当該帯のモデルが存在 → そのモデルを使用
    2. 帯 3 のモデルが存在しない → 帯 2 のモデルを使用
    3. いずれも存在しない → p_wide_harville をそのまま返す
    """
    import math
    if math.isnan(prior_odds):
        return p_wide_harville

    bracket = assign_odds_bracket(prior_odds)
    if bracket == -1:
        return p_wide_harville

    model = bracket_models.get(bracket)
    if model is None and bracket == 3:
        model = bracket_models.get(2)  # 帯 3 フォールバック
    if model is None:
        return p_wide_harville

    return float(model.predict([p_wide_harville])[0])


def save_bracket_calibration(
    models_dir: Path,
    bracket_models: dict[int, object],
    meta: dict,
) -> None:
    """
    帯別キャリブレーションモデルを models/calibration/bracket_isotonic/ に保存する。

    保存先:
        models/calibration/bracket_isotonic/bracket_isotonic_{n}.joblib
        models/calibration/bracket_isotonic/bracket_meta.json
    """
    import joblib

    bracket_dir = models_dir / "calibration" / "bracket_isotonic"
    bracket_dir.mkdir(parents=True, exist_ok=True)

    for bracket_id, model in bracket_models.items():
        p = bracket_dir / f"bracket_isotonic_{bracket_id}.joblib"
        joblib.dump(model, p)
        print(f"  Saved: {p}")

    meta_path = bracket_dir / "bracket_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Saved: {meta_path}")


def load_bracket_calibration(models_dir: Path) -> tuple[dict, dict]:
    """
    保存済み帯別キャリブレーションを読み込む。

    Returns
    -------
    tuple[dict[int, IsotonicRegression], dict]:
        第1要素: bracket_models
        第2要素: meta（境界値・学習年）
    """
    import joblib

    bracket_dir = models_dir / "calibration" / "bracket_isotonic"
    bracket_models: dict[int, object] = {}
    meta: dict = {}

    meta_path = bracket_dir / "bracket_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    for bracket_id in [0, 1, 2, 3]:
        p = bracket_dir / f"bracket_isotonic_{bracket_id}.joblib"
        if p.exists():
            bracket_models[bracket_id] = joblib.load(p)
            print(f"  Loaded: {p}")

    return bracket_models, meta


def run_fit_bracket_calibration(cfg: dict | None = None) -> dict[int, object]:
    """
    帯別 Isotonic キャリブレーションをバリデーション 2024 で学習し保存する。

    train_config.json の calibration.fitted を True に更新する。

    Returns
    -------
    dict[int, IsotonicRegression]: 学習済み bracket_models
    """
    if cfg is None:
        cfg = load_config()

    calib_cfg = cfg.get("calibration", {})
    T_opt = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))
    valid_year = calib_cfg.get("valid_year", "2024")
    min_samples = calib_cfg.get("min_samples_per_bracket", 100)

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    hr_path = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "HR_preprocessed.parquet"
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    odds_dir = PROJECT_ROOT / "common" / "data" / "output" / "odds"

    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)
    valid_start = pd.Timestamp(f"{valid_year}-01-01")
    valid_end = pd.Timestamp(f"{valid_year}-12-31")
    df_valid = df[(df["race_date"] >= valid_start) & (df["race_date"] <= valid_end)].copy()
    print(f"Validation ({valid_year}): {len(df_valid):,} rows, {df_valid['race_id'].nunique():,} races")

    print(f"Loading HR payouts: {hr_path}")
    hr_all = pd.read_parquet(hr_path)
    valid_race_ids = set(df_valid["race_id"].astype(str).unique())
    hr_valid = hr_all[hr_all["race_id"].astype(str).isin(valid_race_ids)].copy()
    print(f"  HR (valid): {len(hr_valid):,} rows")

    print(f"\nLoading WideOdds for {valid_year}...")
    wide_odds_lookup = _build_wide_odds_lookup([int(valid_year)], odds_dir)

    feature_cols = get_feature_cols(df_valid, cfg)
    print(f"\nLoading models from: {models_dir}")
    models = load_models(models_dir)

    print(f"\n[帯別 Isotonic] Collecting wide pair data with odds (T_opt={T_opt})...")
    df_pairs = collect_wide_pair_data_with_odds(
        df_valid, models, feature_cols, hr_valid, wide_odds_lookup, T_opt
    )
    print(f"  Total pairs: {len(df_pairs):,}")
    for b in [0, 1, 2, 3]:
        n_b = int((df_pairs["odds_bracket"] == b).sum())
        n_na = int((df_pairs["odds_bracket"] == -1).sum())
        print(f"  bracket={b}: {n_b:,} pairs")
    print(f"  bracket=-1 (no odds): {n_na:,} pairs")

    print(f"\n[帯別 Isotonic] Fitting bracket isotonic models (min_samples={min_samples})...")
    bracket_models = fit_bracket_isotonic(df_pairs, min_samples=min_samples)

    meta = {
        "bracket_boundaries": [3.0, 8.0, 20.0],
        "valid_year": valid_year,
        "min_samples_per_bracket": min_samples,
        "fitted_brackets": list(bracket_models.keys()),
        "fitted": True,
    }

    print(f"\nSaving bracket calibration models...")
    save_bracket_calibration(models_dir, bracket_models, meta)

    # train_config.json の calibration セクションを更新
    cfg.setdefault("calibration", {})
    cfg["calibration"]["fitted"] = True
    cfg["calibration"]["fitted_brackets"] = list(bracket_models.keys())
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Updated config: {CONFIG_PATH}")

    return bracket_models


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
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="Fit Platt/ROI-T/Isotonic calibration models on validation set (2024)",
    )
    parser.add_argument(
        "--fit-bracket-calibration",
        action="store_true",
        help="Fit bracket-specific Isotonic calibration on validation set (2024)",
    )
    parser.add_argument(
        "--fit-lambda",
        action="store_true",
        help=(
            "Fit Stern-type lam2/lam3 on TRAIN+VALID (<=2024-12-31) via log-likelihood "
            "grid search and save to train_config.json.plackett_luce"
        ),
    )
    parser.add_argument(
        "--fit-market-blend",
        action="store_true",
        help=(
            "Fit market residual blend beta on TRAIN+VALID (<= valid_end) using SE win odds "
            "(betting layer only) and save to train_config.json.plackett_luce"
        ),
    )
    args = parser.parse_args()

    if not any([
        args.calibrate, args.eval, args.race_id,
        args.fit_calibration, args.fit_bracket_calibration, args.fit_lambda,
        args.fit_market_blend,
    ]):
        parser.print_help()
        sys.exit(1)

    cfg = load_config()
    if args.calibrate:
        run_calibrate(cfg)
    if args.eval:
        run_eval(cfg)
    if args.race_id:
        run_race_detail(args.race_id, cfg)
    if args.fit_calibration:
        run_fit_calibration(cfg)
    if args.fit_bracket_calibration:
        run_fit_bracket_calibration(cfg)
    if args.fit_lambda:
        run_fit_lambda(cfg)
    if args.fit_market_blend:
        run_fit_market_blend(cfg)


if __name__ == "__main__":
    main()

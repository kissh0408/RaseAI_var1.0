"""Plackett-Luceモデルによる多馬確率計算。

lambdarankスコアから単勝・複勝・馬連確率を計算する。
Harville公式を使いつつ、温度パラメータによる確率シャープ化を行う。

温度パラメータ T:
  T < 1.0 → 分布が鋭くなる（上位馬の確率が高まる）
  T > 1.0 → 分布が平坦になる
  実証: T=0.70〜0.80 でLog Lossが最小化されることが多い
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# コア確率計算
# ---------------------------------------------------------------------------

def win_probabilities(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Plackett-Luce: 温度付きSoftmaxで単勝確率を計算する。

    P(i wins) = exp(s_i / T) / Σ exp(s_k / T)
    数値安定化のため最大値を引いてからexp。
    T < 1.0 で分布がシャープになり、実際の勝率との乖離が解消される。
    """
    t = max(temperature, 1e-6)  # ゼロ除算防止
    s = (scores - scores.max()) / t
    exp_s = np.exp(s)
    return exp_s / exp_s.sum()


def place_probabilities_harville(win_probs: np.ndarray) -> np.ndarray:
    """Harville公式で複勝確率（3着以内）を計算する。

    P(i in top3) = P(i=1) + P(i=2) + P(i=3)
    各項はHarville展開で計算。
    """
    n = len(win_probs)
    if n == 1:
        return win_probs.copy()

    place_probs = np.zeros(n)
    for i in range(n):
        p = win_probs.copy()

        p_win = p[i]

        p_second = 0.0
        for j in range(n):
            if j == i:
                continue
            remaining = p.copy()
            remaining[j] = 0.0
            remaining_sum = remaining.sum()
            if remaining_sum > 0:
                p_second += p[j] * (p[i] / remaining_sum)

        p_third = 0.0
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                remaining_jk = p.copy()
                remaining_jk[j] = 0.0
                remaining_jk[k] = 0.0
                sum_jk = remaining_jk.sum()
                remaining_j = p.copy()
                remaining_j[j] = 0.0
                sum_j = remaining_j.sum()
                if sum_j > 0 and sum_jk > 0:
                    p_third += p[j] * (p[k] / sum_j) * (p[i] / sum_jk)

        place_probs[i] = p_win + p_second + p_third

    return np.clip(place_probs, 0.0, 1.0)


def exacta_probability_harville(win_probs: np.ndarray, i: int, j: int) -> float:
    """Harville公式で馬単（i→j）の確率を計算する。"""
    p_i = win_probs[i]
    if p_i >= 1.0:
        return 0.0
    p_j_given_i_wins = win_probs[j] / (1.0 - p_i)
    return float(p_i * p_j_given_i_wins)


def quinella_probability(win_probs: np.ndarray, i: int, j: int) -> float:
    """馬連（i-j どちらかが1着・もう片方が2着）確率。"""
    return exacta_probability_harville(win_probs, i, j) + exacta_probability_harville(win_probs, j, i)


def compute_race_probabilities(
    df_race: pd.DataFrame,
    score_col: str = "raw_score",
    temperature: float = 1.0,
) -> pd.DataFrame:
    """1レース分のDataFrameに対して各種確率を追加して返す。"""
    df = df_race.copy().reset_index(drop=True)
    scores = df[score_col].values.astype(float)

    win_probs = win_probabilities(scores, temperature=temperature)
    place_probs = place_probabilities_harville(win_probs)

    df["win_prob"] = win_probs
    df["place_prob"] = place_probs
    return df


def apply_all_races(
    df: pd.DataFrame,
    score_col: str = "raw_score",
    temperature: float = 1.0,
) -> pd.DataFrame:
    """全レースに対してPlackett-Luce確率を計算する。"""
    results = []
    for _, group in df.groupby("race_id"):
        enriched = compute_race_probabilities(group, score_col, temperature=temperature)
        results.append(enriched)
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 温度チューニング
# ---------------------------------------------------------------------------

def tune_temperature(
    df_valid: pd.DataFrame,
    score_col: str = "raw_score",
    label_col: str = "is_win",
    t_range: tuple[float, float] = (0.3, 3.0),
) -> float:
    """バリデーションデータでLog Lossを最小化する温度 T を探索する。

    T=0.70〜0.80 が競馬データでは最適になることが多い。
    scipy が使えない場合は格子探索にフォールバックする。
    """
    def _log_loss_for_T(T: float) -> float:
        total_loss = 0.0
        count = 0
        for _, group in df_valid.groupby("race_id"):
            scores = group[score_col].values
            wins = group[label_col].values
            if wins.sum() == 0:
                continue
            probs = win_probabilities(scores, temperature=T)
            probs = np.clip(probs, 1e-9, 1.0)
            total_loss += -float(np.sum(wins * np.log(probs)))
            count += len(group)
        return total_loss / max(count, 1)

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(
            _log_loss_for_T,
            bounds=t_range,
            method="bounded",
            options={"xatol": 1e-4},
        )
        optimal_t = float(result.x)
    except ImportError:
        # scipy がない場合は格子探索
        candidates = np.linspace(t_range[0], t_range[1], 50)
        losses = [_log_loss_for_T(t) for t in candidates]
        optimal_t = float(candidates[int(np.argmin(losses))])

    print(f"  温度チューニング完了: T={optimal_t:.3f} (Log Loss={_log_loss_for_T(optimal_t):.5f})")
    return optimal_t


# ---------------------------------------------------------------------------
# キャリブレーション評価（Quantile-based ECE）
# ---------------------------------------------------------------------------

def calibration_ece_quantile(
    df: pd.DataFrame,
    prob_col: str = "win_prob",
    label_col: str = "is_win",
    n_bins: int = 10,
) -> dict:
    """Quantile-based binning でキャリブレーション誤差(ECE)を計算する。

    等間隔binは高確率域のサンプル不足で分散が爆発するため、
    各binに同数サンプルを割り当てる分位数binを使用する。

    ECE = Σ (|conf_b - acc_b| × n_b / N)
    """
    df = df.copy().dropna(subset=[prob_col, label_col])
    if len(df) < n_bins:
        return {"ece": np.nan, "n_samples": len(df), "bins": []}

    df["_bin"] = pd.qcut(df[prob_col], q=n_bins, duplicates="drop", labels=False)

    ece = 0.0
    n = len(df)
    bin_details = []

    for bin_id, grp in df.groupby("_bin"):
        if len(grp) == 0:
            continue
        conf = float(grp[prob_col].mean())
        acc = float(grp[label_col].mean())
        weight = len(grp) / n
        ece += weight * abs(conf - acc)
        bin_details.append({
            "bin": int(bin_id),
            "n": len(grp),
            "confidence": conf,
            "accuracy": acc,
            "gap": conf - acc,
        })

    return {
        "ece": float(ece),
        "n_samples": n,
        "n_bins": len(bin_details),
        "bins": bin_details,
    }

"""place_direct: 純関数ライブラリ（target / filters / normalize）。

このモジュールは実験全体（build_dataset.py / train_fold2.py / export_probs.py /
evaluate_place.py / tests/）から import される単一の真実。
学習・エクスポート・評価のロジックがここと食い違わないよう、
target 定義やフィルタ条件を各スクリプトで再実装しない。

本番コード（pure_rank/src/*）・prob_fusion・betting のロジックはコピーせず import する
（CLAUDE.md の「後出し禁止・重複禁止」原則、および本実験仕様書 §7 の指示）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_PURE_RANK_SRC = str(_ROOT / "pure_rank" / "src")


def _import_get_feature_cols():
    """pure_rank/src/common.py がトップレベル common パッケージ（common/data/...）を
    シャドウし、後続テストの `from common.data...` import を壊す
    （2026-07-09に evaluation/odds_loader.py で見つかったのと同じバグパターン）ため、
    挿入をこの関数内に限定し、終了時に必ず後始末する。
    """
    inserted = _PURE_RANK_SRC not in sys.path
    if inserted:
        sys.path.insert(0, _PURE_RANK_SRC)
    try:
        from common import get_feature_cols

        return get_feature_cols
    finally:
        if inserted:
            sys.path.remove(_PURE_RANK_SRC)
            sys.modules.pop("common", None)


get_feature_cols = _import_get_feature_cols()

# 本実験が新規に付与するラベル列。本番 common.FORBIDDEN_COLS には存在しないため、
# 特徴量選択時に明示的に除外しないと label leakage になる（最重要チェック）。
EXPERIMENT_LABEL_COLS: frozenset[str] = frozenset({"target_place"})


def get_experiment_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """place_direct 実験の特徴量列を返す。

    本番 common.get_feature_cols() をそのまま再利用し、実験固有のラベル列
    （target_place）を追加除外する。train_fold2.py / export_probs.py は
    必ずこの関数を通し、独自に特徴量列を組み立てない。
    """
    base_cols = get_feature_cols(df, cfg)
    return [c for c in base_cols if c not in EXPERIMENT_LABEL_COLS]


# ─── target ────────────────────────────────────────────────────────────────


def compute_target_place(finish_rank: pd.Series) -> pd.Series:
    """target_place = 1 if finish_rank <= 3 else 0。

    finish_rank は当該レースの確定着順（結果変数）。shift は不要
    （target は「予測対象」であり特徴量ではないため）。
    """
    return (finish_rank <= 3).astype(int)


# ─── フィルタ（本番標準と同一。pure_rank/config/train_config.json の filters を使う）───


def apply_base_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


# ─── 正規化（レース内合計を 3 に固定。§4） ─────────────────────────────────────


def _normalize_one_race(p: np.ndarray, target_sum: float = 3.0, max_iter: int = 10) -> tuple[np.ndarray, int]:
    """1レース分の p_raw を合計 target_sum に正規化し、1.0 超過分を再配分する。

    手順:
      1. p_norm = p * target_sum / sum(p)
      2. p_norm > 1.0 の馬を 1.0 に clip し、超過分を未 clip 馬へ比例配分
      3. 収束（clip 馬が出なくなる）まで最大 max_iter 回繰り返す

    戻り値: (正規化後確率, clip が発生した馬の頭数)
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    total = p.sum()
    if total <= 0:
        return np.full(n, target_sum / n), 0

    p_norm = p * target_sum / total
    ever_clipped = np.zeros(n, dtype=bool)

    for _ in range(max_iter):
        over = p_norm > 1.0 + 1e-12
        over = over & ~ever_clipped
        if not over.any():
            break
        ever_clipped |= over
        excess = float((p_norm[over] - 1.0).sum())
        p_norm[over] = 1.0

        under = ~ever_clipped
        if not under.any() or excess <= 0:
            break
        weight_sum = p_norm[under].sum()
        if weight_sum <= 0:
            # 残り馬の重みが 0 の場合は均等配分でフォールバック
            p_norm[under] = p_norm[under] + excess / under.sum()
        else:
            p_norm[under] = p_norm[under] + excess * (p_norm[under] / weight_sum)

    return p_norm, int(ever_clipped.sum())


def normalize_place_probs(
    p_raw: np.ndarray, race_id: np.ndarray, max_iter: int = 10
) -> tuple[np.ndarray, int]:
    """レースごとに p_raw をレース内合計 3 へ正規化する（§4 の clip+再配分方式）。

    Parameters
    ----------
    p_raw : 生予測確率（複数レース分をまたいだ 1 次元配列）
    race_id : p_raw と同じ長さの race_id 配列
    max_iter : clip→再配分の反復上限

    Returns
    -------
    (p_norm, total_clip_count) : 正規化後確率（p_raw と同じ順序）、
        レース横断の clip 発生頭数合計
    """
    p_raw = np.asarray(p_raw, dtype=float)
    race_id = np.asarray(race_id)
    n = len(p_raw)
    if n != len(race_id):
        raise ValueError("p_raw と race_id の長さが一致しません")

    p_norm = np.empty(n, dtype=float)
    total_clip = 0
    df = pd.DataFrame({"p_raw": p_raw, "pos": np.arange(n)})
    for _, grp in df.groupby(race_id, sort=False):
        positions = grp["pos"].to_numpy()
        vals = grp["p_raw"].to_numpy()
        normed, clip_ct = _normalize_one_race(vals, max_iter=max_iter)
        p_norm[positions] = normed
        total_clip += clip_ct

    return p_norm, total_clip


# ─── logloss（数値安定化つき。§4 の eps=1e-12） ────────────────────────────────


def place_logloss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

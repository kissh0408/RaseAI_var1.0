"""place_calibration: 純関数ライブラリ（λ logloss fit / 頭数帯割当 / isotonic wrapper / 順位保存）。

このモジュールは実験全体（build_dataset.py / fit_calibrators.py / export_probs.py /
evaluate_calibration.py / tests/）から import される単一の真実。

既存モジュールはコピーせず import して再利用する
（CLAUDE.md の「後出し禁止・重複禁止」原則、および仕様書 §7 の指示）:
  - Stern 式本体（stern_second_prob / stern_third_prob / place_prob_from_p_win）:
    prob_fusion.src.place_prob
  - レース内正規化（B2 用）・logloss:
    pure_rank.experiments.place_direct.place_lib（前フェーズの純関数を import 再利用。コピー禁止）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import optimize

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PLACE_DIRECT_DIR = str(_ROOT / "pure_rank" / "experiments" / "place_direct")


def _import_place_direct_lib():
    """place_direct/place_lib.py を import する。

    place_direct ディレクトリを sys.path に一時的に挿入する。place_direct/tests/ の
    既存パターン（common.py のトップレベル common パッケージ衝突回避）に倣い、
    後始末を必ず行う。place_lib.py 自体は pure_rank/src の common をインポートするため
    その挿入・除去は place_lib 内部（_import_get_feature_cols）で完結している。
    """
    inserted = _PLACE_DIRECT_DIR not in sys.path
    if inserted:
        sys.path.insert(0, _PLACE_DIRECT_DIR)
    try:
        import place_lib

        return place_lib
    finally:
        if inserted:
            sys.path.remove(_PLACE_DIRECT_DIR)
            sys.modules.pop("place_lib", None)


_place_lib = _import_place_direct_lib()
normalize_place_probs = _place_lib.normalize_place_probs
place_logloss = _place_lib.place_logloss

from prob_fusion.src.place_prob import place_prob_from_p_win  # noqa: E402

EPS = 1e-12
LAMBDA_BOUNDS: list[tuple[float, float]] = [(0.1, 3.0), (0.1, 3.0)]
BAND_MAX_LE7 = 7
BANDS: tuple[str, str] = ("le7", "ge8")


# ─── 頭数帯割当（A2用。5–7頭 / 8頭以上の2帯に事前固定。§3.1） ──────────────────────


def band_5to7_8plus(horse_count: np.ndarray | list[int]) -> np.ndarray:
    """頭数帯割当: horse_count<=7 -> 'le7', 8頭以上 -> 'ge8'。

    境界値の扱い: 7頭 -> 'le7'、8頭 -> 'ge8'（仕様書 §8 test_band_split の検証内容）。
    """
    hc = np.asarray(horse_count)
    return np.where(hc <= BAND_MAX_LE7, "le7", "ge8")


# ─── λ logloss 目的関数フィット（A1 / A2） ────────────────────────────────────────


def logloss_objective(
    params: np.ndarray,
    races_p_win: list[np.ndarray],
    races_y: list[np.ndarray],
) -> float:
    """per-horse logloss（レース平均のレース間平均）。仕様書 §3.1 の定義どおり。"""
    lam2, lam3 = params
    if lam2 <= 0 or lam3 <= 0:
        return 1e6
    total = 0.0
    n = 0
    for p_win, y in zip(races_p_win, races_y):
        p_place = place_prob_from_p_win(p_win, lam2, lam3)
        p_place = np.clip(p_place, EPS, 1 - EPS)
        y_arr = np.asarray(y, dtype=float)
        total += float(-np.mean(y_arr * np.log(p_place) + (1 - y_arr) * np.log(1 - p_place)))
        n += 1
    return total / max(n, 1)


def fit_lambda_logloss(
    races_p_win: list[np.ndarray],
    races_y: list[np.ndarray],
    *,
    init_lam2: float = 0.6017839448116524,
    init_lam3: float = 0.6381161426667171,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """logloss 目的で lam2, lam3 を fit する（A1: global）。

    fit_stern_lambda（Brier 目的、prob_fusion/src/place_prob.py）と同型のオプティマイザ
    （L-BFGS-B、bounds [0.1, 3.0]）を使うが目的関数のみ logloss に変える（仕様書 §3.1）。
    """
    b = bounds if bounds is not None else LAMBDA_BOUNDS
    res = optimize.minimize(
        logloss_objective,
        x0=np.array([init_lam2, init_lam3]),
        args=(races_p_win, races_y),
        bounds=b,
        method="L-BFGS-B",
    )
    return float(res.x[0]), float(res.x[1])


def fit_lambda_logloss_banded(
    races_p_win: list[np.ndarray],
    races_y: list[np.ndarray],
    races_band: list[str],
    *,
    bands: tuple[str, ...] = BANDS,
    init_lam2: float = 0.6017839448116524,
    init_lam3: float = 0.6381161426667171,
    bounds: list[tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """頭数帯別に logloss 目的で lam2, lam3 を fit する（A2: 2帯別。パラメータ数4）。"""
    out: dict[str, dict[str, float]] = {}
    for b in bands:
        idx = [i for i, bb in enumerate(races_band) if bb == b]
        p_sub = [races_p_win[i] for i in idx]
        y_sub = [races_y[i] for i in idx]
        if not p_sub:
            out[b] = {"lam2": float("nan"), "lam3": float("nan"), "n_races": 0}
            continue
        lam2, lam3 = fit_lambda_logloss(
            p_sub, y_sub, init_lam2=init_lam2, init_lam3=init_lam3, bounds=bounds
        )
        out[b] = {"lam2": lam2, "lam3": lam3, "n_races": len(idx)}
    return out


# ─── isotonic 事後較正（B1 / B2） ─────────────────────────────────────────────────


def fit_isotonic(p_stern_fit: np.ndarray, y_fit: np.ndarray, *, y_min: float = 0.0, y_max: float = 1.0):
    """fit 期間の (p_stern, y_place) ペアに isotonic regression を fit する（B1/B2共通）。

    sklearn.isotonic.IsotonicRegression(out_of_bounds="clip") を使う。
    lam は formal 値 (0.6018/0.6381) に固定し、これは呼び出し側で p_stern_fit として
    渡す前に確定させておく（再フィットとの直交性。仕様書 §3.2）。
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip")
    iso.fit(np.asarray(p_stern_fit, dtype=float), np.asarray(y_fit, dtype=float))
    return iso


# ─── 順位保存（§6.2の検証項目。isotonic平坦区間・λ変更後もレース内top1がS0と一致するか） ──


def top1_index(p: np.ndarray, tiebreak: np.ndarray | None = None) -> int:
    """レース内 top1 のインデックスを返す。

    tiebreak を渡した場合、p が同値のときは tiebreak の降順で安定的に決める
    （isotonic の平坦区間で p_place が同値になった場合、元の p_stern 順で順位保存する。
    仕様書 §6.2 注記）。
    """
    p = np.asarray(p, dtype=float)
    if tiebreak is None:
        return int(np.argmax(p))
    tb = np.asarray(tiebreak, dtype=float)
    # np.lexsort: 最後のキーが第一ソートキー（昇順）。-p の昇順 = p の降順。
    # 同値時は -tb の昇順 = tb の降順で安定的にタイブレークする。
    order = np.lexsort((-tb, -p))
    return int(order[0])

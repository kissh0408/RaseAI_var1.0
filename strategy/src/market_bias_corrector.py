"""
market_bias_corrector.py
========================
パリミュチュエル市場のFavorite-Longshot Bias（FLB）を補正するモジュール。

問題: 1/odds（暗示確率）はFLBを含む。
  - 本命馬（低オッズ）: 実際の勝率より過小評価される傾向
  - 大穴馬（高オッズ）: 実際の勝率より過大評価される傾向
これを無補正でmarket_shrinkageブレンドに使うとEV計算が歪む。

解決: SE_preprocessed.parquetの過去データ（2024年末まで）で
  x = 1/odds, y = is_win としてIsotonicRegressionをフィット。
  補正済み市場確率をshrinkageブレンドに使用する。
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

# デフォルトパス（プロジェクトルートからの相対パス）
_DEFAULT_SE_PATH = (
    Path(__file__).parents[2]
    / "model_training"
    / "data"
    / "01_preprocessed"
    / "SE_preprocessed.parquet"
)
_DEFAULT_MODEL_PATH = (
    Path(__file__).parents[2]
    / "model_training"
    / "models"
    / "market_bias_corrector_isotonic.pkl"
)
_DEFAULT_META_PATH = (
    Path(__file__).parents[2]
    / "model_training"
    / "models"
    / "market_bias_corrector_isotonic_meta.json"
)


def fit_market_bias_corrector(
    se_path: Optional[Path] = None,
    model_save_path: Optional[Path] = None,
    meta_save_path: Optional[Path] = None,
    cutoff_year: int = 2024,
    min_odds: float = 1.01,
    max_odds: float = 999.9,
    out_of_bounds: str = "clip",
) -> IsotonicRegression:
    """
    SE_preprocessed.parquetの過去データ（cutoff_year以前）で
    1/odds vs is_win の IsotonicRegression をフィットし、保存する。

    テスト期間のデータ汚染防止のため、cutoff_year（デフォルト2024）以前のみ使用。

    Parameters
    ----------
    se_path : Path, optional
        SE_preprocessed.parquetのパス（デフォルト: model_training/data/01_preprocessed/）
    model_save_path : Path, optional
        モデル保存先（デフォルト: model_training/models/market_bias_corrector_isotonic.pkl）
    meta_save_path : Path, optional
        メタ情報保存先（デフォルト: model_training/models/market_bias_corrector_isotonic_meta.json）
    cutoff_year : int
        このyear以前のデータのみ学習に使用（テスト汚染防止）
    min_odds, max_odds : float
        学習データのオッズ範囲フィルタ
    out_of_bounds : str
        IsotonicRegressionのout_of_bounds（"clip" or "nan"）

    Returns
    -------
    IsotonicRegression
        フィット済みモデル
    """
    se_path = se_path or _DEFAULT_SE_PATH
    model_save_path = model_save_path or _DEFAULT_MODEL_PATH
    meta_save_path = meta_save_path or _DEFAULT_META_PATH

    logger.info("Loading SE_preprocessed from %s", se_path)
    df = pd.read_parquet(se_path, columns=["year", "odds", "finish_rank"])

    # テスト期間のデータを除外（時系列リーク防止）
    df_train = df[df["year"] <= cutoff_year].copy()
    logger.info("Total rows: %d, Training rows (year <= %d): %d", len(df), cutoff_year, len(df_train))

    # オッズフィルタ: 有効オッズ範囲のみ
    odds = pd.to_numeric(df_train["odds"], errors="coerce")
    mask_valid = odds.between(min_odds, max_odds) & odds.notna()
    df_train = df_train[mask_valid].copy()
    odds = odds[mask_valid]

    # 1/odds（暗示確率、正規化前の生値）
    raw_implied = (1.0 / odds).clip(lower=1e-6, upper=1.0)
    is_win = (pd.to_numeric(df_train["finish_rank"], errors="coerce") == 1).astype(float)

    # NaN除外
    valid = raw_implied.notna() & is_win.notna()
    x = raw_implied[valid].to_numpy(dtype=np.float64)
    y = is_win[valid].to_numpy(dtype=np.float64)

    logger.info("Fitting IsotonicRegression on %d samples...", len(x))
    iso = IsotonicRegression(increasing=True, out_of_bounds=out_of_bounds)
    iso.fit(x, y)

    # フィット品質ログ: オッズ帯別の補正前後比較
    _log_correction_comparison(x, y, iso)

    # 保存
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(iso, f)
    logger.info("Model saved to %s", model_save_path)

    # メタ情報
    meta = {
        "model_type": "isotonic_regression",
        "purpose": "FLB (Favorite-Longshot Bias) correction for market probabilities",
        "cutoff_year": cutoff_year,
        "n_samples": int(len(x)),
        "n_train_years": int(df_train["year"].nunique()),
        "train_year_range": [int(df_train["year"].min()), int(df_train["year"].max())],
        "odds_range": [float(min_odds), float(max_odds)],
        "out_of_bounds": out_of_bounds,
        "isotonic_x_range": [float(iso.X_min_), float(iso.X_max_)],
    }
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Meta saved to %s", meta_save_path)

    return iso


def _log_correction_comparison(
    x: np.ndarray,
    y: np.ndarray,
    iso: IsotonicRegression,
    n_bins: int = 8,
) -> None:
    """
    オッズ帯別に補正前（1/odds）と補正後の確率、実際の勝率を比較してログ出力。
    """
    corrected = iso.predict(x)
    # 1/oddsを対数スケールで等分割
    bins = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return

    logger.info("=== Market Bias Correction: FLB Analysis ===")
    logger.info("%-15s %-8s %-12s %-12s %-12s %-6s", "odds_range", "n", "raw_implied", "corrected", "actual_win", "bias")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (x >= lo) & (x < hi if i < len(bins) - 2 else x <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        raw_mean = float(x[mask].mean())
        cor_mean = float(corrected[mask].mean())
        actual = float(y[mask].mean())
        bias_dir = "OVER" if raw_mean > actual else "under"
        # 対応するオッズ範囲を表示
        odds_lo = round(1.0 / hi, 1) if hi > 0 else "inf"
        odds_hi = round(1.0 / lo, 1) if lo > 0 else "inf"
        logger.info(
            "%-15s %-8d %-12.4f %-12.4f %-12.4f %-6s",
            f"{odds_lo}-{odds_hi}",
            n,
            raw_mean,
            cor_mean,
            actual,
            bias_dir,
        )
    logger.info("=== End FLB Analysis ===")


def load_market_bias_corrector(
    model_path: Optional[Path] = None,
) -> Optional[IsotonicRegression]:
    """
    保存済みのFLB補正モデルを読み込む。
    モデルが存在しない場合はNoneを返す（補正なしのフォールバック）。
    """
    model_path = model_path or _DEFAULT_MODEL_PATH
    if not model_path.exists():
        logger.warning("Market bias corrector not found at %s. FLB correction disabled.", model_path)
        return None
    with open(model_path, "rb") as f:
        iso = pickle.load(f)
    logger.debug("Market bias corrector loaded from %s", model_path)
    return iso


def correct_market_prob(
    odds: float,
    corrector: Optional[IsotonicRegression] = None,
) -> float:
    """
    単一オッズ値に対してFLB補正を適用した市場確率を返す。

    Parameters
    ----------
    odds : float
        単勝オッズ（1.01以上）
    corrector : IsotonicRegression, optional
        フィット済みFLB補正モデル。Noneの場合は1/oddsをそのまま返す。

    Returns
    -------
    float
        補正済み市場確率 [0, 1]
    """
    raw_implied = float(np.clip(1.0 / max(float(odds), 1.01), 0.0, 1.0))
    if corrector is None:
        return raw_implied
    corrected = float(corrector.predict([raw_implied])[0])
    return float(np.clip(corrected, 0.0, 1.0))


def correct_market_probs_series(
    odds_series: pd.Series,
    corrector: Optional[IsotonicRegression] = None,
) -> pd.Series:
    """
    オッズSeriesに対してベクトル化FLB補正を適用する。
    strategy_engine.py の _market_probabilities() から呼び出す。

    Parameters
    ----------
    odds_series : pd.Series
        単勝オッズのSeries（数値型）
    corrector : IsotonicRegression, optional
        フィット済みFLB補正モデル。Noneの場合は1/oddsをそのまま返す。

    Returns
    -------
    pd.Series
        補正済み市場確率（正規化前）。race_id でグループ正規化は呼び出し側で行う。
    """
    odds_clipped = pd.to_numeric(odds_series, errors="coerce").clip(lower=1.01).fillna(1.01)
    raw_implied = (1.0 / odds_clipped).clip(lower=0.0, upper=1.0)

    if corrector is None:
        return raw_implied

    x_arr = raw_implied.to_numpy(dtype=np.float64)
    corrected = corrector.predict(x_arr)
    return pd.Series(
        np.clip(corrected, 0.0, 1.0),
        index=odds_series.index,
        dtype=np.float64,
    )

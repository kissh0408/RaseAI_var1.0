"""Rank1 勝率キャリブレーション指標とレポート出力。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _binary_labels(finish_rank: pd.Series) -> np.ndarray:
    return (pd.to_numeric(finish_rank, errors="coerce") == 1).astype(np.float64).to_numpy()


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(probs) & np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    p = np.clip(probs[m], 0.0, 1.0)
    return float(np.mean((p - y[m]) ** 2))


def _ece_quantile(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    m = np.isfinite(probs) & np.isfinite(y)
    if m.sum() < n_bins * 5:
        return float("nan")
    p = np.clip(probs[m], 0.0, 1.0)
    yy = y[m]
    try:
        bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 3:
            return float("nan")
    except Exception:
        return float("nan")
    ece = 0.0
    n = len(p)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi if i == len(bins) - 2 else p < hi)
        if mask.sum() == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(yy[mask].mean())
        ece += abs(conf - acc) * (mask.sum() / n)
    return float(ece)


def compute_rank1_calibration_metrics(
    df: pd.DataFrame,
    *,
    score_col: str = "pred_score",
    isotonic_model: IsotonicRegression | None = None,
    ece_gate_max: float = 0.05,
) -> dict:
    """OOF pred_score と 1着ラベルから Brier / ECE を計算。"""
    if score_col not in df.columns or "finish_rank" not in df.columns:
        return {"status": "missing_columns"}

    x = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = _binary_labels(df["finish_rank"])
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 100:
        return {"status": "insufficient_samples", "n": int(len(x))}

    raw_clip = np.clip(x, 0.0, 1.0)
    brier_raw = _brier(raw_clip, y)
    ece_raw = _ece_quantile(raw_clip, y)

    brier_iso = float("nan")
    ece_iso = float("nan")
    if isotonic_model is not None:
        iso_p = np.asarray(isotonic_model.predict(x.reshape(-1)), dtype=np.float64)
        brier_iso = _brier(iso_p, y)
        ece_iso = _ece_quantile(iso_p, y)

    is_degraded = (
        np.isfinite(brier_iso)
        and np.isfinite(brier_raw)
        and brier_iso > brier_raw + 1e-6
    )
    ece_gate_failed = np.isfinite(ece_iso) and ece_iso > ece_gate_max

    if is_degraded:
        status = "degraded"
    elif ece_gate_failed:
        status = "ece_high"
    else:
        status = "ok"

    return {
        "n": int(len(x)),
        "brier_raw": brier_raw,
        "brier_isotonic": brier_iso,
        "ece_raw_quantile": ece_raw,
        "ece_isotonic_quantile": ece_iso,
        "is_degraded": bool(is_degraded),
        "ece_gate_failed": bool(ece_gate_failed),
        "status": status,
    }


def write_calibration_report(
    df: pd.DataFrame,
    feature_set: str,
    output_dir: Path,
    *,
    score_col: str = "pred_score",
    isotonic_model: IsotonicRegression | None = None,
) -> Path:
    """キャリブレーション JSON レポートを output_dir に書き出す。"""
    metrics = compute_rank1_calibration_metrics(
        df, score_col=score_col, isotonic_model=isotonic_model
    )
    out = Path(output_dir) / f"calibration_report_{feature_set}.json"
    payload = {"feature_set": feature_set, **metrics}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

"""
run_mlops_drift.py — Track E: OOF リプレイ ADWIN ドリフト監視（アラートのみ）

監視対象:
  - レース平均 LogLoss（Brier 近似）
  - 予測確率（confidence）分布の平均
  - EV 上位馬（edge>0）の出現率
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPECv2_OOF = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_specv2_oof.csv"
OUT = PROJECT_ROOT / "logs" / "mlops" / "drift_report.json"


class SimpleADWIN:
    """最小 ADWIN 近似（学習期間外監視 MVP）。"""

    def __init__(self, delta: float = 0.002, max_window: int = 400) -> None:
        self.delta = delta
        self.max_window = max_window
        self.window: list[float] = []
        self.drift_points: list[int] = []

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        if len(self.window) > self.max_window:
            self.window = self.window[-self.max_window :]
        n = len(self.window)
        if n < 30:
            return False
        split = n // 2
        w0 = self.window[:split]
        w1 = self.window[split:]
        m0, m1 = float(np.mean(w0)), float(np.mean(w1))
        eps = math.sqrt((1.0 / (2 * n)) * math.log(4 / self.delta))
        if abs(m0 - m1) > eps:
            self.drift_points.append(n)
            self.window = self.window[split:]
            return True
        return False


def _race_series(eval_df: pd.DataFrame, calibrator) -> pd.DataFrame:
    from strategy.src.betting_framework import load_evaluation

    df = eval_df.copy()
    if calibrator is not None:
        df["pred_prob"] = calibrator.transform(df["pred_rank1"]).clip(0.0, 1.0)
    else:
        df["pred_prob"] = pd.to_numeric(df["pred_rank1"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    grp_sum = df.groupby("race_id")["pred_prob"].transform("sum").clip(lower=1e-12)
    df["pred_prob"] = df["pred_prob"] / grp_sum
    rows = []
    for race_id, g in df.groupby("race_id", sort=True):
        p = g["pred_prob"].to_numpy(dtype=float)
        p = p / max(p.sum(), 1e-12)
        y = (g["finish_rank"].astype(int) == 1).to_numpy(dtype=int)
        ll = -float(np.sum(y * np.log(np.clip(p, 1e-12, 1.0))))
        conf_mean = float(np.mean(p))
        conf_var = float(np.var(p))
        odds = pd.to_numeric(g["odds"], errors="coerce").fillna(1.01)
        ev = p * odds.to_numpy(dtype=float)
        top_ev_rate = float((ev >= 1.05).mean())
        rows.append(
            {
                "race_id": race_id,
                "logloss": ll,
                "conf_mean": conf_mean,
                "conf_var": conf_var,
                "top_ev_rate": top_ev_rate,
                "valid_year": int(g["valid_year"].iloc[0]) if "valid_year" in g.columns else None,
            }
        )
    return pd.DataFrame(rows)


def _monitor(series: pd.Series, *, delta: float = 0.002) -> dict:
    adw = SimpleADWIN(delta=delta)
    drifts = []
    for i, v in enumerate(series.to_numpy(dtype=float)):
        if adw.update(float(v)):
            drifts.append(i)
    return {
        "n": int(len(series)),
        "mean": float(series.mean()) if len(series) else 0.0,
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "drift_count": len(drifts),
        "drift_indices": drifts[-5:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from main.pipeline.strategy_pipeline import resolve_strategy_calibration_path
    from strategy.src.betting_framework import ProbabilityCalibrator, load_evaluation

    eval_df = load_evaluation(SPECv2_OOF)
    eval_df = eval_df[pd.to_numeric(eval_df["valid_year"], errors="coerce") == args.year].copy()
    cal_path = resolve_strategy_calibration_path(PROJECT_ROOT)
    calibrator = ProbabilityCalibrator.from_json(cal_path) if cal_path.is_file() else None

    race_df = _race_series(eval_df, calibrator)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "track": "E_mlops_adwin",
        "year": args.year,
        "alert_only": True,
        "monitors": {
            "logloss": _monitor(race_df["logloss"]),
            "confidence_mean": _monitor(race_df["conf_mean"]),
            "confidence_var": _monitor(race_df["conf_var"]),
            "top_ev_horse_rate": _monitor(race_df["top_ev_rate"]),
        },
    }
    any_drift = any(m["drift_count"] > 0 for m in report["monitors"].values())
    report["any_drift_detected"] = any_drift

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out} any_drift={any_drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

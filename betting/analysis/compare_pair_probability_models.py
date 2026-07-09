"""ワイド/馬連ペア確率モデル比較: Stern式(pair_probs.py) vs Harville式(ev_filters.py)。

fold2 OOS L1スコア + 本番fusion_oos_fold2.jsonのformalパラメータ(α,β,λ2,λ3)を使い、
TEST期間(2025+)で実際の着順に対するlogloss / 較正誤差を両モデルで測定して比較する。
本番評価レポート（evaluation/reports/）には書き込まない（判定用の一次資料はこちらのみ）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from betting.src.ev_filters import harville_wide_pair_prob, harville_quinella_pair_prob  # noqa: E402
from betting.src.pair_probs import (  # noqa: E402
    calibration_max_error_pp,
    norm_pair,
    stern_quinella_pair_prob,
    stern_wide_pair_prob,
)
from evaluation.odds_loader import attach_odds_from_se_parquet  # noqa: E402
from prob_fusion.src.fit_fusion import fusion_probs  # noqa: E402
from prob_fusion.src.market_prob import attach_market_q  # noqa: E402
from prob_fusion.src.oos_protocol import TEST_START  # noqa: E402

SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
FUSION_REPORT_PATH = ROOT / "evaluation" / "reports" / "fusion_oos_fold2.json"
OUT_PATH = Path(__file__).resolve().parent / "pair_probability_model_comparison.json"

MAX_HORSES_PER_RACE = 18  # 頭数が多いレースでもO(n^3)が現実的な範囲に収まる上限


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_comparison() -> dict:
    fusion_report = json.loads(FUSION_REPORT_PATH.read_text(encoding="utf-8"))
    formal = fusion_report["formal"]
    alpha, beta = float(formal["alpha"]), float(formal["beta"])
    lam2, lam3 = float(formal["lam2"]), float(formal["lam3"])
    print(f"fusion params: alpha={alpha}, beta={beta}, lam2={lam2}, lam3={lam3}")

    scores = pd.read_parquet(SCORES_PATH)
    scores["race_id"] = scores["race_id"].astype(str)
    if "horse_number" not in scores.columns:
        scores["horse_number"] = scores["horse_num"]
    scores["horse_num"] = scores["horse_number"].astype(int)

    feat = pd.read_parquet(FEATURES_PATH, columns=["race_id", "horse_num", "finish_rank", "race_date"])
    feat["race_id"] = feat["race_id"].astype(str)
    for col in ("finish_rank", "race_date"):
        if col in scores.columns:
            scores = scores.drop(columns=[col])
    df = scores.merge(feat, on=["race_id", "horse_num"], how="inner")

    df = attach_odds_from_se_parquet(df)
    df = attach_market_q(df)
    df = df[pd.to_datetime(df["race_date"]) >= pd.Timestamp(TEST_START)].copy()
    print(f"TEST rows: {len(df):,}, races: {df['race_id'].nunique():,}")

    stern_preds, harville_preds, wide_actual = [], [], []
    stern_q_preds, harville_q_preds, quinella_actual = [], [], []
    n_races_used = 0
    n_races_skipped = 0
    t0 = time.perf_counter()

    for race_id, grp in df.groupby("race_id", sort=False):
        n = len(grp)
        if n < 3 or n > MAX_HORSES_PER_RACE:
            n_races_skipped += 1
            continue
        z = grp["pure_score_z"].astype(float).to_numpy()
        ln_q = grp["ln_market_q"].astype(float).to_numpy()
        horse_nums = grp["horse_num"].astype(int).to_numpy()
        finish = grp["finish_rank"].astype(int).to_numpy()

        p_win = fusion_probs(z, ln_q, alpha, beta)
        p_dict = {int(h): float(p) for h, p in zip(horse_nums, p_win)}

        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = int(horse_nums[i]), int(horse_nums[j])
                actual_wide = 1.0 if finish[i] <= 3 and finish[j] <= 3 else 0.0
                actual_q = 1.0 if finish[i] <= 2 and finish[j] <= 2 else 0.0

                stern_preds.append(stern_wide_pair_prob(p_win, i, j, lam2, lam3))
                harville_preds.append(harville_wide_pair_prob(p_dict, hi, hj))
                wide_actual.append(actual_wide)

                stern_q_preds.append(stern_quinella_pair_prob(p_win, i, j, lam2))
                harville_q_preds.append(harville_quinella_pair_prob(p_dict[hi], p_dict[hj]))
                quinella_actual.append(actual_q)
        n_races_used += 1

    elapsed = time.perf_counter() - t0
    print(f"races used={n_races_used:,}, skipped(頭数外)={n_races_skipped:,}, elapsed={elapsed:.1f}s")

    stern_preds = np.array(stern_preds)
    harville_preds = np.array(harville_preds)
    wide_actual = np.array(wide_actual)
    stern_q_preds = np.array(stern_q_preds)
    harville_q_preds = np.array(harville_q_preds)
    quinella_actual = np.array(quinella_actual)

    result = {
        "protocol": {
            "l1_scores": "fold2-only 5-seed ensemble OOS",
            "test_period": f"{TEST_START}..",
            "fusion_params_source": str(FUSION_REPORT_PATH.relative_to(ROOT)),
            "max_horses_per_race": MAX_HORSES_PER_RACE,
        },
        "n_races_used": n_races_used,
        "n_races_skipped": n_races_skipped,
        "n_pairs": int(len(wide_actual)),
        "wide": {
            "stern": {
                "logloss": _logloss(stern_preds, wide_actual),
                "calibration_max_error_pp": calibration_max_error_pp(stern_preds, wide_actual),
                "mean_pred": float(stern_preds.mean()),
            },
            "harville": {
                "logloss": _logloss(harville_preds, wide_actual),
                "calibration_max_error_pp": calibration_max_error_pp(harville_preds, wide_actual),
                "mean_pred": float(harville_preds.mean()),
            },
            "actual_rate": float(wide_actual.mean()),
        },
        "quinella": {
            "stern": {
                "logloss": _logloss(stern_q_preds, quinella_actual),
                "calibration_max_error_pp": calibration_max_error_pp(stern_q_preds, quinella_actual),
                "mean_pred": float(stern_q_preds.mean()),
            },
            "harville": {
                "logloss": _logloss(harville_q_preds, quinella_actual),
                "calibration_max_error_pp": calibration_max_error_pp(harville_q_preds, quinella_actual),
                "mean_pred": float(harville_q_preds.mean()),
            },
            "actual_rate": float(quinella_actual.mean()),
        },
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_comparison()

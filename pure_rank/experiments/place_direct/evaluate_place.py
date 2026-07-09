"""place_direct: evaluate_place.py

4 系列（Stern逆算 / Harville逆算 / 直接予測raw / 直接予測normalized）の
logloss・Brier・較正誤差を算出し、仕様書 §6 のプロトコルで判定する。

出力: pure_rank/experiments/place_direct/reports/place_direct_comparison.json
（本番 evaluation/reports/ には一切書き込まない）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(EXP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from place_lib import place_logloss  # noqa: E402

from betting.src.pair_probs import calibration_max_error_pp  # noqa: E402

SCORES_PATH = EXP_DIR / "scores" / "probs_place_direct_fold2_oos.parquet"
REPORTS_DIR = EXP_DIR / "reports"
BASELINE_PATH = ROOT / "evaluation" / "reports" / "place_baseline_oos.json"

EPS = 1e-12
SERIES = {
    "stern": "p_stern",
    "harville": "p_harville",
    "direct_raw": "p_direct_raw",
    "direct_norm": "p_direct_norm",
}
LEAK_IMPROVEMENT_THRESHOLD = 0.20  # (a)比 20%以上のlogloss改善はリーク疑い
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _calibration_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    curve = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(mask.sum())
        curve.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": n,
            "mean_pred": float(p[mask].mean()) if n > 0 else None,
            "actual_rate": float(y[mask].mean()) if n > 0 else None,
        })
    return curve


def _series_metrics(df: pd.DataFrame, col: str) -> dict:
    p = np.clip(df[col].to_numpy(dtype=float), EPS, 1 - EPS)
    y = df["target_place"].to_numpy(dtype=float)
    return {
        "logloss": place_logloss(p, y, eps=EPS),
        "brier": _brier(p, y),
        "calibration_max_error_pp": calibration_max_error_pp(p, y),
        "mean_pred": float(p.mean()),
        "actual_rate": float(y.mean()),
        "n": int(len(df)),
    }


def _bootstrap_logloss_diff_ci(
    df: pd.DataFrame, col_a: str, col_b: str, *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict:
    """レース単位ブートストラップで logloss(col_b) - logloss(col_a) の95%CIを計算する。

    負値 = col_b が col_a より logloss が小さい（優れている）ことを意味する。
    """
    race_ids = df["race_id"].unique()
    rng = np.random.default_rng(seed)
    diffs = []
    # レースごとに事前計算しておき、ブートストラップ内は参照のみにする
    grouped = {rid: g for rid, g in df.groupby("race_id", sort=False)}
    for _ in range(samples):
        sampled_ids = rng.choice(race_ids, size=len(race_ids), replace=True)
        sample_df = pd.concat([grouped[rid] for rid in sampled_ids], ignore_index=True)
        p_a = np.clip(sample_df[col_a].to_numpy(dtype=float), EPS, 1 - EPS)
        p_b = np.clip(sample_df[col_b].to_numpy(dtype=float), EPS, 1 - EPS)
        y = sample_df["target_place"].to_numpy(dtype=float)
        ll_a = place_logloss(p_a, y, eps=EPS)
        ll_b = place_logloss(p_b, y, eps=EPS)
        diffs.append(ll_b - ll_a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff_mean": float(np.mean(diffs)),
        "ci_95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def _head_count_breakdown(df: pd.DataFrame) -> dict:
    out = {}
    small = df[df["horse_count"] <= 7]
    large = df[df["horse_count"] >= 8]
    for label, sub in [("le7", small), ("ge8", large)]:
        if len(sub) == 0:
            out[label] = {"n": 0}
            continue
        out[label] = {
            "n": int(len(sub)),
            "n_races": int(sub["race_id"].nunique()),
            **{name: _series_metrics(sub, col) for name, col in SERIES.items()},
        }
    return out


def _top1_hit_rate(df: pd.DataFrame, col: str) -> dict:
    hits = 0
    n_races = 0
    for _, grp in df.groupby("race_id", sort=False):
        top = grp.sort_values(col, ascending=False).iloc[0]
        n_races += 1
        if int(top["target_place"]) == 1:
            hits += 1
    return {"n_races": n_races, "n_hits": hits, "hit_rate": hits / n_races if n_races else None}


def _apply_verdict(metrics: dict) -> dict:
    """仕様書 §6.2 の判定を機械的に適用する。"""
    a = metrics["overall"]["stern"]
    candidates = {}
    for name in ("direct_raw", "direct_norm"):
        m = metrics["overall"][name]
        logloss_better = m["logloss"] < a["logloss"]
        calib_better = m["calibration_max_error_pp"] < a["calibration_max_error_pp"]
        candidates[name] = {
            "logloss_improves_vs_stern": bool(logloss_better),
            "calibration_improves_vs_stern": bool(calib_better),
            "both_improve": bool(logloss_better and calib_better),
        }

    both_ok = [name for name, c in candidates.items() if c["both_improve"]]
    if both_ok:
        # 両方満たす場合は logloss が小さい方を採用候補とする
        best = min(both_ok, key=lambda n: metrics["overall"][n]["logloss"])
        verdict = f"direct_prediction_superior:{best}"
    else:
        verdict = "not_superior_no_reattempt"

    return {"candidates": candidates, "verdict": verdict}


def _leak_check(metrics: dict) -> dict:
    a_ll = metrics["overall"]["stern"]["logloss"]
    flags = {}
    for name in ("direct_raw", "direct_norm"):
        ll = metrics["overall"][name]["logloss"]
        improvement = (a_ll - ll) / a_ll if a_ll > 0 else 0.0
        flags[name] = {
            "logloss": ll,
            "improvement_vs_stern": float(improvement),
            "leak_suspected": bool(improvement >= LEAK_IMPROVEMENT_THRESHOLD),
        }
    any_leak = any(v["leak_suspected"] for v in flags.values())
    return {"per_series": flags, "any_leak_suspected": any_leak}


def evaluate_place_direct() -> dict:
    print(f"Loading: {SCORES_PATH}")
    df = pd.read_parquet(SCORES_PATH)
    df["race_id"] = df["race_id"].astype(str)
    n_races = df["race_id"].nunique()
    print(f"rows={len(df):,}, races={n_races:,}")

    overall = {name: _series_metrics(df, col) for name, col in SERIES.items()}
    calibration_curves = {name: _calibration_curve(
        np.clip(df[col].to_numpy(dtype=float), EPS, 1 - EPS),
        df["target_place"].to_numpy(dtype=float),
    ) for name, col in SERIES.items()}

    print("Bootstrapping logloss diff CIs (race-level, 1000 resamples)...")
    bootstrap = {
        "direct_raw_vs_stern": _bootstrap_logloss_diff_ci(df, "p_stern", "p_direct_raw"),
        "direct_norm_vs_stern": _bootstrap_logloss_diff_ci(df, "p_stern", "p_direct_norm"),
        "harville_vs_stern": _bootstrap_logloss_diff_ci(df, "p_stern", "p_harville"),
    }

    head_count = _head_count_breakdown(df)

    top1 = {
        "direct_raw": _top1_hit_rate(df, "p_direct_raw"),
        "direct_norm": _top1_hit_rate(df, "p_direct_norm"),
    }
    baseline_ref = None
    if BASELINE_PATH.is_file():
        baseline_ref = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        baseline_ref = {
            "model_top1_hit_rate": baseline_ref.get("model_top1", {}).get("hit_rate"),
            "favorite_hit_rate": baseline_ref.get("favorite", {}).get("hit_rate"),
        }

    metrics = {
        "overall": overall,
        "bootstrap_logloss_diff_ci95": bootstrap,
    }
    verdict_info = _apply_verdict(metrics)
    leak_info = _leak_check(metrics)

    result = {
        "protocol": {
            "test_period": "2025-01-01..",
            "n_races": n_races,
            "n_horses": int(len(df)),
            "scores_source": str(SCORES_PATH.relative_to(ROOT)) if SCORES_PATH.is_absolute() else str(SCORES_PATH),
            "eps": EPS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "overall": overall,
        "bootstrap_logloss_diff_ci95": bootstrap,
        "head_count_breakdown_reference_only": head_count,
        "top1_hit_rate_reference_only": top1,
        "baseline_reference": baseline_ref,
        "calibration_curves": calibration_curves,
        "leak_check": leak_info,
        "verdict": verdict_info["verdict"],
        "verdict_detail": verdict_info["candidates"],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "place_direct_comparison.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")

    print("\n=== Overall metrics ===")
    for name, m in overall.items():
        print(f"  {name}: logloss={m['logloss']:.5f} brier={m['brier']:.5f} "
              f"calib_max_err_pp={m['calibration_max_error_pp']:.2f} "
              f"mean_pred={m['mean_pred']:.4f} actual={m['actual_rate']:.4f}")
    print(f"\nVerdict: {result['verdict']}")
    if leak_info["any_leak_suspected"]:
        print("\n" + "!" * 70)
        print("WARNING: リーク疑い（(a)比 logloss 20%以上改善）を検出。evaluatorへ即報告すること。")
        print(json.dumps(leak_info, indent=2, ensure_ascii=False))
        print("!" * 70)

    return result


if __name__ == "__main__":
    evaluate_place_direct()

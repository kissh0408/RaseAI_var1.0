"""place_calibration: evaluate_calibration.py

5 系列（S0/A1/A2/B1/B2）の logloss・Brier・較正誤差を算出し、
仕様書 §6 のプロトコル（ブートストラップCI・順位保存検証・リーク停止閾値・
verdict判定）を適用する。

出力: pure_rank/experiments/place_calibration/reports/place_calibration_comparison.json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from calib_lib import place_logloss, top1_index  # noqa: E402

from betting.src.pair_probs import calibration_max_error_pp  # noqa: E402

DATA_DIR = EXP_DIR / "data"
REPORTS_DIR = EXP_DIR / "reports"
MODELS_DIR = EXP_DIR / "models"
CONFIG_PATH = EXP_DIR / "config.json"

EPS = 1e-12
SERIES = {
    "s0": "p_s0",
    "a1": "p_a1",
    "a2": "p_a2",
    "b1": "p_b1",
    "b2": "p_b2",
}
CANDIDATE_SERIES = ["a1", "a2", "b1", "b2"]


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    """numpy スカラ（np.bool_ / np.float64 等）を JSON 化する。
    calibration_max_error_pp が np.float64 を返し、その比較結果が np.bool_ になるため。"""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


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
    y = df["y_place"].to_numpy(dtype=float)
    return {
        "logloss": place_logloss(p, y, eps=EPS),
        "brier": _brier(p, y),
        "calibration_max_error_pp": calibration_max_error_pp(p, y),
        "mean_pred": float(p.mean()),
        "actual_rate": float(y.mean()),
        "n": int(len(df)),
    }


def _bootstrap_logloss_diff_ci(
    df: pd.DataFrame, col_base: str, col_cand: str, *, samples: int, seed: int
) -> dict:
    """レース単位ブートストラップで logloss(cand) - logloss(base) の95%CIを計算する。

    負値 = cand が base より logloss が小さい（優れている）ことを意味する。
    """
    race_ids = df["race_id"].unique()
    rng = np.random.default_rng(seed)
    diffs = []
    grouped = {rid: g for rid, g in df.groupby("race_id", sort=False)}
    for _ in range(samples):
        sampled_ids = rng.choice(race_ids, size=len(race_ids), replace=True)
        sample_df = pd.concat([grouped[rid] for rid in sampled_ids], ignore_index=True)
        p_base = np.clip(sample_df[col_base].to_numpy(dtype=float), EPS, 1 - EPS)
        p_cand = np.clip(sample_df[col_cand].to_numpy(dtype=float), EPS, 1 - EPS)
        y = sample_df["y_place"].to_numpy(dtype=float)
        ll_base = place_logloss(p_base, y, eps=EPS)
        ll_cand = place_logloss(p_cand, y, eps=EPS)
        diffs.append(ll_cand - ll_base)
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


def _top1_rank_preserved_check(df: pd.DataFrame, *, tie_tol: float = 1e-9) -> dict:
    """全系列で top1 選択馬が S0 と一致することを検証する（仕様書 §6.2）。

    isotonic の平坦区間・A2 の帯別 λ でタイが生じた場合は元の p_stern (S0) 順で
    安定ソートし順位保存を保証する（tiebreak=p_s0）。

    数値タイの扱い（実測に基づく設計判断）: 同一オッズの2頭は p_win が完全一致し、
    Stern place 確率も数学的に同値になる（TEST 2025+ で該当12〜16レースを実測確認。
    p_win 差が厳密に 0、p_place の差は float 総和順序による ~1e-16 のみ）。
    argmax がどちらの馬を返すかは実質未定義のため、「S0 の top1 馬が候補系列でも
    最大値と tie_tol 以内で並んでいる」ことを順位保存と判定する。
    真の順位反転（S0 top1 が候補系列で明確に劣位）のみ mismatch として数え、
    検出時は実装バグとして停止・evaluator へ報告する。
    """
    mismatches: dict[str, int] = {name: 0 for name in CANDIDATE_SERIES}
    numerical_ties: dict[str, int] = {name: 0 for name in CANDIDATE_SERIES}
    mismatch_examples: dict[str, list] = {name: [] for name in CANDIDATE_SERIES}
    n_races = 0
    for race_id, grp in df.groupby("race_id", sort=False):
        n_races += 1
        p_s0 = grp["p_s0"].to_numpy(dtype=float)
        top1_s0 = top1_index(p_s0, tiebreak=p_s0)
        for name in CANDIDATE_SERIES:
            p = grp[SERIES[name]].to_numpy(dtype=float)
            top1_cand = top1_index(p, tiebreak=p_s0)
            if top1_cand != top1_s0:
                # S0 の top1 馬が候補系列でも実質同値（数値タイ）なら順位保存とみなす
                if p[top1_cand] - p[top1_s0] <= tie_tol:
                    numerical_ties[name] += 1
                    continue
                mismatches[name] += 1
                if len(mismatch_examples[name]) < 5:
                    mismatch_examples[name].append(str(race_id))

    all_preserved = all(v == 0 for v in mismatches.values())
    return {
        "n_races_checked": n_races,
        "tie_tolerance": tie_tol,
        "numerical_tie_races_per_series": numerical_ties,
        "mismatches_per_series": mismatches,
        "mismatch_examples_per_series": mismatch_examples,
        "all_rank_preserved": all_preserved,
    }


def _apply_verdict(metrics_overall: dict) -> dict:
    """仕様書 §6.2 の判定を機械的に適用する。"""
    s0 = metrics_overall["s0"]
    candidates = {}
    for name in CANDIDATE_SERIES:
        m = metrics_overall[name]
        logloss_better = m["logloss"] < s0["logloss"]
        calib_better = m["calibration_max_error_pp"] < s0["calibration_max_error_pp"]
        candidates[name] = {
            "logloss_improves_vs_s0": bool(logloss_better),
            "calibration_improves_vs_s0": bool(calib_better),
            "both_improve": bool(logloss_better and calib_better),
        }

    both_ok = [name for name, c in candidates.items() if c["both_improve"]]
    if both_ok:
        best = min(both_ok, key=lambda n: metrics_overall[n]["logloss"])
        verdict = f"adopted_series:{best}"
    else:
        verdict = "no_improvement_no_reattempt"

    return {"candidates": candidates, "verdict": verdict, "adopted_candidates": both_ok}


def _leak_check(metrics_overall: dict, cfg: dict) -> dict:
    s0_ll = metrics_overall["s0"]["logloss"]
    threshold = cfg["leak_threshold"]["logloss_improvement_pct"]
    flags = {}
    for name in CANDIDATE_SERIES:
        ll = metrics_overall[name]["logloss"]
        improvement = (s0_ll - ll) / s0_ll if s0_ll > 0 else 0.0
        flags[name] = {
            "logloss": ll,
            "improvement_vs_s0": float(improvement),
            "leak_suspected": bool(improvement >= threshold),
        }
    any_leak = any(v["leak_suspected"] for v in flags.values())
    return {"per_series": flags, "any_leak_suspected": any_leak, "threshold_pct": threshold}


def _bootstrap_ci_verdict_check(verdict_info: dict, bootstrap: dict) -> dict:
    """§6.2: 採用候補は「両指標の点推定改善 + logloss差CIがゼロを含まない」が必要条件。"""
    final_adopted = None
    ci_gate_passed = {}
    for name in verdict_info["adopted_candidates"]:
        ci = bootstrap[f"{name}_vs_s0"]
        ci_gate_passed[name] = ci["ci_excludes_zero"]
    if verdict_info["verdict"].startswith("adopted_series:"):
        best = verdict_info["verdict"].split(":", 1)[1]
        if ci_gate_passed.get(best, False):
            final_adopted = best
        else:
            final_adopted = None  # CIがゼロを含む場合は「差なし」として不採用
    return {
        "ci_gate_passed_per_candidate": ci_gate_passed,
        "final_verdict": f"adopted_series:{final_adopted}" if final_adopted else "no_improvement_no_reattempt",
    }


def evaluate_place_calibration() -> dict:
    cfg = _load_config()
    print(f"Loading: {DATA_DIR / 'probs_test_2025.parquet'}")
    df = pd.read_parquet(DATA_DIR / "probs_test_2025.parquet")
    df["race_id"] = df["race_id"].astype(str)
    n_races = df["race_id"].nunique()
    print(f"rows={len(df):,}, races={n_races:,}")

    overall = {name: _series_metrics(df, col) for name, col in SERIES.items()}
    calibration_curves = {
        name: _calibration_curve(
            np.clip(df[col].to_numpy(dtype=float), EPS, 1 - EPS),
            df["y_place"].to_numpy(dtype=float),
        )
        for name, col in SERIES.items()
    }

    boot_cfg = cfg["bootstrap"]
    print(f"Bootstrapping logloss diff CIs (race-level, {boot_cfg['samples']} resamples)...")
    bootstrap = {
        f"{name}_vs_s0": _bootstrap_logloss_diff_ci(
            df, "p_s0", SERIES[name], samples=boot_cfg["samples"], seed=boot_cfg["seed"]
        )
        for name in CANDIDATE_SERIES
    }

    head_count = _head_count_breakdown(df)

    print("Checking top1 rank preservation across all series (vs S0, tiebreak on p_s0)...")
    rank_check = _top1_rank_preserved_check(df)
    if not rank_check["all_rank_preserved"]:
        print("!" * 70)
        print("WARNING: top1 順位保存に失敗した系列あり。実装バグの可能性。停止しevaluatorへ報告。")
        print(json.dumps(rank_check["mismatches_per_series"], indent=2))
        print("!" * 70)
    else:
        print(f"OK: all {rank_check['n_races_checked']:,} races -- top1 preserved across all series")

    verdict_info = _apply_verdict(overall)
    ci_check = _bootstrap_ci_verdict_check(verdict_info, bootstrap)
    leak_info = _leak_check(overall, cfg)

    lambda_fit = None
    lambda_fit_path = MODELS_DIR / "lambda_fit.json"
    if lambda_fit_path.is_file():
        lambda_fit = json.loads(lambda_fit_path.read_text(encoding="utf-8"))

    s0_check = {
        "expected_logloss": cfg["s0_baseline"]["expected_logloss"],
        "actual_logloss": overall["s0"]["logloss"],
        "logloss_diff": overall["s0"]["logloss"] - cfg["s0_baseline"]["expected_logloss"],
        "expected_calibration_max_error_pp": cfg["s0_baseline"]["expected_calibration_max_error_pp"],
        "actual_calibration_max_error_pp": overall["s0"]["calibration_max_error_pp"],
        "calibration_diff_pp": (
            overall["s0"]["calibration_max_error_pp"] - cfg["s0_baseline"]["expected_calibration_max_error_pp"]
        ),
        "within_tolerance": (
            abs(overall["s0"]["logloss"] - cfg["s0_baseline"]["expected_logloss"]) <= cfg["s0_baseline"]["logloss_tol"]
            and abs(
                overall["s0"]["calibration_max_error_pp"] - cfg["s0_baseline"]["expected_calibration_max_error_pp"]
            ) <= cfg["s0_baseline"]["calibration_tol_pp"]
        ),
    }

    result = {
        "protocol": {
            "test_period": "2025-01-01..",
            "n_races": n_races,
            "n_horses": int(len(df)),
            "scores_source": str((DATA_DIR / "probs_test_2025.parquet").relative_to(ROOT)),
            "eps": EPS,
            "bootstrap_samples": boot_cfg["samples"],
        },
        "s0_reproduction_check": s0_check,
        "fitted_lambda": lambda_fit,
        "overall": overall,
        "bootstrap_logloss_diff_ci95": bootstrap,
        "head_count_breakdown_reference_only": head_count,
        "top1_rank_preserved_check": rank_check,
        "calibration_curves": calibration_curves,
        "leak_check": leak_info,
        "verdict_raw": verdict_info["verdict"],
        "verdict_detail": verdict_info["candidates"],
        "verdict": ci_check["final_verdict"],
        "ci_gate_check": ci_check,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "place_calibration_comparison.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(f"Saved: {out_path}")

    print("\n=== Overall metrics ===")
    for name, m in overall.items():
        print(
            f"  {name}: logloss={m['logloss']:.5f} brier={m['brier']:.5f} "
            f"calib_max_err_pp={m['calibration_max_error_pp']:.2f} "
            f"mean_pred={m['mean_pred']:.4f} actual={m['actual_rate']:.4f}"
        )
    print(f"\nS0 reproduction within tolerance: {s0_check['within_tolerance']}")
    print(f"Verdict (raw §6.2 point-estimate): {verdict_info['verdict']}")
    print(f"Verdict (final, CI-gated): {result['verdict']}")
    if leak_info["any_leak_suspected"]:
        print("\n" + "!" * 70)
        print("WARNING: リーク疑い（S0比 logloss 10%以上改善）を検出。evaluatorへ即報告すること。")
        print(json.dumps(leak_info, indent=2, ensure_ascii=False))
        print("!" * 70)

    return result


if __name__ == "__main__":
    evaluate_place_calibration()

"""confidence_tiers: run_stage2_valid.py

Stage 2（一次判定材料の測定。VALID = 2024-01-01〜2024-12-31 のみ）:
凍結境界（仕様書§13）で階層 T1〜T4 を割当て、

- §5.2: 階層別 ROI_model / ROI_fav / Δ（同一レース集合のペア比較。1番人気側には
  オッズ除外を適用しない）
- §5.4: 記述的診断（1番人気一致率・平均/中央値オッズ・階層別Top-1的中率ペア・
  margin統計。検定・判定には不使用）
- §6: H1〜H4 片側ペアドクラスタブートストラップ + H_ord = Δ(T4)−Δ(T1) コントラスト
  （B=10000, seed=42, Bonferroni閾値 0.01/5=0.002）
- §5.3: 再現性アンカー（全階層合算 = flat_fraction_valid_2024.json 実測と一致）
- §9: 危険信号チェック（階層Top-1>40% / ROI>100% / |Δ|>20pp）

を results/tiers_valid.json に出力する。一次判定の承認は evaluator が行う。

Rule 3（期間規律）: io 直後に race_date で 2024 年のみへフィルタし、TEST(2025+) の
行には一切触れない。
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
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from betting.src.flat_top1 import DISCLAIMER  # noqa: E402

import tiers_lib as tl  # noqa: E402

CONFIG_PATH = EXP_DIR / "config.json"
DATA_PATH = EXP_DIR / "data" / "bets_dataset.parquet"
OUT_PATH = EXP_DIR / "results" / "tiers_valid.json"

STAKE_YEN = 100.0


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _settle(df: pd.DataFrame) -> pd.DataFrame:
    """モデル側・1番人気側の flat 100円決済列を付与する（ペア比較用）。"""
    out = df.copy()
    out["stake_model"] = STAKE_YEN
    out["payout_model"] = np.where(
        out["finish_rank"].astype(int) == 1, STAKE_YEN * out["odds"].astype(float), 0.0
    )
    out["stake_fav"] = STAKE_YEN
    out["payout_fav"] = np.where(
        out["favorite_finish_rank"].astype(int) == 1,
        STAKE_YEN * out["favorite_odds"].astype(float),
        0.0,
    )
    return out


def _tier_metrics(sub: pd.DataFrame, tier: int, boundaries: list[float], boot_cfg: dict) -> dict:
    """1階層分の §5.2 測定・§5.4 記述的診断・§6.1 検定・§9 危険信号。"""
    n = int(len(sub))
    roi_model = tl.compute_roi(sub["stake_model"], sub["payout_model"])
    roi_fav = tl.compute_roi(sub["stake_fav"], sub["payout_fav"])
    delta = tl.compute_delta(roi_model, roi_fav)

    boot = tl.cluster_bootstrap_delta_p_value(
        sub["stake_model"].to_numpy(),
        sub["payout_model"].to_numpy(),
        sub["stake_fav"].to_numpy(),
        sub["payout_fav"].to_numpy(),
        B=int(boot_cfg["B"]),
        seed=int(boot_cfg["seed"]),
    )

    hit_rate_model = float((sub["finish_rank"].astype(int) == 1).mean()) if n else float("nan")
    hit_rate_fav = float((sub["favorite_finish_rank"].astype(int) == 1).mean()) if n else float("nan")
    agreement = (
        float((sub["horse_num"].astype(int) == sub["favorite_horse_num"].astype(int)).mean())
        if n
        else float("nan")
    )
    odds_vals = sub["odds"].astype(float)
    margins = sub["margin"].astype(float)

    # §9-3: payout集中度ゲート（診断。ROI>100%時の検証材料として常に算出）
    n_hits = int((sub["finish_rank"].astype(int) == 1).sum())
    gate = tl.payout_concentration_gate(sub["payout_model"].to_numpy(), n_hits)

    danger = {
        "leak_review_required": tl.leak_review_flag(hit_rate_model),
        "danger_roi_gt_100": tl.danger_roi_gt_100(roi_model),
        "large_delta_gt_20pp": tl.large_delta_flag(delta),
    }

    return {
        "tier": tier,
        "n_races": n,
        "min_sample_ok": tl.min_sample_ok(n),
        "boundaries_used": boundaries,
        "roi_model_pct": roi_model * 100.0,
        "roi_fav_pct": roi_fav * 100.0,
        "delta_pp": delta * 100.0,
        "bootstrap_p_one_sided": boot["p_value"],
        "ci95_delta_pp": [boot["ci_low"] * 100.0, boot["ci_high"] * 100.0],
        "hit_rate_model": hit_rate_model,
        "hit_rate_fav": hit_rate_fav,
        "favorite_agreement_rate": agreement,
        "mean_odds": float(odds_vals.mean()) if n else None,
        "median_odds": float(odds_vals.median()) if n else None,
        "margin_mean": float(margins.mean()) if n else None,
        "margin_min": float(margins.min()) if n else None,
        "margin_max": float(margins.max()) if n else None,
        "n_hits_model": n_hits,
        "payout_concentration_gate": gate,
        "danger_flags": danger,
    }


def run_stage2_valid() -> dict:
    cfg = _load_config()
    bcfg = cfg["boundaries"]
    assert str(bcfg["status"]).startswith("frozen"), "境界が凍結されていません（Stage 1 → §13追記を先に完了すること）"
    boundaries = [float(bcfg["b1"]), float(bcfg["b2"]), float(bcfg["b3"])]

    period = cfg["protocol"]["stage2_valid_period"]
    start, end = period["start"], period["end"]

    df = pd.read_parquet(DATA_PATH)
    # Rule 3: io 直後に VALID 2024 のみへフィルタ（TEST 2025+ に触れない）
    dates = pd.to_datetime(df["race_date"])
    df = df.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()

    df = _settle(df)
    df["tier"] = tl.assign_tier_batch(df["margin"].to_numpy(dtype=float), boundaries)

    # --- §5.3 再現性アンカー（全階層合算） ---
    anchor = cfg["reproduction_anchors"]["valid_2024"]
    n_total = int(len(df))
    hit_rate_total = float((df["finish_rank"].astype(int) == 1).mean())
    roi_total_pct = tl.compute_roi(df["stake_model"], df["payout_model"]) * 100.0
    tol_pp = float(anchor["tolerance_pp"])
    reproduction_gate = {
        "n_bets": n_total,
        "n_bets_expected": int(anchor["n_bets"]),
        "n_bets_match": n_total == int(anchor["n_bets"]),
        "hit_rate": hit_rate_total,
        "hit_rate_expected": float(anchor["hit_rate"]),
        "hit_rate_match_pm_0p1pp": abs(hit_rate_total - float(anchor["hit_rate"])) * 100.0 <= tol_pp,
        "roi_pct": roi_total_pct,
        "roi_pct_expected": float(anchor["roi_pct"]),
        "roi_match_pm_0p1pp": abs(roi_total_pct - float(anchor["roi_pct"])) <= tol_pp,
        "source": anchor["source"],
    }
    reproduction_gate["pass"] = bool(
        reproduction_gate["n_bets_match"]
        and reproduction_gate["hit_rate_match_pm_0p1pp"]
        and reproduction_gate["roi_match_pm_0p1pp"]
    )

    # --- §5.2/§5.4/§6.1 階層別測定 ---
    boot_cfg = cfg["bootstrap"]
    tiers = []
    for t in (1, 2, 3, 4):
        sub = df.loc[df["tier"] == t]
        tiers.append(_tier_metrics(sub, t, boundaries, boot_cfg))

    n_by_tier = {t["tier"]: t["n_races"] for t in tiers}
    all_min_sample_ok = tl.all_tiers_min_sample_ok(n_by_tier, n_min=int(cfg["min_sample"]["n_min"]))

    # --- §6.2 順序仮説 H_ord（全階層 n>=200 のときのみ実行） ---
    bonf = tl.bonferroni_threshold(k=int(cfg["hypotheses"]["k_hyp"]), alpha=float(cfg["hypotheses"]["bonferroni_alpha"]))
    if all_min_sample_ok:
        t1 = df.loc[df["tier"] == 1]
        t4 = df.loc[df["tier"] == 4]
        ord_res = tl.cluster_bootstrap_ordering_contrast(
            t1["stake_model"].to_numpy(),
            t1["payout_model"].to_numpy(),
            t1["stake_fav"].to_numpy(),
            t1["payout_fav"].to_numpy(),
            t4["stake_model"].to_numpy(),
            t4["payout_model"].to_numpy(),
            t4["stake_fav"].to_numpy(),
            t4["payout_fav"].to_numpy(),
            B=int(boot_cfg["B"]),
            seed=int(boot_cfg["seed"]),
        )
        ordering_contrast = {
            "executed": True,
            "c_pp": ord_res["c_hat"] * 100.0,
            "p_one_sided": ord_res["p_value"],
            "ci95_pp": [ord_res["ci_low"] * 100.0, ord_res["ci_high"] * 100.0],
        }
    else:
        ordering_contrast = {
            "executed": False,
            "reason": "min_sample_hold: some tier n < 200 (spec §4.3)",
            "c_pp": None,
            "p_one_sided": None,
            "ci95_pp": None,
        }

    deltas = [t["delta_pp"] for t in tiers]
    mono = tl.monotonicity_flag(deltas)

    # --- §7.1 一次判定（機械適用。承認は evaluator） ---
    h4_p = tiers[3]["bootstrap_p_one_sided"]
    h4_sig = bool(np.isfinite(h4_p) and h4_p < bonf)
    hord_p = ordering_contrast["p_one_sided"]
    hord_sig = bool(hord_p is not None and np.isfinite(hord_p) and hord_p < bonf)

    if not reproduction_gate["pass"]:
        primary_pass = False
        primary_status = "blocked_reproduction_gate_failed"
    elif not all_min_sample_ok:
        primary_pass = False
        primary_status = "hold_min_sample"
    elif h4_sig or hord_sig:
        primary_pass = True
        primary_status = "primary_pass"
    else:
        primary_pass = False
        primary_status = "primary_fail_all_hypotheses"

    # §7.3 / §8: verdict（一次不通過時の確定記録。最終判定は evaluator）
    delta_t4 = tiers[3]["delta_pp"]
    delta_t1 = tiers[0]["delta_pp"]
    if primary_pass:
        verdict = "primary_pass_pending_evaluator_approval"
    elif primary_status == "blocked_reproduction_gate_failed":
        verdict = "blocked_reproduction_gate_failed"
    elif delta_t4 < 0 or delta_t4 < delta_t1:
        verdict = "confidence_weighting_would_dilute_edge"
    else:
        verdict = "confidence_does_not_predict_market_edge"

    any_danger = any(any(t["danger_flags"].values()) for t in tiers)

    result = {
        "disclaimer": DISCLAIMER,
        "protocol": {
            "stage": "stage2_valid_primary",
            "period": f"{start}..{end}",
            "score_col": cfg["score_col"],
            "boundaries_frozen": {"b1": boundaries[0], "b2": boundaries[1], "b3": boundaries[2]},
            "boundaries_source": "docs/specs/2026-07-11-confidence-tiers-spec.md §13 (stage1_boundaries.json, 2023 outcome-blind)",
            "odds_exclusion_model_side": cfg["odds_filter"],
            "odds_exclusion_favorite_side": "none (spec §5.2: baseline is unexcluded favorite on same race set)",
            "stake_yen_flat": STAKE_YEN,
            "seed": int(boot_cfg["seed"]),
            "B": int(boot_cfg["B"]),
        },
        "K_hyp": int(cfg["hypotheses"]["k_hyp"]),
        "bonferroni_threshold": bonf,
        "tiers": tiers,
        "ordering_contrast": ordering_contrast,
        "monotonicity_flag_descriptive_only": mono,
        "min_sample_all_tiers_ok": all_min_sample_ok,
        "reproduction_gate": reproduction_gate,
        "hypothesis_results": {
            "H1_p": tiers[0]["bootstrap_p_one_sided"],
            "H2_p": tiers[1]["bootstrap_p_one_sided"],
            "H3_p": tiers[2]["bootstrap_p_one_sided"],
            "H4_p": h4_p,
            "H4_significant": h4_sig,
            "H_ord_p": hord_p,
            "H_ord_significant": hord_sig,
        },
        "primary_pass": primary_pass,
        "primary_status": primary_status,
        "verdict": verdict,
        "danger_signals_any": any_danger,
        "caveats": [
            "階層境界はfold2 OOSスコアの2023年（early-stopping弱汚染年）のmargin分布から"
            "outcome-blindに決定した。判定は完全OOSの2024年で実施しており、境界決定への"
            "汚染は判定へのリークにはならない（仕様書§4.2）。",
            "本測定は確定/前日水準オッズベースであり、購入時点のオッズで優位が縮小しうる。",
            "§5.4の記述的診断（favorite_agreement_rate等）は解釈補助であり判定には使用しない。",
            "全体の期待値は負であり、測定対象は市場に対する相対的損失差である。",
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_stage2_valid()

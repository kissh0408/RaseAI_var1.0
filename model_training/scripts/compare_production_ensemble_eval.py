"""
compare_production_ensemble_eval.py — v5 vs specv2 を同一戦略条件でクロス評価

プロファイル:
  production  — strategy_config.json + calibration_isotonic.json + race_num 8-12
  v5_meta     — strategy_config.json + calibration_isotonic.json + race_num なし（旧 v5 記録条件）

モデル:
  specv2 — evaluation_all_non_leak.csv（WF OOF）
  v5     — evaluation_v5_oof.csv（WF OOF、3シード平均）。無ければ evaluation_v5_final_predict.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_BASE = PROJECT_ROOT / "model_training" / "models"
EVAL_CSV = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_all_non_leak.csv"
STRATEGY_CFG_PATH = PROJECT_ROOT / "strategy" / "config" / "strategy_config.json"
CALIB_PATH = PROJECT_ROOT / "strategy" / "models" / "calibration_isotonic.json"
ODDS_DIR = PROJECT_ROOT / "common" / "data" / "output" / "odds"
V5_OOF_CSV = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_v5_oof.csv"
V5_CACHE = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_v5_final_predict.csv"
SPECv2_OOF_CSV = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_specv2_oof.csv"

PROFILES = {
    "production": {
        "label": "strategy_config.json（calibration_path + race_num 8-12）",
        "race_num_min": 8,
        "race_num_max": 12,
        "use_config_calibration": True,
    },
    "v5_meta": {
        "label": "strategy_config + calibrator + race_num なし（旧 v5 記録）",
        "race_num_min": None,
        "race_num_max": None,
        "use_config_calibration": True,
    },
}


def _resolve_calibrator_path(use_config: bool = True) -> Path | None:
    from main.pipeline.strategy_pipeline import resolve_strategy_calibration_path

    if use_config:
        p = resolve_strategy_calibration_path(PROJECT_ROOT, STRATEGY_CFG_PATH)
        return p if p.is_file() else None
    return CALIB_PATH if CALIB_PATH.exists() else None


def _load_runtime_cfg() -> dict:
    return json.loads(STRATEGY_CFG_PATH.read_text(encoding="utf-8"))


def _strategy_config(profile: str):
    from main.pipeline.strategy_pipeline import strategy_config_from_runtime

    runtime = _load_runtime_cfg()
    p = PROFILES[profile]
    runtime = deepcopy(runtime)
    runtime["race_num_min"] = p["race_num_min"]
    runtime["race_num_max"] = p["race_num_max"]
    return strategy_config_from_runtime(runtime)


def _load_ensemble(subdir: str) -> dict[int, list]:
    ens_dir = MODELS_BASE / subdir
    meta = json.loads((ens_dir / "ensemble_meta.json").read_text(encoding="utf-8"))
    models: dict[int, list] = {}
    for rk, paths in meta["model_paths"].items():
        rank = int(str(rk).replace("rank", ""))
        models[rank] = []
        for rel in paths:
            p = ens_dir / Path(rel).name
            with p.open("rb") as f:
                models[rank].append(pickle.load(f))
    return models


def _load_specv2_eval() -> pd.DataFrame:
    from strategy.src.betting_framework import load_evaluation

    path = SPECv2_OOF_CSV if SPECv2_OOF_CSV.exists() else EVAL_CSV
    return load_evaluation(path)


def _load_v5_eval(*, rebuild: bool = False) -> pd.DataFrame:
    from strategy.src.betting_framework import load_evaluation

    if V5_OOF_CSV.exists() and not rebuild:
        return load_evaluation(V5_OOF_CSV)
    if V5_CACHE.exists() and not rebuild:
        return load_evaluation(V5_CACHE)
    return _build_v5_eval(rebuild=rebuild)


def _build_v5_eval(*, rebuild: bool = False) -> pd.DataFrame:
    from main.pipeline.inference_pipeline import predict_ranks_for_frame
    from model_training.src.pipeline_common import load_config

    cfg = load_config()
    feat_rel = cfg.get("production_training", {}).get(
        "feature_file", cfg["training"]["feature_file"]
    )
    feat_path = PROJECT_ROOT / "model_training" / "data" / "02_features" / feat_rel
    if not feat_path.exists():
        feat_path = PROJECT_ROOT / "model_training" / "data" / "02_features" / cfg["training"]["feature_file"]

    probe = pd.read_parquet(feat_path, columns=None)
    use_cols = [
        "race_id", "horse_num", "odds", "finish_rank", "popularity",
        "year", "month_day", "course_code", "kai", "nichi", "race_num",
    ]
    meta_cols = [c for c in use_cols if c in probe.columns]
    df = pd.read_parquet(feat_path)
    df = df[pd.to_numeric(df["finish_rank"], errors="coerce").fillna(0) > 0].copy()
    df["valid_year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[(df["valid_year"] >= 2018) & (df["valid_year"] <= 2025)].copy()

    models = _load_ensemble("ensemble_v5")
    # OOF CSV と同様: rank isotonic は使わず raw predict のみ
    preds = predict_ranks_for_frame(models, df)
    out = df[[c for c in meta_cols if c in df.columns]].copy()
    if "race_id" not in out.columns:
        raise RuntimeError("race_id missing in features")
    for c in ("horse_num", "odds", "finish_rank"):
        if c not in out.columns:
            out[c] = df[c]
    out["valid_year"] = df["valid_year"].astype(int)
    if "race_num" in df.columns:
        out["race_num"] = pd.to_numeric(df["race_num"], errors="coerce").fillna(0).astype(int)
    else:
        from strategy.src.race_filters import attach_race_num

        out = attach_race_num(out)
    for rank in (1, 2, 3):
        out[f"pred_rank{rank}"] = preds[rank]
    out.to_csv(V5_CACHE, index=False, encoding="utf-8-sig")
    return out


def _win_metrics(eval_df: pd.DataFrame, profile: str, year: int | None) -> dict[str, Any]:
    from strategy.src.betting_framework import ProbabilityCalibrator, run_betting_backtest

    df = eval_df.copy()
    if year is not None:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
    if df.empty:
        return {"error": f"empty year={year}"}

    cal_path = _resolve_calibrator_path(PROFILES[profile].get("use_config_calibration", True))
    calibrator = ProbabilityCalibrator.from_json(cal_path) if cal_path is not None else None
    config = _strategy_config(profile)
    _, _, metrics = run_betting_backtest(df, config, calibrator=calibrator)
    roi = float(metrics.get("return_multiple", metrics.get("roi", 0)))
    mdd = float(metrics.get("max_drawdown_rate", metrics.get("mdd", -1)))
    sharpe = float(metrics.get("sharpe", 0))
    n_bets = int(metrics.get("n_bets", 0))
    hit_rate = float(metrics.get("hit_rate", metrics.get("bet_hit_rate", 0)))
    return {
        "roi": roi,
        "mdd": mdd,
        "sharpe": sharpe,
        "n_bets": n_bets,
        "hit_rate": hit_rate,
    }


def _filter_by_profile(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    from strategy.src.race_filters import filter_df_by_race_num

    p = PROFILES[profile]
    return filter_df_by_race_num(
        df,
        race_num_min=p["race_num_min"],
        race_num_max=p["race_num_max"],
    )


def _wide_anchor_hit_rates(eval_df: pd.DataFrame, profile: str, year: int | None) -> dict[str, float]:
    sys.path.insert(0, str(PROJECT_ROOT / "model_training" / "scripts"))
    from combo_rank_hit_rates import race_metrics  # noqa: E402

    df = _filter_by_profile(eval_df, profile)
    if year is not None:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
    if df.empty:
        return {}

    from strategy.src.betting_framework import ProbabilityCalibrator

    cal_path = _resolve_calibrator_path(True)
    calibrator = None
    if cal_path is not None:
        calibrator = ProbabilityCalibrator.from_json(cal_path)
    if calibrator is not None:
        df["_prob"] = calibrator.transform(df["pred_rank1"]).clip(0.0, 1.0)
    else:
        df["_prob"] = pd.to_numeric(df["pred_rank1"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    grp_sum = df.groupby("race_id")["_prob"].transform("sum").clip(lower=1e-12)
    df["_prob"] = df["_prob"] / grp_sum

    rows = []
    for _, g in df.groupby("race_id", sort=False):
        if len(g) < 3:
            continue
        m = race_metrics(g, prob_col="_prob")
        if m:
            rows.append(m)
    if not rows:
        return {}
    agg = pd.DataFrame(rows)
    n = len(agg)
    return {
        "n_races": n,
        "wide_anchor_12": float(agg["wide_anchor_12"].mean()),
        "wide_anchor_13": float(agg["wide_anchor_13"].mean()),
        "wide_anchor_any": float(agg["wide_anchor_any"].mean()),
        "top1_win": float(agg["top1_win"].mean()),
    }


def _ensure_combo_odds(eval_df: pd.DataFrame, years: list[int]) -> Path:
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        y for y in years
        if not (ODDS_DIR / f"QuinellaOdds_{y}.csv").exists()
        or not (ODDS_DIR / f"WideOdds_{y}.csv").exists()
    ]
    if not missing:
        return ODDS_DIR

    from ev_filters import harville_quinella_pair_prob, harville_wide_pair_prob

    takeout = 0.25
    for year in missing:
        sub = eval_df[pd.to_numeric(eval_df["valid_year"], errors="coerce") == year].copy()
        if sub.empty:
            continue
        q_rows: list[dict] = []
        w_rows: list[dict] = []
        for rid, g in sub.groupby("race_id", sort=False):
            odds = pd.to_numeric(g["odds"], errors="coerce")
            valid = odds.notna() & (odds > 1.0)
            if valid.sum() < 2:
                continue
            g2 = g.loc[valid]
            odds2 = odds.loc[valid]
            inv = 1.0 / odds2
            p = (inv / inv.sum()).to_numpy(dtype=float)
            horses = g2["horse_num"].astype(int).to_numpy()
            p_dict = {int(h): float(prob) for h, prob in zip(horses, p)}
            for i in range(len(horses)):
                for j in range(i + 1, len(horses)):
                    h1, h2 = int(horses[i]), int(horses[j])
                    if h1 > h2:
                        h1, h2 = h2, h1
                    q_prob = harville_quinella_pair_prob(float(p[i]), float(p[j]))
                    w_prob = harville_wide_pair_prob(p_dict, h1, h2)
                    q_odds = max((1.0 - takeout) / max(q_prob, 1e-9), 1.01)
                    w_odds = max((1.0 - takeout) / max(w_prob, 1e-9), 1.01)
                    base = {"race_id": int(rid), "horse_num_1": h1, "horse_num_2": h2, "odds_status": "ok"}
                    q_rows.append({**base, "odds": round(q_odds, 1)})
                    w_rows.append({**base, "odds": round(w_odds, 1)})
        if q_rows:
            pd.DataFrame(q_rows).to_csv(ODDS_DIR / f"QuinellaOdds_{year}.csv", index=False, encoding="utf-8-sig")
        if w_rows:
            pd.DataFrame(w_rows).to_csv(ODDS_DIR / f"WideOdds_{year}.csv", index=False, encoding="utf-8-sig")
    return ODDS_DIR


def _wide_anchor_bet_metrics(
    eval_df: pd.DataFrame,
    profile: str,
    year: int | None,
) -> dict[str, Any]:
    from strategy.src.betting_framework import run_combo_betting_backtest

    df = _filter_by_profile(eval_df, profile)
    if year is not None:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
    if df.empty:
        return {"error": f"empty year={year}"}

    years = sorted(pd.to_numeric(df["valid_year"], errors="coerce").dropna().astype(int).unique().tolist())
    _ensure_combo_odds(eval_df, years)

    runtime = _load_runtime_cfg()
    p = PROFILES[profile]
    runtime = deepcopy(runtime)
    runtime["race_num_min"] = p["race_num_min"]
    runtime["race_num_max"] = p["race_num_max"]

    from main.pipeline.strategy_pipeline import strategy_config_from_runtime

    config = strategy_config_from_runtime(runtime)
    pair_top_n = int(runtime.get("pair_top_n", 2))
    wide_top_n = int(runtime.get("wide_top_n", 2))
    rank2_blend = float(runtime.get("rank2_blend", 0.35))

    bet_df, summary = run_combo_betting_backtest(
        df,
        config,
        ODDS_DIR,
        pair_top_n=pair_top_n,
        wide_top_n=wide_top_n,
        rank2_blend=rank2_blend,
    )
    wide = bet_df[bet_df["ticket_type"] == "wide"] if not bet_df.empty else bet_df
    # combo_backtest の wide はすべて pred_rank1 1位軸 × rank2 上位パートナー
    anchor = wide

    invest = float(anchor["actual_stake"].sum()) if not anchor.empty else 0.0
    ret = float(anchor["payout"].sum()) if not anchor.empty else 0.0
    n_bets = len(anchor)
    n_hits = int(anchor["is_hit"].sum()) if not anchor.empty else 0
    return {
        "n_bets": n_bets,
        "n_hits": n_hits,
        "hit_rate": n_hits / n_bets if n_bets else 0.0,
        "roi": ret / invest if invest > 0 else 0.0,
        "invest": invest,
        "return": ret,
        "summary_all_wide": {
            "roi_wide": summary.get("roi_wide"),
            "hit_rate_wide": summary.get("hit_rate_wide"),
            "n_bets_wide": summary.get("n_bets_wide"),
        },
    }


def _evaluate_model(name: str, eval_df: pd.DataFrame, years: list[int | None]) -> dict:
    out: dict[str, Any] = {"model": name, "profiles": {}}
    for profile_id, profile in PROFILES.items():
        prof: dict[str, Any] = {"label": profile["label"], "years": {}}
        for year in years:
            ykey = "all" if year is None else str(year)
            prof["years"][ykey] = {
                "win": _win_metrics(eval_df, profile_id, year),
                "wide_anchor_hit": _wide_anchor_hit_rates(eval_df, profile_id, year),
                "wide_anchor_bet": _wide_anchor_bet_metrics(eval_df, profile_id, year),
            }
        out["profiles"][profile_id] = prof
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-v5", action="store_true")
    parser.add_argument("--years", nargs="+", default=["2025", "all"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    years: list[int | None] = []
    for y in args.years:
        years.append(None if y.lower() == "all" else int(y))

    print("[1/3] load specv2 OOF eval ...")
    specv2_df = _load_specv2_eval()
    print(f"  specv2 rows={len(specv2_df):,} races={specv2_df['race_id'].nunique():,}")

    print("[2/3] load v5 eval (OOF preferred) ...")
    v5_df = _load_v5_eval(rebuild=args.rebuild_v5)
    v5_src = (
        "evaluation_v5_oof.csv"
        if V5_OOF_CSV.exists() and not args.rebuild_v5
        else ("evaluation_v5_final_predict.csv" if V5_CACHE.exists() else "live inference")
    )
    print(f"  v5 source={v5_src} rows={len(v5_df):,} races={v5_df['race_id'].nunique():,}")

    print("[3/3] cross-eval ...")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profiles": PROFILES,
        "specv2": _evaluate_model("ensemble_v5_specv2", specv2_df, years),
        "v5": _evaluate_model("ensemble_v5", v5_df, years),
        "baseline_meta": {},
    }
    backup_meta = MODELS_BASE / "backup_ensemble_v5_20260617" / "ensemble_meta.json"
    if backup_meta.exists():
        report["baseline_meta"] = json.loads(backup_meta.read_text(encoding="utf-8"))

    out_path = args.out or (
        PROJECT_ROOT / "model_training" / "data" / "03_train" / "compare_v5_specv2_eval.json"
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    def _fmt(m: dict) -> str:
        if "error" in m:
            return m["error"]
        return (
            f"ROI={m.get('roi', 0)*100:.1f}% "
            f"Sharpe={m.get('sharpe', 0):.3f} "
            f"MDD={m.get('mdd', 0)*100:.1f}% "
            f"n={m.get('n_bets', 0)} "
            f"hit={m.get('hit_rate', 0)*100:.1f}%"
        )

    for model_key in ("specv2", "v5"):
        block = report[model_key]
        print(f"\n=== {block['model']} ===")
        for pid, prof in block["profiles"].items():
            print(f"\n  [{pid}] {prof['label']}")
            for ykey, ym in prof["years"].items():
                print(f"    year {ykey} WIN: {_fmt(ym['win'])}")
                wh = ym.get("wide_anchor_hit") or {}
                if wh:
                    print(
                        f"    year {ykey} WIDE anchor hit: "
                        f"any={wh.get('wide_anchor_any', 0)*100:.1f}% "
                        f"12={wh.get('wide_anchor_12', 0)*100:.1f}% "
                        f"13={wh.get('wide_anchor_13', 0)*100:.1f}% "
                        f"(n_races={wh.get('n_races', 0)})"
                    )
                wb = ym.get("wide_anchor_bet") or {}
                if wb and "error" not in wb:
                    print(
                        f"    year {ykey} WIDE anchor bet: "
                        f"ROI={wb.get('roi', 0)*100:.1f}% "
                        f"hit={wb.get('hit_rate', 0)*100:.1f}% "
                        f"n={wb.get('n_bets', 0)}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

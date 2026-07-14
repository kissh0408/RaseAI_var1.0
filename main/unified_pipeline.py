"""統合当日パイプライン: LambdaRank（L1）→ prob_fusion（L2）→ betting（L3）。

Layer 1: pure_rank — 市場情報なし着順予測 → main/predictions/
Layer 2: prob_fusion — Benter条件付きロジット統合
Layer 3: betting — EV/Kelly 推奨 → main/results/

仕様: docs/specs/2026-07-08-benter-rebuild-master-plan.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BABA_SCENARIO_JV_CODES: tuple[int, ...] = (1, 2, 3, 4)
BABA_SCENARIO_LABEL_JA: dict[int, str] = {
    1: "良",
    2: "稍重",
    3: "重",
    4: "不良",
}
PREDICTION_OUTPUT_PATH = ROOT / "main" / "results" / "today_predictions_with_bets.csv"
FUSION_PARAMS_PATH = ROOT / "prob_fusion" / "data" / "fusion_params.json"


def _resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _odds_from_se(se_path: Path) -> pd.DataFrame:
    """race_se.csv から race_id, horse_id, odds（decimal）を抽出。"""
    from main.race_id_utils import _make_race_id_vec

    se = pd.read_csv(se_path, dtype=str)
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num", "horse_num"]:
        if col in se.columns:
            se[col] = pd.to_numeric(se[col], errors="coerce").fillna(0).astype(int)
    se["race_id"] = _make_race_id_vec(se).astype(str)
    se["horse_id"] = se["ketto_num"].astype(str)
    se["horse_num"] = se["horse_num"].astype(int)
    se["odds"] = pd.to_numeric(se["odds"], errors="coerce") / 10.0
    return se[["race_id", "horse_id", "horse_num", "odds"]].dropna(subset=["odds"])


def _ensure_export_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    rid = out["race_id"].str.zfill(16)
    if "month_day" not in out.columns:
        out["month_day"] = pd.to_numeric(rid.str[4:8], errors="coerce")
    if "course_code" not in out.columns:
        out["course_code"] = pd.to_numeric(rid.str[8:10], errors="coerce")
    if "race_num" not in out.columns:
        out["race_num"] = pd.to_numeric(rid.str[14:16], errors="coerce")
    if "horse_id" not in out.columns and "ketto_num" in out.columns:
        out["horse_id"] = out["ketto_num"].astype(str)
    return out


def _race_z_from_rank_preds(rank_preds: pd.DataFrame) -> pd.DataFrame:
    """LambdaRank 出力から pure_score / pure_score_z を生成。"""
    from pure_rank.src.score_utils import attach_pure_score_z

    rp = rank_preds.copy()
    rp["race_id"] = rp["race_id"].astype(str)
    score_col = "pred_score" if "pred_score" in rp.columns else "ensemble_score"
    rp["pure_score"] = pd.to_numeric(rp[score_col], errors="coerce")
    if "horse_num" not in rp.columns:
        rp["horse_num"] = pd.to_numeric(rp.get("horse_number", 0), errors="coerce")
    rp = attach_pure_score_z(rp, score_col="pure_score", race_id_col="race_id")
    rp["horse_number"] = rp["horse_num"].astype(int)
    return rp


def load_fusion_params(path: Path | None = None) -> dict[str, float]:
    p = path or FUSION_PARAMS_PATH
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        fold3 = next((f for f in data.get("folds", []) if f.get("fold") == 3), data.get("folds", [{}])[-1])
        return {
            "alpha": float(fold3.get("alpha", 1.0)),
            "beta": float(fold3.get("beta", 1.0)),
            "lam2": float(fold3.get("lam2", 1.0)),
            "lam3": float(fold3.get("lam3", 1.0)),
        }
    return {"alpha": 1.0, "beta": 1.0, "lam2": 1.0, "lam3": 1.0}


def fuse_rank_predictions(
    rank_preds: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """L2: rank scores + odds → fused probabilities."""
    from prob_fusion.src.predict_fusion import fuse_dataframe, load_fusion_config

    cfg = load_fusion_config()
    scored = _race_z_from_rank_preds(rank_preds)
    merged = scored.merge(
        odds_df[["race_id", "horse_num", "odds"]],
        on=["race_id", "horse_num"],
        how="inner",
    )
    params = params or load_fusion_params()
    return fuse_dataframe(
        merged,
        alpha=params["alpha"],
        beta=params["beta"],
        lam2=params["lam2"],
        lam3=params["lam3"],
        q_method=cfg.get("q_method", "proportional"),
        q_power=cfg.get("q_power", 0.81),
    )


def build_unified_export_df(
    odds_df: pd.DataFrame,
    rank_preds: dict[int, pd.DataFrame],
    fused_by_baba: dict[int, pd.DataFrame],
    *,
    primary_tc: int,
    recommendations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Var1×4馬場 + L2/L3 スコアを export 用 wide 形式に組み立てる。"""
    from main.race_runtime import filter_scratched

    base = odds_df.copy()
    base["race_id"] = base["race_id"].astype(str)
    base["horse_id"] = base.get("horse_id", base["race_id"]).astype(str) if "horse_id" in base.columns else base["race_id"]
    if "horse_id" not in base.columns:
        base["horse_id"] = base["race_id"]
    base = filter_scratched(base)
    export_df = _ensure_export_metadata(base)

    keys = ["race_id", "horse_num"]
    for jv in BABA_SCENARIO_JV_CODES:
        if jv not in rank_preds:
            continue
        rp = rank_preds[jv]
        score_col = "pred_score" if "pred_score" in rp.columns else "ensemble_score"
        v1 = rp.copy()
        v1["race_id"] = v1["race_id"].astype(str)
        v1 = v1.rename(columns={score_col: f"pred_rank1_baba{jv}"})[
            ["race_id", "horse_num", f"pred_rank1_baba{jv}"]
        ]
        export_df = export_df.merge(v1, on=keys, how="left")

        if jv in fused_by_baba:
            fused = fused_by_baba[jv].copy()
            fused["race_id"] = fused["race_id"].astype(str)
            fused_sub = fused[["race_id", "horse_number", "p_win", "p_place"]].rename(
                columns={
                    "horse_number": "horse_num",
                    "p_win": f"win_prob_est_baba{jv}",
                    "p_place": f"place_prob_est_baba{jv}",
                }
            )
            export_df = export_df.merge(fused_sub, on=keys, how="left")

    pri = primary_tc if primary_tc in rank_preds else min(rank_preds.keys())
    for suffix, base_col in (
        (f"pred_rank1_baba{pri}", "pred_rank1"),
        (f"win_prob_est_baba{pri}", "win_prob_est"),
        (f"place_prob_est_baba{pri}", "place_prob_est"),
    ):
        if suffix in export_df.columns:
            export_df[base_col] = export_df[suffix]

    if recommendations is not None and not recommendations.empty and "ev" in recommendations.columns:
        # loss_min_top1 mode recs have no EV column (no EV-threshold filter is used;
        # see betting/src/flat_top1.py). expected_return is an ev_filter-mode-only field.
        recs = recommendations.copy()
        recs["race_id"] = recs["race_id"].astype(str)
        export_df = export_df.merge(
            recs.groupby("race_id")["ev"].max().reset_index().rename(columns={"ev": "expected_return"}),
            on="race_id",
            how="left",
        )

    now_jst = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    export_df["odds_timestamp"] = now_jst
    export_df["generated_at"] = now_jst
    return export_df


def export_unified_by_venue_baba(
    export_df: pd.DataFrame,
    *,
    result_root: Path | None = None,
) -> tuple[Path, int]:
    from main.pipeline.export_utils import export_predictions_by_course_baba_race

    if export_df.empty:
        return (result_root or ROOT / "main" / "results"), 0
    root = result_root or ROOT / "main" / "results"
    md = export_df["month_day"].dropna()
    export_month_day = int(md.mode().iloc[0]) if len(md) else None
    return export_predictions_by_course_baba_race(
        export_df,
        BABA_SCENARIO_JV_CODES,
        BABA_SCENARIO_LABEL_JA,
        result_root=root,
        export_month_day=export_month_day,
    )


def _detect_track_condition_code(ra_path: Path) -> int:
    ra = pd.read_csv(ra_path)
    if len(ra) == 0:
        return 1
    row = ra.iloc[0]
    track_code = int(pd.to_numeric(row.get("track_code", 1), errors="coerce") or 1)
    turf = int(pd.to_numeric(row.get("turf_condition", 0), errors="coerce") or 0)
    dirt = int(pd.to_numeric(row.get("dirt_condition", 0), errors="coerce") or 0)
    cond = turf if track_code in (1, 2) else dirt
    return int(cond) if cond in (1, 2, 3, 4) else 1


def run_unified_today(
    *,
    ra_path: Path | str | None = None,
    se_path: Path | str | None = None,
    today_merged: pd.DataFrame | None = None,
    rank_features: dict[int, pd.DataFrame] | None = None,
    rank_preds: dict[int, pd.DataFrame] | None = None,
    track_condition_code: int | None = None,
    odds_csv: Path | str | None = None,
    bankroll: float = 100_000,
    write_rank: bool = True,
    write_bets: bool = True,
    write_venue_export: bool = True,
    rank_cfg: dict | None = None,
) -> dict[str, Any]:
    """当日フロー: L1 着順予測 → L2 確率統合 → L3 EV 推奨。"""
    from betting.src.backtest import load_betting_config
    from betting.src.recommend import run_recommendations, save_recommendations
    from main.notebook_bootstrap import (
        build_today_features as build_rank_features,
        load_config as load_rank_config,
        run_today_predictions,
        write_predictions,
    )

    ra = _resolve_path(ra_path or ROOT / "main" / "data" / "race" / "race_ra.csv")
    se = _resolve_path(se_path or ROOT / "main" / "data" / "race" / "race_se.csv")
    cfg = rank_cfg or load_rank_config()

    if today_merged is None:
        from main.notebook_bootstrap import _load_pure_rank_modules

        race_dir = ra.parent
        preprocessed_dir = ROOT / cfg["data"]["preprocessed_dir"]
        pr_mods = _load_pure_rank_modules()
        today_merged = pr_mods["today_adapter"].build_today_merged(race_dir, preprocessed_dir)

    if rank_features is None:
        rank_features = build_rank_features(today_merged, cfg)
    if rank_preds is None:
        rank_preds = run_today_predictions(rank_features, cfg)

    if write_rank:
        write_predictions(rank_preds, ROOT / "main" / "predictions")

    tc = track_condition_code or _detect_track_condition_code(ra)
    if tc not in rank_preds:
        tc = min(rank_preds.keys())

    odds_df = _odds_from_se(se)
    odds_ts = datetime.now(timezone.utc).isoformat()
    odds_source = "race_se_csv"
    if odds_csv is not None:
        from main.race_id_utils import load_realtime_odds

        # load_realtime_odds() returns exactly race_id/horse_num/odds (no horse_id;
        # O1-equivalent realtime feeds carry no pedigree id). horse_id is not required
        # downstream: build_unified_export_df() already falls back to horse_id=race_id
        # when the column is absent (see below). Selecting "horse_id" here previously
        # raised KeyError unconditionally (bug fixed 2026-07-10).
        rt = load_realtime_odds(_resolve_path(odds_csv))
        rt["race_id"] = rt["race_id"].astype(str)
        odds_df = rt[["race_id", "horse_num", "odds"]].dropna(subset=["odds"])
        odds_ts = datetime.now(timezone.utc).isoformat()
        odds_source = "realtime_o1"

    params = load_fusion_params()
    fused_by_baba: dict[int, pd.DataFrame] = {}
    for jv, rp in rank_preds.items():
        fused = fuse_rank_predictions(rp, odds_df, params=params)
        fused = fused.merge(
            odds_df[["race_id", "horse_num", "odds"]],
            left_on=["race_id", "horse_number"],
            right_on=["race_id", "horse_num"],
            how="left",
        )
        fused_by_baba[jv] = fused

    primary_fused = fused_by_baba[tc].copy()
    bet_cfg = load_betting_config()
    bet_cfg["bankroll"] = bankroll
    mode = bet_cfg.get("mode", "loss_min_top1")

    skipped: pd.DataFrame | None = None
    if mode == "loss_min_top1":
        # L3 loss-minimization path (default, 2026-07-10): model rank-1 (pure_score_z),
        # no EV threshold, flat sizing. See betting/src/flat_top1.py and
        # docs/specs/2026-07-10-loss-minimization-implementation-spec.md.
        from betting.src.flat_top1 import run_loss_min_recommendations

        primary_scored = _race_z_from_rank_preds(rank_preds[tc])
        recs, skipped = run_loss_min_recommendations(
            primary_scored,
            odds_df,
            cfg=bet_cfg,
            odds_timestamp=odds_ts,
            bankroll=bankroll,
            odds_source=odds_source,
        )
    else:
        recs = run_recommendations(primary_fused, cfg=bet_cfg, odds_timestamp=odds_ts)

    if write_bets and len(recs) > 0:
        date_str = datetime.now().strftime("%Y%m%d")
        if mode == "loss_min_top1":
            from betting.src.flat_top1 import save_loss_min_recommendations, save_skipped_races

            save_loss_min_recommendations(recs, ROOT / "main" / "results" / date_str)
            if skipped is not None:
                save_skipped_races(skipped, ROOT / "main" / "results" / date_str)
        else:
            save_recommendations(recs, ROOT / "main" / "results" / date_str)

    venue_files = 0
    venue_root = ROOT / "main" / "results"
    if write_venue_export:
        export_df = build_unified_export_df(
            odds_df,
            rank_preds,
            fused_by_baba,
            primary_tc=tc,
            recommendations=recs,
        )
        if len(export_df) > 0:
            PREDICTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            export_df.to_csv(PREDICTION_OUTPUT_PATH, index=False, encoding="utf-8-sig")
            venue_root, venue_files = export_unified_by_venue_baba(export_df)

    return {
        "track_condition_code": tc,
        "rank_scenarios": list(rank_preds.keys()),
        "fusion_alpha": params.get("alpha"),
        "fusion_beta": params.get("beta"),
        "mode": mode,
        "odds_source": odds_source,
        "recommendations": len(recs),
        "skipped_races": int(len(skipped)) if skipped is not None else None,
        "venue_export_files": venue_files,
        "results_dir": str(venue_root),
        "predictions_with_bets_csv": str(PREDICTION_OUTPUT_PATH),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified today: L1 rank + L2 fusion + L3 EV")
    parser.add_argument("--ra", type=Path, default=None)
    parser.add_argument("--se", type=Path, default=None)
    parser.add_argument("--odds", type=Path, default=None)
    parser.add_argument("--track-condition", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--no-rank", action="store_true")
    parser.add_argument("--no-bets", action="store_true")
    args = parser.parse_args()
    summary = run_unified_today(
        ra_path=args.ra,
        se_path=args.se,
        odds_csv=args.odds,
        track_condition_code=args.track_condition,
        write_rank=not args.no_rank,
        write_bets=not args.no_bets,
    )
    print(summary)

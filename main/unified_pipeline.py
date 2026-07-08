"""統合当日パイプライン: LambdaRank（能力）→ Binary+EV（ベット）。

Layer 1: pure_rank — 市場情報なし着順予測 → main/predictions/
Layer 2: model_training + strategy — init_score(β=0.15) + EV → main/results/

仕様: docs/specs/2026-07-05-var1-integration-architecture-design.md
"""
from __future__ import annotations

import sys
from datetime import datetime
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


def _resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _odds_from_se(se_path: Path) -> pd.DataFrame:
    """race_se.csv から race_id, horse_id, odds（decimal）を抽出。"""
    from main.build_today_features import _make_race_id_vec

    se = pd.read_csv(se_path, dtype=str)
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num", "horse_num"]:
        if col in se.columns:
            se[col] = pd.to_numeric(se[col], errors="coerce").fillna(0).astype(int)
    se["race_id"] = _make_race_id_vec(se).astype(str)
    se["horse_id"] = se["ketto_num"].astype(str)
    se["odds"] = pd.to_numeric(se["odds"], errors="coerce") / 10.0
    return se[["race_id", "horse_id", "odds"]].dropna(subset=["odds"])


def _ensure_export_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """競馬場×馬場シナリオ CSV 出力用に race_id から開催メタを補完する。"""
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


def _var1_display_score_frame(rank_preds: pd.DataFrame) -> pd.DataFrame:
    """export 用 Var1 表示スコア（pred_softmax_prob 優先）。"""
    rp = rank_preds.copy()
    rp["race_id"] = rp["race_id"].astype(str)
    if "horse_id" not in rp.columns and "ketto_num" in rp.columns:
        rp["horse_id"] = rp["ketto_num"].astype(str)
    if "pred_softmax_prob" in rp.columns:
        score = pd.to_numeric(rp["pred_softmax_prob"], errors="coerce")
    else:
        score = pd.to_numeric(rp["pred_score"], errors="coerce")
    return (
        rp.assign(_var1_score=score)[["race_id", "horse_id", "_var1_score"]]
        .drop_duplicates(["race_id", "horse_id"])
    )


def _attach_odds_for_export(base: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """run_strategy と同じキー正規化で odds を付与する。"""
    out = _ensure_export_metadata(base)
    out["race_id"] = out["race_id"].astype(str)
    out["horse_id"] = out["horse_id"].astype(str)
    od = odds_df.copy()
    od["race_id"] = od["race_id"].astype(str)
    od["horse_id"] = od["horse_id"].astype(str)
    out = out.merge(
        od[["race_id", "horse_id", "odds"]],
        on=["race_id", "horse_id"],
        how="left",
        suffixes=("_feat", "_live"),
    )
    if "odds_feat" in out.columns:
        out["odds"] = out["odds_live"].combine_first(out["odds_feat"])
        out = out.drop(columns=["odds_feat", "odds_live"], errors="ignore")
    elif "odds_live" in out.columns:
        out["odds"] = out["odds_live"]
        out = out.drop(columns=["odds_live"], errors="ignore")
    if "odds" not in out.columns:
        raise KeyError("odds column missing after merge with odds_df")
    return out


def compute_wide_ev_race_table(
    rank_preds: pd.DataFrame,
    *,
    rank_cfg: dict | None = None,
    o3_path: Path | None = None,
    apply_bracket: bool | None = None,
) -> pd.DataFrame:
    """Per-race wide EV summary (Layer 1 Harville + optional bracket)."""
    sys.path.insert(0, str(ROOT / "strategy" / "src"))
    sys.path.insert(0, str(ROOT / "pure_rank" / "src"))
    from main.notebook_bootstrap import load_config as load_rank_config
    from wide_ev_core import (
        load_wide_odds_live,
        live_dict_to_race_lookup,
        select_best_pair_by_divergence,
        select_best_pair_by_p_wide,
        o3_odds_path_default,
    )
    from wide_probability import compute_calibrated_wide_probs, load_bracket_models_from_config

    cfg = rank_cfg or load_rank_config()
    pl = cfg.get("plackett_luce", {})
    T_opt = float(pl.get("T_opt", 1.0))
    wide_inf = cfg.get("wide_inference", {})
    use_bracket = wide_inf.get("apply_bracket", True) if apply_bracket is None else apply_bracket
    models_dir = ROOT / cfg["data"]["models_dir"]
    bracket_models = load_bracket_models_from_config(cfg, models_dir) if use_bracket else None

    o3 = o3_path or o3_odds_path_default(ROOT)
    live = load_wide_odds_live(o3)
    wide_lookup = live_dict_to_race_lookup(live)

    rp = rank_preds.copy()
    rp["race_id"] = rp["race_id"].astype(str)
    score_col = "pred_score" if "pred_score" in rp.columns else "ensemble_score"
    if score_col not in rp.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for rid, grp in rp.groupby("race_id"):
        grp = grp.sort_values(score_col, ascending=False)
        if len(grp) < 2:
            continue
        scores = pd.to_numeric(grp[score_col], errors="coerce").fillna(0.0).values
        horses = [int(h) for h in pd.to_numeric(grp["horse_num"], errors="coerce")]
        if len(horses) < 2:
            continue
        p_map = compute_calibrated_wide_probs(
            scores,
            horses,
            T_opt=T_opt,
            bracket_models=bracket_models,
            wide_odds_lookup=wide_lookup,
            race_id=rid,
            apply_bracket=bool(use_bracket and bracket_models),
        )
        if not p_map:
            continue
        div_pick = select_best_pair_by_divergence(p_map, wide_lookup, rid) if wide_lookup else None
        if div_pick:
            pair, p_w, odds, ev, log_div = div_pick
        else:
            best = select_best_pair_by_p_wide(p_map)
            if not best:
                continue
            pair, p_w = best
            odds, ev, log_div = float("nan"), float("nan"), float("nan")
        rows.append(
            {
                "race_id": str(rid),
                "wide_h1": int(pair[0]),
                "wide_h2": int(pair[1]),
                "wide_pair": f"{pair[0]}-{pair[1]}",
                "p_wide": round(float(p_w), 4),
                "wide_odds": round(float(odds), 2) if pd.notna(odds) else np.nan,
                "ev_wide": round(float(ev), 4) if pd.notna(ev) else np.nan,
                "log_divergence": round(float(log_div), 4) if pd.notna(log_div) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_unified_export_df(
    binary_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    rank_preds: dict[int, pd.DataFrame],
    *,
    bankroll: float,
    primary_tc: int,
    primary_recs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Var1×4馬場 + Layer2 EV を main/results 分割 CSV 用の wide 形式に組み立てる。"""
    from main.race_runtime import filter_scratched

    sys.path.insert(0, str(ROOT / "model_training" / "scripts"))
    from merge_var1_pure_scores import attach_var1_z_from_rank_preds  # noqa: E402

    sys.path.insert(0, str(ROOT / "strategy" / "src"))
    from binary_recommendation import run_strategy  # noqa: E402

    merged = _attach_odds_for_export(binary_df, odds_df)
    merged = merged[merged["odds"].notna() & (merged["odds"] > 0)].copy()
    merged = filter_scratched(merged)
    if merged.empty:
        return merged

    export_df = merged.copy()
    keys = ["race_id", "horse_id"]
    for jv in BABA_SCENARIO_JV_CODES:
        if jv not in rank_preds:
            continue
        rp = rank_preds[jv]
        v1 = _var1_display_score_frame(rp).rename(
            columns={"_var1_score": f"pred_rank1_baba{jv}"}
        )
        export_df = export_df.merge(v1, on=keys, how="left")

        if primary_recs is not None and jv == primary_tc:
            recs = primary_recs.copy()
        else:
            df_jv = attach_var1_z_from_rank_preds(merged.copy(), rp)
            recs = run_strategy(df_jv, odds_df, bankroll=bankroll, rank_preds=rp)
        if recs.empty:
            continue
        recs = recs.copy()
        recs["race_id"] = recs["race_id"].astype(str)
        recs["horse_id"] = recs["horse_id"].astype(str)
        recs_sub = recs[keys + ["model_prob", "ev_rate"]].rename(
            columns={
                "model_prob": f"win_prob_est_baba{jv}",
                "ev_rate": f"expected_return_baba{jv}",
            }
        )
        export_df = export_df.merge(recs_sub, on=keys, how="left")

    pri = primary_tc if primary_tc in rank_preds else min(rank_preds.keys())
    for base_col, suffix in (
        ("pred_rank1", f"pred_rank1_baba{pri}"),
        ("win_prob_est", f"win_prob_est_baba{pri}"),
        ("expected_return", f"expected_return_baba{pri}"),
    ):
        if suffix in export_df.columns:
            export_df[base_col] = export_df[suffix]

    if pri in rank_preds:
        wide_tbl = compute_wide_ev_race_table(rank_preds[pri])
        if not wide_tbl.empty:
            wide_tbl["race_id"] = wide_tbl["race_id"].astype(str)
            export_df["race_id"] = export_df["race_id"].astype(str)
            export_df = export_df.merge(wide_tbl, on="race_id", how="left")

    now_jst = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    export_df["odds_timestamp"] = now_jst
    export_df["generated_at"] = now_jst

    sort_col = "win_prob_est" if "win_prob_est" in export_df.columns else "pred_rank1"
    if sort_col in export_df.columns:
        export_df = export_df.sort_values(
            ["race_id", sort_col],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)
    return export_df


def export_unified_by_venue_baba(
    export_df: pd.DataFrame,
    *,
    result_root: Path | None = None,
) -> tuple[Path, int]:
    """``main/results/<競馬場>/馬場_*/<n>R.csv`` に Var1+Layer2 スコアを書き出す。"""
    from main.pipeline.strategy_pipeline import export_predictions_by_course_baba_race

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
    """RA から馬場状態コード（1=良..4=不良）を推定。未確定時は 1。"""
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
    """当日フロー: 着順予測 CSV + EV 推奨 CSV を一括生成。"""
    from main.build_today_features import (
        build_today_features as build_binary_features,
        load_realtime_odds,
        merge_late_info,
    )
    from main.notebook_bootstrap import (
        build_today_features as build_rank_features,
        load_config as load_rank_config,
        run_today_predictions,
        write_predictions,
    )

    sys.path.insert(0, str(ROOT / "model_training" / "scripts"))
    from merge_var1_pure_scores import attach_var1_z_from_rank_preds  # noqa: E402

    sys.path.insert(0, str(ROOT / "strategy" / "src"))
    from binary_recommendation import run_strategy, save_recommendations  # noqa: E402

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

    binary_df = build_binary_features(ra, se)
    odds_df = _odds_from_se(se)
    if odds_csv is not None:
        rt = load_realtime_odds(_resolve_path(odds_csv))
        rt = rt.merge(
            binary_df[["race_id", "horse_id", "horse_num"]].drop_duplicates(),
            on=["race_id", "horse_num"],
            how="left",
        )
        odds_df = rt[["race_id", "horse_id", "odds"]].dropna(subset=["odds"])
    binary_df = merge_late_info(binary_df, odds_df=odds_df)
    binary_df = attach_var1_z_from_rank_preds(binary_df, rank_preds[tc])

    date_str = datetime.now().strftime("%Y%m%d")
    recs = run_strategy(binary_df, odds_df, bankroll=bankroll, rank_preds=rank_preds[tc])
    if write_bets and len(recs) > 0:
        save_recommendations(recs, date_str)

    venue_files = 0
    venue_root = ROOT / "main" / "results"
    if write_venue_export:
        export_df = build_unified_export_df(
            binary_df,
            odds_df,
            rank_preds,
            bankroll=bankroll,
            primary_tc=tc,
            primary_recs=recs,
        )
        if len(export_df) > 0:
            PREDICTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            export_df.to_csv(PREDICTION_OUTPUT_PATH, index=False, encoding="utf-8-sig")
            export_df.to_parquet(PREDICTION_OUTPUT_PATH.with_suffix(".parquet"), index=False)
            venue_root, venue_files = export_unified_by_venue_baba(export_df)

    n_rec = int(recs["is_recommended"].sum()) if len(recs) and "is_recommended" in recs.columns else 0
    return {
        "track_condition_code": tc,
        "rank_scenarios": list(rank_preds.keys()),
        "binary_rows": len(binary_df),
        "recommendations": len(recs),
        "recommended_bets": n_rec,
        "venue_export_files": venue_files,
        "results_dir": str(venue_root),
        "predictions_with_bets_csv": str(PREDICTION_OUTPUT_PATH),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified today: rank + EV recommendations")
    parser.add_argument("--ra", type=Path, default=None)
    parser.add_argument("--se", type=Path, default=None)
    parser.add_argument("--odds", type=Path, default=None, help="realtime_odds/o1_odds.csv")
    parser.add_argument("--track-condition", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--no-rank", action="store_true", help="Skip rank CSV output")
    parser.add_argument("--no-bets", action="store_true", help="Skip EV recommendations")
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

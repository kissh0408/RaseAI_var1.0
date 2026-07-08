"""Fold 別 EV 分布診断（Phase 1 Week 1: D1–D4）。

Arm A 本番設定（blend OFF, bet_tuning OFF）と同一の推論パイプラインで
全頭の model_prob / edge / EV 分布を集計し、Fold 3 枯渇の仮説 H1–H3 を切り分ける。

実行:
    python strategy/src/diagnose_ev_distribution.py
    python strategy/src/diagnose_ev_distribution.py --folds 1 3
    python strategy/src/diagnose_ev_distribution.py --folds 3 --no-isotonic
    python strategy/src/diagnose_ev_distribution.py --folds 3 --isotonic

出力:
    model_training/models/ev_distribution_diagnosis.json
    model_training/models/ev_distribution_diagnosis_YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from backtest import load_features_with_odds  # noqa: E402
from calibration import get_raw_scores  # noqa: E402
from ev_calculator import apply_ev_filters, enrich_predictions  # noqa: E402
from inference_common import (  # noqa: E402
    apply_max_picks_per_race,
    load_ensemble_models,
    normalize_within_race,
    predict_model_probs,
)
from kelly_sizer import apply_kelly_sizing  # noqa: E402
from pipeline_common import MODELS_DIR, load_config  # noqa: E402
from plackett_luce import tune_temperature  # noqa: E402
from train import get_feature_cols  # noqa: E402

FORMAT_VERSION = "1.0"
PERCENTILES = [25, 50, 75, 90, 99]
PROB_HIST_BINS = 20  # [0, 1] 等幅
EV_SWEEP_THRESHOLDS = [0.9, 0.95, 1.0, 1.05, 1.08, 1.1, 1.15, 1.2]
ODDS_BANDS: list[tuple[float, float, str]] = [
    (2.0, 3.0, "2-3"),
    (3.0, 5.0, "3-5"),
    (5.0, 10.0, "5-10"),
    (10.0, 20.0, "10-20"),
    (20.0, 50.0, "20-50"),
]
COURSE_LABELS: dict[str, str] = {
    "1": "札幌",
    "2": "函館",
    "3": "福島",
    "4": "新潟",
    "5": "東京",
    "6": "中山",
    "7": "中京",
    "8": "京都",
    "9": "阪神",
    "10": "小倉",
}
TRACK_LABELS: dict[str, str] = {
    "1": "良",
    "2": "稍重",
    "3": "重",
    "4": "不良",
}


def _percentile_summary(series: pd.Series) -> dict[str, float | int | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": None, "std": None, **{f"p{p}": None for p in PERCENTILES}}
    out: dict[str, float | int | None] = {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(s, p))
    return out


def _histogram(series: pd.Series, *, n_bins: int = PROB_HIST_BINS) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"bin_edges": [], "counts": [], "n": 0}
    counts, edges = np.histogram(s.clip(0.0, 1.0), bins=n_bins, range=(0.0, 1.0))
    return {
        "n": int(len(s)),
        "bin_edges": [float(x) for x in edges],
        "counts": [int(x) for x in counts],
    }


def _filter_attribution(df: pd.DataFrame, t_cfg: dict, ev_threshold: float) -> dict[str, Any]:
    """Arm A 固定閾値で各フィルタが何頭を落とすか（H3）。"""
    work = enrich_predictions(df, model_prob_col="model_prob", odds_col="odds")
    work = apply_kelly_sizing(
        work,
        bankroll=10_000_000,
        kelly_frac=t_cfg["kelly_fraction"],
        max_bet_ratio=t_cfg["max_bet_ratio"],
    )
    n = len(work)
    m_ev = work["ev_rate"] >= ev_threshold
    m_odds_lo = work["odds"] >= t_cfg["min_odds"]
    m_odds_hi = work["odds"] <= t_cfg["max_odds"]
    m_prob = work["model_prob"] >= t_cfg["min_model_prob"]
    m_kelly = work["kelly_bet_yen"] > 0
    base = m_ev & m_odds_lo & m_odds_hi & m_prob & m_kelly

    max_pick = int(t_cfg.get("max_picks_per_race", 2))
    after_max = apply_max_picks_per_race(work, base, max_pick)

    excl_surface = t_cfg.get("exclude_surface_codes", [3])
    if excl_surface and "surface_code" in work.columns:
        m_surface = ~work["surface_code"].isin(excl_surface)
    else:
        m_surface = pd.Series(True, index=work.index)

    excl_abnormal = t_cfg.get("exclude_abnormal_codes", [])
    if excl_abnormal and "abnormal_code" in work.columns:
        m_abnormal = ~work["abnormal_code"].isin(excl_abnormal)
    else:
        m_abnormal = pd.Series(True, index=work.index)

    min_hc = t_cfg.get("min_horse_count")
    if min_hc and "horse_count" in work.columns:
        m_hc = work["horse_count"] >= min_hc
    else:
        m_hc = pd.Series(True, index=work.index)

    final = after_max & m_surface & m_abnormal & m_hc

    n_races = work["race_id"].nunique() if "race_id" in work.columns else None
    n_races_final = (
        work.loc[final, "race_id"].nunique() if final.any() and "race_id" in work.columns else 0
    )

    return {
        "ev_threshold": ev_threshold,
        "n_horses_total": n,
        "n_pass_ev_only": int(m_ev.sum()),
        "n_pass_ev_and_odds": int((m_ev & m_odds_lo & m_odds_hi).sum()),
        "n_pass_ev_odds_prob": int((m_ev & m_odds_lo & m_odds_hi & m_prob).sum()),
        "n_pass_before_max_picks": int(base.sum()),
        "n_pass_after_max_picks": int(after_max.sum()),
        "n_pass_final_arm_a": int(final.sum()),
        "n_races_total": int(n_races) if n_races is not None else None,
        "n_races_with_bet": int(n_races_final),
        "fail_counts": {
            "ev_below_threshold": int((~m_ev).sum()),
            "odds_below_min": int((m_ev & ~m_odds_lo).sum()),
            "odds_above_max": int((m_ev & m_odds_lo & ~m_odds_hi).sum()),
            "model_prob_below_min": int((m_ev & m_odds_lo & m_odds_hi & ~m_prob).sum()),
            "kelly_zero": int((m_ev & m_odds_lo & m_odds_hi & m_prob & ~m_kelly).sum()),
            "max_picks_cap": int((base & ~after_max).sum()),
            "excluded_surface": int((after_max & ~m_surface).sum()),
            "excluded_abnormal": int((after_max & m_surface & ~m_abnormal).sum()),
            "horse_count_below_min": int((after_max & m_surface & m_abnormal & ~m_hc).sum()),
        },
    }


def _ev_threshold_sweep(df: pd.DataFrame, t_cfg: dict) -> list[dict[str, Any]]:
    """D2: 閾値スイープ（件数 + 通過馬の平均オッズ・model_prob）。"""
    work = enrich_predictions(df, model_prob_col="model_prob", odds_col="odds")
    work = apply_kelly_sizing(
        work,
        bankroll=10_000_000,
        kelly_frac=t_cfg["kelly_fraction"],
        max_bet_ratio=t_cfg["max_bet_ratio"],
    )
    rows: list[dict[str, Any]] = []
    for thr in EV_SWEEP_THRESHOLDS:
        mask = apply_ev_filters(
            work,
            ev_threshold=thr,
            min_odds=t_cfg["min_odds"],
            max_odds=t_cfg["max_odds"],
            min_model_prob=t_cfg["min_model_prob"],
        ) & (work["kelly_bet_yen"] > 0)
        masked = apply_max_picks_per_race(work, mask, int(t_cfg.get("max_picks_per_race", 2)))
        sel = work.loc[masked]
        rows.append(
            {
                "ev_threshold": thr,
                "n_bets": int(masked.sum()),
                "n_races_with_bet": int(work.loc[masked, "race_id"].nunique()) if masked.any() else 0,
                "pass_rate_horses": float(masked.mean()) if len(work) else 0.0,
                "mean_odds_pass": float(sel["odds"].mean()) if len(sel) else None,
                "mean_model_prob_pass": float(sel["model_prob"].mean()) if len(sel) else None,
                "mean_edge_pass": float(sel["model_edge"].mean()) if len(sel) else None,
            }
        )
    return rows


def _var1_correlation(df: pd.DataFrame) -> dict[str, Any]:
    """D3: var1_z と model_prob の関係。"""
    z_col = "var1_pure_score_z"
    if z_col not in df.columns:
        return {"available": False}
    sub = df[[z_col, "model_prob", "race_date"]].dropna()
    if len(sub) < 10:
        return {"available": True, "n": int(len(sub)), "pearson_r": None}
    r = float(sub[z_col].corr(sub["model_prob"]))
    by_year: dict[str, Any] = {}
    sub = sub.copy()
    sub["year"] = pd.to_datetime(sub["race_date"]).dt.year
    for yr, g in sub.groupby("year"):
        if len(g) >= 10:
            by_year[str(int(yr))] = {
                "n": int(len(g)),
                "pearson_r": float(g[z_col].corr(g["model_prob"])),
                "var1_z_mean": float(g[z_col].mean()),
                "model_prob_mean": float(g["model_prob"].mean()),
            }
    return {
        "available": True,
        "n": int(len(sub)),
        "pearson_r": r,
        "by_year": by_year,
    }


def _condition_breakdown(df: pd.DataFrame, t_cfg: dict) -> dict[str, Any]:
    """D4: 競馬場・馬場・芝ダ・オッズ帯別 edge / EV 通過率。"""
    work = enrich_predictions(df, model_prob_col="model_prob", odds_col="odds")
    ev_thr = float(t_cfg["ev_threshold"])
    out: dict[str, Any] = {}

    def _seg(name: str, col: str, labels: dict[str, str] | None = None) -> None:
        if col not in work.columns:
            return
        segs: dict[str, Any] = {}
        for val, g in work.groupby(col, observed=True):
            key = str(val)
            label = (labels or {}).get(key, key)
            pass_ev = (g["ev_rate"] >= ev_thr).sum()
            segs[key] = {
                "label": label,
                "n_horses": int(len(g)),
                "model_edge": _percentile_summary(g["model_edge"]),
                "ev_pass_rate_at_fixed_threshold": float(pass_ev / len(g)) if len(g) else 0.0,
            }
        out[name] = segs

    _seg("by_course_code", "course_code", COURSE_LABELS)
    _seg("by_track_condition_code", "track_condition_code", TRACK_LABELS)
    _seg("by_surface_code", "surface_code", {"1": "芝", "2": "ダート", "5": "その他"})

    odds_seg: dict[str, Any] = {}
    for lo, hi, label in ODDS_BANDS:
        g = work[(work["odds"] >= lo) & (work["odds"] < hi)]
        if len(g) == 0:
            continue
        odds_seg[label] = {
            "n_horses": int(len(g)),
            "model_edge": _percentile_summary(g["model_edge"]),
            "ev_pass_rate_at_fixed_threshold": float((g["ev_rate"] >= ev_thr).mean()),
        }
    out["by_odds_band"] = odds_seg
    return out


def _analyze_period(df: pd.DataFrame, t_cfg: dict) -> dict[str, Any]:
    if len(df) == 0:
        return {"n_horses": 0, "n_races": 0}
    work = enrich_predictions(df, model_prob_col="model_prob", odds_col="odds")
    date_min = pd.to_datetime(work["race_date"]).min()
    date_max = pd.to_datetime(work["race_date"]).max()

    d1 = {
        "model_prob": {
            "percentiles": _percentile_summary(work["model_prob"]),
            "histogram": _histogram(work["model_prob"]),
        },
        "implied_prob": {
            "percentiles": _percentile_summary(work["implied_prob"]),
            "histogram": _histogram(work["implied_prob"]),
        },
        "model_edge": {
            "percentiles": _percentile_summary(work["model_edge"]),
            "histogram": _histogram(
                work["model_edge"].clip(-0.5, 0.5).add(0.5)
            ),  # edge を [0,1] にシフトして可視化
        },
        "ev_rate": {
            "percentiles": _percentile_summary(work["ev_rate"]),
            "histogram": _histogram(work["ev_rate"].clip(0, 3) / 3.0),
        },
        "edge_positive_rate": float((work["model_edge"] > 0).mean()),
        "ev_above_1_rate": float((work["ev_rate"] >= 1.0).mean()),
    }
    return {
        "date_range": [str(date_min.date()), str(date_max.date())],
        "n_horses": int(len(work)),
        "n_races": int(work["race_id"].nunique()) if "race_id" in work.columns else None,
        "d1_distribution": d1,
        "d2_ev_sweep": _ev_threshold_sweep(work, t_cfg),
        "d3_var1_correlation": _var1_correlation(work),
        "d4_by_condition": _condition_breakdown(work, t_cfg),
        "h3_filter_attribution_arm_a": _filter_attribution(work, t_cfg, float(t_cfg["ev_threshold"])),
    }


class FoldPredictor:
    """backtest.py と同一の binary 推論（Arm A: blend OFF）。"""

    def __init__(
        self,
        fold: int,
        df_all: pd.DataFrame,
        fold_cfg: dict,
        cfg: dict,
        *,
        use_isotonic: bool | None = None,
    ) -> None:
        self.fold = fold
        self.fold_cfg = fold_cfg
        self.cfg = cfg
        self.t_cfg = cfg["training"]
        self.use_isotonic = use_isotonic
        self.models = load_ensemble_models(fold)
        model = self.models[0]
        model_feature_cols = list(model.feature_name()) if hasattr(model, "feature_name") else []
        resolved = model_feature_cols or get_feature_cols(cfg)
        self.available = [c for c in resolved if c in df_all.columns]
        self.base_margin_col = self.t_cfg.get("base_margin_col")
        self.temperature = self._resolve_temperature(df_all, fold_cfg)
        self.iso = self._fit_isotonic(df_all, fold_cfg)

    def _resolve_temperature(self, df_all: pd.DataFrame, fold_cfg: dict) -> float:
        cal_cfg = self.t_cfg.get("calibration", {})
        temperature = float(cal_cfg.get("temperature", 0.8))
        if not cal_cfg.get("temperature_tune", False):
            return temperature
        valid_start = pd.Timestamp(fold_cfg["valid_start"])
        valid_end = pd.Timestamp(fold_cfg.get("valid_end", fold_cfg["test_start"]))
        valid_df = df_all[
            (df_all["race_date"] >= valid_start) & (df_all["race_date"] <= valid_end)
        ]
        if len(valid_df) == 0:
            return temperature
        try:
            valid_scores = get_raw_scores(self.models[0], valid_df, self.available)
            v = valid_df.copy()
            v["raw_score"] = valid_scores
            v["is_win"] = (v["finish_rank"] == 1).astype(int)
            t_range = tuple(cal_cfg.get("temperature_range", [0.8, 3.0]))
            return tune_temperature(v, score_col="raw_score", label_col="is_win", t_range=t_range)
        except Exception:
            return temperature

    def _fit_isotonic(self, df_all: pd.DataFrame, fold_cfg: dict):
        cal_cfg = self.t_cfg.get("calibration", {})
        iso_on = cal_cfg.get("isotonic", True) if self.use_isotonic is None else self.use_isotonic
        if not iso_on:
            return None
        valid_start = pd.Timestamp(fold_cfg["valid_start"])
        valid_end = pd.Timestamp(fold_cfg.get("valid_end", fold_cfg["test_start"]))
        valid_df = df_all[
            (df_all["race_date"] >= valid_start) & (df_all["race_date"] <= valid_end)
        ]
        if len(valid_df) == 0:
            return None
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            prob = predict_model_probs(
                self.models,
                valid_df,
                self.available,
                self.base_margin_col,
                temperature=self.temperature,
                t_cfg=self.t_cfg,
            )
            iso.fit(prob.values, (valid_df["finish_rank"] == 1).astype(int).values)
            return iso
        except Exception:
            return None

    def predict(self, df_period: pd.DataFrame) -> pd.DataFrame:
        if len(df_period) == 0:
            return pd.DataFrame()
        out = df_period.copy()
        out["model_prob"] = predict_model_probs(
            self.models,
            out,
            self.available,
            self.base_margin_col,
            temperature=self.temperature,
            t_cfg=self.t_cfg,
        )
        if self.iso is not None:
            cal = self.iso.predict(out["model_prob"].values)
            out["model_prob"] = normalize_within_race(cal, out)
        return out


def _split_periods(df_all: pd.DataFrame, fold_cfg: dict) -> dict[str, pd.DataFrame]:
    train_end = pd.Timestamp(fold_cfg["train_end"])
    valid_start = pd.Timestamp(fold_cfg["valid_start"])
    valid_end = pd.Timestamp(fold_cfg.get("valid_end", fold_cfg["test_start"]))
    test_start = pd.Timestamp(fold_cfg["test_start"])
    test_end = pd.Timestamp(fold_cfg["test_end"])
    return {
        "train": df_all[df_all["race_date"] <= train_end].copy(),
        "valid": df_all[(df_all["race_date"] >= valid_start) & (df_all["race_date"] <= valid_end)].copy(),
        "test": df_all[(df_all["race_date"] >= test_start) & (df_all["race_date"] <= test_end)].copy(),
    }


def _cross_fold_compare(folds_out: list[dict]) -> dict[str, Any]:
    by_fold = {f["fold"]: f for f in folds_out}

    def _test_period(fold: int) -> dict | None:
        return by_fold.get(fold, {}).get("periods", {}).get("test")

    t1, t3 = _test_period(1), _test_period(3)
    if not t1 or not t3:
        return {}
    cmp: dict[str, Any] = {
        "fold1_test_vs_fold3_test": {
            "model_prob_p50": {
                "fold1": t1["d1_distribution"]["model_prob"]["percentiles"].get("p50"),
                "fold3": t3["d1_distribution"]["model_prob"]["percentiles"].get("p50"),
            },
            "implied_prob_p50": {
                "fold1": t1["d1_distribution"]["implied_prob"]["percentiles"].get("p50"),
                "fold3": t3["d1_distribution"]["implied_prob"]["percentiles"].get("p50"),
            },
            "model_edge_p50": {
                "fold1": t1["d1_distribution"]["model_edge"]["percentiles"].get("p50"),
                "fold3": t3["d1_distribution"]["model_edge"]["percentiles"].get("p50"),
            },
            "ev_pass_final_arm_a": {
                "fold1": t1["h3_filter_attribution_arm_a"]["n_pass_final_arm_a"],
                "fold3": t3["h3_filter_attribution_arm_a"]["n_pass_final_arm_a"],
            },
            "var1_pearson_r": {
                "fold1": t1["d3_var1_correlation"].get("pearson_r"),
                "fold3": t3["d3_var1_correlation"].get("pearson_r"),
            },
        }
    }
    # 仮説向け簡易スコア（記述のみ、自動判定はしない）
    e1_f3 = t3["d1_distribution"]["implied_prob"]["percentiles"].get("p50")
    e1_f1 = t1["d1_distribution"]["implied_prob"]["percentiles"].get("p50")
    edge_f3 = t3["d1_distribution"]["model_edge"]["percentiles"].get("p50")
    edge_f1 = t1["d1_distribution"]["model_edge"]["percentiles"].get("p50")
    cmp["hypothesis_hints"] = {
        "H1_market_tightening": (
            "implied_prob_p50 increased fold1→fold3"
            if e1_f3 is not None and e1_f1 is not None and e1_f3 > e1_f1
            else "no clear implied_prob shift"
        ),
        "H2_model_conservative": (
            "model_edge_p50 decreased fold1→fold3"
            if edge_f3 is not None and edge_f1 is not None and edge_f3 < edge_f1
            else "no clear edge shrink"
        ),
        "H3_filter_binding": (
            f"fold3 final bets={t3['h3_filter_attribution_arm_a']['n_pass_final_arm_a']} "
            f"vs ev-only={t3['h3_filter_attribution_arm_a']['n_pass_ev_only']}"
        ),
    }
    return cmp


def _extract_ablation_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Fold 別 valid/test の ablation 比較用サマリ。"""
    out: dict[str, Any] = {}
    for fold_block in report.get("folds", []):
        fold = fold_block.get("fold")
        for period in ("valid", "test"):
            p = fold_block.get("periods", {}).get(period, {})
            if not p or p.get("n_horses", 0) == 0:
                continue
            d1 = p.get("d1_distribution", {})
            h3 = p.get("h3_filter_attribution_arm_a", {})
            key = f"fold{fold}_{period}"
            out[key] = {
                "edge_positive_rate": d1.get("edge_positive_rate"),
                "ev_above_1_rate": d1.get("ev_above_1_rate"),
                "model_edge_p50": d1.get("model_edge", {}).get("percentiles", {}).get("p50"),
                "n_pass_ev_only": h3.get("n_pass_ev_only"),
                "n_pass_final_arm_a": h3.get("n_pass_final_arm_a"),
            }
    return out


def run_diagnosis(
    fold_ids: list[int] | None = None,
    *,
    use_isotonic: bool | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    t_cfg = cfg["training"]
    df_all = load_features_with_odds()

    fold_cfgs = cfg["training"]["walkforward_folds"]
    if fold_ids:
        fold_cfgs = [f for f in fold_cfgs if f["fold"] in fold_ids]

    folds_out: list[dict] = []
    for fold_cfg in fold_cfgs:
        fold = int(fold_cfg["fold"])
        print(f"Fold {fold} 診断中...")
        predictor = FoldPredictor(fold, df_all, fold_cfg, cfg, use_isotonic=use_isotonic)
        periods_raw = _split_periods(df_all, fold_cfg)
        periods_out: dict[str, Any] = {}
        for pname, pdf in periods_raw.items():
            if len(pdf) == 0:
                periods_out[pname] = {"n_horses": 0}
                continue
            pred = predictor.predict(pdf)
            periods_out[pname] = _analyze_period(pred, t_cfg)
            print(
                f"  {pname}: n={periods_out[pname]['n_horses']}, "
                f"arm_a_bets={periods_out[pname]['h3_filter_attribution_arm_a']['n_pass_final_arm_a']}"
            )
        folds_out.append(
            {
                "fold": fold,
                "test_year_label": str(pd.Timestamp(fold_cfg["test_start"]).year),
                "periods": periods_out,
            }
        )

    iso_label = (
        "config_default"
        if use_isotonic is None
        else ("isotonic_on" if use_isotonic else "isotonic_off")
    )
    report = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypotheses": {
            "H1": "market_efficiency_implied_prob_rise_edge_shrink",
            "H2": "model_conservative_model_prob_shift_var1_overfit",
            "H3": "filter_too_strict_ev_min_odds_max_picks",
        },
        "meta": {
            "arm_config": "A",
            "calibration_mode": iso_label,
            "isotonic_enabled": (
                use_isotonic
                if use_isotonic is not None
                else t_cfg.get("calibration", {}).get("isotonic", True)
            ),
            "bet_tuning_enabled": t_cfg.get("bet_tuning", {}).get("enabled", False),
            "var1_market_blend_enabled": t_cfg.get("var1_market_blend", {}).get("enabled", False),
            "ev_threshold": t_cfg["ev_threshold"],
            "min_odds": t_cfg["min_odds"],
            "max_odds": t_cfg["max_odds"],
            "min_model_prob": t_cfg["min_model_prob"],
            "max_picks_per_race": t_cfg.get("max_picks_per_race", 2),
            "percentiles_reported": PERCENTILES,
            "prob_histogram_bins": PROB_HIST_BINS,
            "ev_sweep_thresholds": EV_SWEEP_THRESHOLDS,
            "odds_bands": [{"lo": lo, "hi": hi, "label": lb} for lo, hi, lb in ODDS_BANDS],
            "note_test_usage": "TEST metrics are descriptive only; no threshold tuning on TEST.",
        },
        "folds": folds_out,
        "cross_fold_comparison": _cross_fold_compare(folds_out),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold EV distribution diagnosis (D1-D4)")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Fold IDs (default: all)")
    iso_group = parser.add_mutually_exclusive_group()
    iso_group.add_argument(
        "--isotonic",
        action="store_true",
        help="Isotonic 較正を強制 ON（VALID で fit → 全期間に適用）",
    )
    iso_group.add_argument(
        "--no-isotonic",
        action="store_true",
        help="Isotonic 較正を強制 OFF（生 model_prob）",
    )
    args = parser.parse_args()

    use_iso: bool | None = None
    if args.isotonic:
        use_iso = True
    elif args.no_isotonic:
        use_iso = False

    report = run_diagnosis(fold_ids=args.folds, use_isotonic=use_iso)
    iso_suffix = (
        "_iso_on" if use_iso is True else ("_iso_off" if use_iso is False else "")
    )
    out_path = MODELS_DIR / f"ev_distribution_diagnosis{iso_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    stamped = MODELS_DIR / f"ev_distribution_diagnosis_{stamp}.json"
    shutil.copy2(out_path, stamped)
    print(f"\n診断結果保存: {out_path}")
    print(f"  (コピー: {stamped.name})")


if __name__ == "__main__":
    main()

"""戦略・推奨生成系。

EV 計算・Kelly サイジング・買目フォーマット・Notebook 表示ロジックを担う。
データ取得・モデル推論はここに含まない。
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from main.race_runtime import filter_scratched

logger = logging.getLogger(__name__)

_BABA_COL_RE = re.compile(r"_baba[0-9]+$")

_COURSE_FOLDER_JA: dict[int, str] = {
    1: "札幌",
    2: "函館",
    3: "福島",
    4: "新潟",
    5: "東京",
    6: "中山",
    7: "中京",
    8: "京都",
    9: "阪神",
    10: "小倉",
}

# results／Notebook サマリー用CSVのヘッダ（読みやすさのため英語内部列からここへ揃える）
_PREDICT_VIEW_EN_TO_JA: dict[str, str] = {
    "ticket_type": "券種",
    "ticket": "買い目",
    "pred_prob": "モデル勝率シェア",
    "expected_value": "期待値",
    "edge": "エッジ",
    "suggested_stake": "推奨投資額",
    "phase": "フェーズ",
    "odds_timestamp": "オッズ取得時刻",
}
_PREDICT_VIEW_COLUMNS_JA: tuple[str, ...] = (
    "開催日",
    "競馬場",
    "R",
    "券種",
    "買い目",
    "予測スコア",
    "オッズ",
    "モデル勝率シェア",
    "期待値",
    "エッジ",
    "推奨投資額",
    "フェーズ",
    "オッズ取得時刻",
)


# ---------------------------------------------------------------------------
# ユーティリティ（フォーマット・ソート系）
# ---------------------------------------------------------------------------

def _sanitize_path_segment(name: str) -> str:
    for ch in r'<>:"/\|?*':
        name = name.replace(ch, "_")
    s = name.strip().rstrip(".")
    return s or "unknown"


def _course_folder_name(course_code_val) -> str:
    code = pd.to_numeric(pd.Series([course_code_val]), errors="coerce").iloc[0]
    if pd.isna(code):
        return _sanitize_path_segment(str(course_code_val))
    ci = int(code)
    return _sanitize_path_segment(_COURSE_FOLDER_JA.get(ci, f"競馬場コード{ci}"))


def _format_month_day(value) -> str:
    if pd.isna(value):
        return "-"
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return "-"
    digits = digits[-4:].zfill(4)
    mm = int(digits[:2])
    dd = int(digits[2:])
    if mm <= 0 or mm > 12 or dd <= 0 or dd > 31:
        return s
    return f"{mm}月{dd}日"


def _course_name(value) -> str:
    course_map = {
        1: "札幌",
        2: "函館",
        3: "福島",
        4: "新潟",
        5: "東京",
        6: "中山",
        7: "中京",
        8: "京都",
        9: "阪神",
        10: "小倉",
    }
    if pd.isna(value):
        return "-"
    try:
        code = int(float(value))
    except Exception:
        return str(value)
    return course_map.get(code, str(code))


def _sort_by_forecast_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main/results 出力用の並び順。
    win_prob_est があれば高い順、無ければ pred_score 等にフォールバック。
    """
    if df.empty:
        return df
    out = df.copy()
    if "win_prob_est" in out.columns:
        out["_ord0"] = pd.to_numeric(out["win_prob_est"], errors="coerce")
        by = ["_ord0"]
        n = 1
        for k in ("pred_rank1", "pred_rank2", "pred_rank3"):
            if k in out.columns:
                out[f"_ord{n}"] = pd.to_numeric(out[k], errors="coerce")
                by.append(f"_ord{n}")
                n += 1
    elif "pred_score" in out.columns:
        out["_ord0"] = pd.to_numeric(out["pred_score"], errors="coerce")
        by = ["_ord0"]
    elif "pred_rank1" in out.columns:
        out["_ord0"] = pd.to_numeric(out["pred_rank1"], errors="coerce")
        by = ["_ord0"]
        n = 1
        for k in ("pred_rank2", "pred_rank3"):
            if k in out.columns:
                out[f"_ord{n}"] = pd.to_numeric(out[k], errors="coerce")
                by.append(f"_ord{n}")
                n += 1
    else:
        return df
    tmp = out.sort_values(
        by=by, ascending=[False] * len(by), na_position="last"
    ).reset_index(drop=True)
    return tmp.drop(columns=[c for c in tmp.columns if c.startswith("_ord")])


def _slice_predictions_for_scenario_export(df: pd.DataFrame, jv: int) -> pd.DataFrame:
    """``_baba*`` 列を落とし、指定 JV 馬場シナリオの予測列を ``pred_rank*`` 等に複製した DataFrame。"""
    base_cols = [c for c in df.columns if not _BABA_COL_RE.search(c)]
    out = df[base_cols].copy()
    for base in (
        "pred_rank1",
        "pred_rank2",
        "pred_rank3",
        "win_prob_est",
        "expected_return",
    ):
        sc = f"{base}_baba{jv}"
        if sc in df.columns:
            out[base] = df[sc]
    if "win_prob_est" in out.columns:
        out["pred_score"] = out["win_prob_est"]
        out["pred_prob"] = out["win_prob_est"]
    return out


def format_strategy_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "month_day" in out.columns:
        out["開催日"] = out["month_day"].apply(_format_month_day)
    if "course_code" in out.columns:
        out["競馬場"] = out["course_code"].apply(_course_name)
    if "race_num" in out.columns:
        out["R"] = (
            pd.to_numeric(out["race_num"], errors="coerce").astype("Int64").astype(str)
            + "R"
        )
    if "pred_score" in out.columns:
        out["予測スコア"] = pd.to_numeric(out["pred_score"], errors="coerce").round(4)
    if "odds_raw" in out.columns:
        out["オッズ"] = pd.to_numeric(out["odds_raw"], errors="coerce").round(1)
    renames = {
        en_col: ja_col
        for en_col, ja_col in _PREDICT_VIEW_EN_TO_JA.items()
        if en_col in out.columns and en_col != ja_col
    }
    if renames:
        out = out.rename(columns=renames)
    cols = [c for c in _PREDICT_VIEW_COLUMNS_JA if c in out.columns]
    return out[cols]


def format_predictions_export_view(
    df: pd.DataFrame,
    *,
    phase: str = "馬場別",
) -> pd.DataFrame:
    """
    レース単位CSV用。推奨表（format_strategy_view）と同じ列並び・意味に近づける（1頭1行）。
    pred_rank1 を予測スコア、win_prob_est をモデル勝率シェア列、expected_return を期待値列とする。
    """
    if df.empty:
        return df
    n = len(df)
    out = pd.DataFrame(index=df.index)

    out["開催日"] = df["month_day"].apply(_format_month_day) if "month_day" in df.columns else pd.Series(["-"] * n, index=df.index)
    out["競馬場"] = df["course_code"].apply(_course_name) if "course_code" in df.columns else "-"

    if "race_num" in df.columns:
        out["R"] = (
            pd.to_numeric(df["race_num"], errors="coerce").astype("Int64").astype(str) + "R"
        )
    else:
        out["R"] = "-"

    out["券種"] = "単勝"
    if "horse_num" in df.columns:
        hn = pd.to_numeric(df["horse_num"], errors="coerce")
        out["買い目"] = hn.astype("Int64").astype(str)
    else:
        out["買い目"] = ""

    out["予測スコア"] = (
        pd.to_numeric(df["pred_rank1"], errors="coerce").round(4)
        if "pred_rank1" in df.columns
        else np.nan
    )

    if "odds" in df.columns:
        ox = pd.to_numeric(df["odds"], errors="coerce")
        out["オッズ"] = ox.where(ox > 0).round(1)
    else:
        out["オッズ"] = np.nan

    out["モデル勝率シェア"] = (
        pd.to_numeric(df["win_prob_est"], errors="coerce").round(6)
        if "win_prob_est" in df.columns
        else np.nan
    )

    if "expected_return" in df.columns:
        ev = pd.to_numeric(df["expected_return"], errors="coerce")
        out["期待値"] = ev.round(4)
        out["エッジ"] = (ev - 1.0).round(4)
    else:
        out["期待値"] = np.nan
        out["エッジ"] = np.nan

    out["推奨投資額"] = np.nan
    out["フェーズ"] = phase
    out["オッズ取得時刻"] = (
        df["odds_timestamp"] if "odds_timestamp" in df.columns else np.nan
    )

    cols = tuple(c for c in _PREDICT_VIEW_COLUMNS_JA if c in out.columns)
    return out.loc[:, cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

def load_strategy_runtime_config(strategy_config_path: Path) -> dict:
    if not strategy_config_path.exists():
        return {}
    with strategy_config_path.open(encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def resolve_strategy_calibration_path(
    project_root: Path,
    strategy_config_path: Path | None = None,
    *,
    explicit_path: Path | None = None,
) -> Path:
    """
    本番 calibrator パスを解決する。

    優先順: explicit_path → strategy_config.calibration_path → specv2 → legacy
    legacy (calibration_isotonic.json) は上書きしない。不存在時は specv2 を使う。
    """
    if explicit_path is not None and explicit_path.is_file():
        return explicit_path.resolve()

    cfg = (
        load_strategy_runtime_config(strategy_config_path)
        if strategy_config_path is not None
        else {}
    )
    rel = cfg.get("calibration_path")
    if rel:
        configured = (project_root / str(rel).replace("\\", "/")).resolve()
        if configured.is_file():
            return configured

    specv2 = (project_root / "strategy" / "models" / "calibration_isotonic_specv2.json").resolve()
    if specv2.is_file():
        return specv2

    return (project_root / "strategy" / "models" / "calibration_isotonic.json").resolve()


def _resolve_race_date_iso(df: pd.DataFrame) -> str:
    """月次P&Lトラッカー用のレース日付（YYYY-MM-DD）を pred 行から解決する。"""
    import datetime

    if "race_date" in df.columns and df["race_date"].notna().any():
        raw = str(df["race_date"].dropna().iloc[0]).strip()
        return raw[:10]

    if "year" in df.columns and "month_day" in df.columns:
        y = int(pd.to_numeric(df["year"], errors="coerce").dropna().iloc[0])
        md = int(pd.to_numeric(df["month_day"], errors="coerce").dropna().iloc[0])
        md_str = f"{md:04d}"
        return f"{y}-{md_str[:2]}-{md_str[2:]}"

    return datetime.date.today().isoformat()


def strategy_config_from_runtime(runtime_cfg: dict):
    """strategy_config.json から StrategyConfig を構築（Phantom EV フィルタ含む）。"""
    from strategy.src.betting_framework import StrategyConfig

    def _req_int(key: str, default: int) -> int:
        v = runtime_cfg.get(key)
        return int(v if v is not None else default)

    def _req_float(key: str, default: float) -> float:
        v = runtime_cfg.get(key)
        return float(v if v is not None else default)

    def _opt_int(key: str):
        v = runtime_cfg.get(key)
        return int(v) if v is not None else None

    def _opt_float(key: str):
        v = runtime_cfg.get(key)
        return float(v) if v is not None else None

    ev_threshold = _req_float("ev_threshold", 1.05)
    # train_config.json の ev_threshold と整合。min_edge は ev_threshold-1 を下限とする。
    configured_min_edge = runtime_cfg.get("min_edge")
    min_edge = (
        float(configured_min_edge)
        if configured_min_edge is not None
        else ev_threshold - 1.0
    )
    min_edge = max(min_edge, ev_threshold - 1.0)

    max_picks = runtime_cfg.get("max_picks_per_race", runtime_cfg.get("max_selections_per_race", 2))

    bands = runtime_cfg.get("dynamic_edge_bands")
    return StrategyConfig(
        ev_threshold=ev_threshold,
        min_prob=_req_float("min_prob", 0.01),
        min_edge=min_edge,
        min_odds=_req_float("min_odds", 2.0),
        max_odds=_req_float("max_odds", 50.0),
        fractional_kelly=_req_float("kelly_fraction", 0.08),
        max_selections_per_race=int(max_picks) if max_picks is not None else 2,
        initial_bankroll=_req_int("initial_bankroll", 100_000),
        max_stake_per_bet=_req_int("max_stake_per_bet", 3000),
        max_invest_per_race=_req_int("max_invest_per_race", 50_000),
        max_expected_value=_req_float("max_expected_value", 1.5),
        race_num_min=_opt_int("race_num_min"),
        race_num_max=_opt_int("race_num_max"),
        min_win_prob=_opt_float("min_win_prob"),
        max_model_rank=_opt_int("max_model_rank"),
        dynamic_edge_enabled=bool(runtime_cfg.get("dynamic_edge_enabled", False)),
        dynamic_edge_mode=str(runtime_cfg.get("dynamic_edge_mode", "step")),
        dynamic_edge_bands=bands if isinstance(bands, list) else None,
        dynamic_edge_alpha=_req_float("dynamic_edge_alpha", 0.02),
        dynamic_edge_beta=_req_float("dynamic_edge_beta", 0.08),
        min_field_size=_req_int("min_field_size", 9),
        large_field_threshold=_req_int("large_field_threshold", 18),
        large_field_extra_edge=_req_float("large_field_extra_edge", 0.05),
        monthly_drawdown_limit=_req_float("monthly_drawdown_limit", -0.20),
    )


def resolve_recommendation_mode(
    runtime_cfg: dict | None = None,
    *,
    strategy_config_path: Path | None = None,
    use_score_ranked_picks: bool | None = None,
) -> str:
    """
    推奨モードを解決する。許可値: ``ev_kelly`` | ``score_rank``。
    ``use_score_ranked_picks=True`` は後方互換で ``score_rank`` を強制する（非推奨）。
    """
    if use_score_ranked_picks is True:
        return "score_rank"
    if runtime_cfg is None:
        cfg = load_strategy_runtime_config(strategy_config_path) if strategy_config_path else {}
    else:
        cfg = runtime_cfg
    mode = str(cfg.get("recommendation_mode", "ev_kelly")).strip().lower()
    if mode not in {"ev_kelly", "score_rank"}:
        warnings.warn(
            f"Unknown recommendation_mode={mode!r}; falling back to ev_kelly."
        )
        return "ev_kelly"
    return mode


def canonical_race_id_str(series: pd.Series) -> pd.Series:
    """
    ``race_id`` を戦略側・特徴量側で必ず同一の連結キーになるよう正規化する。
    数値化できるものは整数文字列に揃え、それ以外は前後空白除去のみ。
    """
    stripped = series.astype(str).str.strip()
    nums = pd.to_numeric(stripped, errors="coerce")
    out = stripped.copy()
    mask = nums.notna()
    if mask.any():
        out.loc[mask] = nums.loc[mask].round().astype(np.int64).astype(str).to_numpy()
    return out.astype(str)


# ---------------------------------------------------------------------------
# CSV 出力（コース×馬場シナリオ×レース番号別）
# ---------------------------------------------------------------------------

def _resolve_export_month_day(
    df: pd.DataFrame,
    export_month_day: int | None = None,
) -> int | None:
    """分割 CSV 出力対象の month_day（MMDD 整数）を決める。

    export_month_day 明示時はそれを優先。未指定時はカレンダー今日を試し、
    データに行が無ければ df 内の month_day 最頻値にフォールバックする
    （レース翌日に Notebook を再実行しても 614 データを出力できるようにする）。
    """
    if export_month_day is not None:
        return int(export_month_day)
    if "month_day" not in df.columns:
        return None
    md = pd.to_numeric(df["month_day"], errors="coerce")
    if not md.notna().any():
        return None
    import datetime

    today_md = int(datetime.date.today().strftime("%m%d").lstrip("0") or "0")
    if (md == today_md).any():
        return today_md
    fallback = int(md.mode().iloc[0])
    logger.warning(
        "export: calendar today %s has no rows; using month_day=%s from prediction data",
        today_md,
        fallback,
    )
    return fallback


def export_predictions_by_course_baba_race(
    df: pd.DataFrame,
    baba_scenario_jv_codes: tuple[int, ...],
    baba_scenario_label_ja: dict[int, str],
    *,
    result_root: Path,
    max_race_num: int = 12,
    export_month_day: int | None = None,
) -> tuple[Path, int]:
    """
    ``Main/results/<競馬場名>/馬場_<良|稍重|重|不良>/<n>R.csv`` に予測行を書き分ける。
    Returns (root_path, written_file_count)。
    """
    if "course_code" not in df.columns or "race_num" not in df.columns:
        return result_root, 0

    target_md = _resolve_export_month_day(df, export_month_day)
    if target_md is not None and "month_day" in df.columns:
        md = pd.to_numeric(df["month_day"], errors="coerce")
        df = df[md == target_md].copy()
        if df.empty:
            return result_root, 0

    cc_num = pd.to_numeric(df["course_code"], errors="coerce")
    rn_num = pd.to_numeric(df["race_num"], errors="coerce")

    written = 0
    for jv in baba_scenario_jv_codes:
        scenario_label = _sanitize_path_segment(
            "馬場_" + baba_scenario_label_ja.get(jv, f"コード{jv}")
        )
        wide = _slice_predictions_for_scenario_export(df, jv)
        for cc in sorted(cc_num.dropna().astype(int).unique()):
            venue = _course_folder_name(cc)
            for rn in range(1, int(max_race_num) + 1):
                m = cc_num.astype(float).eq(float(cc)) & rn_num.astype(float).eq(float(rn))
                part = _sort_by_forecast_score(wide.loc[m])
                if part.empty:
                    continue
                out_dir = result_root / venue / scenario_label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{rn}R.csv"
                view = format_predictions_export_view(
                    part,
                    phase=f"馬場_{baba_scenario_label_ja.get(jv, str(jv))}",
                )
                view.to_csv(out_path, index=False, encoding="utf-8-sig")
                written += 1

    return result_root, written


# ---------------------------------------------------------------------------
# 推奨保存・空推奨メッセージ
# ---------------------------------------------------------------------------

def persist_recommendations(rec_df: pd.DataFrame, output_path: Path) -> None:
    """推奨 DataFrame を CSV + Parquet に保存する（空 DataFrame も可）。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_out = format_strategy_view(rec_df) if not rec_df.empty else rec_df
    csv_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    rec_df.to_parquet(output_path.with_suffix(".parquet"), index=False)


def build_empty_recommendation_notice(pred_df: pd.DataFrame) -> str:
    """推奨が空のときに Notebook 向けの説明文を生成する。"""
    parts = ["推奨馬券なし"]
    if "month_day" in pred_df.columns and pred_df["month_day"].notna().any():
        md = pred_df["month_day"].dropna().iloc[0]
        day_label = _format_month_day(md)
        if day_label != "-":
            parts.append(day_label)
    if "odds" in pred_df.columns:
        odds = pd.to_numeric(pred_df["odds"], errors="coerce")
        if odds.isna().all():
            parts.append("単勝オッズ未取得")
    return " / ".join(parts)


# ---------------------------------------------------------------------------
# 戦略パイプライン
# ---------------------------------------------------------------------------

def run_today_strategy_pipeline(
    pred_df: pd.DataFrame,
    *,
    strategy_config_path: Path,
    strategy_calibration_path: Path,
    recommendation_output_path: Path,
    o2_odds_path: Path,
    o3_odds_path: Path,
) -> pd.DataFrame:
    from strategy.src.betting_framework import (
        ProbabilityCalibrator,
        run_today_recommendation,
    )
    from main.pipeline.data_pipeline import load_pair_odds_dicts

    from main.pipeline.monthly_dd_tracker import (
        check_monthly_dd_limit,
        record_daily_pnl,
    )

    runtime_cfg = load_strategy_runtime_config(strategy_config_path)
    strategy_cfg = strategy_config_from_runtime(runtime_cfg)

    is_dd_breached, dd_rate = check_monthly_dd_limit(
        float(strategy_cfg.initial_bankroll),
        float(strategy_cfg.monthly_drawdown_limit),
    )
    if is_dd_breached:
        logger.warning(
            "[strategy] 月次DD制限超過 (損益率=%.1f%% < 閾値=%.1f%%)。推奨をスキップします。",
            dd_rate * 100,
            strategy_cfg.monthly_drawdown_limit * 100,
        )
        empty = pd.DataFrame()
        persist_recommendations(empty, recommendation_output_path)
        return empty

    calibrator = None
    if strategy_calibration_path.exists():
        calibrator = ProbabilityCalibrator.from_json(strategy_calibration_path)

    pred_in = pred_df.copy()
    pred_in["race_id"] = canonical_race_id_str(pred_in["race_id"])

    # 戦略パイプライン入口でも再度スクラッチ馬を除外（予測パイプライン後に状態が変わった場合の保険）
    # 除外・再正規化は race_runtime.filter_scratched に集約済み（NaN/0 両方検出）。
    if "odds" in pred_in.columns:
        odds_num = pd.to_numeric(pred_in["odds"], errors="coerce")
        n_s = int(((odds_num == 0) | odds_num.isna()).sum())
        if n_s > 0:
            logger.warning("strategy入口: 出走取消馬 %d 頭を除外", n_s)
            pred_in = filter_scratched(pred_in)

    from strategy.src.race_filters import (
        filter_df_by_race_num,
        filter_df_exclude_courses,
        filter_df_exclude_dirt,
        filter_df_exclude_grades,
        filter_df_exclude_surface,
        filter_df_exclude_age,
    )

    rmin = runtime_cfg.get("race_num_min")
    rmax = runtime_cfg.get("race_num_max")
    if rmin is not None or rmax is not None:
        pred_in = filter_df_by_race_num(
            pred_in,
            race_id_col="race_id",
            race_num_min=int(rmin) if rmin is not None else None,
            race_num_max=int(rmax) if rmax is not None else None,
        )
        logger.info(
            "[strategy] race_num filter %s-%s -> %d races",
            rmin, rmax, pred_in["race_id"].nunique(),
        )

    exclude_courses = runtime_cfg.get("exclude_course_codes") or []
    if exclude_courses:
        n_before = pred_in["race_id"].nunique()
        pred_in = filter_df_exclude_courses(
            pred_in,
            exclude_course_codes=[int(c) for c in exclude_courses],
            race_id_col="race_id",
        )
        n_removed = n_before - pred_in["race_id"].nunique()
        logger.info(
            "[strategy] course exclusion %s: %d races removed -> %d races remaining",
            exclude_courses, n_removed, pred_in["race_id"].nunique(),
        )

    exclude_grades = runtime_cfg.get("exclude_grade_codes") or []
    if exclude_grades:
        n_before = pred_in["race_id"].nunique()
        pred_in = filter_df_exclude_grades(
            pred_in,
            exclude_grade_codes=[int(g) for g in exclude_grades],
            race_id_col="race_id",
        )
        n_removed = n_before - pred_in["race_id"].nunique()
        logger.info(
            "[strategy] grade exclusion %s: %d races removed -> %d races remaining",
            exclude_grades, n_removed, pred_in["race_id"].nunique(),
        )

    # 障害レース除外（本番 rank パス）: grade_code が features で全行7に潰れ
    # exclude_grade_codes:[8,9] が無効化されているため、surface_code=3（障害）を直接除外する。
    # 障害は平地 binary/rank モデルの学習対象外で予測不能（CLAUDE.md「障害は平地モデルで予測不能」）。
    # backtest 側（strategy/src/backtest.py の exclude_surface_codes 既定 [3]）と同一の意味・既定で整合させ、
    # 片側だけの適用による性能乖離を防ぐ（CLAUDE.md §5-1 の精神）。
    excl_surface = runtime_cfg.get("exclude_surface_codes")
    if excl_surface is None:
        excl_surface = [3]
    if excl_surface and "surface_code" in pred_in.columns:
        n_before = pred_in["race_id"].nunique()
        pred_in = filter_df_exclude_surface(
            pred_in,
            exclude_surface_codes=[int(s) for s in excl_surface],
            race_id_col="race_id",
        )
        n_removed = n_before - pred_in["race_id"].nunique()
        if n_removed > 0:
            logger.info(
                "[strategy] surface exclusion %s (障害): %d races removed -> %d races remaining",
                excl_surface, n_removed, pred_in["race_id"].nunique(),
            )

    exclude_dirt = bool(runtime_cfg.get("exclude_dirt_track", False))
    _dtcm = runtime_cfg.get("dirt_track_code_min", 23)
    dirt_min = int(_dtcm if _dtcm is not None else 23)
    if exclude_dirt:
        n_before = pred_in["race_id"].nunique()
        pred_in = filter_df_exclude_dirt(
            pred_in,
            exclude_dirt=True,
            dirt_track_code_min=dirt_min,
            race_id_col="race_id",
        )
        n_removed = n_before - pred_in["race_id"].nunique()
        logger.info(
            "[strategy] dirt exclusion (track_code>=%d): %d races removed -> %d races remaining",
            dirt_min, n_removed, pred_in["race_id"].nunique(),
        )

    exclude_age_max = runtime_cfg.get("exclude_age_max")
    if exclude_age_max is not None and "age" in pred_in.columns:
        n_before = len(pred_in)
        pred_in = filter_df_exclude_age(pred_in, exclude_age_max=int(exclude_age_max))
        n_removed = n_before - len(pred_in)
        if n_removed > 0:
            logger.info(
                "[strategy] age exclusion (age > %d): %d horses removed",
                exclude_age_max, n_removed,
            )

    quinella_odds_dict, wide_odds_dict = load_pair_odds_dicts(o2_odds_path, o3_odds_path)

    def _cfg_int(key: str, default: int) -> int:
        v = runtime_cfg.get(key)
        return int(v if v is not None else default)

    def _cfg_float(key: str, default: float) -> float:
        v = runtime_cfg.get(key)
        return float(v if v is not None else default)

    rec_df = run_today_recommendation(
        pred_in,
        config=strategy_cfg,
        calibrator=calibrator,
        phase=str(runtime_cfg.get("online_phase", "phase1_5")),
        pair_top_n=_cfg_int("pair_top_n", 2),
        wide_top_n=_cfg_int("wide_top_n", 2),
        phase2_enabled=bool(runtime_cfg.get("phase2_enabled", False)),
        save_snapshot_timestamps=bool(runtime_cfg.get("save_snapshot_timestamps", True)),
        probability_policy=str(runtime_cfg.get("probability_policy", "market_shrinkage")),
        market_shrinkage_alpha=_cfg_float("market_shrinkage_alpha", 0.2),
        max_expected_value=_cfg_float("max_expected_value", 1.5),
        max_odds_for_kelly=_cfg_float("max_odds_for_kelly", 30.0),
        min_bucket_count=_cfg_int("min_bucket_count", 100),
        odds_source=str(runtime_cfg.get("odds_source", "unknown")),
        odds_cutoff_policy=str(runtime_cfg.get("odds_cutoff_policy", "unknown")),
        rank2_blend=_cfg_float("rank2_blend", 0.35),
        wide_min_edge=_cfg_float("wide_min_edge", 0.05),
        wide_bets_enabled=bool(runtime_cfg.get("wide_bets_enabled", True)),
        quinella_bets_enabled=bool(runtime_cfg.get("quinella_bets_enabled", True)),
        place_bets_enabled=bool(runtime_cfg.get("place_bets_enabled", True)),
        wide_selection=str(runtime_cfg.get("wide_selection", "harville")),
        wide_ev_threshold=_cfg_float("wide_ev_threshold", 1.05),
        wide_div_threshold=_cfg_float("wide_div_threshold", 0.0),
        portfolio_kelly_enabled=bool(runtime_cfg.get("portfolio_kelly_enabled", False)),
        portfolio_kelly_mode=str(runtime_cfg.get("portfolio_kelly_mode", "portfolio_kelly_fractional")),
        portfolio_growth_ratio_min=_cfg_float("portfolio_growth_ratio_min", 0.5),
        portfolio_ind_cap_ratio=_cfg_float("portfolio_ind_cap_ratio", 0.85),
        portfolio_mc_samples=_cfg_int("portfolio_mc_samples", 500),
        portfolio_mc_seed=_cfg_int("portfolio_mc_seed", 42),
        quinella_odds_dict=quinella_odds_dict or None,
        wide_odds_dict=wide_odds_dict or None,
    )

    if rec_df.empty:
        return rec_df

    if "race_id" not in pred_in.columns:
        raise ValueError("pred_in に race_id 列がありません")
    merge_cols = [
        c for c in ("race_id", "month_day", "course_code", "race_num") if c in pred_in.columns
    ]
    pred_meta = pred_in[merge_cols].drop_duplicates(subset=["race_id"]).copy()
    pred_meta["race_id"] = canonical_race_id_str(pred_meta["race_id"])
    rec_m = rec_df.copy()
    rec_m["race_id"] = canonical_race_id_str(rec_m["race_id"])
    rec_df = rec_m.merge(pred_meta, on="race_id", how="left")

    ev_overrides = runtime_cfg.get("condition_ev_overrides") or []
    if ev_overrides and not rec_df.empty:
        from strategy.src.inference_common import apply_condition_overrides_to_recommendations

        _has_horse_key = "horse_num" in pred_in.columns and "horse_num" in rec_df.columns
        override_merge_cols = ["race_id"]
        if _has_horse_key:
            override_merge_cols.append("horse_num")
        for rule in ev_overrides:
            for key in (
                "surface_code",
                "track_condition_code",
                "course_code",
                "horse_age",
                "grade_code",
                "condition_col",
            ):
                if key in rule and key in pred_in.columns and key not in rec_df.columns:
                    override_merge_cols.append(key)
            col = str(rule.get("condition_col", ""))
            if col in pred_in.columns and col not in rec_df.columns:
                override_merge_cols.append(col)
        override_merge_cols = list(dict.fromkeys(override_merge_cols))

        join_key = [c for c in ["race_id", "horse_num"] if c in override_merge_cols]
        if len(override_merge_cols) > len(join_key):
            grade_meta = (
                pred_in[override_merge_cols]
                .drop_duplicates(subset=join_key)
                .copy()
            )
            grade_meta["race_id"] = canonical_race_id_str(grade_meta["race_id"])
            rec_df = rec_df.merge(grade_meta, on=join_key, how="left", suffixes=("", "_grade"))

        ev_threshold = float(runtime_cfg.get("ev_threshold", 1.05))
        rec_df = apply_condition_overrides_to_recommendations(
            rec_df, ev_overrides, ev_threshold
        )
        n_filtered = int((~rec_df["_conditional_ev_ok"]).sum())
        if n_filtered > 0:
            logger.info(
                "[strategy] conditional EV override: %d recommendations removed",
                n_filtered,
            )
        rec_df = rec_df.loc[rec_df["_conditional_ev_ok"]].drop(
            columns=["_conditional_ev_ok"], errors="ignore"
        ).reset_index(drop=True)

    if not rec_df.empty:
        race_date = _resolve_race_date_iso(pred_in)
        stake_col = "suggested_stake" if "suggested_stake" in rec_df.columns else None
        invested = float(rec_df[stake_col].sum()) if stake_col else 0.0
        record_daily_pnl(race_date, invested=invested, returned=0.0, n_recommendations=len(rec_df))

    persist_recommendations(rec_df, recommendation_output_path)
    return rec_df


def run_today_score_rank_pipeline(
    pred_df: pd.DataFrame,
    *,
    recommendation_output_path: Path,
    score_col: str = "pred_rank1",
    pair_top_n: int = 2,
    wide_top_n: int = 2,
    bet_unit: int = 100,
) -> pd.DataFrame:
    """
    戦略（EV/Kelly）を使わず、score_col（既定 pred_rank1）の高い順で買目を決める。

    未キャリブレーションスコアのため Kelly は使わずフラット買い（bet_unit 固定）のみ。
    """
    warnings.warn(
        "recommendation_mode=score_rank: uncalibrated scores — "
        "Kelly disabled; flat betting forced.",
        UserWarning,
        stacklevel=2,
    )
    need = ("race_id", "horse_num", score_col)
    for c in need:
        if c not in pred_df.columns:
            raise ValueError(f"score_rank 買目には列 {c!r} が必要です")

    df = pred_df.copy()
    df["race_id"] = canonical_race_id_str(df["race_id"])
    df["_sc"] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["_sc", "race_id"]).copy()

    gen = None
    if "generated_at" in df.columns and df["generated_at"].notna().any():
        gen = str(df["generated_at"].dropna().iloc[0])
    now_iso = gen or datetime.now(timezone.utc).isoformat()

    meta_keys = [c for c in ("month_day", "course_code", "race_num") if c in df.columns]
    max_slots = max(int(pair_top_n), int(wide_top_n))
    rows: list[dict] = []

    for rid, g in df.groupby("race_id", sort=False):
        g_sorted = g.sort_values("_sc", ascending=False).reset_index(drop=True)
        if g_sorted.empty:
            continue

        md = {k: g_sorted.iloc[0][k] for k in meta_keys}

        top = g_sorted.iloc[0]
        h1_raw = pd.to_numeric(top["horse_num"], errors="coerce")
        if pd.isna(h1_raw):
            continue
        h1 = int(h1_raw)
        sc1 = float(top["_sc"])
        odds_v = pd.to_numeric(top["odds"], errors="coerce") if "odds" in top.index else np.nan
        wpe = (
            pd.to_numeric(top["win_prob_est"], errors="coerce")
            if "win_prob_est" in top.index
            else np.nan
        )
        ev = (
            float(wpe) * float(odds_v)
            if pd.notna(wpe) and pd.notna(odds_v)
            else np.nan
        )

        rows.append(
            {
                "ticket_type": "単勝",
                "race_id": rid,
                "horse_num": h1,
                "partner_horse_num": np.nan,
                "ticket": str(h1),
                "pred_prob": float(wpe) if pd.notna(wpe) else np.nan,
                "pred_score": sc1,
                "odds_raw": float(odds_v) if pd.notna(odds_v) else np.nan,
                "odds_effective": float(odds_v) if pd.notna(odds_v) else np.nan,
                "expected_value": ev if pd.notna(ev) else np.nan,
                "edge": (ev - 1.0) if pd.notna(ev) else np.nan,
                "suggested_stake": int(bet_unit),
                "is_executable": True,
                "phase": "score_rank",
                "odds_timestamp": top.get("odds_timestamp"),
                "generated_at": now_iso,
                "modeling_note": f"score-ranked ({score_col}, 各レース1着候補)",
                **md,
            }
        )

        for j in range(1, min(len(g_sorted), max_slots + 1)):
            other = g_sorted.iloc[j]
            h2_raw = pd.to_numeric(other["horse_num"], errors="coerce")
            if pd.isna(h2_raw):
                continue
            h2 = int(h2_raw)
            if h1 == h2:
                continue
            sc2 = float(other["_sc"])
            a, b = sorted((h1, h2))
            ticket = f"{a}-{b}"
            prod_sc = sc1 * sc2
            base_pw = {
                "race_id": rid,
                "horse_num": h1,
                "partner_horse_num": h2,
                "ticket": ticket,
                "pred_prob": np.nan,
                "pred_score": prod_sc,
                "odds_raw": np.nan,
                "odds_effective": np.nan,
                "expected_value": np.nan,
                "edge": np.nan,
                "suggested_stake": int(bet_unit),
                "is_executable": True,
                "phase": "score_rank",
                "odds_timestamp": top.get("odds_timestamp"),
                "generated_at": now_iso,
                "modeling_note": f"score-ranked ({score_col} 1位×{j + 1}位の積)",
                **md,
            }
            if j <= int(pair_top_n):
                rows.append({**base_pw, "ticket_type": "馬連"})
            if j <= int(wide_top_n):
                rows.append({**base_pw, "ticket_type": "ワイド"})

    rec_df = pd.DataFrame(rows)
    if not rec_df.empty:
        recommendation_output_path.parent.mkdir(parents=True, exist_ok=True)
        format_strategy_view(rec_df).to_csv(
            recommendation_output_path, index=False, encoding="utf-8-sig"
        )
        rec_df.to_parquet(recommendation_output_path.with_suffix(".parquet"), index=False)
        logger.info(
            "予測スコア順の買目を保存しました: %s (%d 行)",
            recommendation_output_path, len(rec_df),
        )
    return rec_df


def apply_operating_pair_rule_filter(
    pred_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    *,
    recommendation_output_path: Path,
) -> pd.DataFrame:
    """
    ワイド/馬連の推奨について、軸馬・相手馬の pred_rank1 下限で後段フィルタする。
    """
    required_pred_cols = {"race_id", "horse_num", "pred_rank1"}
    required_rec_cols = {"race_id", "horse_num", "partner_horse_num", "ticket_type"}
    if not required_pred_cols.issubset(pred_df.columns):
        return rec_df
    if not required_rec_cols.issubset(rec_df.columns):
        return rec_df

    # pred_rank1 は 0.05〜0.19 の範囲（16頭均等なら 0.0625）
    rule = {
        "ワイド": {"anchor_th": 0.09, "partner_th": 0.06},
        "馬連": {"anchor_th": 0.09, "partner_th": 0.06},
    }

    score_df = pred_df[["race_id", "horse_num", "pred_rank1"]].copy()
    score_df["race_id"] = score_df["race_id"].astype(str)
    score_df["horse_num"] = pd.to_numeric(score_df["horse_num"], errors="coerce")
    score_df["pred_rank1"] = pd.to_numeric(score_df["pred_rank1"], errors="coerce")
    score_df = score_df.dropna(subset=["horse_num", "pred_rank1"]).copy()

    rec2 = rec_df.copy()
    rec2["race_id"] = rec2["race_id"].astype(str)
    rec2["horse_num"] = pd.to_numeric(rec2["horse_num"], errors="coerce")
    rec2["partner_horse_num"] = pd.to_numeric(rec2["partner_horse_num"], errors="coerce")

    anchor_sc = score_df.rename(columns={"pred_rank1": "anchor_score"})
    rec2 = rec2.merge(
        anchor_sc[["race_id", "horse_num", "anchor_score"]],
        on=["race_id", "horse_num"],
        how="left",
    )

    partner_sc = score_df.rename(
        columns={"horse_num": "partner_horse_num", "pred_rank1": "partner_score"}
    )
    rec2 = rec2.merge(
        partner_sc[["race_id", "partner_horse_num", "partner_score"]],
        on=["race_id", "partner_horse_num"],
        how="left",
    )

    keep = pd.Series(True, index=rec2.index)
    for ticket_type in ("ワイド", "馬連"):
        m = rec2["ticket_type"].eq(ticket_type)
        keep &= ~m | (
            (pd.to_numeric(rec2["anchor_score"], errors="coerce") >= rule[ticket_type]["anchor_th"])
            & (
                pd.to_numeric(rec2["partner_score"], errors="coerce")
                >= rule[ticket_type]["partner_th"]
            )
        )

    rec_filtered = rec2[keep].copy()
    drop_cols = [c for c in ("anchor_score", "partner_score") if c in rec_filtered.columns]
    if drop_cols:
        rec_filtered = rec_filtered.drop(columns=drop_cols)

    recommendation_output_path.parent.mkdir(parents=True, exist_ok=True)
    format_strategy_view(rec_filtered).to_csv(
        recommendation_output_path, index=False, encoding="utf-8-sig"
    )
    rec_filtered.to_parquet(recommendation_output_path.with_suffix(".parquet"), index=False)

    logger.info("[applied operating rule]")
    logger.info("  ワイド/馬連: anchor_score >= 0.09 & partner_score >= 0.06 (pred_rank1 基準)")
    logger.info("  before rows: %d", len(rec_df))
    logger.info("  after rows : %d", len(rec_filtered))
    logger.info("推奨保存先: %s", recommendation_output_path)
    return rec_filtered


# ---------------------------------------------------------------------------
# Notebook 表示ヘルパー
# ---------------------------------------------------------------------------

def _rec_race_group_keys(df: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    if "course_code" in df.columns:
        keys.append("course_code")
    if "race_num" in df.columns:
        keys.append("race_num")
    if keys:
        return keys
    if "race_id" in df.columns:
        return ["race_id"]
    return []


def _sort_rec_df_by_race_then_score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_keys: list[str] = []
    ascending: list[bool] = []
    if "course_code" in df.columns:
        sort_keys.append("course_code")
        ascending.append(True)
    if "race_num" in df.columns:
        sort_keys.append("race_num")
        ascending.append(True)
    elif "race_id" in df.columns:
        sort_keys.append("race_id")
        ascending.append(True)
    if "pred_score" in df.columns:
        sort_keys.append("pred_score")
        ascending.append(False)
    if not sort_keys:
        return df
    return df.sort_values(sort_keys, ascending=ascending, kind="mergesort")


def _race_group_label(grp: pd.DataFrame) -> str:
    row = grp.iloc[0]
    course = (
        _course_name(row["course_code"]) if "course_code" in grp.columns else "-"
    )
    rnum = pd.to_numeric(row.get("race_num"), errors="coerce")
    r_label = f"{int(rnum)}R" if pd.notna(rnum) else "-"
    if "month_day" in grp.columns:
        day = _format_month_day(row["month_day"])
        return f"{day} {course} {r_label}"
    return f"{course} {r_label}"


def display_ticket_candidates_by_race(
    df: pd.DataFrame,
    *,
    idisplay,
    drop_race_cols: tuple[str, ...] = ("開催日", "競馬場", "R"),
) -> None:
    """券種候補をレース単位の見出し付きで表示する。"""
    if df.empty:
        idisplay(format_strategy_view(df))
        return
    group_keys = _rec_race_group_keys(df)
    if not group_keys:
        idisplay(format_strategy_view(df))
        return
    sorted_df = _sort_rec_df_by_race_then_score(df)
    for _, grp in sorted_df.groupby(group_keys, sort=False):
        score_col_name = next(
            (c for c in ("pred_score", "heuristic_score") if c in grp.columns), None
        )
        if score_col_name:
            try:
                grp = grp.sort_values(score_col_name, ascending=False)
            except (KeyError, TypeError):
                pass
        label = _race_group_label(grp)
        n = len(grp)
        max_sc = pd.to_numeric(grp[score_col_name], errors="coerce").max() if score_col_name else float("nan")
        max_txt = f"{max_sc:.4f}" if pd.notna(max_sc) else "-"
        logger.info("\n▼ %s（%d件・最高 %s）", label, n, max_txt)
        view = format_strategy_view(grp)
        drop = [c for c in drop_race_cols if c in view.columns]
        if drop:
            view = view.drop(columns=drop)
        idisplay(view.reset_index(drop=True))


def display_top_win_prob_predictions(
    pred_df: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """予測確率（win_prob_est）が高い順に top_n 件を表示する。"""
    try:
        from IPython.display import display as idisplay
    except ImportError:
        def idisplay(obj, **_kwargs):
            print(obj)

    prob_col = "win_prob_est" if "win_prob_est" in pred_df.columns else "pred_rank1"
    if pred_df.empty or prob_col not in pred_df.columns:
        logger.info("\n=== 予測確率 Top%d ===", top_n)
        idisplay(format_predictions_export_view(pred_df, phase=f"Top{top_n}"))
        return pred_df

    top = (
        pred_df.sort_values(prob_col, ascending=False, na_position="last")
        .head(top_n)
        .copy()
    )
    view = format_predictions_export_view(top, phase=f"Top{top_n}")
    view.insert(0, "順位", range(1, len(view) + 1))
    logger.info("\n=== 予測確率 Top%d（全レース・%s 降順） ===", top_n, prob_col)
    idisplay(view.reset_index(drop=True))
    return top


def display_today_recommendation_summaries(
    rec_df: pd.DataFrame,
    *,
    prediction_output_path: Path,
    recommendation_output_path: Path,
) -> None:
    try:
        from IPython.display import display as idisplay
    except ImportError:
        def idisplay(obj, **_kwargs):
            print(obj)

    if rec_df.empty or "ticket_type" not in rec_df.columns:
        return
    ws = rec_df[rec_df["ticket_type"] == "単勝"]
    if not ws.empty and "pred_score" in ws.columns:
        ws = ws.sort_values("pred_score", ascending=False)
    wins = format_strategy_view(ws.head(12))

    pair_display_prob_th = 0.04

    pr = rec_df[rec_df["ticket_type"] == "馬連"]
    if not pr.empty and "pred_prob" in pr.columns:
        pr = pr[pd.to_numeric(pr["pred_prob"], errors="coerce") >= pair_display_prob_th]
    if not pr.empty and "expected_value" in pr.columns:
        pr = pr.sort_values("expected_value", ascending=False)
    pairs = format_strategy_view(pr.head(12))

    wd = rec_df[rec_df["ticket_type"] == "ワイド"]
    if not wd.empty and "pred_prob" in wd.columns:
        wd = wd[pd.to_numeric(wd["pred_prob"], errors="coerce") >= pair_display_prob_th]
    wide_race_keys = _rec_race_group_keys(wd)
    n_wide_races = (
        wd.groupby(wide_race_keys, sort=False).ngroups if wide_race_keys else 0
    )

    logger.info("\n=== 単勝オススメ買目 (最大12件) ===")
    idisplay(wins)
    logger.info("\n=== 馬連候補 (Harville確率>= %s, 最大12件) ===", pair_display_prob_th)
    idisplay(pairs)
    logger.info(
        "\n=== ワイド候補 (Harville確率>= %s, 全%d件 / %dレース) ===",
        pair_display_prob_th, len(wd), n_wide_races,
    )
    display_ticket_candidates_by_race(wd, idisplay=idisplay, drop_race_cols=())
    logger.info("予測保存先: %s", prediction_output_path)
    logger.info("推奨保存先: %s", recommendation_output_path)

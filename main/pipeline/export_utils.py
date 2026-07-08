"""Venue/baba CSV export utilities (no strategy/ model_training dependency)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

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
    course_map = dict(_COURSE_FOLDER_JA)
    if pd.isna(value):
        return "-"
    try:
        code = int(float(value))
    except Exception:
        return str(value)
    return course_map.get(code, str(code))


def _sort_by_forecast_score(df: pd.DataFrame) -> pd.DataFrame:
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
    else:
        return df
    tmp = out.sort_values(by=by, ascending=[False] * len(by), na_position="last").reset_index(drop=True)
    return tmp.drop(columns=[c for c in tmp.columns if c.startswith("_ord")])


def _slice_predictions_for_scenario_export(df: pd.DataFrame, jv: int) -> pd.DataFrame:
    base_cols = [c for c in df.columns if not _BABA_COL_RE.search(c)]
    out = df[base_cols].copy()
    for base in ("pred_rank1", "pred_rank2", "pred_rank3", "win_prob_est", "expected_return"):
        sc = f"{base}_baba{jv}"
        if sc in df.columns:
            out[base] = df[sc]
    if "win_prob_est" in out.columns:
        out["pred_score"] = out["win_prob_est"]
        out["pred_prob"] = out["win_prob_est"]
    return out


def format_predictions_export_view(df: pd.DataFrame, *, phase: str = "馬場別") -> pd.DataFrame:
    if df.empty:
        return df
    n = len(df)
    out = pd.DataFrame(index=df.index)
    out["開催日"] = df["month_day"].apply(_format_month_day) if "month_day" in df.columns else pd.Series(["-"] * n, index=df.index)
    out["競馬場"] = df["course_code"].apply(_course_name) if "course_code" in df.columns else pd.Series(["-"] * n, index=df.index)
    if "race_num" in df.columns:
        out["R"] = pd.to_numeric(df["race_num"], errors="coerce").astype("Int64").astype(str) + "R"
    out["券種"] = "-"
    out["買い目"] = df["horse_id"].astype(str) if "horse_id" in df.columns else "-"
    if "pred_rank1" in df.columns:
        out["予測スコア"] = pd.to_numeric(df["pred_rank1"], errors="coerce").round(4)
    elif "pred_score" in df.columns:
        out["予測スコア"] = pd.to_numeric(df["pred_score"], errors="coerce").round(4)
    if "odds" in df.columns:
        out["オッズ"] = pd.to_numeric(df["odds"], errors="coerce").round(1)
    if "win_prob_est" in df.columns:
        out["モデル勝率シェア"] = pd.to_numeric(df["win_prob_est"], errors="coerce").round(4)
    if "expected_return" in df.columns:
        out["期待値"] = pd.to_numeric(df["expected_return"], errors="coerce").round(4)
    out["エッジ"] = pd.NA
    out["推奨投資額"] = pd.NA
    out["フェーズ"] = phase
    out["オッズ取得時刻"] = df["odds_timestamp"] if "odds_timestamp" in df.columns else pd.NA
    renames = {en: ja for en, ja in _PREDICT_VIEW_EN_TO_JA.items() if en in out.columns and en != ja}
    if renames:
        out = out.rename(columns=renames)
    cols = [c for c in _PREDICT_VIEW_COLUMNS_JA if c in out.columns]
    return out[cols]


def _resolve_export_month_day(df: pd.DataFrame, export_month_day: int | None) -> int | None:
    if export_month_day is not None:
        return int(export_month_day)
    if "month_day" not in df.columns:
        return None
    md = pd.to_numeric(df["month_day"], errors="coerce").dropna()
    if md.empty:
        return None
    return int(md.mode().iloc[0])


def export_predictions_by_course_baba_race(
    df: pd.DataFrame,
    baba_scenario_jv_codes: tuple[int, ...],
    baba_scenario_label_ja: dict[int, str],
    *,
    result_root: Path,
    max_race_num: int = 12,
    export_month_day: int | None = None,
) -> tuple[Path, int]:
    """Write main/results/<venue>/馬場_*/<n>R.csv."""
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
        scenario_label = _sanitize_path_segment("馬場_" + baba_scenario_label_ja.get(jv, f"コード{jv}"))
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
                view = format_predictions_export_view(part, phase=f"馬場_{baba_scenario_label_ja.get(jv, str(jv))}")
                view.to_csv(out_dir / f"{rn}R.csv", index=False, encoding="utf-8-sig")
                written += 1
    return result_root, written

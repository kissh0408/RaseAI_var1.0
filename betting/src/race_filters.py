"""レース番号・レース帯フィルタの共通ユーティリティ。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def parse_race_num_from_race_id(race_id: pd.Series) -> pd.Series:
    """race_id 末尾2桁を R 番号（1-12 等）として解釈する。"""
    rid = race_id.astype(str).str.strip()
    tail = rid.str[-2:]
    nums = pd.to_numeric(tail, errors="coerce")
    return nums


def attach_race_num(
    df: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    race_num_col: str = "race_num",
    overwrite: bool = False,
) -> pd.DataFrame:
    """DataFrame に race_num 列を付与（既存列があれば欠損のみ補完）。"""
    out = df.copy()
    if race_num_col in out.columns and not overwrite:
        existing = pd.to_numeric(out[race_num_col], errors="coerce")
        if existing.notna().any():
            out[race_num_col] = existing
            missing = out[race_num_col].isna()
            if missing.any() and race_id_col in out.columns:
                out.loc[missing, race_num_col] = parse_race_num_from_race_id(
                    out.loc[missing, race_id_col]
                )
            return out

    if race_id_col not in out.columns:
        raise ValueError(f"attach_race_num requires {race_id_col!r}")
    out[race_num_col] = parse_race_num_from_race_id(out[race_id_col])
    return out


def race_num_in_range(
    race_num: float | int | None,
    *,
    race_num_min: Optional[int] = None,
    race_num_max: Optional[int] = None,
) -> bool:
    if race_num is None or (isinstance(race_num, float) and np.isnan(race_num)):
        return race_num_min is None and race_num_max is None
    rn = int(race_num)
    if race_num_min is not None and rn < int(race_num_min):
        return False
    if race_num_max is not None and rn > int(race_num_max):
        return False
    return True


def filter_df_by_race_num(
    df: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    race_num_min: Optional[int] = None,
    race_num_max: Optional[int] = None,
) -> pd.DataFrame:
    """race_num 帯で行をフィルタ（レース単位: 1頭でも範囲外ならレース全体除外）。"""
    if race_num_min is None and race_num_max is None:
        return df.copy()
    work = attach_race_num(df, race_id_col=race_id_col)
    race_nums = (
        work.groupby(race_id_col, sort=False)["race_num"]
        .first()
        .apply(lambda x: race_num_in_range(x, race_num_min=race_num_min, race_num_max=race_num_max))
    )
    keep_ids = set(race_nums[race_nums].index.astype(str))
    out = work[work[race_id_col].astype(str).isin(keep_ids)].copy()
    return out


# ---------------------------------------------------------------------------
# バックテスト弱点に基づく除外フィルタ
# ---------------------------------------------------------------------------

def filter_df_exclude_courses(
    df: pd.DataFrame,
    exclude_course_codes: Optional[List[int]],
    *,
    course_code_col: str = "course_code",
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """
    ROI 基準割れの競馬場コードをレース単位で除外する。

    course_code_col が DataFrame に存在しない場合は何もせず返す。
    レース内の1行でも除外コードに該当すればレース全体を除外する。

    Parameters
    ----------
    exclude_course_codes : list[int] | None
        除外する course_code のリスト（例: [5] = 東京）。None または空リストの場合は除外しない。
    """
    if not exclude_course_codes or course_code_col not in df.columns:
        return df.copy()

    codes = set(int(c) for c in exclude_course_codes)
    cc = pd.to_numeric(df[course_code_col], errors="coerce")
    is_excluded = cc.isin(codes)

    # レース単位の除外: race_id に1頭でも除外コードがあればレース全体を落とす
    if race_id_col in df.columns:
        excluded_race_ids = set(df.loc[is_excluded, race_id_col].astype(str).unique())
        keep_mask = ~df[race_id_col].astype(str).isin(excluded_race_ids)
    else:
        keep_mask = ~is_excluded

    out = df.loc[keep_mask].copy()
    return out


def filter_df_exclude_grades(
    df: pd.DataFrame,
    exclude_grade_codes: Optional[List[int]],
    *,
    grade_code_col: str = "grade_code",
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """
    指定した grade_code のレースをレース単位で除外する。

    grade_code_col が DataFrame に存在しない場合は何もせず返す。
    レース内の1行でも対象 grade_code があればレース全体を除外する。

    Parameters
    ----------
    exclude_grade_codes : list[int] | None
        除外する grade_code のリスト（例: [1] = 新馬・未勝利戦）。None または空リストの場合は除外しない。
    """
    if not exclude_grade_codes or grade_code_col not in df.columns:
        return df.copy()

    codes = set(int(c) for c in exclude_grade_codes)
    gc = pd.to_numeric(df[grade_code_col], errors="coerce")
    is_excluded = gc.isin(codes)

    # レース単位の除外: race_id に1頭でも除外コードがあればレース全体を落とす
    if race_id_col in df.columns:
        excluded_race_ids = set(df.loc[is_excluded, race_id_col].astype(str).unique())
        keep_mask = ~df[race_id_col].astype(str).isin(excluded_race_ids)
    else:
        keep_mask = ~is_excluded

    out = df.loc[keep_mask].copy()
    return out


def filter_df_exclude_surface(
    df: pd.DataFrame,
    exclude_surface_codes: Optional[List[int]],
    *,
    surface_code_col: str = "surface_code",
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """
    指定した surface_code のレースをレース単位で除外する（既定 [3] = 障害）。

    障害レース（surface_code=3）は平地 binary モデルの学習対象外で予測不能なため
    推奨しない（CLAUDE.md「障害は平地モデルで予測不能」）。grade_code は features 層で
    全行7に潰れ exclude_grade_codes:[8,9] が効かないため、surface_code で直接除外する。
    backtest 側（strategy/src/backtest.py の exclude_surface_codes 既定 [3]）と同一意味。

    surface_code_col が DataFrame に存在しない場合は何もせず返す。
    レース内の1行でも対象 surface_code があればレース全体を除外する。

    Parameters
    ----------
    exclude_surface_codes : list[int] | None
        除外する surface_code のリスト（既定運用値 [3] = 障害）。None または空リストの場合は除外しない。
    """
    if not exclude_surface_codes or surface_code_col not in df.columns:
        return df.copy()

    codes = set(int(c) for c in exclude_surface_codes)
    sc = pd.to_numeric(df[surface_code_col], errors="coerce")
    is_excluded = sc.isin(codes)

    # レース単位の除外: race_id に1頭でも除外コードがあればレース全体を落とす
    if race_id_col in df.columns:
        excluded_race_ids = set(df.loc[is_excluded, race_id_col].astype(str).unique())
        keep_mask = ~df[race_id_col].astype(str).isin(excluded_race_ids)
    else:
        keep_mask = ~is_excluded

    out = df.loc[keep_mask].copy()
    return out


def filter_df_exclude_dirt(
    df: pd.DataFrame,
    exclude_dirt: bool,
    dirt_track_code_min: int = 23,
    *,
    track_code_col: str = "track_code",
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """
    ROI 基準割れのダートコース（track_code >= dirt_track_code_min）をレース単位で除外する。

    create_features.py と同一の閾値（デフォルト 23）でダートを判定する。
    track_code_col が DataFrame に存在しない場合は除外しない。

    Parameters
    ----------
    exclude_dirt : bool
        True の場合にダートレースを除外する。
    dirt_track_code_min : int
        ダートと見なす track_code の下限値（23 = JV-Link ダートコース先頭コード）。
    """
    if not exclude_dirt or track_code_col not in df.columns:
        return df.copy()

    tc = pd.to_numeric(df[track_code_col], errors="coerce")
    is_dirt = tc >= dirt_track_code_min

    if race_id_col in df.columns:
        dirt_race_ids = set(df.loc[is_dirt, race_id_col].astype(str).unique())
        keep_mask = ~df[race_id_col].astype(str).isin(dirt_race_ids)
    else:
        keep_mask = ~is_dirt

    out = df.loc[keep_mask].copy()
    return out


def filter_df_exclude_age(
    df: pd.DataFrame,
    exclude_age_max: Optional[int],
    *,
    age_col: str = "age",
) -> pd.DataFrame:
    """
    指定年齢より高齢な馬を馬単位で除外する（レース全体は除外しない）。

    exclude_age_max=6 の場合、age >= 7 の馬行を削除する。
    age_col が DataFrame に存在しない場合は何もせず返す。

    Parameters
    ----------
    exclude_age_max : int | None
        この年齢まで許容（それ以上の馬を除外）。None の場合は除外しない。
    """
    if exclude_age_max is None or age_col not in df.columns:
        return df.copy()

    age = pd.to_numeric(df[age_col], errors="coerce")
    keep_mask = age.le(int(exclude_age_max)) | age.isna()
    return df.loc[keep_mask].copy()


def apply_conditional_ev_overrides(
    df: pd.DataFrame,
    overrides: Optional[List[Dict[str, Any]]],
    *,
    edge_col: str = "edge",
    default_ev_threshold: float = 1.05,
) -> pd.DataFrame:
    """
    条件レースに対して min_edge / min_ev フィルタを追加適用する。

    backtest / 本番共通の ``inference_common.apply_condition_overrides_to_recommendations``
    に委譲する（MS-2: スキーマ統一）。
    """
    from strategy.src.inference_common import apply_condition_overrides_to_recommendations

    out = apply_condition_overrides_to_recommendations(
        df, overrides or [], default_ev_threshold
    )
    if edge_col != "edge" and edge_col in out.columns and "edge" not in out.columns:
        out["edge"] = out[edge_col]
    return out


def apply_long_distance_ev_filter(
    df: pd.DataFrame,
    *,
    min_edge_override: float = 0.30,
    dist_min: int = 2201,
    distance_col: str = "distance",
    edge_col: str = "edge",
) -> pd.DataFrame:
    """
    長距離レース（2201m以上）の min_edge を引き上げる。

    apply_conditional_ev_overrides は等値比較のみのため距離範囲条件は非対応。
    domain-planner 仕様書 (2026-06-05): ROI=53.1% のため min_edge=0.30 を適用。
    ``_conditional_ev_ok`` 列が既に False の行は上書きしない（AND 論理）。

    Parameters
    ----------
    min_edge_override : float
        長距離帯に適用する最小 edge（既定 0.30）。
    dist_min : int
        長距離判定の距離下限（既定 2201m）。
    """
    out = df.copy()
    if edge_col not in out.columns or distance_col not in out.columns:
        return out

    if "_conditional_ev_ok" not in out.columns:
        out["_conditional_ev_ok"] = True

    dist = pd.to_numeric(out[distance_col], errors="coerce")
    edge = pd.to_numeric(out[edge_col], errors="coerce")
    is_long = dist.ge(dist_min)
    violates = is_long & edge.lt(min_edge_override)
    out.loc[violates, "_conditional_ev_ok"] = False
    return out

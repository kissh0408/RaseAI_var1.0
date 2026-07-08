import pandas as pd
import numpy as np
import gc
from pathlib import Path
import warnings
import json
from datetime import datetime
import time
import argparse
from typing import Any
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.utils.common_utils import optimize_dtypes, read_csv_optimized, log_step
from model_training.src.pipeline_common import (
    BASE_LEAK_COLS,
    PAST_EXTRA_LEAK_COLS,
    load_pastfeatures_config,
    load_train_config,
    update_state,
)
from model_training.src.features_prize import add_prize_features
from model_training.src.features_sire_stats import (
    add_sire_stats_features,
    add_nick_win_rate,
    add_sire_long_turf_feature,
)
from model_training.src.features_agari_course import add_agari_course_features
from model_training.src.features_dirt_impute import add_dirt_impute_features
from model_training.src.features_graded import (
    add_graded_jockey_trainer_features,
    add_graded_features,
)
from model_training.src.features_course_dist import add_course_dist_features_v18
from model_training.src.features_training_trend import add_training_trend_features
# Training-Serving Skew 解消: v21 Group A/B/C/D（features_v21_groups.py から共通実装をimport）
from model_training.src.features_v21_groups import (
    add_youshiba_features,
    add_sire_youshiba_features,
    add_kokai_koban_features,
    add_horse_soft_turf_features,
    add_sire_soft_turf_features,
    add_speed_index_features,
    add_pace_dist_style_features,
)
# v16〜v19 で追加されたが推論パイプラインに未追加だったモジュール群
from model_training.src.features_agari_stability import add_agari_stability_features
from model_training.src.features_sex_age import add_sex_age_features
from model_training.src.features_pace_field import add_pace_field_features
from model_training.src.features_soft_turf_impute import add_soft_turf_impute_features
from model_training.src.features_turn_surface import add_turn_surface_features
from model_training.src.features_style_course import add_style_course_features
from model_training.src.features_style_dist_straight import add_style_dist_features
from model_training.src.features_surface_dist_band import add_surface_dist_band_features
from model_training.src.features_rank1_context_v24 import add_rank1_context_v24_features
from model_training.src.features_corner_v25 import add_corner_v25_features
from model_training.src.features_odds_divergence import add_odds_divergence_features

# Group C スピード指数計算に racetime が必要なため _SE_V11_COLS に追加する
_SE_V11_COLS = ["race_id", "ketto_num", "hon_shokin", "fuka_shokin", "time_3f_after", "racetime"]

warnings.simplefilter("ignore")

try:
    import cudf
    import cupy as cp

    _GPU_AVAILABLE = True
except ImportError:
    cudf = None
    cp = None
    _GPU_AVAILABLE = False

INPUT_PATH = PROJECT_ROOT / "model_training/data/02_features/features_basic.csv"
OUTPUT_PATH = PROJECT_ROOT / "model_training/data/02_features/features_past.csv"
MANIFEST_PATH = PROJECT_ROOT / "model_training/data/02_features/features_past_manifest.json"
PED_OUTPUT_DIR = PROJECT_ROOT / "common/data/output"
HC_PREPROCESSED_PATH = PROJECT_ROOT / "model_training/data/01_preprocessed/HC_preprocessed.parquet"
MAIN_FEATURES_BASIC = (
    PROJECT_ROOT / "model_training/data/02_features/main_features_basic.csv"
)


def _features_basic_parquet_path(csv_or_parquet_path: Path) -> Path:
    p = Path(csv_or_parquet_path)
    return p.with_suffix(".parquet") if p.suffix.lower() != ".parquet" else p


def read_features_basic(path: str | Path, *, prefer_gpu: bool = False):
    """
    Prefer Parquet beside the CSV path when present for faster I/O.
    Returns cudf.DataFrame iff prefer_gpu and RAPIDS and a parquet file is used.
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        pq = _features_basic_parquet_path(p)
        read_path = pq if pq.is_file() else p
    elif p.is_file():
        read_path = p
    else:
        read_path = p
    if not read_path.is_file():
        raise FileNotFoundError(f"Features basic not found: {read_path}")
    suf = read_path.suffix.lower()
    if prefer_gpu and _GPU_AVAILABLE and suf == ".parquet":
        return cudf.read_parquet(read_path)
    if suf == ".parquet":
        return pd.read_parquet(read_path)
    return read_csv_optimized(read_path)


CONFIG = load_pastfeatures_config()
TRAIN_CONFIG_DATA = load_train_config()


def _is_gpu_df(df) -> bool:
    return bool(_GPU_AVAILABLE and isinstance(df, cudf.DataFrame))


def _to_gpu_df(df):
    if _is_gpu_df(df):
        return df
    if not _GPU_AVAILABLE:
        return df
    return cudf.from_pandas(df)


def _to_pandas_df(df):
    if _is_gpu_df(df):
        return df.to_pandas()
    return df


def _free_gpu_memory() -> None:
    if not _GPU_AVAILABLE:
        return
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _sort_for_pastfeatures(df):
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    if not sort_cols:
        return df
    return df.sort_values(sort_cols).reset_index(drop=True)


def _prepare_pandas_for_gpu(df: pd.DataFrame) -> pd.DataFrame:
    # cuDF は object 列に mixed types があると DataFrame 変換に失敗しやすい。
    # 変換前に mixed object を string に揃えて GPU へ渡す。
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        try:
            inferred = pd.api.types.infer_dtype(out[col], skipna=True)
        except Exception:
            inferred = "mixed"
        if inferred.startswith("mixed"):
            out[col] = out[col].astype("string")
    return out


def _build_manifest(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    leak_cols = list(dict.fromkeys(BASE_LEAK_COLS + PAST_EXTRA_LEAK_COLS))
    column_dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_columns": len(df.columns),
        "all_columns": list(df.columns),
        "column_dtypes": column_dtypes,
        "numeric_columns": numeric_cols,
        "leak_columns_defined": leak_cols,
        "leak_columns_present": [c for c in leak_cols if c in df.columns],
    }


def _save_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def calculate_win_rates(df, group_cols):
    if "finish_rank" not in df.columns:
        if _is_gpu_df(df):
            nan_series = cudf.Series(np.nan, index=df.index)
            zero_series = cudf.Series(0, index=df.index)
            return nan_series, nan_series, zero_series
        return (
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
            pd.Series(0, index=df.index),
        )

    finish = (
        cudf.to_numeric(df["finish_rank"], errors="coerce")
        if _is_gpu_df(df)
        else pd.to_numeric(df["finish_rank"], errors="coerce")
    )
    win_flag = (finish == 1).astype(int)
    ren_flag = (finish <= 2).astype(int)

    g = df.groupby(group_cols, sort=False)
    # 当該レースを除外した累積
    cum_runs = g.cumcount()
    cum_wins = win_flag.groupby(df[group_cols], sort=False).cumsum() - win_flag
    cum_ren = ren_flag.groupby(df[group_cols], sort=False).cumsum() - ren_flag

    beta = CONFIG["smoothing"]["beta"]
    win_rate = (cum_wins + CONFIG["smoothing"]["prior_default_win"] * beta) / (
        cum_runs + beta
    )
    ren_rate = (cum_ren + CONFIG["smoothing"]["prior_default_ren"] * beta) / (
        cum_runs + beta
    )
    return win_rate, ren_rate, cum_runs


def create_standardization_features(df):
    temp_time = df["racetime"].where(df["racetime"] > 0)
    valid = temp_time.notna().astype(int)

    # 基準タイム計算（過去平均）
    keys = ["course_code", "distance", "track_code"]
    time_for_sum = temp_time.fillna(0)
    tmp = df[keys].copy()
    tmp["__time_for_sum"] = time_for_sum
    tmp["__time_valid"] = valid
    grouped = tmp.groupby(keys, sort=False)
    past_sum = grouped["__time_for_sum"].cumsum() - tmp["__time_for_sum"]
    past_cnt = grouped["__time_valid"].cumsum() - tmp["__time_valid"]
    base_time = past_sum / past_cnt.replace(0, np.nan)
    df["speed_diff"] = base_time - df["racetime"]
    del tmp
    _free_gpu_memory()
    return df


def create_lag_features(df):
    g = df.groupby("ketto_num", sort=False)

    targets = ["finish_rank", "racetime", "speed_diff", "popularity", "odds"]
    for col in tqdm(
        targets, desc="Lag features", leave=False, dynamic_ncols=True
    ):
        if col in df.columns:
            for lag in [1, 2, 3, 4, 5]:
                df[f"lag{lag}_{col}"] = g[col].shift(lag)
    return df


def create_context_features(df):
    # 前走からの日数（interval）が既にある想定。なければdateから計算
    if "interval" not in df.columns and {"ketto_num", "date"}.issubset(df.columns):
        interval = df.groupby("ketto_num", sort=False)["date"].diff().dt.days.fillna(-1)
        df["interval"] = interval

    if "interval" in df.columns:
        # days_since_prev は interval の alias（完全同一）。
        # 下位互換のため列は保持するが、新規コードでは interval を使うこと。
        df["days_since_prev"] = df["interval"]
        df["is_holiday"] = (df["days_since_prev"] >= 90).astype(np.int8)

    # 距離延長・短縮（今回距離 - 前走距離）
    if {"ketto_num", "date", "distance"}.issubset(df.columns):
        df["distance_diff"] = df["distance"] - df.groupby("ketto_num", sort=False)[
            "distance"
        ].shift(1)

    # 騎手乗り替わりフラグ（前走と今走で騎手が変わった場合=1）
    # リーク防止: shift(1) で前走騎手コードを取得
    if "jockey_code" in df.columns:
        _prev_jk = df.groupby("ketto_num", sort=False)["jockey_code"].shift(1)
        df["jockey_change_flag"] = (
            (df["jockey_code"] != _prev_jk) & _prev_jk.notna()
        ).astype("int8")
        # 前走騎手との勝率差（騎手の実力差指数）
        if "jockey_win_rate" in df.columns:
            _prev_jk_wr = df.groupby("ketto_num", sort=False)["jockey_win_rate"].shift(1)
            df["jockey_change_win_rate_diff"] = df["jockey_win_rate"] - _prev_jk_wr

    return df


def create_race_relative_features(df):
    # 【時系列リーク判定: 非該当】
    # transform("mean") は同一 race_id（同一レース）内の出走馬間集計であり、
    # 異なる時点のレース情報を混入しない。
    # speed_deviation / relative_speed_pct は「当該レース完了後に確定する中間生成物」であり、
    # モデルには直接渡さない。必ず shift(1) でlag化した _lag1 サフィックス列のみを使用する。
    # 中間生成物（lag なし版）は PAST_EXTRA_LEAK_COLS に登録済みで、
    # remove_leak_features() により最終出力から削除される二重安全機構がある。
    #   参照: pipeline_common.py PAST_EXTRA_LEAK_COLS = ["speed_deviation", "relative_speed_pct"]
    if "speed_diff" not in df.columns:
        return df

    if "race_id" in df.columns:
        # 同一レース内の平均・標準偏差で偏差値化（race_id 内の相対比較のみ。時系列をまたがない）
        race_mean = df.groupby("race_id", sort=False)["speed_diff"].transform("mean")
        race_std = df.groupby("race_id", sort=False)["speed_diff"].transform("std")
        df["speed_deviation"] = ((df["speed_diff"] - race_mean) / race_std.replace(0, np.nan)) * 10 + 50
        df["relative_speed_pct"] = df.groupby("race_id", sort=False)["speed_diff"].rank(
            pct=True, ascending=False
        )

    if {"ketto_num", "date"}.issubset(df.columns):
        g = df.groupby("ketto_num", sort=False)
        # shift(1) で「前走の相対パフォーマンス」としてlag化。当走情報は含まれない
        if "speed_deviation" in df.columns:
            df["speed_deviation_lag1"] = g["speed_deviation"].shift(1)
        if "relative_speed_pct" in df.columns:
            df["relative_speed_lag1"] = g["relative_speed_pct"].shift(1)
    return df


def create_rolling_features(df):
    windows = [3, 5]
    cols = ["finish_rank", "speed_diff"]
    g = df.groupby("ketto_num", sort=False)

    for col in tqdm(
        cols, desc="Rolling features", leave=False, dynamic_ncols=True
    ):
        if col in df.columns:
            shifted = g[col].shift(1)
            for w in windows:
                if _is_gpu_df(df):
                    valid = shifted.notna().astype("int8")
                    filled = shifted.fillna(0)
                    csum = filled.groupby(df["ketto_num"], sort=False).cumsum()
                    ccnt = valid.groupby(df["ketto_num"], sort=False).cumsum()
                    csum_prev = csum.groupby(df["ketto_num"], sort=False).shift(w).fillna(0)
                    ccnt_prev = ccnt.groupby(df["ketto_num"], sort=False).shift(w).fillna(0)
                    win_sum = csum - csum_prev
                    win_cnt = ccnt - ccnt_prev
                    df[f"avg_{col}_{w}"] = win_sum / win_cnt.replace(0, np.nan)
                else:
                    grouped_shifted = shifted.groupby(df["ketto_num"], sort=False)
                    df[f"avg_{col}_{w}"] = (
                        grouped_shifted.rolling(window=w, min_periods=1)
                        .mean()
                        .droplevel(0)
                    )

    # speed_diff本体は後でリーク回避のため削除するが、
    # 過去情報のみで作る全履歴平均は有効特徴量として残す
    if "speed_diff" in df.columns:
        shifted_speed = g["speed_diff"].shift(1)
        if _is_gpu_df(df):
            df["__shifted_speed_diff_all"] = shifted_speed
            df["__shifted_speed_diff_valid"] = shifted_speed.notna().astype("int8")
            df["__shifted_speed_diff_fill"] = shifted_speed.fillna(0)
            grouped = df.groupby("ketto_num", sort=False)
            csum = grouped["__shifted_speed_diff_fill"].cumsum()
            ccnt = grouped["__shifted_speed_diff_valid"].cumsum()
            df["avg_speed_diff_all"] = csum / ccnt.replace(0, np.nan)
            df = df.drop(
                columns=[
                    "__shifted_speed_diff_all",
                    "__shifted_speed_diff_valid",
                    "__shifted_speed_diff_fill",
                ],
                errors="ignore",
            )
        else:
            df["avg_speed_diff_all"] = (
                shifted_speed.groupby(df["ketto_num"], sort=False)
                .expanding(min_periods=1)
                .mean()
                .droplevel(0)
            )

    # ------------------------------------------------------------------
    # 拡張ローリング: 窓7 + speed_diff の std(5)
    # GPU DataFrame の場合はパンダスに落として計算（rolling/std が限定的）
    # ------------------------------------------------------------------
    source_is_gpu = _is_gpu_df(df)
    if source_is_gpu:
        df = _to_pandas_df(df)
        g = df.groupby("ketto_num", sort=False)

    # avg_finish_rank_7: rolling(7, min_periods=3) of shift(1) finish_rank
    if "finish_rank" in df.columns:
        shifted_fr = g["finish_rank"].shift(1)
        df["avg_finish_rank_7"] = (
            shifted_fr.groupby(df["ketto_num"], sort=False)
            .rolling(window=7, min_periods=3)
            .mean()
            .droplevel(0)
        )

    # avg_speed_diff_7: rolling(7, min_periods=3) of shift(1) speed_diff
    if "speed_diff" in df.columns:
        shifted_sd = g["speed_diff"].shift(1)
        df["avg_speed_diff_7"] = (
            shifted_sd.groupby(df["ketto_num"], sort=False)
            .rolling(window=7, min_periods=3)
            .mean()
            .droplevel(0)
        )
        # speed_diff_std_5: rolling(5, min_periods=3).std() of shift(1) speed_diff
        df["speed_diff_std_5"] = (
            shifted_sd.groupby(df["ketto_num"], sort=False)
            .rolling(window=5, min_periods=3)
            .std()
            .droplevel(0)
        )

    # ------------------------------------------------------------------
    # トレンド特徴量: lag1 - lag3 の差分（方向性・加速度の代理指標）
    # lag3 が確定した直後に計算するため create_rolling_features の末尾に配置
    # ------------------------------------------------------------------
    if "lag1_speed_diff" in df.columns and "lag3_speed_diff" in df.columns:
        # 両方が有効な行のみ計算し、片方がNaNの場合はNaNを維持
        df["speed_diff_trend"] = df["lag1_speed_diff"] - df["lag3_speed_diff"]

    if "lag1_finish_rank" in df.columns and "lag3_finish_rank" in df.columns:
        # 正値=着順悪化（下降トレンド）、負値=着順改善（上昇トレンド）
        df["finish_rank_trend"] = df["lag1_finish_rank"] - df["lag3_finish_rank"]

    if source_is_gpu and _GPU_AVAILABLE:
        try:
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
        except Exception:
            pass

    return df


def create_human_stats(df):
    # 騎手実績
    df["jockey_win_rate"], df["jockey_ren_rate"], _ = calculate_win_rates(
        df, "jockey_code"
    )
    # 調教師実績
    df["trainer_win_rate"], df["trainer_ren_rate"], _ = calculate_win_rates(
        df, "trainer_code"
    )
    return df


def attach_pedigree_features(
    df,
    pedigree_output_dir=None,
    enable=True,
    show_progress=False,
    heartbeat_sec=60,
    log_prefix="[pastfeatures/pedigree]",
    strict=False,
    prefer_bulk_asof=True,
    starts_long_df=None,
    sibling_edges_df=None,
    dam_birth_year_series=None,
    dam_age_df=None,
):
    if not enable:
        return df
    source_is_gpu = _is_gpu_df(df)
    pedigree_output_dir = Path(pedigree_output_dir or PED_OUTPUT_DIR)
    _cfg_cut = TRAIN_CONFIG_DATA.get("training", {}).get("pedigree_train_year_cut")
    if _cfg_cut is not None:
        train_year_cut = int(_cfg_cut)
    elif "year" in df.columns:
        # config 未設定時はデータの最大年から自動導出（特徴量生成期間と常に同期する）
        train_year_cut = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    else:
        train_year_cut = 2024
    try:
        out_df = attach_leak_safe_pedigree(
            df,
            output_dir=pedigree_output_dir,
            train_year_cut=train_year_cut,
            prefer_bulk_asof=prefer_bulk_asof,
            show_progress=show_progress,
            heartbeat_sec=heartbeat_sec,
            log_prefix=log_prefix,
            starts_long_df=starts_long_df,
            sibling_edges_df=sibling_edges_df,
            dam_birth_year_series=dam_birth_year_series,
            dam_age_df=dam_age_df,
        )
        if source_is_gpu:
            gc.collect()
            _free_gpu_memory()
        return out_df
    except Exception as e:
        tqdm.write(f"[warn] Failed to attach leak-safe pedigree features: {e}")
        if strict:
            raise
        return df


def add_condition_specific_features(df):
    """
    条件別・状態系の追加特徴量を生成する。

    生成特徴量:
        horse_course_win_rate      -- ketto_num x course_code 勝率 (ベイズ平滑化 beta=15, prior=0.10)
        horse_surface_win_rate     -- ketto_num x race_type_code 勝率 (beta=10, prior=0.08)
        horse_distance_win_rate    -- ketto_num x distance_category 勝率 (beta=10, prior=0.08)
        corner4_normalized_lag1    -- corner_4 / n_horses の shift(1)
        jockey_trainer_combo_win_rate -- jockey_code x trainer_code 勝率 (beta=20, prior=0.09)
        tm_score_lag1              -- tm_score の shift(1)
        horse_interval_bins        -- days_since_prev の 6 区分カテゴリ (0〜5)

    全特徴量において sort_values('date') + shift(1) でリークを防止する。
    """
    # GPU DataFrame は CPU に落として処理する（cuDF は expanding/rolling が限定的）
    source_is_gpu = _is_gpu_df(df)
    if source_is_gpu:
        df = _to_pandas_df(df)

    # sort 保証（既に sort 済みのはずだが念のため）
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if "finish_rank" in df.columns:
        finish = pd.to_numeric(df["finish_rank"], errors="coerce")
        win_flag = (finish == 1).astype("int8")
    else:
        win_flag = pd.Series(np.zeros(len(df), dtype="int8"), index=df.index)

    # ------------------------------------------------------------------
    # 1. horse_course_win_rate
    #    グループ: ketto_num x course_code
    #    ベイズ: beta=15, prior=0.10
    # ------------------------------------------------------------------
    if "course_code" in df.columns:
        _BETA_COURSE = 15.0
        _PRIOR_COURSE = 0.10
        grp_course = ["ketto_num", "course_code"]
        g_c = df.groupby(grp_course, sort=False)
        cum_runs_c = g_c.cumcount()                          # 0-indexed 出走数（当該レース除外済み）
        cum_wins_c = win_flag.groupby(
            [df["ketto_num"], df["course_code"]], sort=False
        ).cumsum() - win_flag
        df["horse_course_win_rate"] = (
            (cum_wins_c + _BETA_COURSE * _PRIOR_COURSE)
            / (cum_runs_c + _BETA_COURSE)
        ).where(cum_runs_c > 0, _PRIOR_COURSE)
    else:
        df["horse_course_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 2. horse_surface_win_rate
    #    グループ: ketto_num x race_type_code（芝/ダート別）
    #    ベイズ: beta=10, prior=0.08
    # ------------------------------------------------------------------
    if "race_type_code" in df.columns:
        _BETA_SURF = 10.0
        _PRIOR_SURF = 0.08
        cum_runs_s = df.groupby(
            ["ketto_num", "race_type_code"], sort=False
        ).cumcount()
        cum_wins_s = win_flag.groupby(
            [df["ketto_num"], df["race_type_code"]], sort=False
        ).cumsum() - win_flag
        df["horse_surface_win_rate"] = (
            (cum_wins_s + _BETA_SURF * _PRIOR_SURF)
            / (cum_runs_s + _BETA_SURF)
        ).where(cum_runs_s > 0, _PRIOR_SURF)
    else:
        df["horse_surface_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 3. horse_distance_win_rate
    #    前処理: distance_category（既存列があれば利用、なければ生成）
    #    bins: [0, 1400, 1800, 2400, 10000], labels: [0, 1, 2, 3]
    #    ベイズ: beta=10, prior=0.08
    # ------------------------------------------------------------------
    if "distance" in df.columns:
        _BETA_DIST = 10.0
        _PRIOR_DIST = 0.08
        if "distance_category" not in df.columns:
            df["distance_category"] = pd.cut(
                pd.to_numeric(df["distance"], errors="coerce"),
                bins=[0, 1400, 1800, 2400, 10000],
                labels=[0, 1, 2, 3],
                right=True,
            ).astype("Int8")
        cum_runs_d = df.groupby(
            ["ketto_num", "distance_category"], sort=False
        ).cumcount()
        cum_wins_d = win_flag.groupby(
            [df["ketto_num"], df["distance_category"]], sort=False
        ).cumsum() - win_flag
        df["horse_distance_win_rate"] = (
            (cum_wins_d + _BETA_DIST * _PRIOR_DIST)
            / (cum_runs_d + _BETA_DIST)
        ).where(cum_runs_d > 0, _PRIOR_DIST)
    else:
        df["horse_distance_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 4. corner4_normalized_lag1
    #    corner_4 / n_horses を馬ごとに shift(1)
    # ------------------------------------------------------------------
    if "corner_4" in df.columns and "n_horses" in df.columns:
        c4 = pd.to_numeric(df["corner_4"], errors="coerce")
        nh = pd.to_numeric(df["n_horses"], errors="coerce")
        c4_norm = c4 / nh.replace(0, np.nan)
        df["corner4_normalized_lag1"] = (
            c4_norm.groupby(df["ketto_num"], sort=False).shift(1)
        )
    else:
        df["corner4_normalized_lag1"] = np.nan

    # ------------------------------------------------------------------
    # 5. jockey_trainer_combo_win_rate
    #    グループ: jockey_code x trainer_code
    #    ベイズ: beta=20, prior=0.09
    #    観測数 < 20 の場合は NaN → jockey_win_rate と trainer_win_rate の平均で impute
    # ------------------------------------------------------------------
    if "jockey_code" in df.columns and "trainer_code" in df.columns:
        _BETA_JT = 20.0
        _PRIOR_JT = 0.09
        _MIN_OBS_JT = 20
        grp_jt = ["jockey_code", "trainer_code"]
        cum_runs_jt = df.groupby(grp_jt, sort=False).cumcount()
        cum_wins_jt = win_flag.groupby(
            [df["jockey_code"], df["trainer_code"]], sort=False
        ).cumsum() - win_flag
        combo_rate = (cum_wins_jt + _BETA_JT * _PRIOR_JT) / (cum_runs_jt + _BETA_JT)
        # 観測数が少ない場合は NaN
        combo_rate = combo_rate.where(cum_runs_jt >= _MIN_OBS_JT, np.nan)
        df["jockey_trainer_combo_win_rate"] = combo_rate

        # NaN を jockey_win_rate と trainer_win_rate の平均で impute
        impute_mask = df["jockey_trainer_combo_win_rate"].isna()
        if impute_mask.any():
            if "jockey_win_rate" in df.columns and "trainer_win_rate" in df.columns:
                j_rate = pd.to_numeric(df["jockey_win_rate"], errors="coerce")
                t_rate = pd.to_numeric(df["trainer_win_rate"], errors="coerce")
                fallback = (j_rate + t_rate) / 2.0
                df.loc[impute_mask, "jockey_trainer_combo_win_rate"] = fallback[impute_mask]
    else:
        df["jockey_trainer_combo_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 6. tm_score_lag1
    #    ketto_num グループで tm_score を shift(1)
    # ------------------------------------------------------------------
    if "tm_score" in df.columns:
        df["tm_score_lag1"] = (
            df.groupby("ketto_num", sort=False)["tm_score"].shift(1)
        )
    # tm_score が存在しない場合はスキップ（列追加なし）

    # ------------------------------------------------------------------
    # 7. horse_interval_bins
    #    days_since_prev（= interval のコピー）を 6 区分カテゴリ化
    #    days_since_prev が -1（初走マーカー）または欠損の場合はカテゴリ 3（標準間隔）で fill
    # ------------------------------------------------------------------
    _INTERVAL_BINS = [0, 14, 28, 56, 90, 180, 10000]
    _INTERVAL_LABELS = [0, 1, 2, 3, 4, 5]
    _INTERVAL_DEFAULT = 3

    # days_since_prev は create_context_features が interval からコピーするが、
    # 本関数はパイプライン順序に依存せず interval も参照できるようにする
    if "days_since_prev" in df.columns:
        _interval_src = pd.to_numeric(df["days_since_prev"], errors="coerce")
    elif "interval" in df.columns:
        _interval_src = pd.to_numeric(df["interval"], errors="coerce")
    else:
        _interval_src = pd.Series(np.nan, index=df.index)

    # -1 は初走マーカーとして NaN に変換してから cut
    _interval_src = _interval_src.where(_interval_src > 0, np.nan)
    _bins_result = pd.cut(
        _interval_src,
        bins=_INTERVAL_BINS,
        labels=_INTERVAL_LABELS,
        right=True,
    )
    # NaN はカテゴリ 3 で fill
    df["horse_interval_bins"] = (
        _bins_result.cat.add_categories([_INTERVAL_DEFAULT])
        if _INTERVAL_DEFAULT not in _bins_result.cat.categories
        else _bins_result
    ).fillna(_INTERVAL_DEFAULT).astype("int8")

    # ------------------------------------------------------------------
    # 8. jockey_course_win_rate
    #    グループ: jockey_code x course_code
    #    ベイズ: beta=20, prior=0.09
    #    騎手×競馬場のコース適性を時系列安全に累積する
    # ------------------------------------------------------------------
    if "jockey_code" in df.columns and "course_code" in df.columns:
        if "jockey_course_win_rate" not in df.columns:
            _BETA_JC = 20.0
            _PRIOR_JC = 0.09
            cum_runs_jc = df.groupby(
                ["jockey_code", "course_code"], sort=False
            ).cumcount()
            cum_wins_jc = win_flag.groupby(
                [df["jockey_code"], df["course_code"]], sort=False
            ).cumsum() - win_flag
            df["jockey_course_win_rate"] = (
                (cum_wins_jc + _BETA_JC * _PRIOR_JC)
                / (cum_runs_jc + _BETA_JC)
            ).where(cum_runs_jc > 0, _PRIOR_JC)
    else:
        if "jockey_course_win_rate" not in df.columns:
            df["jockey_course_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 9. jockey_surface_win_rate
    #    グループ: jockey_code x race_type_code（芝/ダート別）
    #    ベイズ: beta=15, prior=0.09
    # ------------------------------------------------------------------
    if "jockey_code" in df.columns and "race_type_code" in df.columns:
        if "jockey_surface_win_rate" not in df.columns:
            _BETA_JS = 15.0
            _PRIOR_JS = 0.09
            cum_runs_js = df.groupby(
                ["jockey_code", "race_type_code"], sort=False
            ).cumcount()
            cum_wins_js = win_flag.groupby(
                [df["jockey_code"], df["race_type_code"]], sort=False
            ).cumsum() - win_flag
            df["jockey_surface_win_rate"] = (
                (cum_wins_js + _BETA_JS * _PRIOR_JS)
                / (cum_runs_js + _BETA_JS)
            ).where(cum_runs_js > 0, _PRIOR_JS)
    else:
        if "jockey_surface_win_rate" not in df.columns:
            df["jockey_surface_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 10. trainer_surface_win_rate
    #     グループ: trainer_code x race_type_code（芝/ダート別）
    #     ベイズ: beta=15, prior=0.09
    # ------------------------------------------------------------------
    if "trainer_code" in df.columns and "race_type_code" in df.columns:
        if "trainer_surface_win_rate" not in df.columns:
            _BETA_TS = 15.0
            _PRIOR_TS = 0.09
            cum_runs_ts = df.groupby(
                ["trainer_code", "race_type_code"], sort=False
            ).cumcount()
            cum_wins_ts = win_flag.groupby(
                [df["trainer_code"], df["race_type_code"]], sort=False
            ).cumsum() - win_flag
            df["trainer_surface_win_rate"] = (
                (cum_wins_ts + _BETA_TS * _PRIOR_TS)
                / (cum_runs_ts + _BETA_TS)
            ).where(cum_runs_ts > 0, _PRIOR_TS)
    else:
        if "trainer_surface_win_rate" not in df.columns:
            df["trainer_surface_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 11. trainer_distance_win_rate
    #     グループ: trainer_code x distance_category
    #     ベイズ: beta=15, prior=0.08
    #     distance_category は特徴量3で生成済みのものを再利用する
    # ------------------------------------------------------------------
    if "trainer_code" in df.columns and "distance_category" in df.columns:
        if "trainer_distance_win_rate" not in df.columns:
            _BETA_TD = 15.0
            _PRIOR_TD = 0.08
            cum_runs_td = df.groupby(
                ["trainer_code", "distance_category"], sort=False
            ).cumcount()
            cum_wins_td = win_flag.groupby(
                [df["trainer_code"], df["distance_category"]], sort=False
            ).cumsum() - win_flag
            df["trainer_distance_win_rate"] = (
                (cum_wins_td + _BETA_TD * _PRIOR_TD)
                / (cum_runs_td + _BETA_TD)
            ).where(cum_runs_td > 0, _PRIOR_TD)
    elif "trainer_code" in df.columns and "distance" in df.columns:
        # distance_category が未生成の場合は 4 区分（〜1400m, 1401〜1800m, 1801〜2200m, 2201m〜）で生成
        if "trainer_distance_win_rate" not in df.columns:
            _dc_tmp = pd.cut(
                pd.to_numeric(df["distance"], errors="coerce"),
                bins=[0, 1400, 1800, 2200, 10000],
                labels=[0, 1, 2, 3],
                right=True,
            ).astype("Int8")
            _BETA_TD = 15.0
            _PRIOR_TD = 0.08
            cum_runs_td = df.groupby(
                [df["trainer_code"], _dc_tmp], sort=False
            ).cumcount()
            cum_wins_td = win_flag.groupby(
                [df["trainer_code"], _dc_tmp], sort=False
            ).cumsum() - win_flag
            df["trainer_distance_win_rate"] = (
                (cum_wins_td + _BETA_TD * _PRIOR_TD)
                / (cum_runs_td + _BETA_TD)
            ).where(cum_runs_td > 0, _PRIOR_TD)
    else:
        if "trainer_distance_win_rate" not in df.columns:
            df["trainer_distance_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 12. burden_weight_diff_lag1
    #     前走からの斤量変化（正: 斤量増, 負: 斤量減）
    #     shift(1) で当該レースを除外してリーク防止
    # ------------------------------------------------------------------
    if "burden_weight" in df.columns and "burden_weight_diff_lag1" not in df.columns:
        bw = pd.to_numeric(df["burden_weight"], errors="coerce")
        df["burden_weight_diff_lag1"] = (
            bw - bw.groupby(df["ketto_num"], sort=False).shift(1)
        )

    # ------------------------------------------------------------------
    # 13. n_horses_diff_lag1
    #     前走からの頭数変化（正: 頭数増, 負: 頭数減）
    #     shift(1) で当該レースを除外してリーク防止
    # ------------------------------------------------------------------
    if "n_horses" in df.columns and "n_horses_diff_lag1" not in df.columns:
        nh_num = pd.to_numeric(df["n_horses"], errors="coerce")
        df["n_horses_diff_lag1"] = (
            nh_num - nh_num.groupby(df["ketto_num"], sort=False).shift(1)
        )

    # ------------------------------------------------------------------
    # 14. class_change_code
    #     クラス変化（grade_code の前走差を -2〜+2 にクリップした整数）
    #     正: クラス上昇, 負: クラス降級, 0: 同一クラス
    #     shift(1) で当該レースを除外してリーク防止
    # ------------------------------------------------------------------
    if "grade_code" in df.columns and "class_change_code" not in df.columns:
        grade_num = pd.to_numeric(df["grade_code"], errors="coerce")
        lag_grade = grade_num.groupby(df["ketto_num"], sort=False).shift(1)
        df["class_change_code"] = (
            (grade_num - lag_grade).clip(-2, 2).astype("Int8")
        )

    # ------------------------------------------------------------------
    # 15. jockey_distance_win_rate
    #     グループ: jockey_code x distance_category（4区分）
    #     ベイズ平滑化: beta=20, prior=0.09
    #     騎手の距離帯別適性を時系列安全に累積（cumcount/cumsum でリーク防止）
    # ------------------------------------------------------------------
    if "jockey_code" in df.columns and "distance_category" in df.columns:
        if "jockey_distance_win_rate" not in df.columns:
            _BETA_JD = 20.0
            _PRIOR_JD = 0.09
            cum_runs_jd = df.groupby(
                ["jockey_code", "distance_category"], sort=False
            ).cumcount()
            cum_wins_jd = win_flag.groupby(
                [df["jockey_code"], df["distance_category"]], sort=False
            ).cumsum() - win_flag
            df["jockey_distance_win_rate"] = (
                (cum_wins_jd + _BETA_JD * _PRIOR_JD)
                / (cum_runs_jd + _BETA_JD)
            ).where(cum_runs_jd > 0, _PRIOR_JD)
    elif "jockey_code" in df.columns and "distance" in df.columns:
        if "jockey_distance_win_rate" not in df.columns:
            # distance_category が未生成の場合は動的に生成して処理
            _dc_jd_tmp = pd.cut(
                pd.to_numeric(df["distance"], errors="coerce"),
                bins=[0, 1400, 1800, 2400, 10000],
                labels=[0, 1, 2, 3],
                right=True,
            ).astype("Int8")
            _BETA_JD = 20.0
            _PRIOR_JD = 0.09
            cum_runs_jd = df.groupby(
                [df["jockey_code"], _dc_jd_tmp], sort=False
            ).cumcount()
            cum_wins_jd = win_flag.groupby(
                [df["jockey_code"], _dc_jd_tmp], sort=False
            ).cumsum() - win_flag
            df["jockey_distance_win_rate"] = (
                (cum_wins_jd + _BETA_JD * _PRIOR_JD)
                / (cum_runs_jd + _BETA_JD)
            ).where(cum_runs_jd > 0, _PRIOR_JD)
    else:
        if "jockey_distance_win_rate" not in df.columns:
            df["jockey_distance_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 16. trainer_course_win_rate
    #     グループ: trainer_code x course_code
    #     ベイズ平滑化: beta=15, prior=0.08
    #     調教師の競馬場別適性を時系列安全に累積
    # ------------------------------------------------------------------
    if "trainer_code" in df.columns and "course_code" in df.columns:
        if "trainer_course_win_rate" not in df.columns:
            _BETA_TC = 15.0
            _PRIOR_TC = 0.08
            cum_runs_tc = df.groupby(
                ["trainer_code", "course_code"], sort=False
            ).cumcount()
            cum_wins_tc = win_flag.groupby(
                [df["trainer_code"], df["course_code"]], sort=False
            ).cumsum() - win_flag
            df["trainer_course_win_rate"] = (
                (cum_wins_tc + _BETA_TC * _PRIOR_TC)
                / (cum_runs_tc + _BETA_TC)
            ).where(cum_runs_tc > 0, _PRIOR_TC)
    else:
        if "trainer_course_win_rate" not in df.columns:
            df["trainer_course_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 17. corner_flow_lag1
    #     (corner_4 - corner_1) / n_horses を馬ごとに shift(1)
    #     前走のコーナー位置変化（コーナー流れ）を表す指標
    # ------------------------------------------------------------------
    if (
        "corner_4" in df.columns
        and "corner_1" in df.columns
        and "n_horses" in df.columns
        and "corner_flow_lag1" not in df.columns
    ):
        c4_num = pd.to_numeric(df["corner_4"], errors="coerce")
        c1_num = pd.to_numeric(df["corner_1"], errors="coerce")
        nh_num = pd.to_numeric(df["n_horses"], errors="coerce")
        corner_flow = (c4_num - c1_num) / nh_num.replace(0, np.nan)
        df["corner_flow_lag1"] = (
            corner_flow.groupby(df["ketto_num"], sort=False).shift(1)
        )

    # ------------------------------------------------------------------
    # 18. horse_heavy_win_rate
    #     重馬場（稍重・重・不良）のみを対象とした ketto_num 別ベイズ平滑化勝率
    #     beta=10, prior=0.08
    #     良馬場時は NaN とする
    # ------------------------------------------------------------------
    if (
        ("turf_condition" in df.columns or "dirt_condition" in df.columns)
        and "horse_heavy_win_rate" not in df.columns
    ):
        _BETA_HEAVY = 10.0
        _PRIOR_HEAVY = 0.08
        turf_cond = pd.to_numeric(df.get("turf_condition", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        dirt_cond = pd.to_numeric(df.get("dirt_condition", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        # 稍重=2, 重=3, 不良=4
        is_heavy = ((turf_cond >= 2) | (dirt_cond >= 2)).astype("int8")

        # 重馬場レースのみの win_flag（良馬場は 0 として扱う）
        heavy_win = (win_flag * is_heavy).astype("int8")

        # ketto_num × 全レース（重馬場フラグ付き）で累積
        g_heavy = df.groupby("ketto_num", sort=False)
        cum_heavy_runs = is_heavy.groupby(df["ketto_num"], sort=False).cumsum() - is_heavy
        cum_heavy_wins = heavy_win.groupby(df["ketto_num"], sort=False).cumsum() - heavy_win

        heavy_rate = (
            (cum_heavy_wins + _BETA_HEAVY * _PRIOR_HEAVY)
            / (cum_heavy_runs + _BETA_HEAVY)
        ).where(cum_heavy_runs > 0, _PRIOR_HEAVY)

        # 良馬場時は NaN
        df["horse_heavy_win_rate"] = heavy_rate.where(is_heavy == 1, np.nan)

    # ------------------------------------------------------------------
    # 19. horse_weight_trend
    #     horse_weight_change の shift(1) 後に rolling(3, min_periods=2).sum()
    #     直近3走の体重変動方向の累積（正=増加トレンド、負=減少トレンド）
    # ------------------------------------------------------------------
    if "horse_weight_change" in df.columns and "horse_weight_trend" not in df.columns:
        hwc = pd.to_numeric(df["horse_weight_change"], errors="coerce")
        shifted_hwc = hwc.groupby(df["ketto_num"], sort=False).shift(1)
        df["horse_weight_trend"] = (
            shifted_hwc.groupby(df["ketto_num"], sort=False)
            .rolling(window=3, min_periods=2)
            .sum()
            .droplevel(0)
        )

    # ------------------------------------------------------------------
    # 20. same_distance_win_rate
    #     ketto_num × distance（完全一致）でのベイズ平滑化勝率
    #     beta=10, prior=0.08
    #     horse_distance_win_rate（距離カテゴリ別）より粒度が細かい
    # ------------------------------------------------------------------
    if "distance" in df.columns and "same_distance_win_rate" not in df.columns:
        _BETA_SD = 10.0
        _PRIOR_SD = 0.08
        dist_num = pd.to_numeric(df["distance"], errors="coerce")
        cum_runs_sd = df.groupby(
            [df["ketto_num"], dist_num], sort=False
        ).cumcount()
        cum_wins_sd = win_flag.groupby(
            [df["ketto_num"], dist_num], sort=False
        ).cumsum() - win_flag
        df["same_distance_win_rate"] = (
            (cum_wins_sd + _BETA_SD * _PRIOR_SD)
            / (cum_runs_sd + _BETA_SD)
        ).where(cum_runs_sd > 0, _PRIOR_SD)

    # ------------------------------------------------------------------
    # 21. agari3f_rank_in_race_lag1
    #     前走レース内の上がり3F順位（1が最速）を ketto_num ごとに shift(1) でlag化する。
    #     time_3f_after（秒）が小さいほど順位が上位（ascending=True）。
    #     当走の time_3f_after はリーク列のため remove_leak_features で削除されるが、
    #     本関数はその前に呼ばれるため安全に参照できる。
    # ------------------------------------------------------------------
    if "time_3f_after" in df.columns and "race_id" in df.columns and "agari3f_rank_in_race_lag1" not in df.columns:
        agari = pd.to_numeric(df["time_3f_after"], errors="coerce")
        # レース内で昇順 rank（3Fタイムが短いほど1位）
        agari_rank_in_race = agari.groupby(df["race_id"], sort=False).rank(
            method="min", ascending=True
        )
        # 前走の値として lag 化（当走を除外）
        df["agari3f_rank_in_race_lag1"] = (
            agari_rank_in_race.groupby(df["ketto_num"], sort=False).shift(1)
        )
    elif "agari3f_rank_in_race_lag1" not in df.columns:
        df["agari3f_rank_in_race_lag1"] = np.nan

    # ------------------------------------------------------------------
    # 22. agari3f_vs_winner_diff_lag1
    #     前走の自馬上がり3F - 前走1着馬の上がり3F（秒単位）。
    #     0またはマイナスは1着馬と同等以上の切れ味を意味する。
    #     リーク防止: レース内で1着馬の time_3f_after を参照してから ketto_num で shift(1)。
    # ------------------------------------------------------------------
    if "time_3f_after" in df.columns and "finish_rank" in df.columns and "race_id" in df.columns and "agari3f_vs_winner_diff_lag1" not in df.columns:
        agari_num = pd.to_numeric(df["time_3f_after"], errors="coerce")
        finish_num = pd.to_numeric(df["finish_rank"], errors="coerce")

        # 各レースの1着馬の time_3f_after を取得する
        # 1着馬が複数いる（同着）場合は最小値（最速）を採用
        winner_agari = (
            agari_num.where(finish_num == 1)
            .groupby(df["race_id"], sort=False)
            .transform("min")
        )
        # 自馬上がり3F - 1着馬上がり3F（負値 = 1着馬より速い）
        agari_diff = agari_num - winner_agari

        # 前走値として lag 化（当走を除外）
        df["agari3f_vs_winner_diff_lag1"] = (
            agari_diff.groupby(df["ketto_num"], sort=False).shift(1)
        )
    elif "agari3f_vs_winner_diff_lag1" not in df.columns:
        df["agari3f_vs_winner_diff_lag1"] = np.nan

    # ------------------------------------------------------------------
    # 23. best_finish_rank_l5
    #     lag1_finish_rank〜lag5_finish_rank の行方向 min（過去5走の最高着順）。
    #     一部 NaN は無視し、全列 NaN の場合のみ NaN。
    # ------------------------------------------------------------------
    lag_rank_cols = [f"lag{i}_finish_rank" for i in range(1, 6)]
    available_lag_rank = [c for c in lag_rank_cols if c in df.columns]
    if available_lag_rank and "best_finish_rank_l5" not in df.columns:
        df["best_finish_rank_l5"] = df[available_lag_rank].min(axis=1, skipna=True)
    elif "best_finish_rank_l5" not in df.columns:
        df["best_finish_rank_l5"] = np.nan

    # ------------------------------------------------------------------
    # 24. finish_rank_trend_l2
    #     lag1_finish_rank - lag2_finish_rank（既存列から算出）。
    #     負値 = 近走改善傾向（着順が良くなっている）。
    #     lag1 か lag2 が NaN の場合は NaN。
    #     旧: finish_rank_trend_l3 (lag1-lag3, NaN率30.5%) から変更。
    #     lag2を使うことで出走2走以上から計算可能（NaN率約21%）。
    # ------------------------------------------------------------------
    if "lag1_finish_rank" in df.columns and "lag2_finish_rank" in df.columns and "finish_rank_trend_l2" not in df.columns:
        df["finish_rank_trend_l2"] = (
            df["lag1_finish_rank"] - df["lag2_finish_rank"]
        )
    elif "finish_rank_trend_l2" not in df.columns:
        df["finish_rank_trend_l2"] = np.nan

    # ------------------------------------------------------------------
    # 25. race_upset_rate_l12m
    #     過去12ヶ月（365日）の同一コース×距離カテゴリ×馬場状態において
    #     1番人気が1着にならなかったレースの割合（レース単位の集計）。
    #     リーク防止: 当該レース自体を除外（当日以降のデータは参照しない）。
    #     安定性確保: 過去データが5件未満の場合は NaN。
    # ------------------------------------------------------------------
    if (
        "popularity" in df.columns
        and "finish_rank" in df.columns
        and "race_id" in df.columns
        and "date" in df.columns
        and "distance" in df.columns
        and "race_upset_rate_l12m" not in df.columns
    ):
        # 距離カテゴリを生成（distance_category が既にある場合は使い回す）
        if "distance_category" in df.columns:
            dist_cat_for_upset = df["distance_category"].astype(str)
        else:
            dist_cat_for_upset = pd.cut(
                pd.to_numeric(df["distance"], errors="coerce"),
                bins=[0, 1400, 1800, 2200, 10000],
                labels=["sprint", "mile", "middle", "long"],
                right=False,
            ).astype(str)

        # 馬場状態コード（芝 turf_condition / ダート dirt_condition）
        # race_type_code が 1=芝, 2=ダート の場合に対応した馬場コードを合成
        if "turf_condition" in df.columns and "dirt_condition" in df.columns:
            turf_cond_u = pd.to_numeric(df.get("turf_condition", 0), errors="coerce").fillna(0).astype(int)
            dirt_cond_u = pd.to_numeric(df.get("dirt_condition", 0), errors="coerce").fillna(0).astype(int)
            # 芝=1*, ダート=2* とし、最大値で代表させる
            track_cond_key = turf_cond_u.astype(str) + "_" + dirt_cond_u.astype(str)
        else:
            track_cond_key = pd.Series("0_0", index=df.index)

        course_key_u = pd.to_numeric(df.get("course_code", 0), errors="coerce").fillna(0).astype(int).astype(str)

        # レース単位の「1番人気が1着にならなかった」フラグ
        pop_num = pd.to_numeric(df["popularity"], errors="coerce")
        rank_num = pd.to_numeric(df["finish_rank"], errors="coerce")
        # 1番人気の馬の finish_rank != 1 をレースごとに判定
        is_fav = (pop_num == 1)
        fav_win = (is_fav & (rank_num == 1))

        # レースごとの集約（1レース1行）
        # drop_duplicates はインデックスを保持するため reindex で整合させる
        _upset_tmp = df[["race_id", "date"]].copy()
        _upset_tmp["_dist_cat"] = dist_cat_for_upset.values
        _upset_tmp["_cond_key"] = track_cond_key.values
        _upset_tmp["_course_key"] = course_key_u.values
        race_level_df = _upset_tmp.drop_duplicates("race_id").copy().reset_index(drop=True)

        # レースで1番人気が存在するかチェック
        fav_win_per_race = fav_win.groupby(df["race_id"]).any()
        is_fav_present = is_fav.groupby(df["race_id"]).any()
        # 1番人気が存在するレースのみ upset フラグを付ける
        race_level_df["_has_fav"] = is_fav_present.reindex(race_level_df["race_id"]).values
        race_level_df["_fav_won"] = fav_win_per_race.reindex(race_level_df["race_id"]).values
        # 1番人気が存在しないレースは除外（NaN）
        race_level_df["_is_upset"] = np.where(
            race_level_df["_has_fav"],
            (~race_level_df["_fav_won"]).astype(float),
            np.nan,
        )

        race_level_df["_group_key"] = (
            race_level_df["_course_key"] + "_" + race_level_df["_dist_cat"] + "_" + race_level_df["_cond_key"]
        )
        race_level_df = race_level_df.sort_values("date").reset_index(drop=True)

        # 365日ローリング集計（当日未満・strict、当日は除外）
        # ベクトル化実装: グループ×日付でソート済みの状態で
        # shift(1) + expanding で「当該レースを除いた過去」を cumsum し、
        # 365日前の cumsum との差分から rolling mean を算出する。
        # NaN の _is_upset（1番人気不在レース）は 0/1 の分母に含めないため
        # 別途 count_valid を追跡する。
        _MIN_RACE_COUNT_UPSET = 5

        # グループ×時系列ソートで expanding cumsum を取得
        race_level_df = race_level_df.sort_values(["_group_key", "date"]).reset_index(drop=True)

        # NaN でない行のみを 1 とするカウントフラグ
        is_valid = race_level_df["_is_upset"].notna().astype("int32")
        upset_val_filled = race_level_df["_is_upset"].fillna(0.0)

        grp_key_col = race_level_df["_group_key"]

        # shift(1) で当該レースを除外した累積和・累積カウントを取得
        cumsum_upset = (
            upset_val_filled.groupby(grp_key_col, sort=False).cumsum()
            - upset_val_filled
        )
        cumcnt_valid = (
            is_valid.groupby(grp_key_col, sort=False).cumsum()
            - is_valid
        )

        # 365日前時点の累積値（merge_asof で過去365日分の先頭を特定）
        # date の数値変換で日付差を高速計算
        date_num = race_level_df["date"].astype("int64")  # nanoseconds

        # グループごとに 365日前境界での cumsum を lookup するため
        # 365日前時点の行インデックスを特定し、そこでの累積値を引く
        ns_per_day = 86_400 * 10**9
        window_ns = 365 * ns_per_day

        # グループ単位で先頭境界の累積値を差し引く（pandas groupby + merge_asof 相当）
        # shift(1) 済み cumsum を使うことで当該レース自体は既に除外されている
        result_vals = np.full(len(race_level_df), np.nan)

        for gk, grp_idx in race_level_df.groupby("_group_key", sort=False).groups.items():
            grp_idx_arr = grp_idx  # DatetimeIndex → array
            dn = date_num.iloc[grp_idx_arr].values
            cs = cumsum_upset.iloc[grp_idx_arr].values
            cc = cumcnt_valid.iloc[grp_idx_arr].values

            # 各行 i に対して、365日前以降の累積開始点を二分探索で特定
            # 開始点の cumsum を引くことで [window_start, cutoff) の統計量を得る
            for i in range(len(dn)):
                cutoff_ns = dn[i]
                window_start_ns = cutoff_ns - window_ns

                # 窓開始以前（strict: date < cutoff かつ date >= window_start）
                # cumsum はすでに shift(1) 済みで当該行を除外している
                # 窓の左端: date_num[j] >= window_start_ns となる最初の j
                lo = np.searchsorted(dn, window_start_ns, side="left")

                # shift(1) 済み cumsum[i] は cutoff 未満の全累積
                # lo-1 時点の cumsum を引けば [window_start, cutoff) の集計になる
                total_sum = cs[i]
                total_cnt = cc[i]

                if lo > 0:
                    # window_start より前の累積分を差し引く
                    # lo-1 の shift(1) 済み cumsum ではなく実際の lo-1 の値を使う:
                    # cs[lo-1] は lo-1 番目のレースを除いた累積 → lo-1 のレース自体は含まれない。
                    # lo-1 のレース（window_start より前）を含めて引くため cs[lo-1] + value[lo-1]
                    window_sum = total_sum - cs[lo - 1] - upset_val_filled.iloc[grp_idx_arr[lo - 1]]
                    window_cnt = total_cnt - cc[lo - 1] - is_valid.iloc[grp_idx_arr[lo - 1]]
                else:
                    window_sum = total_sum
                    window_cnt = total_cnt

                if window_cnt >= _MIN_RACE_COUNT_UPSET:
                    result_vals[grp_idx_arr[i]] = window_sum / window_cnt

        race_upset_series = pd.Series(result_vals, index=race_level_df["race_id"].values)

        # 馬行に race_id でマッピング
        df["race_upset_rate_l12m"] = df["race_id"].map(race_upset_series)
    elif "race_upset_rate_l12m" not in df.columns:
        df["race_upset_rate_l12m"] = np.nan

    # ------------------------------------------------------------------
    # 26. class_jump_flag
    #     前走のクラス（race_level）に対する当走クラスの変化。
    #     昇格=+1、降格=-1、維持=0。
    #     前走クラスは shift(1) で取得するためリーク防止済み。
    #     当走の race_level（出走クラス）は事前公開情報のため参照可。
    # ------------------------------------------------------------------
    if "race_level" in df.columns and "class_jump_flag" not in df.columns:
        rl = pd.to_numeric(df["race_level"], errors="coerce")
        lag_rl = rl.groupby(df["ketto_num"], sort=False).shift(1)
        diff_rl = rl - lag_rl
        # +1: 昇格（race_level値が上昇）、-1: 降格、0: 同一
        df["class_jump_flag"] = np.where(
            lag_rl.isna(),
            np.nan,
            np.where(diff_rl > 0, 1.0, np.where(diff_rl < 0, -1.0, 0.0)),
        )
    elif "class_jump_flag" not in df.columns:
        df["class_jump_flag"] = np.nan

    # ------------------------------------------------------------------
    # 27. opening_week_flag
    #     競馬場の今開催期間で最初の2日間（開幕バイアス）を表すフラグ。
    #     3月のROI -20.6%の主因（開幕函館・阪神のコース替わり直後）。
    #     年×競馬場の最小日付との差分なので当日情報のみを使用（リークなし）。
    # ------------------------------------------------------------------
    if "course_code" in df.columns and "date" in df.columns:
        # 年×競馬場の最初の開催日を計算
        df["_race_year"] = pd.to_datetime(df["date"]).dt.year
        first_date_of_year_course = (
            df.groupby(["_race_year", "course_code"])["date"]
            .transform("min")
        )
        days_from_first = (
            pd.to_datetime(df["date"]) - pd.to_datetime(first_date_of_year_course)
        ).dt.days
        df["opening_week_flag"] = (days_from_first <= 7).astype("int8")
        df.drop(columns=["_race_year"], inplace=True, errors="ignore")

    if source_is_gpu and _GPU_AVAILABLE:
        try:
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
        except Exception:
            pass  # GPU 変換失敗時は pandas のまま返す

    return df


def remove_leak_features(df):
    leak_cols = list(dict.fromkeys(BASE_LEAK_COLS + PAST_EXTRA_LEAK_COLS))
    df = df.drop(columns=[c for c in leak_cols if c in df.columns])
    return df


def _add_interaction_features_v22(df: pd.DataFrame) -> pd.DataFrame:
    """v22 交差特徴量：Training-Serving Skew 解消のため、学習データに存在した
    交差特徴量を推論パイプラインでも再現する。

    生成する列:
    - tm_score_x_jk_style_dist  = tm_score × jockey_style_dist_win_rate
      (モデル gain 18.73% — 最重要の欠落列)
    - corner4_x_jk_style_dist   = corner4_normalized_lag1 × jockey_style_dist_win_rate
      (モデル gain 0.47%)
    - horse_course_dist_band_win_rate_v17 = horse_course_dist_band_win_rate の v17 退避列
      (add_course_dist_features_v18 実行後、v17 列が存在しない場合に NaN で補完)

    Note: corner_advance_rate / corner_position_change は実装ソースが存在しないため省略。
    going_heavy_aptitude_score / going_dirt_heavy_aptitude_score は
    _add_going_aptitude_features_v23 として別実装済み（gain合計 3.5%）。
    """
    # tm_score × jockey_style_dist_win_rate（両列が存在する場合のみ）
    if "tm_score" in df.columns and "jockey_style_dist_win_rate" in df.columns:
        df["tm_score_x_jk_style_dist"] = (
            df["tm_score"].astype("float32") * df["jockey_style_dist_win_rate"].astype("float32")
        )
    else:
        df["tm_score_x_jk_style_dist"] = np.nan

    # corner4_normalized_lag1 × jockey_style_dist_win_rate
    if "corner4_normalized_lag1" in df.columns and "jockey_style_dist_win_rate" in df.columns:
        df["corner4_x_jk_style_dist"] = (
            df["corner4_normalized_lag1"].astype("float32")
            * df["jockey_style_dist_win_rate"].astype("float32")
        )
    else:
        df["corner4_x_jk_style_dist"] = np.nan

    # horse_course_dist_band_win_rate_v17: v18 step が実行済みの場合は v18 列をコピー
    # （v18 と v17 は同一の horse_course_dist_band_win_rate 列を共有するため等価）
    if "horse_course_dist_band_win_rate_v17" not in df.columns:
        if "horse_course_dist_band_win_rate" in df.columns:
            df["horse_course_dist_band_win_rate_v17"] = df["horse_course_dist_band_win_rate"].astype("float32")
        else:
            df["horse_course_dist_band_win_rate_v17"] = np.nan

    return df


def _add_going_aptitude_features_v23(df: pd.DataFrame) -> pd.DataFrame:
    """v23: 馬の重馬場・不良馬場適性スコア（Training-Serving Skew 解消）

    v21 まで存在していたが v22 で誤って除外された 2 列を復元する。
    推論時の弱点（dirt_condition=3 ROI 82.5%, turf_condition=2/3/4 弱点）を改善するため
    add_going_condition_features (v7) で計算済みの Bayesian 勝率列を再利用する。

    生成する列:
        going_heavy_aptitude_score     -- 芝重馬場適性（horse_turf_heavy_win_rate の別名）
                                          条件: is_turf==1 & turf_condition in {3, 4}
                                          NaN: 芝重馬場の過去出走なし
        going_dirt_heavy_aptitude_score -- ダート重馬場適性（horse_dirt_heavy_win_rate の別名）
                                           条件: is_dirt==1 & dirt_condition >= 3
                                           NaN: ダート重馬場の過去出走なし

    注意: add_going_condition_features (v7) の実行後に呼ぶこと。
    前提列 horse_turf_heavy_win_rate / horse_dirt_heavy_win_rate が存在しない場合は NaN を設定。
    """
    # 芝重馬場適性: 芝重(turf_condition=3)+不良(turf_condition=4) の Bayesian 累積勝率
    # horse_turf_heavy_win_rate は v7 で条件 turf_condition>=3 かつ is_turf==1 で計算済み
    if "going_heavy_aptitude_score" not in df.columns:
        if "horse_turf_heavy_win_rate" in df.columns:
            df["going_heavy_aptitude_score"] = df["horse_turf_heavy_win_rate"].astype("float32")
        else:
            df["going_heavy_aptitude_score"] = np.nan

    # ダート重馬場適性: ダート重(dirt_condition=3) の Bayesian 累積勝率
    # horse_dirt_heavy_win_rate は v7 で条件 dirt_condition>=3 かつ is_dirt==1 で計算済み
    if "going_dirt_heavy_aptitude_score" not in df.columns:
        if "horse_dirt_heavy_win_rate" in df.columns:
            df["going_dirt_heavy_aptitude_score"] = df["horse_dirt_heavy_win_rate"].astype("float32")
        else:
            df["going_dirt_heavy_aptitude_score"] = np.nan

    return df


def create_pastfeatures_main(
    input_path=None,
    output_path=None,
    save=True,
    df=None,
    manifest_path=None,
    attach_pedigree=True,
    pedigree_output_dir=None,
    strict_pedigree=False,
    show_progress=False,
    heartbeat_sec=60,
    log_prefix="[pastfeatures]",
    prefer_bulk_asof=True,
    pedigree_cache=None,
    use_gpu=True,
):
    if df is None:
        inp = input_path or INPUT_PATH
        prefer = bool(use_gpu and _GPU_AVAILABLE)
        if prefer:
            try:
                df = read_features_basic(inp, prefer_gpu=True)
            except Exception as e:
                tqdm.write(
                    f"{log_prefix or '[pastfeatures]'} [warn] GPU parquet read failed "
                    f"({e}); fallback pandas+CPU read."
                )
                df = read_features_basic(inp, prefer_gpu=False)
        else:
            df = read_features_basic(inp, prefer_gpu=False)
        if use_gpu and _GPU_AVAILABLE and not _is_gpu_df(df):
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
    else:
        df = df.copy()
    if _is_gpu_df(df):
        try:
            df["date"] = cudf.to_datetime(df["date"])
        except Exception:
            pass
    else:
        df["date"] = pd.to_datetime(df["date"])
    df = _sort_for_pastfeatures(df)

    if use_gpu and _GPU_AVAILABLE:
        try:
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
            if show_progress:
                tqdm.write(f"{log_prefix} backend=gpu(cudf/cupy)")
        except Exception:
            try:
                normalized_df = _prepare_pandas_for_gpu(_to_pandas_df(df))
                df = _to_gpu_df(normalized_df)
                if show_progress:
                    tqdm.write(f"{log_prefix} backend=gpu(cudf/cupy, normalized)")
            except Exception as e:
                tqdm.write(
                    f"{log_prefix} [warn] GPU conversion failed, fallback to CPU: {e}"
                )
                df = _to_pandas_df(df)
    elif show_progress:
        tqdm.write(f"{log_prefix} backend=cpu(pandas/numpy)")

    _se_v11 = None
    _se_path = PROJECT_ROOT / "model_training/data/01_preprocessed/SE_preprocessed.parquet"
    if _se_path.exists():
        try:
            _se_v11 = pd.read_parquet(_se_path, columns=_SE_V11_COLS)
        except Exception as _e:
            tqdm.write(f"{log_prefix} [warn] SE v11 load failed ({_e}); prize/agari-course features skipped")

    pipeline = [
        (create_standardization_features, "Standardization"),
        (create_context_features, "Context features"),
        (create_race_relative_features, "Race relative"),
        (create_human_stats, "Human stats"),
        (create_lag_features, "Lag calculation"),
        (create_rolling_features, "Rolling calculation"),
        (
            lambda x: attach_pedigree_features(
                x,
                pedigree_output_dir=pedigree_output_dir,
                enable=attach_pedigree,
                show_progress=show_progress,
                heartbeat_sec=heartbeat_sec,
                log_prefix=f"{log_prefix}/pedigree",
                strict=strict_pedigree,
                prefer_bulk_asof=prefer_bulk_asof,
                starts_long_df=(pedigree_cache or {}).get("starts_long_df"),
                sibling_edges_df=(pedigree_cache or {}).get("sibling_edges_df"),
                dam_birth_year_series=(pedigree_cache or {}).get("dam_birth_year_series"),
                dam_age_df=(pedigree_cache or {}).get("dam_age_df"),
            ),
            "Pedigree leak-safe",
        ),
        (add_condition_specific_features, "Condition-specific features"),
        (add_surface_course_features, "Surface/course features v6"),
        (add_going_condition_features, "Going condition features v7"),
        (add_going_condition_v8_features, "Going condition features v8"),
        (add_going_condition_v9_features, "Going condition features v9"),
        (add_going_condition_v10_features, "Going condition features v10"),
        (lambda x: add_prize_features(x, _se_v11) if _se_v11 is not None else x, "Prize features v11"),
        (add_sire_stats_features, "Sire stats features v11"),
        (lambda x: add_agari_course_features(x, _se_v11) if _se_v11 is not None else x, "Agari course features v11"),
        (add_dirt_impute_features, "Dirt impute features v11"),
        # v18特徴量: 騎手/調教師の重賞限定勝率、馬×距離帯勝率（リーク防止済み: cumsum-currentパターン）
        (add_graded_jockey_trainer_features, "Graded jockey/trainer win rate v18"),
        (add_course_dist_features_v18, "Horse course/dist band win rate v18"),
        # v19特徴量: 調教マルチセッション傾向（直近3セッション time_1f 変化量・最終調教日からの日数）
        # HC.train_date < race_date を厳守（当日調教を含まない）
        (
            lambda x: add_training_trend_features(x, HC_PREPROCESSED_PATH)
            if HC_PREPROCESSED_PATH.exists()
            else x,
            "Training trend features v19",
        ),
        # =====================================================================
        # v22: Training-Serving Skew 解消（features_past_v21 との列数一致）
        # 以下のモジュールは build_features_v21.py のビルドパイプラインでは呼ばれていたが
        # 推論パイプライン（create_main_pastfeatures）では欠落していた特徴量群を追加する。
        # =====================================================================
        # Group A/B: 洋芝適性・芝稍重適性（v21 で追加された12列を推論側でも生成）
        (add_youshiba_features, "v21 Group A: Youshiba win rate"),
        (add_sire_youshiba_features, "v21 Group A: Sire youshiba win rate"),
        (add_kokai_koban_features, "v21 Group A: Kokai/koban win rate"),
        (add_horse_soft_turf_features, "v21 Group B: Horse soft turf features"),
        (add_sire_soft_turf_features, "v21 Group B: Sire soft turf win rate"),
        # Group C: スピード指数（speed_index_course_adj はリーク列のため除外済み）
        # SE_preprocessed.parquet から racetime を参照して過去走分のみ計算する
        # _se_v11 に racetime 列が含まれている（_SE_V11_COLS を参照）
        (
            lambda x: add_speed_index_features(x, _se_path)
            if _se_path.exists()
            else x,
            "v21 Group C: Speed index features (past runs only, no leak)",
        ),
        # Group D: ペース圧力×脚質交差
        (add_pace_dist_style_features, "v21 Group D: Pace dist style win rate"),
        # v16〜v19 で追加されたが推論パイプラインに未追加だったモジュール群（41列相当）
        # 重賞勝率・格変動フラグ（add_graded_jockey_trainer_features とは別の関数）
        (add_graded_features, "v16: Graded race win rate and grade step flags"),
        # ニックス勝率・芝長距離父馬勝率
        (add_nick_win_rate, "v20: Nick win rate (sire x dam_sire)"),
        (add_sire_long_turf_feature, "v20: Sire long turf win rate"),
        # 芝稍重補完特徴量・父馬芝重不良適性
        (add_soft_turf_impute_features, "v16: Soft turf impute features"),
        # 上がり3F安定性・傾向（agari3f_rank_in_race_lag1 が前提）
        (add_agari_stability_features, "v15: Agari 3F stability features"),
        # 性別×距離帯適性・騸馬年齢
        (add_sex_age_features, "v16: Sex/age deviation features"),
        # ペース構成（レース内の逃げ・先行比率）
        (add_pace_field_features, "v15: Pace field composition features"),
        # 騎手・調教師の回り×馬場別勝率
        (add_turn_surface_features, "v17: Turn surface win rate features"),
        # 脚質×コース相性
        (add_style_course_features, "v17: Style course compatibility features"),
        # 脚質×距離帯適性
        (add_style_dist_features, "v17: Style dist band features"),
        # 性別×芝ダート×距離帯（v16）
        (add_surface_dist_band_features, "v16: Sex surface dist band features"),
        # =====================================================================
        # v22: 交差特徴量（Training-Serving Skew 残7列のうち実装可能な3列）
        # tm_score_x_jk_style_dist (gain 18.73%) = tm_score × jockey_style_dist_win_rate
        # corner4_x_jk_style_dist  (gain  0.47%) = corner4_normalized_lag1 × jockey_style_dist_win_rate
        # horse_course_dist_band_win_rate_v17 = v18適用前の旧バージョン退避列（現在はv18が有効）
        # corner_advance_rate / corner_position_change は実装ソースが存在しないため省略。
        # going_heavy_aptitude_score / going_dirt_heavy_aptitude_score は v23 で復元済み（下記）。
        # =====================================================================
        (
            lambda df: _add_interaction_features_v22(df),
            "v22: Interaction features (tm_score×jk_style_dist, corner4×jk_style_dist)",
        ),
        # =====================================================================
        # v23: 重馬場適性スコア復元（v21 → v22 で誤って削除された 2 列）
        # going_heavy_aptitude_score     (gain 3.08%) = horse_turf_heavy_win_rate の別名
        # going_dirt_heavy_aptitude_score (gain 0.40%) = horse_dirt_heavy_win_rate の別名
        # add_going_condition_features (v7) 実行後に呼ぶ必要があるため、ここで追加。
        # =====================================================================
        (
            lambda df: _add_going_aptitude_features_v23(df),
            "v23: Going aptitude score restoration (heavy/dirt-heavy win rate alias)",
        ),
        (
            add_rank1_context_v24_features,
            "v24: Rank1 context (traffic, handicap, field competition)",
        ),
        # =====================================================================
        # v25: コーナー位置取り変化特徴量
        # corner_position_change_lag1  = 前走の4角順位 - 1角順位（追い込み方向=負値）
        # corner_advance_rate          = abs(位置変化) / 頭数（0〜1の追い込み指数）
        # =====================================================================
        (
            add_corner_v25_features,
            "v25: Corner position change (advance rate, position delta)",
        ),
        # =====================================================================
        # v25_odds: オッズ乖離特徴量
        # odds_rank_divergence  = AIスコア順位 - 市場人気順位（正=AI高評価・低人気）
        # field_odds_entropy    = レース内オッズ分布のShannonエントロピー
        # =====================================================================
        (
            add_odds_divergence_features,
            "v25_odds: Odds rank divergence and field entropy",
        ),
        # =====================================================================
        # v26: 馬場適性 delta 特徴量（良 vs 重/稍重の差分ベクトル）
        # =====================================================================
        (
            _add_going_delta_features_v26,
            "v26: Going delta aptitude features",
        ),
        # =====================================================================
        # v27: トラックバリアント + 馬場補正済み tm_score
        # =====================================================================
        (
            _add_track_variant_features_v27,
            "v27: Daily track variant and surface-adjusted tm_score",
        ),
        (remove_leak_features, "Leak removal"),
    ]

    n_steps = len(pipeline)
    with tqdm(
        pipeline,
        desc="Past features",
        dynamic_ncols=True,
        smoothing=0.05,
    ) as pbar:
        for i, (func, desc) in enumerate(pbar):
            pbar.set_description(f"Step {i + 1}/{n_steps}: {desc}")
            step_start = time.perf_counter()
            rows_in = len(df)

            df = func(df)

            step_elapsed = time.perf_counter() - step_start
            log_step(
                desc,
                rows_in=rows_in,
                rows_out=len(df),
                started_at=step_start,
                prefix=log_prefix,
            )
            if show_progress:
                tqdm.write(
                    f"{log_prefix} [{desc}] rows_in={rows_in} rows_out={len(df)} "
                    f"cols={len(df.columns)} elapsed={step_elapsed:.2f}s"
                )
            pbar.set_postfix_str(
                f"rows={len(df):,} last={step_elapsed:.1f}s",
                refresh=True,
            )
            if desc == "Pedigree leak-safe":
                # 大型 tx / merge_asof 直後の VRAM を後続ステップに残さない
                if _is_gpu_df(df):
                    gc.collect()
                    _free_gpu_memory()
            else:
                _free_gpu_memory()

    if _is_gpu_df(df):
        df = df.to_pandas()
        _free_gpu_memory()

    tqdm.write(f"{log_prefix} Starting dtype optimization...")
    step_start = time.perf_counter()
    rows_in = len(df)
    df = optimize_dtypes(df)
    log_step(
        "Dtype optimization",
        rows_in=rows_in,
        rows_out=len(df),
        started_at=step_start,
        prefix=log_prefix,
    )

    if save:
        out_p = Path(output_path or OUTPUT_PATH)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        try:
            df.to_parquet(out_p.with_suffix(".parquet"), index=False)
        except Exception as e:
            tqdm.write(f"[warn] Failed to save parquet next to {out_p}: {e}")
        _save_manifest(manifest_path or MANIFEST_PATH, _build_manifest(df))
    return df


def _add_going_delta_features_v26(df: pd.DataFrame) -> pd.DataFrame:
    from model_training.src.builders.going_delta import add_going_delta_features

    return add_going_delta_features(df)


def _add_track_variant_features_v27(df: pd.DataFrame) -> pd.DataFrame:
    from model_training.src.builders.track_variant import add_track_variant_features

    return add_track_variant_features(df)


def create_main_pastfeatures(
    *,
    attach_pedigree: bool = True,
    strict_pedigree: bool = True,
):
    # 予測用はHistoryと結合してLagを計算する（Parquet があれば優先）
    hist_df = read_features_basic(INPUT_PATH, prefer_gpu=False)
    main_df = read_features_basic(MAIN_FEATURES_BASIC, prefer_gpu=False)

    hist_df["_is_main"] = 0
    main_df["_is_main"] = 1

    # ターゲット馬の履歴のみに絞って結合（メモリ節約）
    target_horses = main_df["ketto_num"].unique()
    hist_df = hist_df[hist_df["ketto_num"].isin(target_horses)]

    combined = pd.concat([hist_df, main_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # 特徴量作成（共通処理）— 学習時と同じ attach_pedigree を CLI から渡す
    combined = create_pastfeatures_main(
        df=combined,
        save=False,
        attach_pedigree=attach_pedigree,
        strict_pedigree=strict_pedigree,
    )

    # main のみ抽出
    out_df = combined[combined["_is_main"] == 1].drop(columns=["_is_main"])
    out_csv = PROJECT_ROOT / "model_training/data/02_features/main_features_past.csv"
    out_pq = out_csv.with_suffix(".parquet")
    out_df.to_csv(out_csv, index=False)
    # v18モデルは parquet を優先読み込みするため parquet も必ず保存する
    # （run_today_prediction_pipeline が prefer_parquet=True 時に parquet を使用）
    try:
        out_df.to_parquet(out_pq, index=False)
    except Exception as _e:
        tqdm.write(f"[warn] main_features_past.parquet の保存に失敗しました: {_e}")
    return out_df


def _parse_years_arg(years: str | None) -> list[int] | None:
    if not years:
        return None
    vals = []
    for p in str(years).split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    return vals or None


def create_pastfeatures_chunked_by_year(
    input_path=None,
    output_path=None,
    manifest_path=None,
    chunks_dir=None,
    resume=True,
    years: list[int] | None = None,
    attach_pedigree=True,
    strict_pedigree=True,
    show_progress=True,
    heartbeat_sec=60,
    log_prefix="[pastfeatures/chunked]",
    prefer_bulk_asof=True,
):
    pedigree_cache = None
    if attach_pedigree:
        try:
            source_output_dir = Path(PED_OUTPUT_DIR)
            pedigree_cache = {
                "starts_long_df": load_or_build_starts_cumulative(source_output_dir),
                "sibling_edges_df": load_sibling_edges(source_output_dir),
                "dam_birth_year_series": load_dam_birth_year(source_output_dir),
                "dam_age_df": load_dam_age_features(source_output_dir),
            }
            tqdm.write(f"{log_prefix} pedigree cache loaded for chunked execution")
        except Exception as e:
            tqdm.write(f"{log_prefix} [warn] Failed to preload pedigree cache: {e}")

    in_path = Path(input_path or INPUT_PATH)
    out_path = Path(output_path or OUTPUT_PATH)
    man_path = Path(manifest_path or MANIFEST_PATH)
    chunks_dir = Path(
        chunks_dir
        or (PROJECT_ROOT / "model_training/data/02_features/chunks/features_past_by_year")
    )
    chunks_dir.mkdir(parents=True, exist_ok=True)

    df_all = read_features_basic(in_path, prefer_gpu=False)
    if "date" not in df_all.columns:
        raise ValueError(f"Input data has no date column: {in_path}")
    if "year" not in df_all.columns:
        raise ValueError(f"Input data has no year column: {in_path}")
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["year"] = pd.to_numeric(df_all["year"], errors="coerce").astype("Int64")
    df_all = df_all.sort_values(["date", "race_id"]).reset_index(drop=True)
    df_all["__chunk_row_id"] = np.arange(len(df_all), dtype=np.int64)

    target_years = sorted([int(y) for y in df_all["year"].dropna().unique().tolist()])
    if years:
        years_set = set(int(y) for y in years)
        target_years = [y for y in target_years if y in years_set]
    if not target_years:
        raise ValueError("No target years found to process.")

    tqdm.write(f"{log_prefix} target_years={target_years}")

    chunk_files: list[Path] = []
    for y in target_years:
        chunk_path = chunks_dir / f"features_past_{y}.csv"
        if resume and chunk_path.exists():
            tqdm.write(f"{log_prefix} skip year={y} (exists: {chunk_path.name})")
            chunk_files.append(chunk_path)
            continue

        tqdm.write(f"{log_prefix} start year={y}")
        context = df_all[df_all["year"] <= y].copy()
        context["__chunk_target"] = (context["year"] == y).astype(np.int8)

        processed = create_pastfeatures_main(
            df=context,
            save=False,
            attach_pedigree=attach_pedigree,
            strict_pedigree=strict_pedigree,
            show_progress=show_progress,
            heartbeat_sec=heartbeat_sec,
            log_prefix=f"{log_prefix}/y{y}",
            prefer_bulk_asof=prefer_bulk_asof,
            pedigree_cache=pedigree_cache,
        )

        out_chunk = processed[processed["__chunk_target"] == 1].copy()
        out_chunk = out_chunk.drop(columns=["__chunk_target"], errors="ignore")
        out_chunk = out_chunk.sort_values("__chunk_row_id").reset_index(drop=True)
        out_chunk.to_csv(chunk_path, index=False)
        try:
            out_chunk.to_parquet(chunk_path.with_suffix(".parquet"), index=False)
        except Exception as e:
            tqdm.write(f"[warn] Failed to save chunk parquet for year={y}: {e}")

        chunk_files.append(chunk_path)
        tqdm.write(
            f"{log_prefix} done year={y} rows={len(out_chunk)} cols={len(out_chunk.columns)}"
        )

    if not chunk_files:
        raise ValueError("No chunk files available to merge.")

    def _read_chunk(p: Path):
        pq = _features_basic_parquet_path(p)
        if pq.is_file():
            return pd.read_parquet(pq)
        return read_csv_optimized(p)

    merged = pd.concat([_read_chunk(p) for p in sorted(chunk_files)], ignore_index=True)
    merged = merged.sort_values("__chunk_row_id").reset_index(drop=True)
    merged = merged.drop(columns=["__chunk_row_id"], errors="ignore")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    try:
        merged.to_parquet(out_path.with_suffix(".parquet"), index=False)
    except Exception as e:
        tqdm.write(f"[warn] Failed to save merged parquet next to {out_path}: {e}")
    _save_manifest(man_path, _build_manifest(merged))
    tqdm.write(
        f"{log_prefix} merged rows={len(merged)} cols={len(merged.columns)} -> {out_path}"
    )
    return merged


def update_pastfeatures(
    *,
    mode: str = "all",
    state_path: str | None = None,
) -> dict:
    """
    過去参照特徴量（past）を更新する統一エントリポイント。

    - mode="train": features_basic から features_past を再生成
    - mode="main": main_features_basic から main_features_past を再生成
    - mode="all": 上記両方

    Args:
        mode: "train" | "main" | "all"
        state_path: 状態ファイルのパス（省略時: model_training/data/state/pastfeatures_last_update.json）

    Returns:
        保存した状態 dict
    """
    if state_path is None:
        state_file = (
            PROJECT_ROOT
            / "model_training"
            / "data"
            / "state"
            / "pastfeatures_last_update.json"
        )
    else:
        state_file = Path(state_path)

    mode = str(mode).lower().strip()
    if mode not in {"train", "main", "all"}:
        raise ValueError('mode must be one of: "train", "main", "all"')

    out: dict = {}

    if mode in {"train", "all"}:
        create_pastfeatures_main()
        out["train_pastfeatures_updated"] = True

    if mode in {"main", "all"}:
        create_main_pastfeatures()
        out["main_pastfeatures_updated"] = True

    return update_state(state_file, mode=mode, updates=out)



# --- Pedigree (leak-safe row features) ---

STARTS_LONG_CACHE_SUBDIR = "model_training/data/02_features/cache"
STARTS_LONG_CACHE_FILENAME = "starts_long_cumulative.parquet"
STARTS_LONG_CACHE_META = "starts_long_cumulative.meta.json"


def _cache_paths() -> tuple[Path, Path]:
    base = PROJECT_ROOT / STARTS_LONG_CACHE_SUBDIR
    return base / STARTS_LONG_CACHE_FILENAME, base / STARTS_LONG_CACHE_META


def fingerprint_race_sources(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    se = sorted((output_dir / "race_se").glob("race_se_*.csv"))
    ra = sorted((output_dir / "race_ra").glob("race_ra_*.csv"))

    def ent(p: Path) -> dict[str, Any]:
        st = p.stat()
        return {"path": str(p.resolve()), "mtime_ns": st.st_mtime_ns, "size": st.st_size}

    return {"race_se": [ent(x) for x in se], "race_ra": [ent(x) for x in ra]}


def jv_registration_to_int64_pd(s: pd.Series) -> pd.Series:
    """JV 登録キーを int64 で統一（先頭ゼロ文字列も数値化で整合）。"""
    x = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
    return x


def load_or_build_starts_cumulative(
    output_dir: Path | str,
    *,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """load_starts_long + add_cumulative をキャッシュし、入力 CSV が新しければ再計算。"""
    output_dir = Path(output_dir)
    pq_path, meta_path = _cache_paths()
    fp = fingerprint_race_sources(output_dir)
    pq_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and pq_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp:
                return pd.read_parquet(pq_path)
        except Exception:
            pass

    s = load_starts_long(output_dir)
    s = add_cumulative_per_horse(s)
    meta = {"fingerprint": fp, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    s.to_parquet(pq_path, index=False)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return s


def _read_yearly_csvs(
    dir_path: Path, prefix: str, usecols: list[str] | None
) -> pd.DataFrame:
    files = sorted(dir_path.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files matched: {dir_path / (prefix + '_*.csv')}")
    return pd.concat(
        [read_csv_optimized(p, usecols=usecols) for p in files], ignore_index=True
    )


def load_starts_long(output_dir: Path) -> pd.DataFrame:
    se_cols = [
        "ketto_num",
        "finish_rank",
        "abnormal_code",
        "hon_shokin",
        "year",
        "month_day",
        "course_code",
        "kai",
        "nichi",
        "race_num",
    ]
    ra_cols = [
        "year",
        "month_day",
        "course_code",
        "kai",
        "nichi",
        "race_num",
        "track_code",
        "turf_condition",
        "dirt_condition",
        "distance",
    ]
    se = _read_yearly_csvs(output_dir / "race_se", "race_se", se_cols)
    ra = _read_yearly_csvs(output_dir / "race_ra", "race_ra", ra_cols)
    keys = ["year", "month_day", "course_code", "kai", "nichi", "race_num"]
    se = se.merge(
        ra[keys + ["track_code", "turf_condition", "dirt_condition", "distance"]],
        on=keys,
        how="left",
    )
    for c in ["abnormal_code", "track_code", "turf_condition", "dirt_condition"]:
        se[c] = se[c].astype(str).str.strip()
    kn = jv_registration_to_int64_pd(se["ketto_num"])
    se = se.loc[kn.notna()].copy()
    se["ketto_num"] = kn.loc[se.index].astype("int64")
    se["finish_rank"] = pd.to_numeric(se["finish_rank"], errors="coerce")
    se["hon_shokin"] = pd.to_numeric(se["hon_shokin"], errors="coerce").fillna(0).astype(
        "float64"
    )
    se["distance"] = pd.to_numeric(se["distance"], errors="coerce")
    se["abnormal_flag"] = np.where(se["abnormal_code"].eq("0"), np.int8(0), np.int8(1))
    se["is_valid_start"] = (se["abnormal_flag"] == 0).astype(np.int8)
    se["is_win"] = ((se["finish_rank"] == 1) & (se["is_valid_start"] == 1)).astype(np.int8)
    se["is_top3"] = ((se["finish_rank"] <= 3) & (se["is_valid_start"] == 1)).astype(
        np.int8
    )
    track = se["track_code"].astype(str).str.strip()
    se["is_turf"] = track.str.startswith("1")
    se["is_dirt"] = track.str.startswith("2")
    se["turf_t3"] = (se["is_top3"].astype(bool) & se["is_turf"]).astype(np.int8)
    se["turf_st"] = (
        se["is_valid_start"].astype(int).eq(1) & se["is_turf"]
    ).astype(np.int8)
    se["dirt_t3"] = (se["is_top3"].astype(bool) & se["is_dirt"]).astype(np.int8)
    se["dirt_st"] = (
        se["is_valid_start"].astype(int).eq(1) & se["is_dirt"]
    ).astype(np.int8)
    for name, code in [("soft", "2"), ("heavy", "3"), ("bad", "4")]:
        hit = se["turf_condition"].eq(code) | se["dirt_condition"].eq(code)
        se[f"{name}_st"] = (
            se["is_valid_start"].astype(int).eq(1) & hit
        ).astype(np.int8)
        se[f"{name}_t3"] = (se["is_top3"].astype(int).eq(1) & hit).astype(np.int8)
    se["race_dt"] = pd.to_datetime(
        se["year"].astype(str) + se["month_day"].astype(str).str.zfill(4),
        format="%Y%m%d",
        errors="coerce",
    )
    se["win_dist"] = np.where(
        (se["is_win"] == 1) & (se["is_valid_start"] == 1), se["distance"], np.nan
    )
    return se.sort_values(["ketto_num", "race_dt"]).reset_index(drop=True)


def load_sibling_edges(output_dir: Path) -> pd.DataFrame:
    sk = read_csv_optimized(
        output_dir / "blod_sk" / "blod_sk.csv",
        usecols=["ketto_num", "p_dam"],
    )
    hid = jv_registration_to_int64_pd(sk["ketto_num"])
    sk["horse_id"] = hid.astype("Int64")
    sk["dam_id"] = sk["p_dam"].astype(str).str.strip()
    sk = sk.loc[sk["horse_id"].notna()].copy()
    sk["horse_id"] = sk["horse_id"].astype("int64")
    base = sk[["horse_id", "dam_id"]].drop_duplicates()
    sib = base[["horse_id", "dam_id"]].rename(columns={"horse_id": "sibling_id"})
    net = base.merge(sib, on="dam_id", how="left")
    net = net[net["horse_id"] != net["sibling_id"]].copy()
    return net[["horse_id", "sibling_id"]].drop_duplicates()


def load_dam_birth_year(output_dir: Path) -> pd.Series:
    hn = read_csv_optimized(
        output_dir / "blod_hn" / "blod_hn.csv",
        usecols=["breeding_reg_num", "birth_year"],
    )
    hn["breeding_reg_num"] = hn["breeding_reg_num"].astype(str).str.strip()
    return hn.drop_duplicates("breeding_reg_num").set_index("breeding_reg_num")[
        "birth_year"
    ]


def load_dam_age_features(output_dir: Path) -> pd.DataFrame:
    skd = read_csv_optimized(
        output_dir / "blod_sk" / "blod_sk.csv",
        usecols=["ketto_num", "p_dam"],
    )
    kk = jv_registration_to_int64_pd(skd["ketto_num"])
    skd = skd.loc[kk.notna()].copy()
    skd["ketto_num"] = kk.loc[skd.index].astype("int64")
    skd["dam_id"] = skd["p_dam"].astype(str).str.strip()
    skd = skd.drop_duplicates("ketto_num")
    skd["foal_year_proxy"] = (skd["ketto_num"] // 10**6).astype("float64")
    return skd


def _safe_div(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    out = np.full_like(n, np.nan, dtype=np.float64)
    return np.divide(n, d, out=out, where=d > 0)


SHRINK_K_DEFAULT = 5.0
PRIOR_SIBLING_WIN_RATE = 0.10
PRIOR_SIBLING_TOP3_RATE = 0.22
PRIOR_WIN_DIST_M = 1600.0


def _shrink_toward_prior(
    rate: np.ndarray,
    n_eff: np.ndarray,
    prior: float,
    k: float,
) -> np.ndarray:
    n_eff = np.asarray(n_eff, dtype=float)
    rate = np.asarray(rate, dtype=float)
    w = n_eff / (n_eff + k)
    out = np.full(np.shape(rate), np.nan, dtype=float)
    ok = (n_eff > 0) & np.isfinite(rate)
    out[ok] = prior + (rate[ok] - prior) * w[ok]
    return out


def add_cumulative_per_horse(s: pd.DataFrame) -> pd.DataFrame:
    out = s.sort_values(["ketto_num", "race_dt"]).copy()
    g = out.groupby("ketto_num", sort=False)
    out["prior_start_count"] = g.cumcount().astype(np.int32)
    out["cum_wins"] = g["is_win"].cumsum().astype(np.int32)
    out["cum_starts"] = g["is_valid_start"].cumsum().astype(np.int32)
    out["cum_money"] = g["hon_shokin"].cumsum().astype(np.float64)
    for cname in ["turf_t3", "turf_st", "dirt_t3", "dirt_st", "soft_t3", "soft_st"]:
        out[f"cum_{cname}"] = g[cname].cumsum().astype(np.int32)
    for cname in ["heavy_t3", "heavy_st", "bad_t3", "bad_st"]:
        out[f"cum_{cname}"] = g[cname].cumsum().astype(np.int32)
    z = out["win_dist"].fillna(0).groupby(out["ketto_num"], sort=False).cumsum()
    out["cum_win_dist_sum"] = z.astype(np.float64)
    return out


def _asof_prep_tx_sort(
    tx: pd.DataFrame,
    *,
    tie_cols_ml: list[str],
) -> pd.DataFrame:
    sort_keys = ["sibling_id", "target_dt"] + tie_cols_ml + ["_row_id"]
    tie_r = "__asof_r_tie_within_dt"
    r = tx.copy()
    r[tie_r] = r.groupby(["sibling_id", "target_dt"], sort=False).cumcount().astype(np.int64)
    return r.sort_values(sort_keys + [tie_r], kind="mergesort").reset_index(drop=True)


def _asof_prep_right_sort(right: pd.DataFrame) -> pd.DataFrame:
    r = right.sort_values(["sibling_id", "race_dt"], kind="mergesort").reset_index(
        drop=True
    )
    r["__asof_r_tie"] = r.groupby(["sibling_id", "race_dt"], sort=False).cumcount().astype(
        np.int64
    )
    return r.sort_values(["sibling_id", "race_dt", "__asof_r_tie"], kind="mergesort")


def _merge_asof_sibling_edges_cpu(
    tx: pd.DataFrame,
    right: pd.DataFrame,
    *,
    prefer_bulk: bool = True,
    show_progress: bool = False,
    bulk_chunk_siblings: int = 2000,
) -> pd.DataFrame:
    tie_cols_ml = [c for c in ("race_id", "horse_num", "umaban") if c in tx.columns]

    tx_s = _asof_prep_tx_sort(tx, tie_cols_ml=tie_cols_ml)
    right_s = _asof_prep_right_sort(right)
    right_s = right_s.drop(columns=["__asof_r_tie"], errors="ignore")
    tx_s = tx_s.drop(
        columns=[c for c in tx_s.columns if c.startswith("__asof")], errors="ignore"
    )
    if tx_s.empty:
        return pd.DataFrame()
    if prefer_bulk and not show_progress:
        try:
            merged = pd.merge_asof(
                tx_s,
                right_s,
                left_on="target_dt",
                right_on="race_dt",
                by="sibling_id",
                direction="backward",
            )
            return merged
        except (ValueError, TypeError):
            pass
    if prefer_bulk and show_progress:
        try:
            from tqdm.auto import tqdm

            tx_s2 = tx_s
            right_s2 = right_s
            sibling_ids = tx_s2["sibling_id"].dropna().unique().tolist()
            if not sibling_ids:
                return pd.DataFrame()
            chunks: list[pd.DataFrame] = []
            pbar = tqdm(
                total=len(sibling_ids),
                desc="pedigree merge_asof",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            for i in range(0, len(sibling_ids), max(1, int(bulk_chunk_siblings))):
                ids = sibling_ids[i : i + max(1, int(bulk_chunk_siblings))]
                tx_chunk = tx_s2[tx_s2["sibling_id"].isin(ids)]
                if tx_chunk.empty:
                    pbar.update(len(ids))
                    continue
                right_chunk = right_s2[right_s2["sibling_id"].isin(ids)]
                if right_chunk.empty:
                    pbar.update(len(ids))
                    continue
                chunks.append(
                    pd.merge_asof(
                        tx_chunk,
                        right_chunk,
                        left_on="target_dt",
                        right_on="race_dt",
                        by="sibling_id",
                        direction="backward",
                    )
                )
                pbar.update(len(ids))
            pbar.close()
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        except Exception:
            pass
    chunks: list[pd.DataFrame] = []
    _grp = tx.groupby("sibling_id", sort=False)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            _grp = tqdm(
                _grp,
                total=tx["sibling_id"].nunique(),
                desc="pedigree merge_asof(fallback)",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )
        except ImportError:
            pass
    for sid, g in _grp:
        rr = right.loc[right["sibling_id"] == sid]
        if rr.empty:
            continue
        chunks.append(
            pd.merge_asof(
                g.sort_values("target_dt", kind="mergesort"),
                rr,
                left_on="target_dt",
                right_on="race_dt",
                direction="backward",
            )
        )
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _merge_asof_sibling_edges_gpu(
    tx: Any,
    right: Any,
    *,
    tie_cols_ml: list[str],
) -> Any:
    if tx.empty:
        return cudf.DataFrame()
    tcol = "__asof_r_tie_within_dt"
    tt = tx.copy()
    tt[tcol] = tt.groupby(["sibling_id", "target_dt"]).cumcount().astype("int64")
    sort_keys = ["sibling_id", "target_dt"] + tie_cols_ml + ["_row_id", tcol]
    tx_s = tt.sort_values(sort_keys).reset_index(drop=True)

    rr = right.copy()
    rr["__asof_r_tie"] = rr.groupby(["sibling_id", "race_dt"]).cumcount().astype(
        "int64"
    )
    right_s = rr.sort_values(["sibling_id", "race_dt", "__asof_r_tie"]).reset_index(
        drop=True
    )
    kwargs = dict(
        left_on="target_dt",
        right_on="race_dt",
        by="sibling_id",
        direction="backward",
    )
    try:
        merged = cudf.merge_asof(tx_s, right_s, allow_exact_matches=True, **kwargs)
    except TypeError:
        merged = cudf.merge_asof(tx_s, right_s, **kwargs)
    merged = merged.drop(columns=[tcol, "__asof_r_tie"], errors="ignore")
    return merged


def _merge_asof_sibling_edges(
    tx: Any,
    right: Any,
    *,
    prefer_bulk: bool = True,
    show_progress: bool = False,
    bulk_chunk_siblings: int = 2000,
) -> Any:
    if _is_gpu_df(tx) or _is_gpu_df(right):
        if not _GPU_AVAILABLE:
            raise RuntimeError("cuDF が利用できません。")
        tie_cols_ml: list[str] = []
        for c in ("race_id", "horse_num", "umaban"):
            if c in tx.columns:
                tie_cols_ml.append(c)
        return _merge_asof_sibling_edges_gpu(tx, right, tie_cols_ml=tie_cols_ml)
    return _merge_asof_sibling_edges_cpu(
        tx,
        right,
        prefer_bulk=prefer_bulk,
        show_progress=show_progress,
        bulk_chunk_siblings=bulk_chunk_siblings,
    )


def _surf_from_track(tc: pd.Series) -> pd.Series:
    t = tc.astype(str).str.strip()
    return pd.Series(
        np.where(
            t.str.startswith("1"),
            1.0,
            np.where(t.str.startswith("2"), 2.0, np.nan),
        ),
        index=tc.index,
    )


def _surf_from_track_gpu(tc: Any) -> Any:
    s = tc.astype(str).fillna("")
    s = s.str.strip()
    p1 = s.str.startswith("1").fillna(False)
    p2 = s.str.startswith("2").fillna(False)
    neither = (~p1) & (~p2)
    out = (
        p1.astype("float64") * 1.0 + (~p1 & p2).astype("float64") * 2.0
    )
    return out.mask(neither)


def _ketto_join_key_pd(col: pd.Series) -> pd.Series:
    """JV 統一キー（欠損は NA）。"""
    return pd.to_numeric(col.astype(str).str.strip(), errors="coerce").astype("Int64")


def _apply_sibling_money_rel_z_pd(
    out: pd.DataFrame,
    *,
    train_year_cut: int,
    raw_col: str = "sibling_avg_money_ls",
) -> pd.DataFrame:
    raw = out[raw_col]
    train_mask = out["year"] < train_year_cut
    stats = (
        out.loc[train_mask, ["year", raw_col]]
        .groupby("year", observed=True)[raw_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "_money_ym", "std": "_money_ys"})
        .reset_index()
    )
    out = out.merge(stats, on="year", how="left")
    ok_year = out["_money_ys"].notna() & (out["_money_ys"] > 0)
    z_vals = np.where(
        ok_year.to_numpy(dtype=bool),
        (raw.values - out["_money_ym"].values) / out["_money_ys"].values,
        np.nan,
    )
    out["sibling_money_rel_z"] = z_vals
    mu_all = float(out.loc[train_mask, raw_col].mean())
    sd_all = float(out.loc[train_mask, raw_col].std(ddof=0))
    fallback_sd = sd_all if sd_all > 0 else 1.0
    need_fb = out["sibling_money_rel_z"].isna() & raw.notna()
    out.loc[need_fb, "sibling_money_rel_z"] = (
        raw.loc[need_fb] - mu_all
    ) / fallback_sd
    return out.drop(columns=["_money_ym", "_money_ys"], errors="ignore")


def _apply_sibling_money_rel_z_cudf(
    out: Any,
    *,
    train_year_cut: int,
    raw_col: str = "sibling_avg_money_ls",
) -> Any:
    tcut = int(train_year_cut)
    train_part = out.loc[out["year"] < tcut, ["year", raw_col]]
    stats = (
        train_part.groupby("year", observed=True)[raw_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "_money_ym", "std": "_money_ys"})
    )
    merged = out.merge(stats, on="year", how="left")
    ok_year = merged["_money_ys"].notna() & (merged["_money_ys"] > 0)
    ym = merged["_money_ym"].astype("float64")
    ys = merged["_money_ys"].astype("float64")
    r = merged[raw_col].astype("float64")
    z_year = ((r - ym) / ys).mask(~ok_year)
    merged["sibling_money_rel_z"] = z_year
    tr = merged.loc[merged["year"] < tcut, raw_col]
    mu_all = float(tr.mean()) if len(tr) else float("nan")
    sd_all = float(tr.std(ddof=0)) if len(tr) else float("nan")
    fallback_sd = sd_all if sd_all > 0 else 1.0
    need_fb = merged["sibling_money_rel_z"].isna() & merged[raw_col].notna()
    fb = (merged[raw_col].astype("float64") - mu_all) / fallback_sd
    merged["sibling_money_rel_z"] = fb.where(need_fb, merged["sibling_money_rel_z"])
    merged = merged.drop(columns=["_money_ym", "_money_ys"], errors="ignore")
    return merged


def _cup_shrink(
    rate: Any,
    n_eff: Any,
    prior: float,
    k: float,
) -> Any:
    n = n_eff.astype("float64").fillna(0).to_cupy()
    r = rate.astype("float64").fillna(cp.nan).to_cupy()
    out = cp.full_like(r, cp.nan, dtype=cp.float64)
    kk = float(k)
    pp = float(prior)
    ok = (n > 0) & cp.isfinite(r)
    w = n / (n + kk)
    out = cp.where(ok, pp + (r - pp) * w, out)
    return cudf.Series(out, index=rate.index)


def _cup_safe_div(numer: Any, denom: Any) -> Any:
    a = numer.astype("float64")
    b = denom.astype("float64")
    return (a / b).where((b > 0) & (~b.isna()))


def _resolve_starts_pandas(
    output_dir: Path,
    *,
    starts_long_df: pd.DataFrame | None,
    force_reload_src: bool = False,
) -> pd.DataFrame:
    if starts_long_df is not None:
        s = starts_long_df.copy()
    else:
        s = (
            load_starts_long(output_dir)
            if force_reload_src
            else load_or_build_starts_cumulative(output_dir)
        )
    if "cum_wins" not in s.columns:
        s = add_cumulative_per_horse(s)
    return s


def _attach_leak_safe_pedigree_cpu(
    df_ml: pd.DataFrame,
    output_dir: Path,
    *,
    train_year_cut: int = 2024,
    shrink_k: float = SHRINK_K_DEFAULT,
    shrink_prior_win: float = PRIOR_SIBLING_WIN_RATE,
    shrink_prior_top3: float = PRIOR_SIBLING_TOP3_RATE,
    shrink_prior_win_dist: float = PRIOR_WIN_DIST_M,
    prefer_bulk_asof: bool = True,
    show_progress: bool = False,
    heartbeat_sec: int = 60,
    log_prefix: str = "[pedigree]",
    starts_long_df: pd.DataFrame | None = None,
    sibling_edges_df: pd.DataFrame | None = None,
    dam_birth_year_series: pd.Series | None = None,
    dam_age_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    start_ts = time.perf_counter()
    last_hb = start_ts

    def _heartbeat(msg: str, force: bool = False) -> None:
        nonlocal last_hb
        now = time.perf_counter()
        if force or (heartbeat_sec > 0 and now - last_hb >= heartbeat_sec):
            line = f"{log_prefix} t+{int(now - start_ts)}s {msg}"
            if show_progress:
                try:
                    from tqdm.auto import tqdm

                    tqdm.write(line)
                except Exception:
                    print(line)
            else:
                print(line)
            last_hb = now

    _heartbeat("start attach_leak_safe_pedigree (pandas)", force=True)
    s = _resolve_starts_pandas(output_dir, starts_long_df=starts_long_df)
    _heartbeat(f"starts rows={len(s)} cumulative=1", force=True)
    net = (
        sibling_edges_df.copy()
        if sibling_edges_df is not None
        else load_sibling_edges(output_dir)
    )
    _heartbeat(f"loaded sibling edges rows={len(net)}", force=True)

    work = df_ml.copy()
    if "date" not in work.columns:
        raise ValueError("df_ml に date 列が必要です")
    if "year" not in work.columns:
        raise ValueError("df_ml に year 列が必要です")

    # 既存の blood 中間列・出力列を除去して再計算の列名衝突を防ぐ
    # （v21 以前の parquet には sib_sum_* が残存している場合がある）
    _PEDIGREE_REGEN_COLS = [
        "sib_sum_wins", "sib_sum_starts", "sib_sum_money",
        "sib_sum_turf_t3", "sib_sum_turf_st", "sib_sum_dirt_t3", "sib_sum_dirt_st",
        "sib_sum_soft_t3", "sib_sum_soft_st", "sib_sum_heavy_t3", "sib_sum_heavy_st",
        "sib_sum_bad_t3", "sib_sum_bad_st", "sib_sum_win_dist",
        "sibling_confidence_ls", "sibling_winup_rate_ls", "sibling_surface_bias_ls",
        "sibling_heavy_score_ls", "sibling_avg_money_ls", "sibling_avg_win_dist_ls",
        "sibling_money_rel_z",
        "pedigree_prior_starts_self", "pedigree_prev_surf", "pedigree_surface_switch_flag",
        "pedigree_debut_flag",
        "interact_switch_x_sibling_bias_ls", "interact_debut_x_sibling_bias_ls",
        "interact_shallow_x_winup_ls",
        "dam_age_at_birth",
    ]
    _cols_to_drop = [c for c in _PEDIGREE_REGEN_COLS if c in work.columns]
    if _cols_to_drop:
        work = work.drop(columns=_cols_to_drop)

    work["_row_id"] = np.arange(len(work), dtype=np.int64)
    work["_ketto_key"] = _ketto_join_key_pd(work["ketto_num"])

    work["target_dt"] = pd.to_datetime(work["date"])

    own = s[["ketto_num", "race_dt", "prior_start_count"]].copy()
    own_m = work.merge(
        own,
        left_on=["_ketto_key", "target_dt"],
        right_on=["ketto_num", "race_dt"],
        how="left",
    )
    work["pedigree_prior_starts_self"] = own_m["prior_start_count"]

    sw = s[["ketto_num", "race_dt", "track_code"]].copy()
    sw = sw.sort_values(["ketto_num", "race_dt"])
    sw["prev_track_code"] = sw.groupby("ketto_num")["track_code"].shift(1)
    sw["prev_surf"] = _surf_from_track(sw["prev_track_code"])
    prev_m = work.merge(
        sw[["ketto_num", "race_dt", "prev_surf"]],
        left_on=["_ketto_key", "target_dt"],
        right_on=["ketto_num", "race_dt"],
        how="left",
    )
    work["pedigree_prev_surf"] = prev_m["prev_surf"]

    cum_cols = [
        "race_dt",
        "cum_wins",
        "cum_starts",
        "cum_money",
        "cum_turf_t3",
        "cum_turf_st",
        "cum_dirt_t3",
        "cum_dirt_st",
        "cum_soft_t3",
        "cum_soft_st",
        "cum_heavy_t3",
        "cum_heavy_st",
        "cum_bad_t3",
        "cum_bad_st",
        "cum_win_dist_sum",
    ]
    s_cum = s[["ketto_num"] + cum_cols].copy()
    s_cum = s_cum.sort_values(["ketto_num", "race_dt"])

    tx = work.merge(net, left_on="_ketto_key", right_on="horse_id", how="left")
    tx = tx.dropna(subset=["sibling_id", "target_dt"]).copy()
    _heartbeat(
        f"prepared tx rows={len(tx)} unique_siblings={tx['sibling_id'].nunique() if len(tx) else 0}",
        force=True,
    )

    right = s_cum.rename(columns={"ketto_num": "sibling_id"})
    merged = _merge_asof_sibling_edges_cpu(
        tx,
        right,
        prefer_bulk=prefer_bulk_asof,
        show_progress=show_progress,
    )
    _heartbeat(f"merge_asof finished rows={len(merged)}", force=True)
    agg_sum = {
        "cum_wins": "sum",
        "cum_starts": "sum",
        "cum_money": "sum",
        "cum_turf_t3": "sum",
        "cum_turf_st": "sum",
        "cum_dirt_t3": "sum",
        "cum_dirt_st": "sum",
        "cum_soft_t3": "sum",
        "cum_soft_st": "sum",
        "cum_heavy_t3": "sum",
        "cum_heavy_st": "sum",
        "cum_bad_t3": "sum",
        "cum_bad_st": "sum",
        "cum_win_dist_sum": "sum",
    }
    rename_map = {
        "cum_wins": "sib_sum_wins",
        "cum_starts": "sib_sum_starts",
        "cum_money": "sib_sum_money",
        "cum_turf_t3": "sib_sum_turf_t3",
        "cum_turf_st": "sib_sum_turf_st",
        "cum_dirt_t3": "sib_sum_dirt_t3",
        "cum_dirt_st": "sib_sum_dirt_st",
        "cum_soft_t3": "sib_sum_soft_t3",
        "cum_soft_st": "sib_sum_soft_st",
        "cum_heavy_t3": "sib_sum_heavy_t3",
        "cum_heavy_st": "sib_sum_heavy_st",
        "cum_bad_t3": "sib_sum_bad_t3",
        "cum_bad_st": "sib_sum_bad_st",
        "cum_win_dist_sum": "sib_sum_win_dist",
    }
    if merged.empty:
        grp = pd.DataFrame({"_row_id": work["_row_id"].values})
        for c in rename_map.values():
            grp[c] = np.nan
    else:
        merged = merged.loc[merged["race_dt"] < merged["target_dt"]].copy()
        if merged.empty:
            grp = pd.DataFrame({"_row_id": work["_row_id"].values})
            for c in rename_map.values():
                grp[c] = np.nan
        else:
            grp = merged.groupby("_row_id", as_index=False).agg(agg_sum).rename(
                columns=rename_map
            )

    out = work.merge(grp, on="_row_id", how="left")
    _heartbeat("aggregated sibling cumulative features", force=True)

    k = float(shrink_k)
    n_all = out["sib_sum_starts"].to_numpy(dtype=float)
    out["sibling_confidence_ls"] = np.where(n_all > 0, n_all / (n_all + k), np.nan)

    raw_wr = _safe_div(out["sib_sum_wins"].values, n_all)
    out["sibling_winup_rate_ls"] = _shrink_toward_prior(raw_wr, n_all, shrink_prior_win, k)

    nt = out["sib_sum_turf_st"].to_numpy(dtype=float)
    nd = out["sib_sum_dirt_st"].to_numpy(dtype=float)
    tr_raw = _safe_div(out["sib_sum_turf_t3"].values, nt)
    dr_raw = _safe_div(out["sib_sum_dirt_t3"].values, nd)
    tr_s = _shrink_toward_prior(tr_raw, nt, shrink_prior_top3, k)
    dr_s = _shrink_toward_prior(dr_raw, nd, shrink_prior_top3, k)
    out["sibling_surface_bias_ls"] = np.where(
        np.isfinite(tr_s) & np.isfinite(dr_s),
        tr_s - dr_s,
        np.nan,
    )

    heavy_t3 = (
        out["sib_sum_soft_t3"].fillna(0)
        + out["sib_sum_heavy_t3"].fillna(0)
        + out["sib_sum_bad_t3"].fillna(0)
    )
    heavy_st = (
        out["sib_sum_soft_st"].fillna(0)
        + out["sib_sum_heavy_st"].fillna(0)
        + out["sib_sum_bad_st"].fillna(0)
    )
    hst = heavy_st.to_numpy(dtype=float)
    raw_h = _safe_div(heavy_t3.to_numpy(dtype=float), hst)
    out["sibling_heavy_score_ls"] = _shrink_toward_prior(raw_h, hst, shrink_prior_top3, k)

    out["sibling_avg_money_ls"] = _safe_div(out["sib_sum_money"].values, n_all)

    nw = out["sib_sum_wins"].to_numpy(dtype=float)
    raw_wd = _safe_div(out["sib_sum_win_dist"].values, nw)
    out["sibling_avg_win_dist_ls"] = _shrink_toward_prior(
        raw_wd, nw, shrink_prior_win_dist, k
    )

    out = _apply_sibling_money_rel_z_pd(out, train_year_cut=train_year_cut)

    cur_surf = _surf_from_track(out["track_code"])
    out["pedigree_surface_switch_flag"] = (
        (cur_surf != out["pedigree_prev_surf"])
        & cur_surf.notna()
        & out["pedigree_prev_surf"].notna()
    ).astype(float)
    out["interact_switch_x_sibling_bias_ls"] = (
        out["pedigree_surface_switch_flag"] * out["sibling_surface_bias_ls"].fillna(0)
    )

    if "lag1_finish_rank" in out.columns:
        out["pedigree_debut_flag"] = out["lag1_finish_rank"].isna().astype(float)
    else:
        out["pedigree_debut_flag"] = (
            out["pedigree_prior_starts_self"].fillna(0) == 0
        ).astype(float)
    out["interact_debut_x_sibling_bias_ls"] = (
        out["pedigree_debut_flag"] * out["sibling_surface_bias_ls"].fillna(0)
    )

    shallow = out["pedigree_prior_starts_self"].fillna(0) <= 3
    out["interact_shallow_x_winup_ls"] = (
        shallow.astype(float) * out["sibling_winup_rate_ls"].fillna(0)
    )

    try:
        dam_by = (
            dam_birth_year_series.copy()
            if dam_birth_year_series is not None
            else load_dam_birth_year(output_dir)
        )
        skd = dam_age_df.copy() if dam_age_df is not None else load_dam_age_features(output_dir)
        skd["dam_birth_year"] = skd["dam_id"].map(dam_by)
        skd["dam_age_at_birth"] = skd["foal_year_proxy"] - skd["dam_birth_year"]
        m = skd[["ketto_num", "dam_age_at_birth"]].rename(columns={"ketto_num": "_dam_join_k"})
        out = out.merge(
            m,
            left_on="_ketto_key",
            right_on="_dam_join_k",
            how="left",
        )
        out = out.drop(columns=["_dam_join_k"], errors="ignore")
    except Exception:
        out["dam_age_at_birth"] = np.nan

    drop_tmp = ["_row_id", "_ketto_key", "target_dt"]
    out = out.drop(columns=[c for c in drop_tmp if c in out.columns], errors="ignore")
    _heartbeat(f"done attach rows={len(out)} cols={len(out.columns)}", force=True)
    return out


def _ketto_join_key_cudf(col: Any) -> Any:
    if getattr(col.dtype, "kind", None) in "iu":
        return col.astype("int64")
    return cudf.to_numeric(col.astype(str).str.strip(), errors="coerce").astype("int64")


def _attach_leak_safe_pedigree_gpu(
    df_ml: Any,
    output_dir: Path,
    *,
    train_year_cut: int = 2024,
    shrink_k: float = SHRINK_K_DEFAULT,
    shrink_prior_win: float = PRIOR_SIBLING_WIN_RATE,
    shrink_prior_top3: float = PRIOR_SIBLING_TOP3_RATE,
    shrink_prior_win_dist: float = PRIOR_WIN_DIST_M,
    prefer_bulk_asof: bool = True,
    show_progress: bool = False,
    heartbeat_sec: int = 60,
    log_prefix: str = "[pedigree]",
    starts_long_df: pd.DataFrame | None = None,
    sibling_edges_df: pd.DataFrame | None = None,
    dam_birth_year_series: pd.Series | None = None,
    dam_age_df: pd.DataFrame | None = None,
) -> Any:
    if not _GPU_AVAILABLE:
        raise RuntimeError("cuDF が利用できません。")
    _ = prefer_bulk_asof  # GPU は常に一括 merge_asof（フラグ無視）
    start_ts = time.perf_counter()
    last_hb = start_ts

    def _heartbeat(msg: str, force: bool = False) -> None:
        nonlocal last_hb
        now = time.perf_counter()
        if force or (heartbeat_sec > 0 and now - last_hb >= heartbeat_sec):
            line = f"{log_prefix} t+{int(now - start_ts)}s {msg}"
            if show_progress:
                try:
                    from tqdm.auto import tqdm

                    tqdm.write(line)
                except Exception:
                    print(line)
            else:
                print(line)
            last_hb = now

    _heartbeat("start attach_leak_safe_pedigree (cudf)", force=True)
    s_pd = _resolve_starts_pandas(output_dir, starts_long_df=starts_long_df)
    s = cudf.from_pandas(s_pd)
    _heartbeat(f"starts rows={len(s)} on device", force=True)
    del s_pd
    gc.collect()

    net_pd = (
        sibling_edges_df.copy()
        if sibling_edges_df is not None
        else load_sibling_edges(output_dir)
    )
    net = cudf.from_pandas(net_pd)
    del net_pd
    _heartbeat(f"loaded sibling edges rows={len(net)}", force=True)

    work = df_ml.copy()
    if "date" not in work.columns:
        raise ValueError("df_ml に date 列が必要です")
    if "year" not in work.columns:
        raise ValueError("df_ml に year 列が必要です")

    # 既存の blood 中間列・出力列を除去して再計算の列名衝突を防ぐ
    _PEDIGREE_REGEN_COLS_GPU = [
        "sib_sum_wins", "sib_sum_starts", "sib_sum_money",
        "sib_sum_turf_t3", "sib_sum_turf_st", "sib_sum_dirt_t3", "sib_sum_dirt_st",
        "sib_sum_soft_t3", "sib_sum_soft_st", "sib_sum_heavy_t3", "sib_sum_heavy_st",
        "sib_sum_bad_t3", "sib_sum_bad_st", "sib_sum_win_dist",
        "sibling_confidence_ls", "sibling_winup_rate_ls", "sibling_surface_bias_ls",
        "sibling_heavy_score_ls", "sibling_avg_money_ls", "sibling_avg_win_dist_ls",
        "sibling_money_rel_z",
        "pedigree_prior_starts_self", "pedigree_prev_surf", "pedigree_surface_switch_flag",
        "pedigree_debut_flag",
        "interact_switch_x_sibling_bias_ls", "interact_debut_x_sibling_bias_ls",
        "interact_shallow_x_winup_ls",
        "dam_age_at_birth",
    ]
    _cols_to_drop_gpu = [c for c in _PEDIGREE_REGEN_COLS_GPU if c in work.columns]
    if _cols_to_drop_gpu:
        work = work.drop(columns=_cols_to_drop_gpu)

    work["_row_id"] = cudf.Series(np.arange(len(work), dtype=np.int64))
    work["_ketto_key"] = _ketto_join_key_cudf(work["ketto_num"])
    work["target_dt"] = cudf.to_datetime(work["date"])

    own = s[["ketto_num", "race_dt", "prior_start_count"]]
    own_m = work.merge(
        own,
        left_on=["_ketto_key", "target_dt"],
        right_on=["ketto_num", "race_dt"],
        how="left",
    )
    work["pedigree_prior_starts_self"] = own_m["prior_start_count"]

    sw = s[["ketto_num", "race_dt", "track_code"]].sort_values(
        ["ketto_num", "race_dt"]
    )
    sw["prev_track_code"] = sw.groupby("ketto_num")["track_code"].shift(1)
    sw["prev_surf"] = _surf_from_track_gpu(sw["prev_track_code"])
    prev_m = work.merge(
        sw[["ketto_num", "race_dt", "prev_surf"]],
        left_on=["_ketto_key", "target_dt"],
        right_on=["ketto_num", "race_dt"],
        how="left",
    )
    work["pedigree_prev_surf"] = prev_m["prev_surf"]

    cum_cols = [
        "race_dt",
        "cum_wins",
        "cum_starts",
        "cum_money",
        "cum_turf_t3",
        "cum_turf_st",
        "cum_dirt_t3",
        "cum_dirt_st",
        "cum_soft_t3",
        "cum_soft_st",
        "cum_heavy_t3",
        "cum_heavy_st",
        "cum_bad_t3",
        "cum_bad_st",
        "cum_win_dist_sum",
    ]
    s_cum = s[["ketto_num"] + cum_cols].sort_values(["ketto_num", "race_dt"])

    tx = work.merge(net, left_on="_ketto_key", right_on="horse_id", how="left")
    tx = tx.dropna(subset=["sibling_id", "target_dt"])
    _heartbeat(
        f"prepared tx rows={len(tx)} unique_siblings={int(tx['sibling_id'].nunique()) if len(tx) else 0}",
        force=True,
    )

    right = s_cum.rename(columns={"ketto_num": "sibling_id"})
    tie_cols_ml = [c for c in ("race_id", "horse_num", "umaban") if c in tx.columns]
    merged = _merge_asof_sibling_edges_gpu(tx, right, tie_cols_ml=tie_cols_ml)
    _heartbeat(f"merge_asof finished rows={len(merged)}", force=True)

    agg_sum = {
        "cum_wins": "sum",
        "cum_starts": "sum",
        "cum_money": "sum",
        "cum_turf_t3": "sum",
        "cum_turf_st": "sum",
        "cum_dirt_t3": "sum",
        "cum_dirt_st": "sum",
        "cum_soft_t3": "sum",
        "cum_soft_st": "sum",
        "cum_heavy_t3": "sum",
        "cum_heavy_st": "sum",
        "cum_bad_t3": "sum",
        "cum_bad_st": "sum",
        "cum_win_dist_sum": "sum",
    }
    rename_map = {
        "cum_wins": "sib_sum_wins",
        "cum_starts": "sib_sum_starts",
        "cum_money": "sib_sum_money",
        "cum_turf_t3": "sib_sum_turf_t3",
        "cum_turf_st": "sib_sum_turf_st",
        "cum_dirt_t3": "sib_sum_dirt_t3",
        "cum_dirt_st": "sib_sum_dirt_st",
        "cum_soft_t3": "sib_sum_soft_t3",
        "cum_soft_st": "sib_sum_soft_st",
        "cum_heavy_t3": "sib_sum_heavy_t3",
        "cum_heavy_st": "sib_sum_heavy_st",
        "cum_bad_t3": "sib_sum_bad_t3",
        "cum_bad_st": "sib_sum_bad_st",
        "cum_win_dist_sum": "sib_sum_win_dist",
    }
    if merged.empty:
        grp = cudf.DataFrame({"_row_id": work["_row_id"]})
        for c in rename_map.values():
            grp[c] = np.nan
    else:
        merged = merged.loc[merged["race_dt"] < merged["target_dt"]]
        if merged.empty:
            grp = cudf.DataFrame({"_row_id": work["_row_id"]})
            for c in rename_map.values():
                grp[c] = np.nan
        else:
            grp = merged.groupby("_row_id", as_index=False).agg(agg_sum).rename(
                columns=rename_map
            )

    out = work.merge(grp, on="_row_id", how="left")
    _heartbeat("aggregated sibling cumulative features", force=True)

    k = float(shrink_k)
    n_all = out["sib_sum_starts"].astype("float64")
    out["sibling_confidence_ls"] = (n_all / (n_all + k)).where(n_all > 0)

    out["sibling_winup_rate_ls"] = _cup_shrink(
        _cup_safe_div(out["sib_sum_wins"], n_all),
        n_all,
        shrink_prior_win,
        k,
    )

    nt = out["sib_sum_turf_st"].astype("float64")
    nd = out["sib_sum_dirt_st"].astype("float64")
    tr_s = _cup_shrink(_cup_safe_div(out["sib_sum_turf_t3"], nt), nt, shrink_prior_top3, k)
    dr_s = _cup_shrink(_cup_safe_div(out["sib_sum_dirt_t3"], nd), nd, shrink_prior_top3, k)
    out["sibling_surface_bias_ls"] = (tr_s.astype("float64") - dr_s.astype("float64")).where(
        tr_s.notna() & dr_s.notna()
    )

    heavy_t3 = (
        out["sib_sum_soft_t3"].fillna(0)
        + out["sib_sum_heavy_t3"].fillna(0)
        + out["sib_sum_bad_t3"].fillna(0)
    )
    heavy_st = (
        out["sib_sum_soft_st"].fillna(0)
        + out["sib_sum_heavy_st"].fillna(0)
        + out["sib_sum_bad_st"].fillna(0)
    )
    hst = heavy_st.astype("float64")
    raw_h = _cup_safe_div(heavy_t3.astype("float64"), hst)
    out["sibling_heavy_score_ls"] = _cup_shrink(raw_h, hst, shrink_prior_top3, k)

    out["sibling_avg_money_ls"] = _cup_safe_div(
        out["sib_sum_money"].astype("float64"), n_all
    )

    nw = out["sib_sum_wins"].astype("float64")
    out["sibling_avg_win_dist_ls"] = _cup_shrink(
        _cup_safe_div(out["sib_sum_win_dist"].astype("float64"), nw),
        nw,
        shrink_prior_win_dist,
        k,
    )

    out = _apply_sibling_money_rel_z_cudf(out, train_year_cut=train_year_cut)

    cur_surf = _surf_from_track_gpu(out["track_code"]).astype("float64")
    prev_surf = out["pedigree_prev_surf"].astype("float64")
    surf_sw = cur_surf.ne(prev_surf) & cur_surf.notna() & prev_surf.notna()
    out["pedigree_surface_switch_flag"] = surf_sw.astype("float64")
    out["interact_switch_x_sibling_bias_ls"] = (
        out["pedigree_surface_switch_flag"] * out["sibling_surface_bias_ls"].fillna(0).astype(
            "float64"
        )
    )

    if "lag1_finish_rank" in out.columns:
        out["pedigree_debut_flag"] = out["lag1_finish_rank"].isna().astype("float64")
    else:
        out["pedigree_debut_flag"] = (
            out["pedigree_prior_starts_self"].fillna(0).astype("int64") == 0
        ).astype("float64")
    out["interact_debut_x_sibling_bias_ls"] = (
        out["pedigree_debut_flag"] * out["sibling_surface_bias_ls"].fillna(0).astype("float64")
    )

    shallow = out["pedigree_prior_starts_self"].fillna(0).astype("int64") <= 3
    out["interact_shallow_x_winup_ls"] = (
        shallow.astype("float64") * out["sibling_winup_rate_ls"].fillna(0).astype("float64")
    )

    try:
        dam_by = (
            dam_birth_year_series.copy()
            if dam_birth_year_series is not None
            else load_dam_birth_year(output_dir)
        )
        skd_pd = dam_age_df.copy() if dam_age_df is not None else load_dam_age_features(output_dir)
        skd_pd["dam_birth_year"] = skd_pd["dam_id"].map(dam_by)
        skd_pd["dam_age_at_birth"] = skd_pd["foal_year_proxy"] - skd_pd["dam_birth_year"]
        m = skd_pd[["ketto_num", "dam_age_at_birth"]].rename(columns={"ketto_num": "_dam_join_k"})
        m_cd = cudf.from_pandas(m)
        out = out.merge(m_cd, left_on="_ketto_key", right_on="_dam_join_k", how="left")
        out = out.drop(columns=["_dam_join_k"], errors="ignore")
    except Exception:
        out["dam_age_at_birth"] = float("nan")

    drop_tmp = ["_row_id", "_ketto_key", "target_dt"]
    out = out.drop(columns=[c for c in drop_tmp if c in out.columns], errors="ignore")
    _heartbeat(f"done attach rows={len(out)} cols={len(out.columns)}", force=True)
    return out


def attach_leak_safe_pedigree(
    df_ml: Any,
    output_dir: Path | str,
    *,
    train_year_cut: int = 2024,
    shrink_k: float = SHRINK_K_DEFAULT,
    shrink_prior_win: float = PRIOR_SIBLING_WIN_RATE,
    shrink_prior_top3: float = PRIOR_SIBLING_TOP3_RATE,
    shrink_prior_win_dist: float = PRIOR_WIN_DIST_M,
    prefer_bulk_asof: bool = True,
    show_progress: bool = False,
    heartbeat_sec: int = 60,
    log_prefix: str = "[pedigree]",
    starts_long_df: pd.DataFrame | None = None,
    sibling_edges_df: pd.DataFrame | None = None,
    dam_birth_year_series: pd.Series | None = None,
    dam_age_df: pd.DataFrame | None = None,
) -> Any:
    """
    df_ml に血統列を付与（行順・index は維持）。

    pandas DataFrame は CPU 経路、cudf は GPU merge_asof 一括（sibling Python フォールバックなし）。
    """
    kw = dict(
        train_year_cut=train_year_cut,
        shrink_k=shrink_k,
        shrink_prior_win=shrink_prior_win,
        shrink_prior_top3=shrink_prior_top3,
        shrink_prior_win_dist=shrink_prior_win_dist,
        prefer_bulk_asof=prefer_bulk_asof,
        show_progress=show_progress,
        heartbeat_sec=heartbeat_sec,
        log_prefix=log_prefix,
        starts_long_df=starts_long_df,
        sibling_edges_df=sibling_edges_df,
        dam_birth_year_series=dam_birth_year_series,
        dam_age_df=dam_age_df,
    )
    if _is_gpu_df(df_ml):
        return _attach_leak_safe_pedigree_gpu(df_ml, Path(output_dir), **kw)
    if not isinstance(df_ml, pd.DataFrame):
        raise TypeError("df_ml must be pandas.DataFrame or cudf.DataFrame")
    return _attach_leak_safe_pedigree_cpu(df_ml, Path(output_dir), **kw)
def add_surface_course_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v5用の新規特徴量10件を追加する。
    入力DFはv4.parquetをそのまま渡す（既存列はすべて保持）。

    追加特徴量:
        A1. horse_dirt_win_rate   -- ダート限定累積勝率
        A2. horse_turf_win_rate   -- 芝限定累積勝率
        A3. surface_win_rate_diff -- 芝勝率 - ダート勝率
        A4. horse_course_win_rate_v5 -- ketto_num x course_code 純累積勝率（NaN=0出走）
        B1. running_style_surface_win_rate -- 走法x馬場種別の全馬集計勝率
        C1. surface_switch_flag   -- 前走と今走の馬場種別切り替えフラグ
        D1. agari3f_same_surface_rank_lag1 -- 同一馬場種別の前走上がり3Fレース内順位
        D2. same_distance_same_surface_win_rate -- 距離カテゴリx馬場種別の累積勝率
        E1. age_distance_win_rate  -- 全馬集計の年齢x距離カテゴリ勝率
        E2. jockey_horse_combo_count -- 騎手x馬の過去コンビ回数

    リーク防止: 全特徴量で sort_values(['ketto_num', 'date']) + shift(1) / cumcount() - 当該行 を使用。
    """
    # sort 保証（ketto_num + date 順）
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 馬場種別フラグ（is_dirt / is_turf）を生成
    # track_code: 10-19=芝, 20-29=ダート, 50-59=障害
    # 障害レース（50-59）は芝・ダート両方とも0扱いにしてNaNで処理
    # ------------------------------------------------------------------
    tc = pd.to_numeric(df["track_code"], errors="coerce")
    is_dirt = (tc.between(20, 29)).astype("int8")   # ダート: 1, それ以外: 0
    is_turf = (tc.between(10, 19)).astype("int8")   # 芝: 1, それ以外: 0
    # 障害レースは is_dirt=0, is_turf=0 として距離・馬場種別特徴量はNaNになる

    # ------------------------------------------------------------------
    # 勝利フラグ生成
    # ------------------------------------------------------------------
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    # ------------------------------------------------------------------
    # 距離カテゴリ生成（D2・E1共用）
    # sprint<1400, mile 1400-1800, middle 1800-2200, long>2200
    # ------------------------------------------------------------------
    dist_num = pd.to_numeric(df["distance"], errors="coerce")
    dist_cat = pd.cut(
        dist_num,
        bins=[0, 1400, 1800, 2200, 100000],
        labels=[0, 1, 2, 3],
        right=False,
    ).astype("Int8")

    # ==================================================================
    # A1. horse_dirt_win_rate
    #     ダート(is_dirt=1)レースのみを対象とした ketto_num 別累積勝率
    #     当該レース除外: cumsum/cumcount から当該行を引く
    #     ダート出走0回の場合は NaN
    # ==================================================================
    dirt_win = (win_flag * is_dirt).astype("int8")      # ダート勝利フラグ
    dirt_run = is_dirt.astype("int8")                    # ダート出走フラグ

    g_dirt = df.groupby("ketto_num", sort=False)
    cum_dirt_runs = dirt_run.groupby(df["ketto_num"], sort=False).cumsum() - dirt_run
    cum_dirt_wins = dirt_win.groupby(df["ketto_num"], sort=False).cumsum() - dirt_win
    # Bayesian smoothing: 未経験馬はprior(0.10)に収束、NaN=0
    _BETA_DIRT, _PRIOR_DIRT = 5.0, 0.10
    df["horse_dirt_win_rate"] = (cum_dirt_wins + _BETA_DIRT * _PRIOR_DIRT) / (cum_dirt_runs + _BETA_DIRT)
    df["horse_dirt_n_runs"] = cum_dirt_runs.astype("int16")

    # ==================================================================
    # A2. horse_turf_win_rate
    #     芝(is_turf=1)レースのみを対象とした ketto_num 別累積勝率
    # ==================================================================
    turf_win = (win_flag * is_turf).astype("int8")
    turf_run = is_turf.astype("int8")

    cum_turf_runs = turf_run.groupby(df["ketto_num"], sort=False).cumsum() - turf_run
    cum_turf_wins = turf_win.groupby(df["ketto_num"], sort=False).cumsum() - turf_win
    _BETA_TURF, _PRIOR_TURF = 5.0, 0.10
    df["horse_turf_win_rate"] = (cum_turf_wins + _BETA_TURF * _PRIOR_TURF) / (cum_turf_runs + _BETA_TURF)
    df["horse_turf_n_runs"] = cum_turf_runs.astype("int16")

    # ==================================================================
    # A3. surface_win_rate_diff = horse_turf_win_rate - horse_dirt_win_rate
    #     芝有利なら正, ダート有利なら負 (A1/A2がBayesian smoothing済みのためNaN=0)
    # ==================================================================
    df["surface_win_rate_diff"] = df["horse_turf_win_rate"] - df["horse_dirt_win_rate"]

    # ==================================================================
    # A4. horse_course_win_rate_v5
    #     ketto_num x course_code の純累積勝率（NaN=当コース初出走）
    #     既存の horse_course_win_rate はベイズ平滑化済みで上書きしないため新列名
    # ==================================================================
    if "course_code" in df.columns:
        cc = df["course_code"].astype(str)
        grp_keys_cc = [df["ketto_num"].astype(str), cc]
        cum_cc_runs = df.groupby(["ketto_num", "course_code"], sort=False).cumcount()
        cum_cc_wins = win_flag.groupby(
            [df["ketto_num"], df["course_code"]], sort=False
        ).cumsum() - win_flag
        _BETA_CC, _PRIOR_CC = 5.0, 0.10
        df["horse_course_win_rate_v5"] = (cum_cc_wins + _BETA_CC * _PRIOR_CC) / (cum_cc_runs + _BETA_CC)
        df["horse_course_n_runs"] = cum_cc_runs.astype("int16")
    else:
        df["horse_course_win_rate_v5"] = np.nan
        df["horse_course_n_runs"] = np.zeros(len(df), dtype="int16")

    # ==================================================================
    # B0. horse_modal_running_style
    #     馬の直近 N 走の最頻脚質コード（shift(1) でリーク防止）
    #     running_style_code=0（未確定）は集計対象外。
    #     推論時は当日レースの running_style_code=0 なので、過去履歴のみで計算される。
    #     不明/過去レースなしの場合は 0。
    #     設定: pastfeatures_config.json の modal_running_style_lookback（デフォルト10）
    # ==================================================================
    if "running_style_code" in df.columns:
        _config_modal = load_pastfeatures_config()
        _n_lookback_modal = int(_config_modal.get("modal_running_style_lookback", 10))
        # running_style_code=0（未確定・不明）は NaN として扱い、集計対象外にする
        df["__rs_modal_src"] = pd.to_numeric(df["running_style_code"], errors="coerce").replace(0, np.nan)

        def _modal_shift(x: "pd.Series") -> "pd.Series":
            # shift(1) で当該レースを除外してから直近 N 走の最頻値を計算
            shifted = x.shift(1)
            return shifted.rolling(window=_n_lookback_modal, min_periods=1).apply(
                lambda v: pd.Series(v[~np.isnan(v)]).mode().iloc[0] if (~np.isnan(v)).any() else 0,
                raw=True,
            )

        df["horse_modal_running_style"] = (
            df.groupby("ketto_num")["__rs_modal_src"]
            .transform(_modal_shift)
            .fillna(0)
            .astype(int)
        )
        df = df.drop(columns=["__rs_modal_src"], errors="ignore")
    else:
        df["horse_modal_running_style"] = 0

    # ==================================================================
    # B1. running_style_surface_win_rate
    #     走法 x 馬場種別(is_dirt) の全馬集計勝率
    #     horse_modal_running_style を優先使用: 推論時に running_style_code=0 でも
    #     過去履歴から脚質を取得できるため NaN 分岐を回避できる。
    #     フォールバック: horse_modal_running_style が未計算の場合は running_style_code を使用。
    #     時系列安全: race_date 順で当該レース自身を除外 (cumsum/cumcount パターン)
    #     DFはすでに date 順でソート済みであることが前提
    # ==================================================================
    # horse_modal_running_style は直前の B0 セクションで計算済み（0=不明/実績なし）
    if "horse_modal_running_style" in df.columns:
        # 0=不明（推論時初出走など）は除外対象、1-4 が有効値
        rs = pd.to_numeric(df["horse_modal_running_style"], errors="coerce").fillna(0).astype(int)
    elif "running_style_code" in df.columns:
        rs = pd.to_numeric(df["running_style_code"], errors="coerce").fillna(0).astype(int)
    else:
        rs = pd.Series(0, index=df.index, dtype=int)

    if rs.any():
        rs_valid = (rs >= 1).astype("int8")  # 0=不明は除外対象

        # グループキー文字列: modal_running_style_is_dirt
        df["__rs_surf_key"] = rs.astype(str) + "_" + is_dirt.astype(str)
        cum_rs_count = df.groupby("__rs_surf_key", sort=False).cumcount()  # 0-indexed, リーク防止済み
        cum_rs_wins = win_flag.groupby(df["__rs_surf_key"], sort=False).cumsum() - win_flag

        df["running_style_surface_win_rate"] = (
            (cum_rs_wins / cum_rs_count.replace(0, np.nan)).where(
                (cum_rs_count > 0) & (rs_valid == 1)
            )
        )
        df = df.drop(columns=["__rs_surf_key"], errors="ignore")
    else:
        df["running_style_surface_win_rate"] = np.nan

    # ==================================================================
    # C1. surface_switch_flag
    #     前走の馬場種別と今走が異なる場合は 1, 同じなら 0
    #     初走（前走なし）は NaN
    # ==================================================================
    is_dirt_lag1 = is_dirt.groupby(df["ketto_num"], sort=False).shift(1)
    is_turf_lag1 = is_turf.groupby(df["ketto_num"], sort=False).shift(1)

    # 前走が有効（芝かダート）かつ今走と異なる場合は1
    prev_is_dirt = is_dirt_lag1
    curr_is_dirt = is_dirt.astype(float)
    has_prev = is_dirt_lag1.notna() & (is_dirt_lag1 + is_turf_lag1 > 0)  # 前走が芝 or ダート
    surface_switch = (prev_is_dirt != curr_is_dirt).astype(float)
    df["surface_switch_flag"] = surface_switch.where(has_prev)

    # ==================================================================
    # D1. agari3f_same_surface_rank_lag1
    #     同一馬場種別（is_dirt が今走と同じ）の前走上がり3Fレース内順位
    #     agari3f_rank_in_race_lag1 を使わず、馬ごとに is_dirt が一致する行のみでshift
    # ==================================================================
    if "agari3f_rank_in_race_lag1" in df.columns:
        agari_rank = pd.to_numeric(df["agari3f_rank_in_race_lag1"], errors="coerce")
        # 今走の is_dirt
        # ketto_num でgroupby後、is_dirt が一致する行のみ前走値を取り出す
        # 実装: 各馬について、過去の同馬場種別レースの agari3f_rank_in_race_lag1 を
        #       agari3f_rank_in_race_lag1 は既に shift済みなので、
        #       「同馬場種別の前走値」は is_dirt が同じ前走を探す必要がある。
        # 実装方針: 元の time_3f_after はリーク列削除済みで存在しないため、
        #           agari3f_rank_in_race_lag1 を再利用して同一馬場フィルタshiftする。
        # 具体的には:
        #   (1) 元 agari3f_rank_in_race_lag1 は shift(1)で lag化されている（前走の値）。
        #   (2) 同一馬場種別の前走値 = 今走と同じ is_dirt のときの lag1 値。
        #   しかし agari3f_rank_in_race_lag1 は前走の値なので、前走の is_dirt と
        #   今走の is_dirt を比較するために前走の is_dirt が必要。
        # 前走の is_dirt を取得
        prev_is_dirt_for_agari = is_dirt.groupby(df["ketto_num"], sort=False).shift(1)
        # 前走の agari3f_rank_in_race_lag1 は既存列（前走の値が格納済み）
        # 同一馬場種別の前走値: 前走 is_dirt == 今走 is_dirt の場合のみ採用
        same_surf_mask = (prev_is_dirt_for_agari == is_dirt.astype(float))
        df["agari3f_same_surface_rank_lag1"] = agari_rank.where(same_surf_mask)
    else:
        df["agari3f_same_surface_rank_lag1"] = np.nan

    # ==================================================================
    # D2. same_distance_same_surface_win_rate
    #     ketto_num x dist_cat x is_dirt の累積勝率
    # ==================================================================
    df["__dist_cat_tmp"] = dist_cat
    df["__is_dirt_tmp"] = is_dirt

    grp_d2 = df.groupby(["ketto_num", "__dist_cat_tmp", "__is_dirt_tmp"], sort=False)
    cum_d2_runs = grp_d2.cumcount()
    cum_d2_wins = win_flag.groupby(
        [df["ketto_num"], df["__dist_cat_tmp"], df["__is_dirt_tmp"]], sort=False
    ).cumsum() - win_flag
    _BETA_D2, _PRIOR_D2 = 5.0, 0.10
    df["same_distance_same_surface_win_rate"] = (cum_d2_wins + _BETA_D2 * _PRIOR_D2) / (cum_d2_runs + _BETA_D2)
    df["same_distance_same_surface_n_runs"] = cum_d2_runs.astype("int16")
    df = df.drop(columns=["__dist_cat_tmp", "__is_dirt_tmp"], errors="ignore")

    # ==================================================================
    # E1. age_distance_win_rate
    #     全馬集計の age x dist_cat の累積勝率
    #     当該レース除外: cumsum - win_flag
    # ==================================================================
    if "age" in df.columns:
        age_num = pd.to_numeric(df["age"], errors="coerce").astype("Int8")
        df["__age_tmp"] = age_num
        df["__dist_cat_e1"] = dist_cat

        cum_e1_count = df.groupby(
            ["__age_tmp", "__dist_cat_e1"], sort=False
        ).cumcount()
        cum_e1_wins = win_flag.groupby(
            [df["__age_tmp"], df["__dist_cat_e1"]], sort=False
        ).cumsum() - win_flag
        df["age_distance_win_rate"] = (
            (cum_e1_wins / cum_e1_count.replace(0, np.nan)).where(cum_e1_count > 0)
        )
        df = df.drop(columns=["__age_tmp", "__dist_cat_e1"], errors="ignore")
    else:
        df["age_distance_win_rate"] = np.nan

    # ==================================================================
    # E2. jockey_horse_combo_count
    #     騎手(jockey_code) x 馬(ketto_num) の過去コンビ回数
    #     shift(1)で当該レースを除外 → 初コンビは0
    # ==================================================================
    if "jockey_code" in df.columns:
        combo_cumcount = df.groupby(
            ["ketto_num", "jockey_code"], sort=False
        ).cumcount()  # 0-indexed: 当該レースを含む回数-1
        # cumcount() は当該行が0-indexed のため、そのまま shift(1) は不要
        # cumcount() = 今走までの出走回数-1 = 前走までの回数 → リーク防止済み
        df["jockey_horse_combo_count"] = combo_cumcount.astype("int16")
    else:
        df["jockey_horse_combo_count"] = 0

    return df


def add_going_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v7 用の馬場状態別過去実績特徴量を追加する。

    全特徴量において sort_values(['date', 'race_id', 'ketto_num']) + shift(1) または
    cumsum/cumcount から当該行を引く方式でリークを防止する。

    追加特徴量:
        horse_dirt_heavy_win_rate   -- ダート重馬場(dirt_condition>=3)累積勝率 (Bayesian, prior=0.08, beta=5)
        horse_dirt_heavy_n_runs     -- 同上出走数
        horse_turf_heavy_win_rate   -- 芝重馬場(turf_condition>=3)累積勝率 (Bayesian, prior=0.07, beta=5)
        horse_turf_heavy_n_runs     -- 同上出走数
        horse_dirt_light_win_rate   -- ダート良馬場(dirt_condition==1)累積勝率 (Bayesian, prior=0.08, beta=5)
        horse_dirt_light_n_runs     -- 同上出走数
        horse_turf_light_win_rate   -- 芝良馬場(turf_condition==1)累積勝率 (Bayesian, prior=0.07, beta=5)
        horse_turf_light_n_runs     -- 同上出走数
        horse_going_preference      -- 重馬場勝率 - 良馬場勝率（同一surface内）
        jockey_heavy_win_rate       -- 騎手の重馬場累積勝率 (Bayesian, prior=0.08, beta=20)

    NaN 許容: 条件に合うレースが過去に存在しない場合は NaN を返す。
    """
    # GPU DataFrame は CPU に落として処理する
    source_is_gpu = _is_gpu_df(df)
    if source_is_gpu:
        df = _to_pandas_df(df)

    # sort 保証（date → race_id → ketto_num 順）
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 勝利フラグ生成（リーク防止のため finish_rank を参照）
    # ------------------------------------------------------------------
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    # ------------------------------------------------------------------
    # track_code から芝/ダートを判定
    # track_code: 10-19=芝, 20-29=ダート（50-59=障害は両方0扱い）
    # ------------------------------------------------------------------
    tc = pd.to_numeric(df["track_code"], errors="coerce")
    is_dirt = (tc.between(20, 29)).astype("int8")
    is_turf = (tc.between(10, 19)).astype("int8")

    # ------------------------------------------------------------------
    # 馬場状態コード（0=未記録は各条件フィルタから除外する）
    # ------------------------------------------------------------------
    turf_cond = pd.to_numeric(
        df.get("turf_condition", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0).astype("int8")
    dirt_cond = pd.to_numeric(
        df.get("dirt_condition", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0).astype("int8")

    # ==================================================================
    # 1. horse_dirt_heavy_win_rate
    #    条件: ダートレース(is_dirt=1) かつ dirt_condition >= 3 かつ dirt_condition != 0
    #    Bayesian smoothing: prior=0.08, beta=5
    #    NaN: 条件合致レースが0回の場合は NaN（prior への収束なし）
    # ==================================================================
    _PRIOR_DH, _BETA_DH = 0.08, 5.0
    is_dirt_heavy = ((is_dirt == 1) & (dirt_cond >= 3)).astype("int8")
    dirt_heavy_win = (win_flag * is_dirt_heavy).astype("int8")

    cum_dh_runs = is_dirt_heavy.groupby(df["ketto_num"], sort=False).cumsum() - is_dirt_heavy
    cum_dh_wins = dirt_heavy_win.groupby(df["ketto_num"], sort=False).cumsum() - dirt_heavy_win

    dirt_heavy_rate = (
        (cum_dh_wins + _BETA_DH * _PRIOR_DH) / (cum_dh_runs + _BETA_DH)
    )
    # 過去に条件合致レースが0回の場合は NaN（プライアーへ収束させない）
    df["horse_dirt_heavy_win_rate"] = dirt_heavy_rate.where(cum_dh_runs > 0, np.nan)
    df["horse_dirt_heavy_n_runs"] = cum_dh_runs.astype("int16")

    # ==================================================================
    # 2. horse_turf_heavy_win_rate
    #    条件: 芝レース(is_turf=1) かつ turf_condition >= 3 かつ turf_condition != 0
    #    Bayesian smoothing: prior=0.07, beta=5
    # ==================================================================
    _PRIOR_TH, _BETA_TH = 0.07, 5.0
    is_turf_heavy = ((is_turf == 1) & (turf_cond >= 3)).astype("int8")
    turf_heavy_win = (win_flag * is_turf_heavy).astype("int8")

    cum_th_runs = is_turf_heavy.groupby(df["ketto_num"], sort=False).cumsum() - is_turf_heavy
    cum_th_wins = turf_heavy_win.groupby(df["ketto_num"], sort=False).cumsum() - turf_heavy_win

    turf_heavy_rate = (
        (cum_th_wins + _BETA_TH * _PRIOR_TH) / (cum_th_runs + _BETA_TH)
    )
    df["horse_turf_heavy_win_rate"] = turf_heavy_rate.where(cum_th_runs > 0, np.nan)
    df["horse_turf_heavy_n_runs"] = cum_th_runs.astype("int16")

    # ==================================================================
    # 3. horse_dirt_light_win_rate
    #    条件: ダートレース(is_dirt=1) かつ dirt_condition == 1（良馬場）
    #    Bayesian smoothing: prior=0.08, beta=5
    # ==================================================================
    _PRIOR_DL, _BETA_DL = 0.08, 5.0
    is_dirt_light = ((is_dirt == 1) & (dirt_cond == 1)).astype("int8")
    dirt_light_win = (win_flag * is_dirt_light).astype("int8")

    cum_dl_runs = is_dirt_light.groupby(df["ketto_num"], sort=False).cumsum() - is_dirt_light
    cum_dl_wins = dirt_light_win.groupby(df["ketto_num"], sort=False).cumsum() - dirt_light_win

    dirt_light_rate = (
        (cum_dl_wins + _BETA_DL * _PRIOR_DL) / (cum_dl_runs + _BETA_DL)
    )
    df["horse_dirt_light_win_rate"] = dirt_light_rate.where(cum_dl_runs > 0, np.nan)
    df["horse_dirt_light_n_runs"] = cum_dl_runs.astype("int16")

    # ==================================================================
    # 4. horse_turf_light_win_rate
    #    条件: 芝レース(is_turf=1) かつ turf_condition == 1（良馬場）
    #    Bayesian smoothing: prior=0.07, beta=5
    # ==================================================================
    _PRIOR_TL, _BETA_TL = 0.07, 5.0
    is_turf_light = ((is_turf == 1) & (turf_cond == 1)).astype("int8")
    turf_light_win = (win_flag * is_turf_light).astype("int8")

    cum_tl_runs = is_turf_light.groupby(df["ketto_num"], sort=False).cumsum() - is_turf_light
    cum_tl_wins = turf_light_win.groupby(df["ketto_num"], sort=False).cumsum() - turf_light_win

    turf_light_rate = (
        (cum_tl_wins + _BETA_TL * _PRIOR_TL) / (cum_tl_runs + _BETA_TL)
    )
    df["horse_turf_light_win_rate"] = turf_light_rate.where(cum_tl_runs > 0, np.nan)
    df["horse_turf_light_n_runs"] = cum_tl_runs.astype("int16")

    # ==================================================================
    # 5. horse_going_preference
    #    同一 surface 内での 重馬場勝率 - 良馬場勝率
    #    芝レース(track_code < 23): horse_turf_heavy_win_rate - horse_turf_light_win_rate
    #    ダートレース(track_code >= 23): horse_dirt_heavy_win_rate - horse_dirt_light_win_rate
    #    どちらかが NaN の場合は NaN のまま
    #    仕様書の条件: track_code < 23 を芝、track_code >= 23 をダートとして判定
    # ==================================================================
    is_turf_for_pref = (tc < 23).fillna(False)  # 芝判定: track_code < 23
    is_dirt_for_pref = (tc >= 23).fillna(False)  # ダート判定: track_code >= 23

    turf_pref = df["horse_turf_heavy_win_rate"] - df["horse_turf_light_win_rate"]
    dirt_pref = df["horse_dirt_heavy_win_rate"] - df["horse_dirt_light_win_rate"]

    # 芝レースでは芝の差分、ダートレースではダートの差分を採用
    going_pref = pd.Series(np.nan, index=df.index)
    going_pref = going_pref.where(~is_turf_for_pref, turf_pref)
    going_pref = going_pref.where(~is_dirt_for_pref, dirt_pref)
    # どちらの条件にも該当しない（障害など）は NaN のまま
    df["horse_going_preference"] = going_pref

    # ==================================================================
    # 6. jockey_heavy_win_rate
    #    jockey_code ごとの重馬場（condition_code = max(turf_condition, dirt_condition) >= 3）累積勝率
    #    Bayesian smoothing: prior=0.08, beta=20
    #    turf_condition または dirt_condition が 0 の列は除外（max で処理）
    # ==================================================================
    if "jockey_code" in df.columns:
        _PRIOR_JH, _BETA_JH = 0.08, 20.0
        # max(turf_condition, dirt_condition) を計算（0=未記録は考慮済み）
        max_cond = pd.concat([turf_cond.rename("t"), dirt_cond.rename("d")], axis=1).max(axis=1)
        # max_cond >= 3 かつ少なくとも一方が 0 でないこと（両方0=条件未記録は除外）
        both_zero = (turf_cond == 0) & (dirt_cond == 0)
        is_heavy_race = ((max_cond >= 3) & ~both_zero).astype("int8")
        jockey_heavy_win = (win_flag * is_heavy_race).astype("int8")

        cum_jh_runs = is_heavy_race.groupby(df["jockey_code"], sort=False).cumsum() - is_heavy_race
        cum_jh_wins = jockey_heavy_win.groupby(df["jockey_code"], sort=False).cumsum() - jockey_heavy_win

        jockey_heavy_rate = (
            (cum_jh_wins + _BETA_JH * _PRIOR_JH) / (cum_jh_runs + _BETA_JH)
        )
        df["jockey_heavy_win_rate"] = jockey_heavy_rate.where(cum_jh_runs > 0, np.nan)
    else:
        df["jockey_heavy_win_rate"] = np.nan

    if source_is_gpu and _GPU_AVAILABLE:
        try:
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
        except Exception:
            pass

    return df


def add_going_condition_v8_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v8 用の one-hot flags + 稍重専用 Bayesian 特徴量 + 交互作用特徴量を追加する。

    v7 の add_going_condition_features() が呼ばれた後に実行することを前提とする。
    以下の v7 列が df に存在すること:
        horse_turf_heavy_win_rate, horse_turf_light_win_rate,
        horse_dirt_heavy_win_rate, horse_dirt_light_win_rate

    Group B: one-hot flags（6列）
        turf_cond_2  -- 芝稍重フラグ (is_turf==1 & turf_condition==2)
        turf_cond_3  -- 芝重フラグ   (is_turf==1 & turf_condition==3)
        turf_cond_4  -- 芝不良フラグ (is_turf==1 & turf_condition==4)
        dirt_cond_2  -- ダート稍重フラグ
        dirt_cond_3  -- ダート重フラグ
        dirt_cond_4  -- ダート不良フラグ

    Group A: 稍重専用 Bayesian 特徴量（4列）
        horse_turf_soft_win_rate  -- 芝稍重累積勝率 (prior=0.07, beta=5)
        horse_turf_soft_n_runs    -- 芝稍重出走数
        horse_dirt_soft_win_rate  -- ダート稍重累積勝率 (prior=0.07, beta=5)
        horse_dirt_soft_n_runs    -- ダート稍重出走数

    Group A: interaction features（6列）
        going_x_turf_heavy_winrate  -- turf_cond_3 * horse_turf_heavy_win_rate
        going_x_turf_light_winrate  -- (良馬場フラグ) * horse_turf_light_win_rate
        going_x_turf_soft_winrate   -- turf_cond_2 * horse_turf_soft_win_rate
        going_x_dirt_heavy_winrate  -- dirt_cond_3 * horse_dirt_heavy_win_rate
        going_match_score_turf      -- 稍重・重・不良それぞれの条件付き芝勝率合算
        going_match_score_dirt      -- ダート版

    NaN 伝播: 掛け合わせる片方が NaN の場合は結果も NaN（pandas デフォルト動作）。
    one-hot flags はリークなし（当日の馬場状態コードであり過去集計ではない）。
    """
    # GPU DataFrame は CPU に落として処理する
    source_is_gpu = _is_gpu_df(df)
    if source_is_gpu:
        df = _to_pandas_df(df)

    # sort 保証
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # track_code から芝/ダートを判定（v7 と同じロジック）
    # ------------------------------------------------------------------
    tc = pd.to_numeric(df.get("track_code", pd.Series(0, index=df.index)), errors="coerce")
    is_turf = (tc.between(10, 19)).astype("float32")
    is_dirt = (tc.between(20, 29)).astype("float32")

    turf_cond = pd.to_numeric(
        df.get("turf_condition", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)
    dirt_cond = pd.to_numeric(
        df.get("dirt_condition", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)

    # ==================================================================
    # Group B: one-hot flags（当日の馬場状態コード — リークなし）
    # turf_condition/dirt_condition が 0 の場合は 0 扱い（欠損ではない）
    # ==================================================================
    if "turf_cond_2" not in df.columns:
        df["turf_cond_2"] = ((is_turf == 1) & (turf_cond == 2)).astype("float32")
    if "turf_cond_3" not in df.columns:
        df["turf_cond_3"] = ((is_turf == 1) & (turf_cond == 3)).astype("float32")
    if "turf_cond_4" not in df.columns:
        df["turf_cond_4"] = ((is_turf == 1) & (turf_cond == 4)).astype("float32")
    if "dirt_cond_2" not in df.columns:
        df["dirt_cond_2"] = ((is_dirt == 1) & (dirt_cond == 2)).astype("float32")
    if "dirt_cond_3" not in df.columns:
        df["dirt_cond_3"] = ((is_dirt == 1) & (dirt_cond == 3)).astype("float32")
    if "dirt_cond_4" not in df.columns:
        df["dirt_cond_4"] = ((is_dirt == 1) & (dirt_cond == 4)).astype("float32")

    # ==================================================================
    # Group A: 稍重専用 Bayesian 特徴量（v7 の heavy と同じパターン）
    # 勝利フラグは finish_rank を参照
    # ==================================================================
    if "finish_rank" in df.columns:
        finish = pd.to_numeric(df["finish_rank"], errors="coerce")
        win_flag = (finish == 1).astype("int8")
    else:
        win_flag = pd.Series(np.zeros(len(df), dtype="int8"), index=df.index)

    _PRIOR_SOFT, _BETA_SOFT = 0.07, 5.0

    # horse_turf_soft_win_rate: 芝稍重 (is_turf==1 & turf_condition==2)
    if "horse_turf_soft_win_rate" not in df.columns:
        is_turf_soft = ((is_turf == 1) & (turf_cond == 2)).astype("int8")
        turf_soft_win = (win_flag * is_turf_soft).astype("int8")

        cum_ts_runs = is_turf_soft.groupby(df["ketto_num"], sort=False).cumsum() - is_turf_soft
        cum_ts_wins = turf_soft_win.groupby(df["ketto_num"], sort=False).cumsum() - turf_soft_win

        turf_soft_rate = (cum_ts_wins + _BETA_SOFT * _PRIOR_SOFT) / (cum_ts_runs + _BETA_SOFT)
        # 条件合致が0回の場合は NaN（プライアーへ収束させない）
        df["horse_turf_soft_win_rate"] = turf_soft_rate.where(cum_ts_runs > 0, np.nan)
        df["horse_turf_soft_n_runs"] = cum_ts_runs.astype("int16")

    # horse_dirt_soft_win_rate: ダート稍重 (is_dirt==1 & dirt_condition==2)
    if "horse_dirt_soft_win_rate" not in df.columns:
        is_dirt_soft = ((is_dirt == 1) & (dirt_cond == 2)).astype("int8")
        dirt_soft_win = (win_flag * is_dirt_soft).astype("int8")

        cum_ds_runs = is_dirt_soft.groupby(df["ketto_num"], sort=False).cumsum() - is_dirt_soft
        cum_ds_wins = dirt_soft_win.groupby(df["ketto_num"], sort=False).cumsum() - dirt_soft_win

        dirt_soft_rate = (cum_ds_wins + _BETA_SOFT * _PRIOR_SOFT) / (cum_ds_runs + _BETA_SOFT)
        df["horse_dirt_soft_win_rate"] = dirt_soft_rate.where(cum_ds_runs > 0, np.nan)
        df["horse_dirt_soft_n_runs"] = cum_ds_runs.astype("int16")

    # ==================================================================
    # Group A: interaction features（v7 特徴量 × Group B one-hot）
    # 片方が NaN の場合は積も NaN（pandas デフォルト動作）
    # ==================================================================

    # going_x_turf_heavy_winrate: 芝重時の重馬場勝率
    if "going_x_turf_heavy_winrate" not in df.columns:
        if "horse_turf_heavy_win_rate" in df.columns:
            df["going_x_turf_heavy_winrate"] = (
                df["turf_cond_3"] * df["horse_turf_heavy_win_rate"]
            )
        else:
            df["going_x_turf_heavy_winrate"] = np.nan

    # going_x_turf_light_winrate: 芝良馬場時の良馬場勝率
    # 良馬場フラグ = 稍重・重・不良以外（turf_cond_2 + turf_cond_3 + turf_cond_4 = 0 かつ is_turf=1）
    if "going_x_turf_light_winrate" not in df.columns:
        if "horse_turf_light_win_rate" in df.columns:
            turf_good_flag = (
                (df["turf_cond_2"] + df["turf_cond_3"] + df["turf_cond_4"]).clip(0, 1)
            )
            turf_light_flag = (is_turf * (1.0 - turf_good_flag)).clip(0, 1)
            df["going_x_turf_light_winrate"] = (
                turf_light_flag * df["horse_turf_light_win_rate"]
            )
        else:
            df["going_x_turf_light_winrate"] = np.nan

    # going_x_turf_soft_winrate: 芝稍重時の稍重勝率
    if "going_x_turf_soft_winrate" not in df.columns:
        if "horse_turf_soft_win_rate" in df.columns:
            df["going_x_turf_soft_winrate"] = (
                df["turf_cond_2"] * df["horse_turf_soft_win_rate"]
            )
        else:
            df["going_x_turf_soft_winrate"] = np.nan

    # going_x_dirt_heavy_winrate: ダート重時の重馬場勝率
    if "going_x_dirt_heavy_winrate" not in df.columns:
        if "horse_dirt_heavy_win_rate" in df.columns:
            df["going_x_dirt_heavy_winrate"] = (
                df["dirt_cond_3"] * df["horse_dirt_heavy_win_rate"]
            )
        else:
            df["going_x_dirt_heavy_winrate"] = np.nan

    # going_match_score_turf: 芝の馬場状態別勝率合算スコア
    # 稍重×稍重勝率 + 重×重勝率 + 不良×重勝率（不良の専用勝率がないため重勝率で代用）
    if "going_match_score_turf" not in df.columns:
        if "horse_turf_soft_win_rate" in df.columns and "horse_turf_heavy_win_rate" in df.columns:
            df["going_match_score_turf"] = (
                df["turf_cond_2"] * df["horse_turf_soft_win_rate"]
                + df["turf_cond_3"] * df["horse_turf_heavy_win_rate"]
                + df["turf_cond_4"] * df.get(
                    "horse_turf_very_heavy_win_rate", df["horse_turf_heavy_win_rate"]
                )
            )
        else:
            df["going_match_score_turf"] = np.nan

    # going_match_score_dirt: ダートの馬場状態別勝率合算スコア
    if "going_match_score_dirt" not in df.columns:
        if "horse_dirt_soft_win_rate" in df.columns and "horse_dirt_heavy_win_rate" in df.columns:
            df["going_match_score_dirt"] = (
                df["dirt_cond_2"] * df["horse_dirt_soft_win_rate"]
                + df["dirt_cond_3"] * df["horse_dirt_heavy_win_rate"]
                + df["dirt_cond_4"] * df.get(
                    "horse_dirt_very_heavy_win_rate", df["horse_dirt_heavy_win_rate"]
                )
            )
        else:
            df["going_match_score_dirt"] = np.nan

    if source_is_gpu and _GPU_AVAILABLE:
        try:
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
        except Exception:
            pass

    return df


def add_going_condition_v9_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v9 用の芝稍重スコア分化特徴量（8列）を追加する。

    v8 の add_going_condition_v8_features() が呼ばれた後に実行することを前提とする。
    以下の v8 列が df に存在すること:
        horse_turf_soft_n_runs, horse_turf_soft_win_rate,
        horse_turf_light_win_rate, turf_cond_2, turf_cond_3, turf_cond_4

    追加特徴量（8列）:
        horse_turf_soft_win_rate_bayes  -- 稍重未経験馬を Bayesian impute した芝稍重勝率
        horse_turf_soft_top3_rate       -- 芝稍重3着内率（prior=0.20, beta=5）
        soft_minus_good_rank_diff       -- 稍重着順平均 - 良着順平均（負=稍重得意）
        soft_experience_flag            -- 稍重3走以上経験フラグ
        soft_exp_x_win_rate             -- 経験フラグ × Bayesian勝率（交互作用）
        trainer_turf_soft_win_rate      -- 調教師の芝稍重勝率（prior=0.08, beta=10）
        jockey_turf_soft_win_rate       -- 騎手の芝稍重勝率（prior=0.09, beta=15）
        going_x_soft_win_rate_imputed   -- turf_cond_2 × impute済み勝率

    リークチェック: 各列と is_win の Pearson 相関が 0.30 以上の場合 ValueError を raise。
    """
    # GPU DataFrame は CPU に落として処理する
    source_is_gpu = _is_gpu_df(df)
    if source_is_gpu:
        df = _to_pandas_df(df)

    # sort 保証（v8 と同じ）
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 共通前処理: track_code・馬場コード・勝利フラグ
    # ------------------------------------------------------------------
    tc = pd.to_numeric(df.get("track_code", pd.Series(0, index=df.index)), errors="coerce")
    is_turf = (tc.between(10, 19)).astype("float32")

    turf_cond = pd.to_numeric(
        df.get("turf_condition", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)

    if "finish_rank" in df.columns:
        finish_num = pd.to_numeric(df["finish_rank"], errors="coerce")
        win_flag = (finish_num == 1).astype("int8")
    else:
        finish_num = pd.Series(np.nan, index=df.index)
        win_flag = pd.Series(np.zeros(len(df), dtype="int8"), index=df.index)

    # 芝稍重フラグ（is_turf==1 & turf_condition==2）
    is_turf_soft = ((is_turf == 1) & (turf_cond == 2)).astype("int8")

    # v8 で計算済みの cumsum を再利用するか、ここで再計算する
    # v8 の horse_turf_soft_n_runs = cumsum(is_turf_soft) - is_turf_soft なので同じ値
    cum_ts_runs = is_turf_soft.groupby(df["ketto_num"], sort=False).cumsum() - is_turf_soft
    cum_ts_wins = (win_flag * is_turf_soft).astype("int8").groupby(df["ketto_num"], sort=False).cumsum() - (win_flag * is_turf_soft).astype("int8")

    # ==================================================================
    # 1. horse_turf_soft_win_rate_bayes（Bayesian impute 版勝率）
    #    prior=0.07, beta=5。稍重経験0の馬は horse_turf_light_win_rate で fill、
    #    それも NaN なら 0.07 で fill。
    # ==================================================================
    _PRIOR_SOFT_BAYES, _BETA_SOFT_BAYES = 0.07, 5.0
    soft_rate_raw = (cum_ts_wins + _BETA_SOFT_BAYES * _PRIOR_SOFT_BAYES) / (cum_ts_runs + _BETA_SOFT_BAYES)

    # 稍重経験0の馬: horse_turf_light_win_rate → 0.07 の順で fill
    if "horse_turf_light_win_rate" in df.columns:
        fallback = df["horse_turf_light_win_rate"].fillna(0.07)
    else:
        fallback = pd.Series(0.07, index=df.index)

    horse_turf_soft_win_rate_bayes = soft_rate_raw.where(cum_ts_runs > 0, fallback)
    df["horse_turf_soft_win_rate_bayes"] = horse_turf_soft_win_rate_bayes.astype("float32")

    # ==================================================================
    # 2. horse_turf_soft_top3_rate（稍重3着内率）
    #    prior=0.20, beta=5。経験0の馬は 0.20 で fill。
    # ==================================================================
    _PRIOR_TOP3, _BETA_TOP3 = 0.20, 5.0
    is_top3 = (finish_num <= 3).fillna(False).astype("int8")
    top3_soft = (is_top3 * is_turf_soft).astype("int8")
    cum_ts_top3 = top3_soft.groupby(df["ketto_num"], sort=False).cumsum() - top3_soft
    top3_rate = (cum_ts_top3 + _BETA_TOP3 * _PRIOR_TOP3) / (cum_ts_runs + _BETA_TOP3)
    df["horse_turf_soft_top3_rate"] = top3_rate.where(cum_ts_runs > 0, _PRIOR_TOP3).astype("float32")

    # ==================================================================
    # 3. soft_minus_good_rank_diff（稍重着順平均 - 良着順平均）
    #    負値 = 稍重のほうが着順が上 = 稍重得意。
    #    shift(1).expanding().mean() で時系列リーク防止。
    #    片方 or 両方 NaN → 0（中立仮定）。
    # ==================================================================
    is_turf_good = ((is_turf == 1) & (turf_cond == 1)).astype(bool)
    is_turf_soft_bool = is_turf_soft.astype(bool)

    # 稍重・良それぞれの着順だけを残し、それ以外は NaN
    finish_soft_series = finish_num.where(is_turf_soft_bool)
    finish_good_series = finish_num.where(is_turf_good)

    soft_mean = (
        finish_soft_series
        .groupby(df["ketto_num"], sort=False)
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    good_mean = (
        finish_good_series
        .groupby(df["ketto_num"], sort=False)
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["soft_minus_good_rank_diff"] = (soft_mean - good_mean).fillna(0.0).astype("float32")

    # ==================================================================
    # 4. soft_experience_flag（稍重3走以上経験フラグ）
    # ==================================================================
    # horse_turf_soft_n_runs = cum_ts_runs（v8 で計算済みの値と同一）
    if "horse_turf_soft_n_runs" in df.columns:
        n_runs_ref = df["horse_turf_soft_n_runs"]
    else:
        n_runs_ref = cum_ts_runs
    soft_experience_flag = (n_runs_ref >= 3).astype("int8")
    df["soft_experience_flag"] = soft_experience_flag

    # ==================================================================
    # 5. soft_exp_x_win_rate（経験フラグ × Bayesian勝率）
    # ==================================================================
    df["soft_exp_x_win_rate"] = (soft_experience_flag * horse_turf_soft_win_rate_bayes).astype("float32")

    # ==================================================================
    # 6. trainer_turf_soft_win_rate（調教師の芝稍重勝率）
    #    prior=0.08, beta=10。経験0 → 0.08 で fill。
    # ==================================================================
    _PRIOR_TR, _BETA_TR = 0.08, 10.0
    if "trainer_code" in df.columns:
        cum_tr_runs = is_turf_soft.groupby(df["trainer_code"], sort=False).cumsum() - is_turf_soft
        cum_tr_wins = (win_flag * is_turf_soft).astype("int8").groupby(df["trainer_code"], sort=False).cumsum() - (win_flag * is_turf_soft).astype("int8")
        tr_rate = (cum_tr_wins + _BETA_TR * _PRIOR_TR) / (cum_tr_runs + _BETA_TR)
        df["trainer_turf_soft_win_rate"] = tr_rate.where(cum_tr_runs > 0, _PRIOR_TR).astype("float32")
    else:
        df["trainer_turf_soft_win_rate"] = np.float32(_PRIOR_TR)

    # ==================================================================
    # 7. jockey_turf_soft_win_rate（騎手の芝稍重勝率）
    #    prior=0.09, beta=15。経験0 → 0.09 で fill。
    # ==================================================================
    _PRIOR_JK, _BETA_JK = 0.09, 15.0
    if "jockey_code" in df.columns:
        cum_jk_runs = is_turf_soft.groupby(df["jockey_code"], sort=False).cumsum() - is_turf_soft
        cum_jk_wins = (win_flag * is_turf_soft).astype("int8").groupby(df["jockey_code"], sort=False).cumsum() - (win_flag * is_turf_soft).astype("int8")
        jk_rate = (cum_jk_wins + _BETA_JK * _PRIOR_JK) / (cum_jk_runs + _BETA_JK)
        df["jockey_turf_soft_win_rate"] = jk_rate.where(cum_jk_runs > 0, _PRIOR_JK).astype("float32")
    else:
        df["jockey_turf_soft_win_rate"] = np.float32(_PRIOR_JK)

    # ==================================================================
    # 8. going_x_soft_win_rate_imputed（turf_cond_2 × impute済み勝率）
    #    v8 の turf_cond_2 one-hot フラグを使う。
    # ==================================================================
    turf_cond_2_col = df.get("turf_cond_2", pd.Series(0.0, index=df.index))
    df["going_x_soft_win_rate_imputed"] = (turf_cond_2_col * horse_turf_soft_win_rate_bayes).astype("float32")

    if source_is_gpu and _GPU_AVAILABLE:
        try:
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
        except Exception:
            pass

    return df


def generate_features_past_v9(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    v8 parquet を読み込み、v9 特徴量（芝稍重スコア分化 8列）を追加して
    features_past_v9.parquet として保存する。

    追加特徴量:
        horse_turf_soft_win_rate_bayes  -- Bayesian impute 版稍重勝率
        horse_turf_soft_top3_rate       -- 稍重3着内率
        soft_minus_good_rank_diff       -- 稍重着順平均 - 良着順平均
        soft_experience_flag            -- 稍重3走以上経験フラグ
        soft_exp_x_win_rate             -- 経験フラグ × Bayesian勝率
        trainer_turf_soft_win_rate      -- 調教師の芝稍重勝率
        jockey_turf_soft_win_rate       -- 騎手の芝稍重勝率
        going_x_soft_win_rate_imputed   -- turf_cond_2 × impute済み勝率

    安全対策:
        - features_past_v8.parquet を上書きしない（assert で保証）
        - 既存 v9 ファイルは _bak.parquet へバックアップ
        - リークチェック: 新規列と is_win の Pearson 相関 >= 0.30 で ValueError
    """
    import shutil

    _FEAT_DIR = PROJECT_ROOT / "model_training/data/02_features"

    inp = Path(input_path or (_FEAT_DIR / "features_past_v8.parquet"))
    out = Path(output_path or (_FEAT_DIR / "features_past_v9.parquet"))
    man = Path(manifest_path or (_FEAT_DIR / "features_past_v9_manifest.json"))

    print(f"[v9] Reading v8 from: {inp}")
    if not inp.is_file():
        raise FileNotFoundError(f"[v9] 入力ファイルが存在しません: {inp}")

    # v8 を上書きしない（絶対パスで比較）
    assert out.resolve() != inp.resolve(), (
        "[v9] 出力パスが入力パスと同じです。v8 の上書きを防ぐため中断します。"
    )

    # 既存 v9 ファイルをバックアップ（破壊的上書き防止）
    bak = out.with_name(out.stem + "_bak" + out.suffix)
    if out.is_file():
        shutil.copy2(out, bak)
        print(f"[v9] Backed up existing v9 to: {bak}")

    df = pd.read_parquet(inp)
    print(f"[v9] v8 shape: {df.shape}")

    df["date"] = pd.to_datetime(df["date"])

    # v8 の going condition 列が存在しない場合は先に生成する
    v8_required = [
        "horse_turf_soft_n_runs",
        "horse_turf_soft_win_rate",
        "turf_cond_2",
    ]
    missing_v8 = [c for c in v8_required if c not in df.columns]
    if missing_v8:
        print(f"[v9] v8 列が不足しているため add_going_condition_v8_features を実行: {missing_v8}")
        # v7 列も必要な場合は先に補完
        v7_required = [
            "horse_turf_heavy_win_rate",
            "horse_turf_light_win_rate",
            "horse_dirt_heavy_win_rate",
            "horse_dirt_light_win_rate",
        ]
        missing_v7 = [c for c in v7_required if c not in df.columns]
        if missing_v7:
            print(f"[v9] v7 列も不足しているため add_going_condition_features を実行: {missing_v7}")
            df = add_going_condition_features(df)
        df = add_going_condition_v8_features(df)

    print("[v9] Adding v9 features (turf-soft score differentiation)...")
    df = add_going_condition_v9_features(df)

    NEW_COLS = [
        "horse_turf_soft_win_rate_bayes",
        "horse_turf_soft_top3_rate",
        "soft_minus_good_rank_diff",
        "soft_experience_flag",
        "soft_exp_x_win_rate",
        "trainer_turf_soft_win_rate",
        "jockey_turf_soft_win_rate",
        "going_x_soft_win_rate_imputed",
    ]
    present_new_cols = [c for c in NEW_COLS if c in df.columns]
    print(f"[v9] Final shape: {df.shape}")
    print(f"[v9] New cols added ({len(present_new_cols)}): {present_new_cols}")

    # --- NaN 率レポート ---
    nan_report: dict = {}
    print("[v9] NaN rates for new features:")
    for col in present_new_cols:
        nan_rate = float(df[col].isna().mean())
        nan_report[col] = nan_rate
        print(f"  {col}: {nan_rate:.3%}")

    # --- リークチェック: 各新規特徴量と is_win の Pearson 相関 ---
    print("[v9] Leak check (Pearson correlation with is_win):")
    is_win_series = (pd.to_numeric(df["finish_rank"], errors="coerce") == 1).astype(float)
    leak_report: dict = {}
    for col in present_new_cols:
        valid_mask = df[col].notna() & is_win_series.notna()
        if valid_mask.sum() > 100:
            corr = float(df.loc[valid_mask, col].astype(float).corr(is_win_series[valid_mask]))
            leak_report[col] = corr
            flag = "WARN(>=0.30)" if abs(corr) >= 0.30 else "OK"
            print(f"  {col}: corr={corr:.4f} [{flag}]")
            if abs(corr) >= 0.30:
                raise ValueError(
                    f"[v9] リーク検出: {col} の is_win 相関が {corr:.4f} >= 0.30 です。"
                    f" shift(1)/cumsum-self の実装を確認してください。"
                )
        else:
            print(f"  {col}: サンプル不足 (valid_mask.sum={valid_mask.sum()}) — スキップ")

    # --- 保存 ---
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[v9] Saved parquet: {out}")

    # --- マニフェスト生成 ---
    manifest = {
        "name": "features_past_v9",
        "source": str(inp.name),
        "rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "new_columns": present_new_cols,
        "nan_rates_new_features": nan_report,
        "leak_check_corr_with_iswin": leak_report,
        "date_range": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[v9] Manifest saved: {man}")

    # --- 読み込み確認（保存後の shape を verify） ---
    verify = pd.read_parquet(out)
    print(f"[v9] Verified shape from saved file: {verify.shape}")

    return df


def add_going_condition_v10_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v10 馬場relay特徴量を追加する（計19列）。

    v7/v8/v9 の列が df に存在することを前提とする。
    不足している場合は当該関数を呼び出して補完する。

    【グループ1】4段階個別勝率（relay featureの素材 / 11列）:
        horse_turf_heavy3_win_rate      -- 芝重(code==3)のみの勝率 (prior=0.07, beta=5)
        horse_turf_heavy3_n_runs        -- 芝重(code==3)出走数
        horse_turf_very_heavy_win_rate  -- 芝不良(code==4)のみの勝率 (prior=0.06, beta=3)
        horse_turf_very_heavy_n_runs    -- 芝不良(code==4)出走数
        horse_turf_soft_win_rate_v10    -- 芝稍重(code==2)勝率（v9の再実装版 prior=0.07, beta=5）
        horse_turf_soft_top3_rate_v10   -- 芝稍重3着内率 (prior=0.20, beta=5)
        horse_turf_heavy3_top3_rate     -- 芝重3着内率 (prior=0.22, beta=5)
        horse_dirt_heavy3_win_rate      -- ダート重(code==3)勝率 (prior=0.08, beta=5)
        horse_dirt_heavy3_n_runs        -- ダート重(code==3)出走数
        horse_dirt_very_heavy_win_rate  -- ダート不良(code==4)勝率 (prior=0.07, beta=3)
        horse_dirt_very_heavy_n_runs    -- ダート不良出走数

    【グループ2】relay features（zero-injection回避 / 3列）:
        current_going_win_rate_turf     -- 現在の芝馬場コードに対応する個別勝率
        current_going_win_rate_dirt     -- 現在のダート馬場コードに対応する個別勝率
        current_going_top3_rate_turf    -- 現在の芝馬場コードに対応する3着内率

    【グループ3】変化系特徴量（3列）:
        going_change_lag1               -- 前走からの馬場コード差分（正=悪化、負=改善）
        going_worsening_flag            -- 前走より馬場が悪化したフラグ
        horse_going_recovery_rate       -- 馬場悪化時勝率/全体勝率（1以上=得意）

    【グループ4】騎手・調教師特徴量（2列）:
        jockey_turf_heavy3_win_rate     -- 騎手の芝重勝率 (prior=0.09, beta=15)
        trainer_turf_heavy3_win_rate    -- 調教師の芝重勝率 (prior=0.08, beta=10)

    リークチェック: 各列と is_win の Pearson 相関 >= 0.30 で ValueError を raise。
    """
    # GPU DataFrame は CPU に落として処理する
    source_is_gpu = _is_gpu_df(df)
    if source_is_gpu:
        df = _to_pandas_df(df)

    # sort 保証（v8/v9 と同じ）
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 共通前処理: track_code・馬場コード・勝利フラグ・3着以内フラグ
    # ------------------------------------------------------------------
    tc = pd.to_numeric(df.get("track_code", pd.Series(0, index=df.index)), errors="coerce")
    is_turf = (tc.between(10, 19)).astype("float32")
    is_dirt = (tc.between(20, 29)).astype("float32")

    turf_cond = pd.to_numeric(
        df.get("turf_condition", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)
    dirt_cond = pd.to_numeric(
        df.get("dirt_condition", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)

    if "finish_rank" in df.columns:
        finish_num = pd.to_numeric(df["finish_rank"], errors="coerce")
        win_flag = (finish_num == 1).astype("int8")
        top3_flag = (finish_num <= 3).fillna(False).astype("int8")
    else:
        finish_num = pd.Series(np.nan, index=df.index)
        win_flag = pd.Series(np.zeros(len(df), dtype="int8"), index=df.index)
        top3_flag = pd.Series(np.zeros(len(df), dtype="int8"), index=df.index)

    horse_id = df["ketto_num"]

    # ==================================================================
    # グループ1: 4段階個別勝率
    # cumsum - self でshift(1)相当のリーク防止を実現する
    # ==================================================================

    # --- 芝重(code==3) ---
    _PRIOR_TH3, _BETA_TH3 = 0.07, 5.0
    is_turf_heavy3 = ((is_turf == 1) & (turf_cond == 3)).astype("int8")
    cum_th3_runs = is_turf_heavy3.groupby(horse_id, sort=False).cumsum() - is_turf_heavy3
    cum_th3_wins = (win_flag * is_turf_heavy3).astype("int8").groupby(horse_id, sort=False).cumsum() \
                   - (win_flag * is_turf_heavy3).astype("int8")
    th3_rate = (cum_th3_wins + _BETA_TH3 * _PRIOR_TH3) / (cum_th3_runs + _BETA_TH3)
    df["horse_turf_heavy3_win_rate"] = th3_rate.where(cum_th3_runs > 0, np.nan).astype("float32")
    df["horse_turf_heavy3_n_runs"] = cum_th3_runs.astype("int16")

    # --- 芝不良(code==4) ---
    _PRIOR_TVH, _BETA_TVH = 0.06, 3.0
    is_turf_vh = ((is_turf == 1) & (turf_cond == 4)).astype("int8")
    cum_tvh_runs = is_turf_vh.groupby(horse_id, sort=False).cumsum() - is_turf_vh
    cum_tvh_wins = (win_flag * is_turf_vh).astype("int8").groupby(horse_id, sort=False).cumsum() \
                   - (win_flag * is_turf_vh).astype("int8")
    tvh_rate = (cum_tvh_wins + _BETA_TVH * _PRIOR_TVH) / (cum_tvh_runs + _BETA_TVH)
    df["horse_turf_very_heavy_win_rate"] = tvh_rate.where(cum_tvh_runs > 0, np.nan).astype("float32")
    df["horse_turf_very_heavy_n_runs"] = cum_tvh_runs.astype("int16")

    # --- 芝稍重(code==2) v10版（v9のhorse_turf_soft_win_rate_bayesと別実装） ---
    _PRIOR_TS2, _BETA_TS2 = 0.07, 5.0
    is_turf_soft2 = ((is_turf == 1) & (turf_cond == 2)).astype("int8")
    cum_ts2_runs = is_turf_soft2.groupby(horse_id, sort=False).cumsum() - is_turf_soft2
    cum_ts2_wins = (win_flag * is_turf_soft2).astype("int8").groupby(horse_id, sort=False).cumsum() \
                   - (win_flag * is_turf_soft2).astype("int8")
    ts2_rate = (cum_ts2_wins + _BETA_TS2 * _PRIOR_TS2) / (cum_ts2_runs + _BETA_TS2)
    # 経験0の馬は horse_turf_light_win_rate → 0.07 の順でfallback
    if "horse_turf_light_win_rate" in df.columns:
        fallback_ts2 = df["horse_turf_light_win_rate"].fillna(0.07)
    else:
        fallback_ts2 = pd.Series(0.07, index=df.index)
    df["horse_turf_soft_win_rate_v10"] = ts2_rate.where(cum_ts2_runs > 0, fallback_ts2).astype("float32")

    # --- 芝稍重3着内率 v10版 ---
    _PRIOR_TS2_TOP3, _BETA_TS2_TOP3 = 0.20, 5.0
    cum_ts2_top3 = (top3_flag * is_turf_soft2).astype("int8").groupby(horse_id, sort=False).cumsum() \
                   - (top3_flag * is_turf_soft2).astype("int8")
    ts2_top3_rate = (cum_ts2_top3 + _BETA_TS2_TOP3 * _PRIOR_TS2_TOP3) / (cum_ts2_runs + _BETA_TS2_TOP3)
    df["horse_turf_soft_top3_rate_v10"] = ts2_top3_rate.where(cum_ts2_runs > 0, _PRIOR_TS2_TOP3).astype("float32")

    # --- 芝重3着内率 ---
    _PRIOR_TH3_TOP3, _BETA_TH3_TOP3 = 0.22, 5.0
    cum_th3_top3 = (top3_flag * is_turf_heavy3).astype("int8").groupby(horse_id, sort=False).cumsum() \
                   - (top3_flag * is_turf_heavy3).astype("int8")
    th3_top3_rate = (cum_th3_top3 + _BETA_TH3_TOP3 * _PRIOR_TH3_TOP3) / (cum_th3_runs + _BETA_TH3_TOP3)
    df["horse_turf_heavy3_top3_rate"] = th3_top3_rate.where(cum_th3_runs > 0, _PRIOR_TH3_TOP3).astype("float32")

    # --- ダート重(code==3) ---
    _PRIOR_DH3, _BETA_DH3 = 0.08, 5.0
    is_dirt_heavy3 = ((is_dirt == 1) & (dirt_cond == 3)).astype("int8")
    cum_dh3_runs = is_dirt_heavy3.groupby(horse_id, sort=False).cumsum() - is_dirt_heavy3
    cum_dh3_wins = (win_flag * is_dirt_heavy3).astype("int8").groupby(horse_id, sort=False).cumsum() \
                   - (win_flag * is_dirt_heavy3).astype("int8")
    dh3_rate = (cum_dh3_wins + _BETA_DH3 * _PRIOR_DH3) / (cum_dh3_runs + _BETA_DH3)
    df["horse_dirt_heavy3_win_rate"] = dh3_rate.where(cum_dh3_runs > 0, np.nan).astype("float32")
    df["horse_dirt_heavy3_n_runs"] = cum_dh3_runs.astype("int16")

    # --- ダート不良(code==4) ---
    _PRIOR_DVH, _BETA_DVH = 0.07, 3.0
    is_dirt_vh = ((is_dirt == 1) & (dirt_cond == 4)).astype("int8")
    cum_dvh_runs = is_dirt_vh.groupby(horse_id, sort=False).cumsum() - is_dirt_vh
    cum_dvh_wins = (win_flag * is_dirt_vh).astype("int8").groupby(horse_id, sort=False).cumsum() \
                   - (win_flag * is_dirt_vh).astype("int8")
    dvh_rate = (cum_dvh_wins + _BETA_DVH * _PRIOR_DVH) / (cum_dvh_runs + _BETA_DVH)
    df["horse_dirt_very_heavy_win_rate"] = dvh_rate.where(cum_dvh_runs > 0, np.nan).astype("float32")
    df["horse_dirt_very_heavy_n_runs"] = cum_dvh_runs.astype("int16")

    # ==================================================================
    # グループ2: relay features（zero-injection回避）
    # 現在の馬場コードに応じて対応する過去勝率を「中継ぎ」する。
    # fallback chain で全馬に値が入るためレース内分散=0を回避できる。
    # ==================================================================

    # relay helper: 芝版
    # コード1(良): horse_turf_light_win_rate → 0.07
    # コード2(稍重): horse_turf_soft_win_rate_v10 → light → 0.07
    # コード3(重): horse_turf_heavy3_win_rate → soft → light → 0.07
    # コード4(不良): horse_turf_very_heavy_win_rate → heavy3 → soft → light → 0.06
    turf_cond_int = pd.to_numeric(df.get("turf_condition", pd.Series(1, index=df.index)), errors="coerce").fillna(1).astype(int)

    light_rate = df.get("horse_turf_light_win_rate", pd.Series(np.nan, index=df.index))
    soft_v10_rate = df["horse_turf_soft_win_rate_v10"]
    heavy3_rate = df["horse_turf_heavy3_win_rate"]
    very_heavy_rate = df["horse_turf_very_heavy_win_rate"]

    relay_turf = pd.Series(np.nan, index=df.index, dtype="float32")

    mask1 = (turf_cond_int == 1)
    relay_turf[mask1] = light_rate[mask1].fillna(0.07)

    mask2 = (turf_cond_int == 2)
    relay_turf[mask2] = soft_v10_rate[mask2].fillna(light_rate[mask2]).fillna(0.07)

    mask3 = (turf_cond_int == 3)
    relay_turf[mask3] = (
        heavy3_rate[mask3]
        .fillna(soft_v10_rate[mask3])
        .fillna(light_rate[mask3])
        .fillna(0.07)
    )

    mask4 = (turf_cond_int == 4)
    relay_turf[mask4] = (
        very_heavy_rate[mask4]
        .fillna(heavy3_rate[mask4])
        .fillna(soft_v10_rate[mask4])
        .fillna(light_rate[mask4])
        .fillna(0.06)
    )

    # 芝以外のレース（ダート等）では NaN のまま（LightGBM が NaN 処理）
    df["current_going_win_rate_turf"] = relay_turf

    # relay helper: ダート版
    # コード1(良): horse_dirt_light_win_rate → 0.08
    # コード2(稍重): horse_dirt_soft_win_rate → light → 0.08
    # コード3(重): horse_dirt_heavy3_win_rate → soft → light → 0.08
    # コード4(不良): horse_dirt_very_heavy_win_rate → heavy3 → soft → light → 0.07
    dirt_cond_int = pd.to_numeric(df.get("dirt_condition", pd.Series(1, index=df.index)), errors="coerce").fillna(1).astype(int)

    dirt_light_rate = df.get("horse_dirt_light_win_rate", pd.Series(np.nan, index=df.index))
    dirt_soft_rate = df.get("horse_dirt_soft_win_rate", pd.Series(np.nan, index=df.index))
    dirt_heavy3_rate = df["horse_dirt_heavy3_win_rate"]
    dirt_vh_rate = df["horse_dirt_very_heavy_win_rate"]

    relay_dirt = pd.Series(np.nan, index=df.index, dtype="float32")

    dmask1 = (dirt_cond_int == 1)
    relay_dirt[dmask1] = dirt_light_rate[dmask1].fillna(0.08)

    dmask2 = (dirt_cond_int == 2)
    relay_dirt[dmask2] = dirt_soft_rate[dmask2].fillna(dirt_light_rate[dmask2]).fillna(0.08)

    dmask3 = (dirt_cond_int == 3)
    relay_dirt[dmask3] = (
        dirt_heavy3_rate[dmask3]
        .fillna(dirt_soft_rate[dmask3])
        .fillna(dirt_light_rate[dmask3])
        .fillna(0.08)
    )

    dmask4 = (dirt_cond_int == 4)
    relay_dirt[dmask4] = (
        dirt_vh_rate[dmask4]
        .fillna(dirt_heavy3_rate[dmask4])
        .fillna(dirt_soft_rate[dmask4])
        .fillna(dirt_light_rate[dmask4])
        .fillna(0.07)
    )

    # ダート以外のレース（芝等）では NaN のまま
    df["current_going_win_rate_dirt"] = relay_dirt

    # relay helper: 芝3着内率版
    # コード1(良): horse_turf_light_win_rate で代用（3着内率専用列がなければ0.30）→ 0.30
    # コード2(稍重): horse_turf_soft_top3_rate_v10 → horse_turf_soft_top3_rate（v9）→ 0.20
    # コード3(重): horse_turf_heavy3_top3_rate → soft_top3 → 0.22
    # コード4(不良): horse_turf_heavy3_top3_rate（不良専用なし） → 0.22
    soft_top3_v10 = df["horse_turf_soft_top3_rate_v10"]
    # v9 で作成した horse_turf_soft_top3_rate があれば使う
    soft_top3_v9 = df.get("horse_turf_soft_top3_rate", pd.Series(np.nan, index=df.index))
    heavy3_top3 = df["horse_turf_heavy3_top3_rate"]

    relay_top3_turf = pd.Series(np.nan, index=df.index, dtype="float32")

    relay_top3_turf[mask1] = 0.30  # 良馬場の3着内率は全馬共通の prior で初期化

    relay_top3_turf[mask2] = (
        soft_top3_v10[mask2]
        .fillna(soft_top3_v9[mask2])
        .fillna(0.20)
    )

    relay_top3_turf[mask3] = (
        heavy3_top3[mask3]
        .fillna(soft_top3_v10[mask3])
        .fillna(soft_top3_v9[mask3])
        .fillna(0.22)
    )

    relay_top3_turf[mask4] = (
        heavy3_top3[mask4]
        .fillna(soft_top3_v10[mask4])
        .fillna(soft_top3_v9[mask4])
        .fillna(0.22)
    )

    df["current_going_top3_rate_turf"] = relay_top3_turf

    # ==================================================================
    # グループ3: 変化系特徴量
    # going_change_lag1: 前走からの馬場コード差分
    # 芝レースは turf_condition を、ダートレースは dirt_condition を使う。
    # 芝・ダート混在馬は surface ごとに別管理が理想だが実装簡易化のため
    # 直近レースで使用された馬場コードを継承する。
    # ==================================================================

    # 馬場コードを1列に統合（芝優先。芝でない場合はダート条件を使用）
    unified_cond = turf_cond.where(is_turf == 1, dirt_cond).replace(0, np.nan)

    # 前走の馬場コード（shift(1)でリーク防止）
    prev_cond = (
        unified_cond
        .groupby(horse_id, sort=False)
        .transform(lambda x: x.shift(1))
    )

    going_change = (unified_cond - prev_cond).astype("float32")
    df["going_change_lag1"] = going_change.fillna(0.0)

    # going_worsening_flag: 前走より馬場が悪化（コード増加）したフラグ
    df["going_worsening_flag"] = (going_change > 0).astype("int8")

    # horse_going_recovery_rate: 馬場悪化時の勝率 / 全体勝率
    # 悪化レース（going_change_lag1 > 0）での累積勝率 / 全累積勝率
    # 全累積勝率は v7 の horse_turf_heavy_win_rate 等を使うより
    # ここで再計算する（surface非依存・全レース通算）
    worsening_flag_int = (going_change > 0).fillna(False).astype("int8")
    cum_worsening_runs = worsening_flag_int.groupby(horse_id, sort=False).cumsum() - worsening_flag_int
    cum_worsening_wins = (win_flag * worsening_flag_int).astype("int8").groupby(horse_id, sort=False).cumsum() \
                         - (win_flag * worsening_flag_int).astype("int8")

    # 全体勝率（累積 / 出走数 — プライアー付き）
    cum_total_runs = pd.Series(np.ones(len(df), dtype="int8"), index=df.index).groupby(horse_id, sort=False).cumsum() \
                     - 1  # shift(1)相当
    cum_total_wins = win_flag.groupby(horse_id, sort=False).cumsum() - win_flag

    _PRIOR_RECOVERY, _BETA_RECOVERY = 0.07, 5.0
    overall_win_rate = (cum_total_wins + _BETA_RECOVERY * _PRIOR_RECOVERY) / (cum_total_runs + _BETA_RECOVERY)
    worsening_win_rate = (cum_worsening_wins + _BETA_RECOVERY * _PRIOR_RECOVERY) / (cum_worsening_runs + _BETA_RECOVERY)

    # 悪化経験が0回の馬は NaN（中立仮定として 1.0 でも可だが NaN にして LightGBM に委ねる）
    recovery_rate = (worsening_win_rate / overall_win_rate.replace(0, np.nan)).where(
        cum_worsening_runs > 0, np.nan
    )
    df["horse_going_recovery_rate"] = recovery_rate.astype("float32")

    # ==================================================================
    # グループ4: 騎手・調教師の芝重勝率
    # ==================================================================

    # jockey_turf_heavy3_win_rate（騎手の芝重勝率 / prior=0.09, beta=15）
    _PRIOR_JK_H3, _BETA_JK_H3 = 0.09, 15.0
    if "jockey_code" in df.columns:
        jk_id = df["jockey_code"]
        cum_jkh3_runs = is_turf_heavy3.groupby(jk_id, sort=False).cumsum() - is_turf_heavy3
        cum_jkh3_wins = (win_flag * is_turf_heavy3).astype("int8").groupby(jk_id, sort=False).cumsum() \
                        - (win_flag * is_turf_heavy3).astype("int8")
        jkh3_rate = (cum_jkh3_wins + _BETA_JK_H3 * _PRIOR_JK_H3) / (cum_jkh3_runs + _BETA_JK_H3)
        df["jockey_turf_heavy3_win_rate"] = jkh3_rate.where(cum_jkh3_runs > 0, _PRIOR_JK_H3).astype("float32")
    else:
        df["jockey_turf_heavy3_win_rate"] = np.float32(_PRIOR_JK_H3)

    # trainer_turf_heavy3_win_rate（調教師の芝重勝率 / prior=0.08, beta=10）
    _PRIOR_TR_H3, _BETA_TR_H3 = 0.08, 10.0
    if "trainer_code" in df.columns:
        tr_id = df["trainer_code"]
        cum_trh3_runs = is_turf_heavy3.groupby(tr_id, sort=False).cumsum() - is_turf_heavy3
        cum_trh3_wins = (win_flag * is_turf_heavy3).astype("int8").groupby(tr_id, sort=False).cumsum() \
                        - (win_flag * is_turf_heavy3).astype("int8")
        trh3_rate = (cum_trh3_wins + _BETA_TR_H3 * _PRIOR_TR_H3) / (cum_trh3_runs + _BETA_TR_H3)
        df["trainer_turf_heavy3_win_rate"] = trh3_rate.where(cum_trh3_runs > 0, _PRIOR_TR_H3).astype("float32")
    else:
        df["trainer_turf_heavy3_win_rate"] = np.float32(_PRIOR_TR_H3)

    # ==================================================================
    # NaN 率レポート
    # ==================================================================
    V10_NEW_COLS = [
        # グループ1
        "horse_turf_heavy3_win_rate",
        "horse_turf_heavy3_n_runs",
        "horse_turf_very_heavy_win_rate",
        "horse_turf_very_heavy_n_runs",
        "horse_turf_soft_win_rate_v10",
        "horse_turf_soft_top3_rate_v10",
        "horse_turf_heavy3_top3_rate",
        "horse_dirt_heavy3_win_rate",
        "horse_dirt_heavy3_n_runs",
        "horse_dirt_very_heavy_win_rate",
        "horse_dirt_very_heavy_n_runs",
        # グループ2
        "current_going_win_rate_turf",
        "current_going_win_rate_dirt",
        "current_going_top3_rate_turf",
        # グループ3
        "going_change_lag1",
        "going_worsening_flag",
        "horse_going_recovery_rate",
        # グループ4
        "jockey_turf_heavy3_win_rate",
        "trainer_turf_heavy3_win_rate",
    ]
    present_new_cols = [c for c in V10_NEW_COLS if c in df.columns]
    print("[v10] NaN rates for new features:")
    for col in present_new_cols:
        nan_rate = float(df[col].isna().mean())
        flag = "(HIGH)" if nan_rate > 0.50 else ""
        print(f"  {col}: {nan_rate:.3%} {flag}")

    # ==================================================================
    # リークチェック: 各新規列と is_win の Pearson 相関 >= 0.30 で ValueError
    # ==================================================================
    print("[v10] Leak check (Pearson correlation with is_win):")
    is_win_series = (pd.to_numeric(df.get("finish_rank", pd.Series(np.nan, index=df.index)), errors="coerce") == 1).astype(float)
    for col in present_new_cols:
        valid_mask = df[col].notna() & is_win_series.notna()
        if valid_mask.sum() > 100:
            corr = float(df.loc[valid_mask, col].astype(float).corr(is_win_series[valid_mask]))
            flag = "WARN(>=0.30)" if abs(corr) >= 0.30 else "OK"
            print(f"  {col}: corr={corr:.4f} [{flag}]")
            if abs(corr) >= 0.30:
                raise ValueError(
                    f"[v10] リーク検出: {col} の is_win 相関が {corr:.4f} >= 0.30 です。"
                    f" shift(1)/cumsum-self の実装を確認してください。"
                )
        else:
            print(f"  {col}: サンプル不足 (valid_mask.sum={valid_mask.sum()}) — スキップ")

    if source_is_gpu and _GPU_AVAILABLE:
        try:
            df = _to_gpu_df(_prepare_pandas_for_gpu(df))
        except Exception:
            pass

    return df


def generate_features_past_v10(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    v9 parquet を読み込み、v10 特徴量（馬場relay特徴量 19列）を追加して
    features_past_v10.parquet として保存する。

    追加特徴量（19列）:
        [グループ1] horse_turf_heavy3_win_rate, horse_turf_heavy3_n_runs,
                    horse_turf_very_heavy_win_rate, horse_turf_very_heavy_n_runs,
                    horse_turf_soft_win_rate_v10, horse_turf_soft_top3_rate_v10,
                    horse_turf_heavy3_top3_rate,
                    horse_dirt_heavy3_win_rate, horse_dirt_heavy3_n_runs,
                    horse_dirt_very_heavy_win_rate, horse_dirt_very_heavy_n_runs,
        [グループ2] current_going_win_rate_turf, current_going_win_rate_dirt,
                    current_going_top3_rate_turf,
        [グループ3] going_change_lag1, going_worsening_flag, horse_going_recovery_rate,
        [グループ4] jockey_turf_heavy3_win_rate, trainer_turf_heavy3_win_rate

    安全対策:
        - features_past_v9.parquet を上書きしない（assert で保証）
        - 既存 v10 ファイルは _bak.parquet へバックアップ
        - リークチェック: 新規列と is_win の Pearson 相関 >= 0.30 で ValueError
    """
    import shutil

    _FEAT_DIR = PROJECT_ROOT / "model_training/data/02_features"

    inp = Path(input_path or (_FEAT_DIR / "features_past_v9.parquet"))
    out = Path(output_path or (_FEAT_DIR / "features_past_v10.parquet"))
    man = Path(manifest_path or (_FEAT_DIR / "features_past_v10_manifest.json"))

    print(f"[v10] Reading v9 from: {inp}")
    if not inp.is_file():
        raise FileNotFoundError(f"[v10] 入力ファイルが存在しません: {inp}")

    # v9 を上書きしない（絶対パスで比較）
    assert out.resolve() != inp.resolve(), (
        "[v10] 出力パスが入力パスと同じです。v9 の上書きを防ぐため中断します。"
    )

    # 既存 v10 ファイルをバックアップ（破壊的上書き防止）
    bak = out.with_name(out.stem + "_bak" + out.suffix)
    if out.is_file():
        shutil.copy2(out, bak)
        print(f"[v10] Backed up existing v10 to: {bak}")

    df = pd.read_parquet(inp)
    print(f"[v10] v9 shape: {df.shape}")

    df["date"] = pd.to_datetime(df["date"])

    # v9 の going condition 列が存在しない場合は先に生成する
    v9_required = [
        "horse_turf_soft_win_rate_bayes",
        "horse_turf_soft_top3_rate",
        "turf_cond_2",
        "turf_cond_3",
        "turf_cond_4",
    ]
    missing_v9 = [c for c in v9_required if c not in df.columns]
    if missing_v9:
        print(f"[v10] v9 列が不足しているため add_going_condition_v9_features を実行: {missing_v9}")
        # v8 列も必要な場合は先に補完
        v8_required = [
            "horse_turf_soft_n_runs",
            "horse_turf_soft_win_rate",
        ]
        missing_v8 = [c for c in v8_required if c not in df.columns]
        if missing_v8:
            print(f"[v10] v8 列も不足しているため add_going_condition_v8_features を実行: {missing_v8}")
            v7_required = [
                "horse_turf_heavy_win_rate",
                "horse_turf_light_win_rate",
                "horse_dirt_heavy_win_rate",
                "horse_dirt_light_win_rate",
            ]
            missing_v7 = [c for c in v7_required if c not in df.columns]
            if missing_v7:
                print(f"[v10] v7 列も不足しているため add_going_condition_features を実行: {missing_v7}")
                df = add_going_condition_features(df)
            df = add_going_condition_v8_features(df)
        df = add_going_condition_v9_features(df)

    print("[v10] Adding v10 features (going relay features)...")
    df = add_going_condition_v10_features(df)

    V10_NEW_COLS = [
        "horse_turf_heavy3_win_rate",
        "horse_turf_heavy3_n_runs",
        "horse_turf_very_heavy_win_rate",
        "horse_turf_very_heavy_n_runs",
        "horse_turf_soft_win_rate_v10",
        "horse_turf_soft_top3_rate_v10",
        "horse_turf_heavy3_top3_rate",
        "horse_dirt_heavy3_win_rate",
        "horse_dirt_heavy3_n_runs",
        "horse_dirt_very_heavy_win_rate",
        "horse_dirt_very_heavy_n_runs",
        "current_going_win_rate_turf",
        "current_going_win_rate_dirt",
        "current_going_top3_rate_turf",
        "going_change_lag1",
        "going_worsening_flag",
        "horse_going_recovery_rate",
        "jockey_turf_heavy3_win_rate",
        "trainer_turf_heavy3_win_rate",
    ]
    present_new_cols = [c for c in V10_NEW_COLS if c in df.columns]
    print(f"[v10] Final shape: {df.shape}")
    print(f"[v10] New cols added ({len(present_new_cols)}): {present_new_cols}")

    # --- NaN 率レポート ---
    nan_report: dict = {}
    print("[v10] NaN rates for new features:")
    for col in present_new_cols:
        nan_rate = float(df[col].isna().mean())
        nan_report[col] = nan_rate
        print(f"  {col}: {nan_rate:.3%}")

    # --- リークチェック: 各新規特徴量と is_win の Pearson 相関 ---
    print("[v10] Leak check (Pearson correlation with is_win):")
    is_win_series = (pd.to_numeric(df["finish_rank"], errors="coerce") == 1).astype(float)
    leak_report: dict = {}
    for col in present_new_cols:
        valid_mask = df[col].notna() & is_win_series.notna()
        if valid_mask.sum() > 100:
            corr = float(df.loc[valid_mask, col].astype(float).corr(is_win_series[valid_mask]))
            leak_report[col] = corr
            flag = "WARN(>=0.30)" if abs(corr) >= 0.30 else "OK"
            print(f"  {col}: corr={corr:.4f} [{flag}]")
            if abs(corr) >= 0.30:
                raise ValueError(
                    f"[v10] リーク検出: {col} の is_win 相関が {corr:.4f} >= 0.30 です。"
                    f" shift(1)/cumsum-self の実装を確認してください。"
                )
        else:
            print(f"  {col}: サンプル不足 (valid_mask.sum={valid_mask.sum()}) — スキップ")

    # --- 保存 ---
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[v10] Saved parquet: {out}")

    # --- マニフェスト生成 ---
    manifest = {
        "name": "features_past_v10",
        "source": str(inp.name),
        "rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "new_columns": present_new_cols,
        "nan_rates_new_features": nan_report,
        "leak_check_corr_with_iswin": leak_report,
        "date_range": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[v10] Manifest saved: {man}")

    # --- 読み込み確認（保存後の shape を verify） ---
    verify = pd.read_parquet(out)
    print(f"[v10] Verified shape from saved file: {verify.shape}")

    return df


def generate_features_past_v7(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    v6 parquet を読み込み、going condition 特徴量 10 件を追加して v7 として保存する。

    追加特徴量:
        horse_dirt_heavy_win_rate, horse_dirt_heavy_n_runs,
        horse_turf_heavy_win_rate, horse_turf_heavy_n_runs,
        horse_dirt_light_win_rate, horse_dirt_light_n_runs,
        horse_turf_light_win_rate, horse_turf_light_n_runs,
        horse_going_preference, jockey_heavy_win_rate

    既存 features_past_v7.parquet は上書きしない。
    上書き前に features_past_v7_bak.parquet へバックアップする。
    """
    _FEAT_DIR = PROJECT_ROOT / "model_training/data/02_features"

    inp = Path(input_path or (_FEAT_DIR / "features_past_v6.parquet"))
    out = Path(output_path or (_FEAT_DIR / "features_past_v7.parquet"))
    man = Path(manifest_path or (_FEAT_DIR / "features_past_v7_manifest.json"))

    print(f"[v7] Reading v6 from: {inp}")
    if not inp.is_file():
        raise FileNotFoundError(f"[v7] 入力ファイルが存在しません: {inp}")

    df = pd.read_parquet(inp)
    print(f"[v7] v6 shape: {df.shape}")

    # 出力パスが入力パスと同一の場合は中断（v6 上書き防止）
    assert out.resolve() != inp.resolve(), (
        "[v7] 出力パスが入力パスと同じです。v6 の上書きを防ぐため中断します。"
    )

    # 既存 v7 ファイルをバックアップ（破壊的上書き防止）
    bak = out.with_name(out.stem + "_bak" + out.suffix)
    if out.is_file():
        import shutil
        shutil.copy2(out, bak)
        print(f"[v7] Backed up existing v7 to: {bak}")

    df["date"] = pd.to_datetime(df["date"])

    print("[v7] Adding going condition features...")
    df = add_going_condition_features(df)

    NEW_COLS = [
        "horse_dirt_heavy_win_rate",
        "horse_dirt_heavy_n_runs",
        "horse_turf_heavy_win_rate",
        "horse_turf_heavy_n_runs",
        "horse_dirt_light_win_rate",
        "horse_dirt_light_n_runs",
        "horse_turf_light_win_rate",
        "horse_turf_light_n_runs",
        "horse_going_preference",
        "jockey_heavy_win_rate",
    ]
    present_new_cols = [c for c in NEW_COLS if c in df.columns]
    print(f"[v7] Final shape: {df.shape}")
    print(f"[v7] New cols added: {present_new_cols}")

    # --- NaN 率レポート ---
    nan_report: dict = {}
    for col in present_new_cols:
        nan_rate = float(df[col].isna().mean())
        nan_report[col] = nan_rate
    print("[v7] NaN rates for new features:")
    for col, rate in nan_report.items():
        print(f"  {col}: {rate:.3%}")

    # --- リークチェック: 各新規特徴量と is_win の Pearson 相関 ---
    print("[v7] Leak check (Pearson correlation with is_win):")
    is_win_series = (pd.to_numeric(df["finish_rank"], errors="coerce") == 1).astype(float)
    leak_report: dict = {}
    for col in present_new_cols:
        valid_mask = df[col].notna() & is_win_series.notna()
        if valid_mask.sum() > 100:
            corr = float(df.loc[valid_mask, col].astype(float).corr(is_win_series[valid_mask]))
            leak_report[col] = corr
            flag = "WARN(>=0.30)" if abs(corr) >= 0.30 else "OK"
            print(f"  {col}: corr={corr:.4f} [{flag}]")
            if abs(corr) >= 0.30:
                raise ValueError(
                    f"[v7] リーク検出: {col} の is_win 相関が {corr:.4f} >= 0.30 です。"
                    f" shift(1) の実装を確認してください。"
                )

    # --- 保存 ---
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[v7] Saved parquet: {out}")

    # --- マニフェスト生成 ---
    manifest = {
        "name": "features_past_v7",
        "source": str(inp.name),
        "rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "new_columns": present_new_cols,
        "nan_rates_new_features": nan_report,
        "leak_check_corr_with_iswin": leak_report,
        "date_range": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[v7] Manifest saved: {man}")

    # --- 読み込み確認 ---
    verify = pd.read_parquet(out)
    print(f"[v7] Verified shape from saved file: {verify.shape}")

    return df


def generate_features_past_v8(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    v7 parquet を読み込み、v8 特徴量（one-hot flags 6列 + 稍重 Bayesian 4列 + interaction 6列 = 計16列）を追加して
    features_past_v8.parquet として保存する。

    追加特徴量（Group B: one-hot flags）:
        turf_cond_2, turf_cond_3, turf_cond_4,
        dirt_cond_2, dirt_cond_3, dirt_cond_4

    追加特徴量（Group A: 稍重専用 Bayesian）:
        horse_turf_soft_win_rate, horse_turf_soft_n_runs,
        horse_dirt_soft_win_rate, horse_dirt_soft_n_runs

    追加特徴量（Group A: interaction features）:
        going_x_turf_heavy_winrate, going_x_turf_light_winrate,
        going_x_turf_soft_winrate, going_x_dirt_heavy_winrate,
        going_match_score_turf, going_match_score_dirt

    安全対策:
        - features_past_v7.parquet を上書きしない（assert で保証）
        - 既存 v8 ファイルは _bak.parquet へバックアップ
        - リークチェック: 新規列と is_win の Pearson 相関 >= 0.30 で ValueError
    """
    import shutil

    _FEAT_DIR = PROJECT_ROOT / "model_training/data/02_features"

    inp = Path(input_path or (_FEAT_DIR / "features_past_v7.parquet"))
    out = Path(output_path or (_FEAT_DIR / "features_past_v8.parquet"))
    man = Path(manifest_path or (_FEAT_DIR / "features_past_v8_manifest.json"))

    print(f"[v8] Reading v7 from: {inp}")
    if not inp.is_file():
        raise FileNotFoundError(f"[v8] 入力ファイルが存在しません: {inp}")

    # v7 を上書きしない（絶対パスで比較）
    assert out.resolve() != inp.resolve(), (
        "[v8] 出力パスが入力パスと同じです。v7 の上書きを防ぐため中断します。"
    )

    # 既存 v8 ファイルをバックアップ（破壊的上書き防止）
    bak = out.with_name(out.stem + "_bak" + out.suffix)
    if out.is_file():
        shutil.copy2(out, bak)
        print(f"[v8] Backed up existing v8 to: {bak}")

    df = pd.read_parquet(inp)
    print(f"[v8] v7 shape: {df.shape}")

    df["date"] = pd.to_datetime(df["date"])

    # v7 の going condition 特徴量が存在しない場合は先に生成する
    v7_required = [
        "horse_turf_heavy_win_rate",
        "horse_turf_light_win_rate",
        "horse_dirt_heavy_win_rate",
        "horse_dirt_light_win_rate",
    ]
    missing_v7 = [c for c in v7_required if c not in df.columns]
    if missing_v7:
        print(f"[v8] v7 列が不足しているため add_going_condition_features を実行: {missing_v7}")
        df = add_going_condition_features(df)

    print("[v8] Adding v8 features (one-hot flags + soft Bayesian + interactions)...")
    df = add_going_condition_v8_features(df)

    NEW_COLS = [
        # Group B: one-hot flags
        "turf_cond_2",
        "turf_cond_3",
        "turf_cond_4",
        "dirt_cond_2",
        "dirt_cond_3",
        "dirt_cond_4",
        # Group A: 稍重専用 Bayesian
        "horse_turf_soft_win_rate",
        "horse_turf_soft_n_runs",
        "horse_dirt_soft_win_rate",
        "horse_dirt_soft_n_runs",
        # Group A: interaction features
        "going_x_turf_heavy_winrate",
        "going_x_turf_light_winrate",
        "going_x_turf_soft_winrate",
        "going_x_dirt_heavy_winrate",
        "going_match_score_turf",
        "going_match_score_dirt",
    ]
    present_new_cols = [c for c in NEW_COLS if c in df.columns]
    print(f"[v8] Final shape: {df.shape}")
    print(f"[v8] New cols added ({len(present_new_cols)}): {present_new_cols}")

    # --- NaN 率レポート ---
    nan_report: dict = {}
    print("[v8] NaN rates for new features:")
    for col in present_new_cols:
        nan_rate = float(df[col].isna().mean())
        nan_report[col] = nan_rate
        print(f"  {col}: {nan_rate:.3%}")

    # --- リークチェック: 各新規特徴量と is_win の Pearson 相関 ---
    print("[v8] Leak check (Pearson correlation with is_win):")
    is_win_series = (pd.to_numeric(df["finish_rank"], errors="coerce") == 1).astype(float)
    leak_report: dict = {}
    for col in present_new_cols:
        valid_mask = df[col].notna() & is_win_series.notna()
        if valid_mask.sum() > 100:
            corr = float(df.loc[valid_mask, col].astype(float).corr(is_win_series[valid_mask]))
            leak_report[col] = corr
            flag = "WARN(>=0.30)" if abs(corr) >= 0.30 else "OK"
            print(f"  {col}: corr={corr:.4f} [{flag}]")
            if abs(corr) >= 0.30:
                raise ValueError(
                    f"[v8] リーク検出: {col} の is_win 相関が {corr:.4f} >= 0.30 です。"
                    f" shift(1)/cumsum-self の実装を確認してください。"
                )
        else:
            print(f"  {col}: サンプル不足 (valid_mask.sum={valid_mask.sum()}) — スキップ")

    # --- 保存 ---
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[v8] Saved parquet: {out}")

    # --- マニフェスト生成 ---
    manifest = {
        "name": "features_past_v8",
        "source": str(inp.name),
        "rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "new_columns": present_new_cols,
        "nan_rates_new_features": nan_report,
        "leak_check_corr_with_iswin": leak_report,
        "date_range": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[v8] Manifest saved: {man}")

    # --- 読み込み確認（保存後の shape を verify） ---
    verify = pd.read_parquet(out)
    print(f"[v8] Verified shape from saved file: {verify.shape}")

    return df


def generate_features_past_v5(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    features_past_v4.parquet を読み込み、新規特徴量10件を追加し、
    不要列7件を削除して features_past_v5.parquet を生成する。

    入力の v4 ファイルは破壊的に上書きしない（別パスへ保存）。

    削除列（Gain=0）:
        breed_code_encoded, is_ritto_slope, is_past_abnormal, is_holiday,
        has_training_data, pedigree_debut_flag, distance_category

    追加列（10件）:
        horse_dirt_win_rate, horse_turf_win_rate, surface_win_rate_diff,
        horse_course_win_rate_v5, running_style_surface_win_rate,
        surface_switch_flag, agari3f_same_surface_rank_lag1,
        same_distance_same_surface_win_rate, age_distance_win_rate,
        jockey_horse_combo_count
    """
    _FEAT_DIR = PROJECT_ROOT / "model_training/data/02_features"

    inp = Path(input_path or (_FEAT_DIR / "features_past_v4.parquet"))
    out = Path(output_path or (_FEAT_DIR / "features_past_v5.parquet"))
    man = Path(manifest_path or (_FEAT_DIR / "features_past_v5_manifest.json"))

    print(f"[v5] Reading v4 from: {inp}")
    df = pd.read_parquet(inp)
    print(f"[v5] v4 shape: {df.shape}")

    # v4 ファイルが変更されていないことを確認（上書き防止チェック）
    assert out.resolve() != inp.resolve(), "出力パスが入力パスと同じです。上書きを防ぐため中断します。"

    df["date"] = pd.to_datetime(df["date"])

    # --- 新規特徴量を追加 ---
    print("[v5] Adding surface/course features...")
    df = add_surface_course_features(df)

    # --- 不要列を削除 ---
    DROP_COLS_V5 = [
        "breed_code_encoded",
        "is_ritto_slope",
        "is_past_abnormal",
        "is_holiday",
        "has_training_data",
        "pedigree_debut_flag",
        "distance_category",
    ]
    existing_drops = [c for c in DROP_COLS_V5 if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"[v5] Dropped {len(existing_drops)} columns: {existing_drops}")

    NEW_COLS = [
        "horse_dirt_win_rate",
        "horse_turf_win_rate",
        "surface_win_rate_diff",
        "horse_course_win_rate_v5",
        "running_style_surface_win_rate",
        "surface_switch_flag",
        "agari3f_same_surface_rank_lag1",
        "same_distance_same_surface_win_rate",
        "age_distance_win_rate",
        "jockey_horse_combo_count",
    ]
    print(f"[v5] Final shape: {df.shape}")
    print(f"[v5] New cols added: {[c for c in NEW_COLS if c in df.columns]}")

    # --- NaN率レポート ---
    nan_report = {}
    for col in NEW_COLS:
        if col in df.columns:
            nan_rate = float(df[col].isna().mean())
            nan_report[col] = nan_rate
    print("[v5] NaN rates for new features:")
    for col, rate in nan_report.items():
        print(f"  {col}: {rate:.3%}")

    # --- リークチェック: 各新規特徴量と finish_rank==1 の相関 ---
    print("[v5] Leak check (correlation with is_win):")
    is_win = (pd.to_numeric(df["finish_rank"], errors="coerce") == 1).astype(float)
    leak_report = {}
    for col in NEW_COLS:
        if col in df.columns:
            valid_mask = df[col].notna() & is_win.notna()
            if valid_mask.sum() > 100:
                corr = float(df.loc[valid_mask, col].astype(float).corr(is_win[valid_mask]))
                leak_report[col] = corr
                flag = "WARN(>0.3)" if abs(corr) > 0.3 else "OK"
                print(f"  {col}: corr={corr:.4f} [{flag}]")

    # --- 保存 ---
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[v5] Saved parquet: {out}")

    # マニフェスト生成
    manifest = {
        "name": "features_past_v5",
        "source": str(inp.name),
        "rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dropped_columns": existing_drops,
        "new_columns": [c for c in NEW_COLS if c in df.columns],
        "nan_rates_new_features": nan_report,
        "leak_check_corr_with_iswin": leak_report,
        "date_range": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[v5] Manifest saved: {man}")

    return df


def generate_features_past_v6(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    features_past_v4.parquet を読み込み、Bayesian smoothing 済みの特徴量 14 件を追加し、
    features_past_v6.parquet を生成する。v5 との差分:

    変更（NaN率改善）:
        horse_dirt_win_rate      未経験馬 NaN→prior(0.10) に収束
        horse_turf_win_rate      同上
        surface_win_rate_diff    A1/A2 に依存するため自動的に NaN=0
        horse_course_win_rate_v5 当コース初出走 NaN→prior に収束
        same_distance_same_surface_win_rate 同上

    追加（count companion）:
        horse_dirt_n_runs
        horse_turf_n_runs
        horse_course_n_runs
        same_distance_same_surface_n_runs
    """
    _FEAT_DIR = PROJECT_ROOT / "model_training/data/02_features"

    inp = Path(input_path or (_FEAT_DIR / "features_past_v4.parquet"))
    out = Path(output_path or (_FEAT_DIR / "features_past_v6.parquet"))
    man = Path(manifest_path or (_FEAT_DIR / "features_past_v6_manifest.json"))

    print(f"[v6] Reading v4 from: {inp}")
    df = pd.read_parquet(inp)
    print(f"[v6] v4 shape: {df.shape}")

    assert out.resolve() != inp.resolve(), "出力パスが入力パスと同じです。上書きを防ぐため中断します。"

    df["date"] = pd.to_datetime(df["date"])

    print("[v6] Adding surface/course features (Bayesian smoothing)...")
    df = add_surface_course_features(df)

    DROP_COLS_V6 = [
        "breed_code_encoded",
        "is_ritto_slope",
        "is_past_abnormal",
        "is_holiday",
        "has_training_data",
        "distance_category",
        # pedigree_debut_flag は _validate_pedigree_requirements で必要なため保持
    ]
    existing_drops = [c for c in DROP_COLS_V6 if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"[v6] Dropped {len(existing_drops)} columns: {existing_drops}")

    NEW_COLS = [
        "horse_dirt_win_rate",
        "horse_dirt_n_runs",
        "horse_turf_win_rate",
        "horse_turf_n_runs",
        "surface_win_rate_diff",
        "horse_course_win_rate_v5",
        "horse_course_n_runs",
        "running_style_surface_win_rate",
        "surface_switch_flag",
        "agari3f_same_surface_rank_lag1",
        "same_distance_same_surface_win_rate",
        "same_distance_same_surface_n_runs",
        "age_distance_win_rate",
        "jockey_horse_combo_count",
    ]
    print(f"[v6] Final shape: {df.shape}")
    print(f"[v6] New cols added: {[c for c in NEW_COLS if c in df.columns]}")

    nan_report = {}
    for col in NEW_COLS:
        if col in df.columns:
            nan_report[col] = float(df[col].isna().mean())
    print("[v6] NaN rates for new features:")
    for col, rate in nan_report.items():
        print(f"  {col}: {rate:.3%}")

    print("[v6] Leak check (correlation with is_win):")
    is_win = (pd.to_numeric(df["finish_rank"], errors="coerce") == 1).astype(float)
    leak_report = {}
    for col in NEW_COLS:
        if col in df.columns:
            valid_mask = df[col].notna() & is_win.notna()
            if valid_mask.sum() > 100:
                corr = float(df.loc[valid_mask, col].astype(float).corr(is_win[valid_mask]))
                leak_report[col] = corr
                flag = "WARN(>0.3)" if abs(corr) > 0.3 else "OK"
                print(f"  {col}: corr={corr:.4f} [{flag}]")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[v6] Saved parquet: {out}")

    manifest = {
        "name": "features_past_v6",
        "source": str(inp.name),
        "rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dropped_columns": existing_drops,
        "new_columns": [c for c in NEW_COLS if c in df.columns],
        "nan_rates_new_features": nan_report,
        "leak_check_corr_with_iswin": leak_report,
        "date_range": [
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[v6] Manifest saved: {man}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create past features.")
    parser.add_argument(
        "--mode",
        choices=["single", "main", "chunked", "v5", "v6", "v7", "v8", "v9", "v10"],
        default="single",
        help="single: normal full run, main: main prediction run, chunked: per-year chunk run, v5: build features_past_v5 from v4, v6: build features_past_v6 from v4 (Bayesian smoothing), v7: build features_past_v7 from v6 (going condition features), v8: build features_past_v8 from v7 (one-hot flags + soft Bayesian + interaction features), v9: build features_past_v9 from v8 (turf-soft score differentiation 8 features), v10: build features_past_v10 from v9 (going relay features 19 columns)",
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--chunks-dir", default=None)
    parser.add_argument("--years", default=None, help="comma separated years, e.g. 2021,2022")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-attach-pedigree", action="store_true")
    parser.add_argument("--strict-pedigree", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--heartbeat-sec", type=int, default=60)
    parser.add_argument(
        "--disable-bulk-asof",
        action="store_true",
        help="Disable bulk merge_asof and use sibling-wise loop (slower but shows finer progress).",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Suffix appended to output filename stem (e.g. '_v2' -> features_past_v2.parquet). "
            "Prevents overwriting existing features_past.parquet."
        ),
    )
    args = parser.parse_args()

    # --output-suffix が指定された場合は出力パスを派生させる（既存ファイル上書き防止）
    _out_path = args.output_path
    _man_path = args.manifest_path
    if args.output_suffix and _out_path is None:
        _base = OUTPUT_PATH.with_stem(OUTPUT_PATH.stem + args.output_suffix)
        _out_path = str(_base)
        _man_path = str(_base.parent / (_base.stem + "_manifest.json"))

    if args.mode == "v10":
        generate_features_past_v10(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
        )
    elif args.mode == "v9":
        generate_features_past_v9(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
        )
    elif args.mode == "v8":
        generate_features_past_v8(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
        )
    elif args.mode == "v7":
        generate_features_past_v7(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
        )
    elif args.mode == "v6":
        generate_features_past_v6(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
        )
    elif args.mode == "v5":
        generate_features_past_v5(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
        )
    elif args.mode == "main":
        create_main_pastfeatures(
            attach_pedigree=not args.no_attach_pedigree,
            strict_pedigree=args.strict_pedigree,
        )
    elif args.mode == "chunked":
        create_pastfeatures_chunked_by_year(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
            chunks_dir=args.chunks_dir,
            resume=not args.no_resume,
            years=_parse_years_arg(args.years),
            attach_pedigree=not args.no_attach_pedigree,
            strict_pedigree=args.strict_pedigree,
            show_progress=args.show_progress,
            heartbeat_sec=args.heartbeat_sec,
            prefer_bulk_asof=not args.disable_bulk_asof,
        )
    else:
        create_pastfeatures_main(
            input_path=args.input_path,
            output_path=_out_path,
            manifest_path=_man_path,
            attach_pedigree=not args.no_attach_pedigree,
            strict_pedigree=args.strict_pedigree,
            show_progress=args.show_progress,
            heartbeat_sec=args.heartbeat_sec,
            prefer_bulk_asof=not args.disable_bulk_asof,
        )

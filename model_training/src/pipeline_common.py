"""
Shared config, constants, I/O, row filters, pipeline state, and leak checks
for model_training pipeline stages.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.utils.common_utils import read_csv_optimized

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT = PROJECT_ROOT
CONFIG_DIR = PROJECT_ROOT / "model_training" / "config"
FEATURES_DIR = PROJECT_ROOT / "model_training" / "data" / "02_features"
PREPROCESSED_DIR = PROJECT_ROOT / "model_training" / "data" / "01_preprocessed"
MODELS_DIR = PROJECT_ROOT / "model_training" / "models"
TRAIN_CONFIG_PATH = CONFIG_DIR / "train_config.json"
PASTFEATURES_CONFIG_PATH = (
    PROJECT_ROOT / "model_training" / "config" / "pastfeatures_config.json"
)

# --- feature_constants ---
BASE_LEAK_COLS = [
    "racetime",
    "margin",
    "time_diff",
    "time_3f_after",
    "time_4f_after",
    "hon_shokin",
    "fuka_shokin",
    "speed_diff",
    "pci",
    "rank_ratio",
    "rank_deviation",
    # v21追加: 当該レースの実走タイムを使った偏差スピード指数（リーク）
    # speed_index_3run_avg / speed_index_trend は shift(1) 済みで問題なし
    "speed_index_course_adj",
    # v27追加（A2監査）: いずれも当該レースの base_time（prepare_db で当日勝者の
    # 走破タイムに上書きされる確定後の値）を未shiftで参照するためCLAUDE.md禁止#7違反。
    #  - daily_track_variant: 当日同キーの base_time 中央値との差（自身の base_time が分子）
    #  - tm_score_surface_adj: daily_track_variant を引き算で取り込むため間接汚染
    #  - base_time_cond_zscore: z-score 分子 x が当該レースの base_time（過去統計はshift済みでも分子が当日結果）
    # いずれも学習リーク＋本番では base_time が未確定で計算不能（train/serve skew）。
    # experiment D（ensemble_v6 / features_past_v27_track_variant）で学習混入を防ぐため leak 網に登録。
    "daily_track_variant",
    "tm_score_surface_adj",
    "base_time_cond_zscore",
    # DA-1: レース内確定上がり3F Z（shift 前の中間列。出力禁止）
    "agari_z_race",
    # DA-3: 当該レース実測ラップ由来（create_features_v4 経由で混入しうる）
    "lap_time_std",
    "early_pace_ratio",
    # DA-4: 当該レース finish_time 由来の生スピード指数
    "speed_index",
]

PAST_EXTRA_LEAK_COLS = [
    "speed_deviation",
    "relative_speed_pct",
]

ID_COLS = ["race_id", "ketto_num"]

TRAIN_ONLY_EXCLUDE_COLS = ["finish_rank", "target", "weight", "odds", "popularity"]

CATEGORICAL_FEATURES = [
    "course_code",
    "wakuban",
    "horse_num",
    "sex_code",
    "grade_code",
    "race_type_code",
    "track_code",
    "weather_code",
    # turf_condition と dirt_condition は数値（ordinal）として扱う。
    # GPU学習時に categorical_feature=[] が返りスプリットが≈0になるバグを回避するため除外。
    # 閾値 1.5/2.5/3.5 による良/稍重/重/不良 の正しい分岐が GPU/CPU 両方で機能する。
    "course_kubun",
    "jockey_code_encoded",
    "trainer_code_encoded",
    "owner_code_encoded",
    "sire_id_encoded",
    "running_style_code_encoded",
    "horse_interval_bins",
]

DROP_COLUMNS = [
    "registered_count",
    "cos_date_plus1",
    "mining_best_time",
    "mining_worst_time",
]

DEFAULT_FEATURE_PATH = PROJECT_ROOT / "model_training/data/02_features/features_past.parquet"
DEFAULT_FEATURE_CSV_FALLBACK = (
    PROJECT_ROOT / "model_training/data/02_features/features_past.csv"
)

# --- config_loader ---
DEFAULT_FILTER_CONFIG = {"default_exclude_abnormal_codes": [1, 3, 4]}

DEFAULT_TRAIN_CONFIG = {
    "training": {
        "seed": 42,
        "n_trials": 50,
        "default_feature_set": "all_non_leak",
        "use_gpu": True,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "gpu_use_dp": False,
        "max_bin": 255,
        "rank1_label_mode": "binary",
        "rank1_soft_targets": {"1": 1.0, "2": 0.6, "3": 0.4},
        "rank1_time_decay_alpha": 0.175,
        "rank1_objective": "cross_entropy",
        "rank1_weight_min": 0.5,
        "rank1_weight_max": 5.0,
        "rank1_time_diff_missing_fill": 5.0,
        "rank1_binary_loser_weight": 1.0,
        "rank1_log_weight_quantiles": True,
        "rank1_isotonic_enabled": True,
        "rank1_isotonic_fit_years": None,
        "rank1_log_optuna_top3_proxy": False,
        "rank1_log_fold_top3_proxy": True,
        "show_progress": True,
    },
    "shap": {"enabled": False, "sample_size": 1000, "top_ev_n": 30},
    "filters": DEFAULT_FILTER_CONFIG,
}

DEFAULT_PASTFEATURES_CONFIG = {
    "smoothing": {
        "beta": 10.0,
        "use_race_level_prior": True,
        "prior_default_win": 0.10,
        "prior_default_ren": 0.20,
    },
    "transport": {
        "hokkaido_courses": [1, 2],
        "ritto_long_courses": [1, 2, 3, 4, 5, 6],
        "miho_long_courses": [1, 2, 7, 8, 9, 10],
        "ritto_code": 1,
        "miho_code": 2,
    },
    "distance_category": {
        "bins": [0, 1400, 1800, 2400, 10000],
        "labels": [1, 2, 3, 4],
        "right": False,
    },
    "main_build": {
        "filter_history_for_main": True,
        "filter_keys": ["ketto_num", "jockey_code", "trainer_code"],
    },
}


def load_parquet_or_csv(
    parquet_path: str | Path, csv_fallback_path: str | Path
) -> pd.DataFrame:
    parquet = Path(parquet_path)
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception as exc:
            print(f"[warn] Failed to read parquet {parquet}: {exc}")
    csv_fallback = Path(csv_fallback_path)
    if csv_fallback.exists():
        return read_csv_optimized(csv_fallback)
    raise FileNotFoundError(
        f"Feature file not found. parquet={parquet} csv={csv_fallback}"
    )


def _deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for key, value in override.items():
        if key in out:
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_json_with_default(path: Path, default: dict, warn_prefix: str) -> dict:
    if not path.exists():
        return dict(default)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] Failed to load config {path}: {exc}. Use default {warn_prefix}.")
        return dict(default)
    if not isinstance(loaded, dict):
        print(f"[warn] Invalid config shape in {path}. Use default {warn_prefix}.")
        return dict(default)
    return _deep_merge(default, loaded)


def load_train_config(config_path: Path | None = None) -> dict:
    return _load_json_with_default(
        Path(config_path) if config_path else TRAIN_CONFIG_PATH,
        DEFAULT_TRAIN_CONFIG,
        "train_config",
    )


def load_config(config_path: Path | None = None) -> dict:
    """strategy / backtest 向けエイリアス（デフォルト付き train_config）。"""
    return load_train_config(config_path)


def load_pastfeatures_config(config_path: Path | None = None) -> dict:
    return _load_json_with_default(
        Path(config_path) if config_path else PASTFEATURES_CONFIG_PATH,
        DEFAULT_PASTFEATURES_CONFIG,
        "pastfeatures_config",
    )


def load_filter_config(config_path: Path | None = None) -> dict:
    cfg = load_train_config(config_path)
    filters = cfg.get("filters", {})
    if not isinstance(filters, dict):
        return dict(DEFAULT_FILTER_CONFIG)
    return _deep_merge(DEFAULT_FILTER_CONFIG, filters)


def apply_row_filters_for_training(
    df: pd.DataFrame,
    *,
    abnormal_exclude_codes: tuple[int, ...] | list[int] = tuple(DEFAULT_FILTER_CONFIG["default_exclude_abnormal_codes"]),
    min_horses: int | None = None,
    exempt_track_codes: list[int] | tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    meta = {
        "abnormal_exclude_codes": list(abnormal_exclude_codes or []),
        "min_horses": int(min_horses) if min_horses is not None else None,
        "exempt_track_codes": list(exempt_track_codes or []),
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "abnormal_rows_dropped": 0,
        "small_field_rows_dropped": 0,
    }
    out = df.copy()

    if abnormal_exclude_codes and "abnormal_code" in out.columns:
        abnormal_vals = pd.to_numeric(out["abnormal_code"], errors="coerce")
        drop_mask = abnormal_vals.isin(set(abnormal_exclude_codes))
        meta["abnormal_rows_dropped"] = int(drop_mask.sum())
        out = out[~drop_mask].copy()

    if min_horses is not None and "race_id" in out.columns:
        race_sizes = out.groupby("race_id")["race_id"].transform("count")
        if exempt_track_codes and "track_code" in out.columns:
            exempt_mask = pd.to_numeric(out["track_code"], errors="coerce").isin(
                set(exempt_track_codes)
            )
        else:
            exempt_mask = pd.Series(False, index=out.index)
        small_mask = (race_sizes < int(min_horses)) & (~exempt_mask)
        meta["small_field_rows_dropped"] = int(small_mask.sum())
        out = out[~small_mask].copy()

    meta["rows_after"] = int(len(out))
    return out, meta


def load_state(path: str | Path) -> dict:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: str | Path, state: dict) -> dict:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def update_state(
    path: str | Path, *, mode: str | None = None, updates: dict | None = None
) -> dict:
    state = load_state(path)
    state["last_update_date"] = datetime.now().strftime("%Y%m%d")
    state["last_run_at"] = datetime.now().strftime("%Y%m%d%H%M%S")
    if mode is not None:
        state["mode"] = mode
    if updates:
        state.update(updates)
    return save_state(path, state)


def assert_no_leak_columns(
    features: list[str], leak_columns: list[str] | tuple[str, ...]
) -> None:
    leak_set = set(leak_columns)
    overlap = sorted([c for c in features if c in leak_set])
    if overlap:
        raise SystemExit(
            "Leak columns detected in training features: " + ", ".join(overlap)
        )


def run_leak_check(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    csv_fallback_path: Path = DEFAULT_FEATURE_CSV_FALLBACK,
    leak_columns: list[str] | tuple[str, ...] = BASE_LEAK_COLS,
) -> None:
    df = load_parquet_or_csv(feature_path, csv_fallback_path)
    assert_no_leak_columns(list(df.columns), list(leak_columns))
    print("[ok] Leak column check passed.")


def get_db_connection() -> sqlite3.Connection:
    cfg = load_config()
    db_path = PROJECT_ROOT / cfg["db"]["path"]
    return sqlite3.connect(db_path)


def save_features(df: pd.DataFrame, name: str) -> None:
    """特徴量をParquet + CSV + manifestとして保存する。"""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = FEATURES_DIR / f"{name}.parquet"
    csv_path = FEATURES_DIR / f"{name}.csv"
    manifest_path = FEATURES_DIR / f"{name}_manifest.json"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest: dict[str, Any] = {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": [
            str(df["race_date"].min()) if "race_date" in df.columns else None,
            str(df["race_date"].max()) if "race_date" in df.columns else None,
        ],
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved {name}: {len(df)} rows, {len(df.columns)} cols → {parquet_path}")


def make_target(finish_rank: pd.Series) -> pd.Series:
    """着順をlambdarank用スコアに変換する（値が大きいほど上位）。"""
    target = pd.Series(0, index=finish_rank.index, dtype=int)
    target[finish_rank == 1] = 3
    target[(finish_rank == 2) | (finish_rank == 3)] = 2
    target[(finish_rank == 4) | (finish_rank == 5)] = 1
    return target


def shift_expanding_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding().mean()


def shift_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def validate_no_leakage(df: pd.DataFrame, feature_cols: list[str]) -> None:
    nan_rates = df[feature_cols].isnull().mean()
    high_nan = nan_rates[nan_rates > 0.5]
    if len(high_nan) > 0:
        print("WARNING: NaN率50%超の特徴量（新馬戦は許容範囲）:")
        print(high_nan.to_string())
    else:
        print("OK: NaN率チェック通過（50%超なし）")


def encode_surface(surface_str: pd.Series) -> pd.Series:
    mapping = {"芝": 1, "ダート": 2, "障害": 3}
    return surface_str.map(mapping).fillna(0).astype(int)


def encode_track_condition(cond_str: pd.Series) -> pd.Series:
    mapping = {"良": 1, "稍重": 2, "重": 3, "不良": 4}
    return cond_str.map(mapping).fillna(1).astype(int)

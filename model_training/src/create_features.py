import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
import warnings
import json
import joblib
import time
import sys
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.utils.common_utils import optimize_dtypes, read_csv_optimized, log_step
from model_training.src.pipeline_common import (
    BASE_LEAK_COLS,
    DROP_COLUMNS,
    ID_COLS,
    apply_row_filters_for_training,
    load_filter_config,
    update_state,
)

warnings.simplefilter("ignore")

INPUT_PATH = PROJECT_ROOT / "model_training/data/01_preprocessed/horse_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "model_training/data/02_features/features_basic.csv"
ENCODER_PATH = PROJECT_ROOT / "model_training/data/02_features/features_basic_encoders.json"
ENCODER_JOBLIB_PATH = PROJECT_ROOT / "model_training/data/02_features/features_basic_encoders.joblib"
MANIFEST_PATH = PROJECT_ROOT / "model_training/data/02_features/features_basic_manifest.json"


RUNTIME_FILTERS = load_filter_config()
DEFAULT_EXCLUDE_ABNORMAL_CODES = tuple(
    RUNTIME_FILTERS.get("default_exclude_abnormal_codes", [1, 3, 4])
)

def load_data(path):
    path = Path(path)
    print(f"Loading data from {path}...")
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            pass
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")
    return read_csv_optimized(path, optimize=True)


def process_dates(df):
    if "year" in df.columns and "month_day" in df.columns:
        date_series = pd.to_datetime(
            df["year"].astype(str) + df["month_day"].astype(str).str.zfill(4),
            format="%Y%m%d",
            errors="coerce",
        )
        df["date"] = date_series
        df["month"] = date_series.dt.month
        df["weekday"] = date_series.dt.weekday
        df["season"] = (df["month"] % 12 + 3) // 3

        month = df["month"].astype(float)
        df["sin_date"] = np.sin(2 * np.pi * month / 12.0)
        df["cos_date"] = np.cos(2 * np.pi * month / 12.0)
    return df


def _normalize_category_series(series):
    return (
        series.astype(str)
        .fillna("unknown")
        .replace(["nan", "None", "-1", "-1.0"], "unknown")
    )


def _load_encoder_maps(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_encoder_maps(path, encoder_maps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(encoder_maps, f, ensure_ascii=False, indent=2)


def _load_encoder_bundle(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        loaded = joblib.load(path)
        return loaded if isinstance(loaded, dict) else {}
    except Exception as e:
        warnings.warn(f"Failed to load encoder bundle {path}: {e}")
        return {}


def _save_encoder_bundle(path, encoder_bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder_bundle, path)


def _build_feature_manifest(df, filter_metadata=None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    blocklist = set(BASE_LEAK_COLS + ID_COLS + ["finish_rank", "target", "weight"])
    available_leak_cols = [c for c in BASE_LEAK_COLS if c in df.columns]
    available_id_cols = [c for c in ID_COLS if c in df.columns]
    safe_numeric_features = [c for c in numeric_cols if c not in blocklist]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_columns": len(df.columns),
        "all_columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "leak_columns_defined": BASE_LEAK_COLS,
        "leak_columns_present": available_leak_cols,
        "id_columns_defined": ID_COLS,
        "id_columns_present": available_id_cols,
        "excluded_for_training": sorted(list(blocklist)),
        "safe_numeric_features": safe_numeric_features,
        "safe_numeric_feature_count": len(safe_numeric_features),
    }
    if filter_metadata is not None:
        manifest["filter_metadata"] = filter_metadata
    return manifest


def _save_feature_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def process_categorical(df, encoder_maps=None, encoder_bundle=None, fit_encoders=True):
    if encoder_maps is None:
        encoder_maps = {}
    if encoder_bundle is None:
        encoder_bundle = {}
    target_cols = [
        "jockey_code",
        "trainer_code",
        "owner_code",
        "breeder_code",
        "place_code",
        "weather_code",
        "course_code",
        "breed_code",
        "running_style_code",
        "sire_id",
        "bms_id",
        "ketto_num",
        "p_sire_sys_id",
        "p_dam_sire_sys_id",
        "p_sire_sire_sys_id",
        "p_dam_dam_sire_sys_id",
    ]
    for col in tqdm(target_cols, desc="Categorical encoding", leave=False):
        if col in df.columns:
            series = _normalize_category_series(df[col])
            if fit_encoders:
                oe = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
                )
                encoded = oe.fit_transform(series.to_frame()).reshape(-1)
                categories = [str(v) for v in oe.categories_[0]]
                encoder_maps[col] = {cls: int(i) for i, cls in enumerate(categories)}
                encoder_bundle[col] = oe
                df[f"{col}_encoded"] = encoded.astype(np.int32)
            else:
                enc = encoder_bundle.get(col)
                if enc is not None:
                    transformed = enc.transform(series.to_frame()).reshape(-1)
                    df[f"{col}_encoded"] = transformed.astype(np.int32)
                else:
                    col_map = encoder_maps.get(col, {})
                    df[f"{col}_encoded"] = (
                        series.map(col_map).fillna(-1).astype(np.int32)
                    )
    return df


def process_race_context(df):
    if "race_id" in df.columns:
        df["n_horses"] = df.groupby("race_id")["race_id"].transform("count").astype(int)
    if "ketto_num" in df.columns and "date" in df.columns:
        # 重複インデックス時に reindex がNaNを生成するため、位置ベースで元順を復元する
        orig_pos = np.arange(len(df))
        sorted_df = df.copy()
        sorted_df["_orig_pos"] = orig_pos
        sorted_df = sorted_df.sort_values(["ketto_num", "date"])
        sorted_df["interval"] = (
            sorted_df.groupby("ketto_num")["date"].diff().dt.days.fillna(-1)
        )
        sorted_df["interval"] = sorted_df["interval"].clip(lower=-1, upper=999)
        sorted_df = sorted_df.sort_values("_orig_pos")
        df["interval"] = sorted_df["interval"].values
    return df


def process_abnormal_history(df):
    if "is_past_abnormal" not in df.columns:
        df["is_past_abnormal"] = 0
    if not {"ketto_num", "date", "abnormal_code"}.issubset(df.columns):
        return df

    tmp = df.copy()
    tmp["_orig_pos"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["ketto_num", "date", "race_id"])
    abnormal_numeric = pd.to_numeric(tmp["abnormal_code"], errors="coerce").fillna(0)
    tmp["_abnormal_flag"] = (abnormal_numeric != 0).astype(np.int8)
    tmp["is_past_abnormal"] = (
        tmp.groupby("ketto_num", sort=False)["_abnormal_flag"].shift(1).fillna(0).astype(np.int8)
    )
    # 重複インデックス時に reindex がNaNを生成するため、位置ベースで元順に復元する
    tmp = tmp.sort_values("_orig_pos")
    df["is_past_abnormal"] = tmp["is_past_abnormal"].values
    return df


def process_interactions(df, encoder_bundle=None, fit_encoders=True):
    if encoder_bundle is None:
        encoder_bundle = {}
    if "sex_code" in df.columns and "season" in df.columns:
        sex_season = (
            _normalize_category_series(df["sex_code"])
            + "_"
            + _normalize_category_series(df["season"])
        )
        if fit_encoders:
            oe = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
            )
            df["sex_season_encoded"] = (
                oe.fit_transform(sex_season.to_frame()).reshape(-1).astype(np.int32)
            )
            encoder_bundle["sex_season"] = oe
        else:
            enc = encoder_bundle.get("sex_season")
            if enc is not None:
                df["sex_season_encoded"] = (
                    enc.transform(sex_season.to_frame()).reshape(-1).astype(np.int32)
                )
            else:
                df["sex_season_encoded"] = -1
    if "wakuban" in df.columns and "track_code" in df.columns:
        # JV-Link: track_code >= 50 = ダート（52-57）。23は誤り。
        is_dirt = (pd.to_numeric(df["track_code"], errors="coerce") >= 50).astype(int)
        df["wakuban_track_type"] = df["wakuban"] * 10 + is_dirt
    if "sex_code" in df.columns and "sin_date" in df.columns:
        is_female = (df["sex_code"] == 2).astype(int)
        df["female_season_sin"] = df["sin_date"] * is_female
        df["female_season_cos"] = df["cos_date"] * is_female
    return df


def process_going_conditions_numeric(df):
    """
    JV 馬場コード（芝/ダート条件）を数値として残す。
    ``all_non_leak`` 学習は ``select_dtypes(np.number)`` のため、ここが object のままだと特徴に入らない。

    未発表(0)は prepare_db / build_today_features と同様、当該サーフェスのみ良(1)に正規化する。
    """
    for col in ("turf_condition", "dirt_condition"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "track_code" in df.columns:
        tc = pd.to_numeric(df["track_code"], errors="coerce").fillna(0)
        is_dirt = tc >= 23
        if "turf_condition" in df.columns:
            t = pd.to_numeric(df["turf_condition"], errors="coerce")
            mask = (~is_dirt) & (t == 0)
            df.loc[mask, "turf_condition"] = 1
        if "dirt_condition" in df.columns:
            d = pd.to_numeric(df["dirt_condition"], errors="coerce")
            mask = is_dirt & (d == 0)
            df.loc[mask, "dirt_condition"] = 1
    else:
        for col in ("turf_condition", "dirt_condition"):
            if col in df.columns:
                df[col] = df[col].replace(0, 1)

    return df


def process_physical(df):
    if "horse_weight" in df.columns:
        hw = df["horse_weight"].replace(0, np.nan)
        if "burden_weight" in df.columns:
            df["weight_ratio"] = df["burden_weight"] / hw
        if "horse_weight_change" in df.columns:
            prev_weight = (hw - df["horse_weight_change"]).replace(0, np.nan)
            df["weight_change_ratio"] = df["horse_weight_change"] / prev_weight
            df["weight_change_ratio"] = df["weight_change_ratio"].replace(
                [np.inf, -np.inf], np.nan
            )
    return df


def process_training(df):
    if "time_4f" in df.columns and "time_1f" in df.columns:
        df["training_acceleration"] = (df["time_4f"] / 4.0) - df["time_1f"]
        df["time_first_3f"] = df["time_4f"] - df["time_1f"]
        df["has_training_data"] = df["time_4f"].notna().astype(int)
        if "center_code" in df.columns:
            df["is_ritto_slope"] = (
                (df["center_code"] == 1) & df["time_4f"].notna()
            ).astype(int)
    return df


def process_mining(df):
    if "mining_uncertainty" in df.columns:
        df["mining_confidence"] = 1.0 / (1.0 + df["mining_uncertainty"].fillna(0))
    return df


def drop_redundant_columns(df):
    existing_drops = [c for c in DROP_COLUMNS if c in df.columns]
    return df.drop(columns=existing_drops)


def create_features_main(
    input_path=None,
    output_path=None,
    return_df=True,
    encoder_path=None,
    fit_encoders=True,
    manifest_path=None,
    apply_training_filters=True,
    abnormal_exclude_codes=DEFAULT_EXCLUDE_ABNORMAL_CODES,
    min_horses=6,
    exempt_track_codes=None,
):
    input_path = Path(input_path or INPUT_PATH)
    output_path = Path(output_path or OUTPUT_PATH)
    encoder_path = Path(encoder_path or ENCODER_PATH)
    encoder_joblib_path = encoder_path.with_suffix(".joblib")
    manifest_path = Path(manifest_path or MANIFEST_PATH)
    df = load_data(input_path)
    filter_meta = {
        "apply_filters": bool(apply_training_filters),
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "abnormal_rows_dropped": 0,
        "small_field_rows_dropped": 0,
    }
    if apply_training_filters:
        df, base_meta = apply_row_filters_for_training(
            df,
            abnormal_exclude_codes=abnormal_exclude_codes,
            min_horses=min_horses,
            exempt_track_codes=exempt_track_codes,
        )
        filter_meta.update(base_meta)
        print(
            "[info] Training filters applied: "
            f"rows {filter_meta['rows_before']} -> {filter_meta['rows_after']} "
            f"(abnormal_drop={filter_meta['abnormal_rows_dropped']}, "
            f"small_field_drop={filter_meta['small_field_rows_dropped']})"
        )
    encoder_maps = {} if fit_encoders else _load_encoder_maps(encoder_path)
    encoder_bundle = {} if fit_encoders else _load_encoder_bundle(encoder_joblib_path)

    pipeline = [
        (process_dates, "Processing dates"),
        (process_going_conditions_numeric, "Going conditions (numeric)"),
        (process_abnormal_history, "Processing abnormal history"),
        (process_race_context, "Processing race context"),
        (
            lambda x: process_categorical(
                x,
                encoder_maps=encoder_maps,
                encoder_bundle=encoder_bundle,
                fit_encoders=fit_encoders,
            ),
            "Processing categorical",
        ),
        (
            lambda x: process_interactions(
                x, encoder_bundle=encoder_bundle, fit_encoders=fit_encoders
            ),
            "Processing interactions",
        ),
        (process_physical, "Processing physical"),
        (process_training, "Processing training"),
        (process_mining, "Processing mining"),
        (drop_redundant_columns, "Dropping redundant columns"),
    ]

    with tqdm(pipeline, desc="Feature engineering pipeline") as pbar:
        for func, desc in pbar:
            pbar.set_postfix_str(desc)
            step_started = time.perf_counter()
            rows_in = len(df)
            df = func(df)
            log_step(
                desc,
                rows_in=rows_in,
                rows_out=len(df),
                started_at=step_started,
                prefix="[features]",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    try:
        df.to_parquet(output_path.with_suffix(".parquet"), index=False)
    except Exception as e:
        print(f"[warn] Failed to save parquet: {e}")
    _save_encoder_maps(encoder_path, encoder_maps)
    _save_encoder_bundle(encoder_joblib_path, encoder_bundle)
    _save_feature_manifest(manifest_path, _build_feature_manifest(df, filter_meta))
    return df if return_df else None


def create_main_features():
    input_path = (
        PROJECT_ROOT / "model_training/data/01_preprocessed/main_horse_data.csv"
    )
    output_path = (
        PROJECT_ROOT / "model_training/data/02_features/main_features_basic.csv"
    )
    enc = Path(ENCODER_PATH)
    enc_joblib = Path(ENCODER_JOBLIB_PATH)
    # 学習用 features_basic で保存したエンコーダを流用（本番馬のカテゴリを学習データと揃える）
    return create_features_main(
        input_path,
        output_path,
        fit_encoders=not (enc.is_file() and enc_joblib.is_file()),
        apply_training_filters=False,
    )


def update_features(
    *,
    mode: str = "all",
    state_path: str | None = None,
) -> dict:
    """
    特徴量（basic）を更新する統一エントリポイント。

    - mode="train": horse_data から features_basic を再生成（カテゴリエンコーダを学習し直す）
    - mode="main": main_horse_data から main_features_basic を再生成
    - mode="all": 上記両方

    Args:
        mode: "train" | "main" | "all"
        state_path: 状態ファイルのパス（省略時: model_training/data/state/features_last_update.json）

    Returns:
        保存した状態 dict
    """
    if state_path is None:
        state_file = (
            PROJECT_ROOT
            / "model_training"
            / "data"
            / "state"
            / "features_last_update.json"
        )
    else:
        state_file = Path(state_path)

    mode = str(mode).lower().strip()
    if mode not in {"train", "main", "all"}:
        raise ValueError('mode must be one of: "train", "main", "all"')

    out: dict = {}
    if mode in {"train", "all"}:
        create_features_main(return_df=False)
        out["train_features_updated"] = True

    if mode in {"main", "all"}:
        create_main_features()
        out["main_features_updated"] = True

    return update_state(state_file, mode=mode, updates=out)


def main():
    create_features_main(return_df=False)


if __name__ == "__main__":
    main()

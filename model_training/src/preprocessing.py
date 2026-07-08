"""
データ前処理モジュール
race_seデータとrace_raデータを読み込み、前処理を行い、SE_preprocessed.csvとRA_preprocessed.csvを出力する
【修正版】mining_predicted_timeの0をNaNに変換する処理を適用済み、エイリアス関数の追加
【最適化版】データ型最適化とParquet形式サポート追加
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from common.utils.common_utils import (
    log_step,
    normalize_ketto_num,
    optimize_dtypes,
    read_csv_optimized,
)
from model_training.src.pipeline_common import (
    apply_row_filters_for_training,
    load_filter_config,
    update_state,
)
# 馬体重の値域ガード閾値は validate_horse_weight と共有し、前処理層と本番判定で整合させる。
from main.race_runtime import HORSE_WEIGHT_MAX_KG, HORSE_WEIGHT_MIN_KG


RUNTIME_FILTERS = load_filter_config()
DEFAULT_EXCLUDE_ABNORMAL_CODES = tuple(
    RUNTIME_FILTERS.get("default_exclude_abnormal_codes", [1, 3, 4])
)
READ_CHUNK_SIZE = 250000
SE_READ_DTYPES = {
    "ketto_num": "string",
    "race_num": "Int64",
    "year": "Int64",
    "month_day": "Int64",
    "course_code": "Int64",
    "kai": "Int64",
    "nichi": "Int64",
}
RA_READ_DTYPES = {
    "race_num": "Int64",
    "year": "Int64",
    "month_day": "Int64",
    "course_code": "Int64",
    "kai": "Int64",
    "nichi": "Int64",
}


def convert_time_to_seconds(time_value):
    """
    time列の値を秒に変換
    例: 1152 → 75.2 (1分15秒2 = 60 + 15 + 0.2)
    """
    if pd.isna(time_value):
        return np.nan

    try:
        time_str = str(int(time_value)).zfill(4)  # 4桁にゼロパディング
    except Exception:
        return np.nan

    # 各桁を抽出
    minutes = int(time_str[0])  # 千の位: 分
    seconds = int(time_str[1:3])  # 百・十の位: 秒
    deciseconds = int(time_str[3])  # 一の位: 0.1秒単位

    # 秒に変換
    total_seconds = (minutes * 60) + seconds + (deciseconds * 0.1)

    return total_seconds


def parse_margin_code(x):
    """
    着差コード（margin_code）を馬身単位の数値（float）に変換する
    """
    if pd.isna(x) or str(x).strip() == "":
        return np.nan

    s = str(x).strip().upper()

    # 1. 特殊コードの変換マップ
    special_map = {
        "H": 0.05,  # ハナ
        "A": 0.1,  # アタマ
        "K": 0.2,  # クビ
        "D": 0.0,  # 同着
        "Z": 10.0,  # 10馬身以上
        "T": 10.0,  # 大差
    }

    # マップにあればそれを返す（前方一致でチェック）
    for key, val in special_map.items():
        if s.startswith(key):
            return val

    # 2. 数値コードの解析
    try:
        # 単純な数値（例: "2", "10"）の場合
        if s.replace(".", "").isdigit() and len(s) <= 2:
            return float(s)

        # 3桁の分数形式 (例: 112 -> 1 1/2, _34 -> 0 3/4)
        if len(s) == 3:
            integer_part = 0
            if s[0].isdigit():
                integer_part = int(s[0])
            elif s[0] == "_":
                integer_part = 0
            else:
                return np.nan

            if s[1].isdigit() and s[2].isdigit():
                numerator = int(s[1])  # 分子
                denominator = int(s[2])  # 分母

                if denominator != 0:
                    return float(integer_part + (numerator / denominator))

    except (ValueError, TypeError, IndexError):
        pass

    return np.nan


def convert_mining_predicted_time_to_seconds(time_value):
    """
    mining_predicted_time列の値を秒に変換
    例: 11552 → 75.52
    【修正済み】0 は「データなし」なので NaN を返す
    """
    if pd.isna(time_value):
        return np.nan

    try:
        int_val = int(time_value)
        # ★ここが重要：0なら計算せずNaNを返す
        if int_val == 0:
            return np.nan

        time_str = str(int_val).zfill(5)
    except Exception:
        return np.nan

    # 各桁を抽出
    minutes = int(time_str[0])
    seconds = int(time_str[1:3])
    centiseconds = int(time_str[3:5])

    total_seconds = (minutes * 60) + seconds + (centiseconds * 0.01)

    return total_seconds


def parse_mining_error(x):
    """
    マイニング誤差コードを秒（float）に変換する関数
    仕様: '0082' -> 0.82秒
    【重要】0 は「信頼度MAX」ではなく「不明」とみなす
    """
    if pd.isna(x) or str(x).strip() == "":
        return np.nan

    try:
        val = float(x)
        if val == 0:
            return np.nan
        return val / 100.0
    except (ValueError, TypeError):
        return np.nan


def save_with_parquet(df, output_path):
    """
    CSVとParquet形式の両方で保存（互換性と高速化の両立）
    指数表記を避けるため、CSV保存時にfloat_formatを使用
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parquet 生成を先に保証し、失敗時は即時停止する。
    parquet_path = output_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False, compression="snappy")
    except ImportError as e:
        raise RuntimeError(
            f"Parquet export failed because pyarrow is missing. Install with "
            f"`pip install pyarrow` and retry. target={parquet_path}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Parquet export failed: target={parquet_path}, reason={e}") from e

    # CSV形式で保存（互換性のため）
    # 注意: float_format="%.0f" は小数情報を破壊するため使用しない。
    df.to_csv(output_path, index=False)


def create_race_id(df: pd.DataFrame) -> pd.Series:
    """JRA標準のレースIDを作成する"""
    year = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int).astype(str)
    month_day = (
        pd.to_numeric(df["month_day"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    course = (
        pd.to_numeric(df["course_code"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    kai = pd.to_numeric(df["kai"], errors="coerce").fillna(0).astype(int).astype(str)
    nichi = (
        pd.to_numeric(df["nichi"], errors="coerce").fillna(0).astype(int).astype(str)
    )
    race_num = (
        pd.to_numeric(df["race_num"], errors="coerce").fillna(0).astype(int).astype(str)
    )

    race_id = (
        year
        + month_day.str.zfill(4)
        + course.str.zfill(2)
        + kai.str.zfill(2)
        + nichi.str.zfill(2)
        + race_num.str.zfill(2)
    )
    return race_id


def preprocess_se_data(input_path=None, output_path=None):
    """race_seデータ前処理のメイン関数"""
    project_root = PROJECT_ROOT

    if input_path is None:
        se_dir = project_root / "common" / "data" / "output" / "race_se"
        csv_files = sorted(se_dir.glob("race_se_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"SEデータファイルが見つかりません: {se_dir}")
        print(f"読み込むSEファイル数: {len(csv_files)}")
        df_list = []
        for f in csv_files:
            df_list.append(
                read_csv_optimized(
                    f,
                    dtype=SE_READ_DTYPES,
                    chunksize=READ_CHUNK_SIZE,
                )
            )
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = read_csv_optimized(
            input_path,
            dtype=SE_READ_DTYPES,
            chunksize=READ_CHUNK_SIZE,
        )

    if output_path is None:
        output_path = (
            project_root
            / "model_training"
            / "data"
            / "01_preprocessed"
            / "SE_preprocessed.csv"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df["race_id"] = create_race_id(df)

    df["horse_weight_change"] = df.apply(
        lambda row: (
            np.nan
            if pd.isna(row["weight_change"])
            else (
                float(row["weight_change"])
                if row["weight_change_sign"] == "+"
                else (
                    -float(row["weight_change"])
                    if row["weight_change_sign"] == "-"
                    else float(row["weight_change"])
                )
            )
        ),
        axis=1,
    )

    # 馬体重の値域ガード（P2）: validate_horse_weight と同一閾値で「異常値のみ」を NaN 化する。
    # 物理的にあり得ない値（<300kg / >650kg、入力誤りやデータ破損）が特徴量や
    # 体重差計算に伝播するのを防ぐ。正常な体重・既存の挙動は変えない（異常値のみ処理）。
    if "horse_weight" in df.columns:
        hw = pd.to_numeric(df["horse_weight"], errors="coerce")
        abnormal_hw = hw.notna() & (
            (hw < HORSE_WEIGHT_MIN_KG) | (hw > HORSE_WEIGHT_MAX_KG)
        )
        if "horse_weight_change" in df.columns:
            # 体重本体が異常なら、その行の体重差も信頼できないため無効化する。
            df.loc[abnormal_hw, "horse_weight_change"] = np.nan
        df["horse_weight"] = hw.where(~abnormal_hw, np.nan)

    df["racetime"] = df["time"].apply(convert_time_to_seconds).replace(0, np.nan)
    df["margin"] = df["margin_code"].apply(parse_margin_code)
    df["mining_times"] = df["mining_predicted_time"].apply(
        convert_mining_predicted_time_to_seconds
    )

    if "mining_predicted_rank" in df.columns:
        df["mining_predicted_rank"] = pd.to_numeric(
            df["mining_predicted_rank"], errors="coerce"
        )
        df["mining_predicted_rank"] = df["mining_predicted_rank"].replace(0, np.nan)

    if "mining_error_plus" in df.columns:
        df["mining_error_plus_sec"] = df["mining_error_plus"].apply(parse_mining_error)
    else:
        df["mining_error_plus_sec"] = np.nan
    if "mining_error_minus" in df.columns:
        df["mining_error_minus_sec"] = df["mining_error_minus"].apply(
            parse_mining_error
        )
    else:
        df["mining_error_minus_sec"] = np.nan

    if "mining_times" in df.columns:
        mask = df["mining_times"].notna()
        both_valid = (
            df.loc[mask, "mining_error_plus_sec"].notna()
            & df.loc[mask, "mining_error_minus_sec"].notna()
        )
        df.loc[mask & both_valid, "mining_uncertainty"] = (
            df.loc[mask & both_valid, "mining_error_plus_sec"]
            + df.loc[mask & both_valid, "mining_error_minus_sec"]
        )
        df.loc[mask, "mining_best_time"] = (
            df.loc[mask, "mining_times"] - df.loc[mask, "mining_error_plus_sec"]
        )
        df.loc[mask, "mining_worst_time"] = (
            df.loc[mask, "mining_times"] + df.loc[mask, "mining_error_minus_sec"]
        )

    if "odds" in df.columns:
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        df["odds"] = df["odds"].replace(0, np.nan)
        df["odds"] = df["odds"] / 10.0

    if "time_3f_after" in df.columns:
        df["time_3f_after"] = pd.to_numeric(df["time_3f_after"], errors="coerce")
        df["time_3f_after"] = df["time_3f_after"].replace(0, np.nan)
        df["time_3f_after"] = df["time_3f_after"] / 10.0

    if "burden_weight" in df.columns:
        df["burden_weight"] = df["burden_weight"] / 10.0

    for c in ["corner_1", "corner_2", "corner_3", "corner_4"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(0, np.nan)

    if "margin" in df.columns and "finish_rank" in df.columns:
        first_place_mask = df["finish_rank"] == 1
        df.loc[first_place_mask & df["margin"].isna(), "margin"] = 0.0

    if "time_diff" in df.columns:
        df["time_diff"] = pd.to_numeric(df["time_diff"], errors="coerce") / 10.0
        if "finish_rank" in df.columns:
            mask_winner = df["finish_rank"] == 1
            df.loc[mask_winner, "time_diff"] = 0.0

    col_list = [
        "race_id",
        "year",
        "month_day",
        "course_code",
        "kai",
        "nichi",
        "race_num",
        "wakuban",
        "horse_num",
        "ketto_num",
        "horse_mark_code",
        "sex_code",
        "breed_code",
        "age",
        "region_code",
        "trainer_code",
        "owner_code",
        "burden_weight",
        "burden_weight_prev",
        "blinker_code",
        "jockey_code",
        "horse_weight",
        "horse_weight_change",
        "abnormal_code",
        "finish_rank",
        "dead_heat_flag",
        "dead_heat_count",
        "racetime",
        "margin",
        "corner_1",
        "corner_2",
        "corner_3",
        "corner_4",
        "odds",
        "popularity",
        "hon_shokin",
        "fuka_shokin",
        "time_4f_after",
        "time_3f_after",
        "time_diff",
        "mining_times",
        "mining_error_plus_sec",
        "mining_error_minus_sec",
        "mining_uncertainty",
        "mining_best_time",
        "mining_worst_time",
        "mining_predicted_rank",
        "running_style_code",
    ]
    available_cols = [col for col in col_list if col in df.columns]
    df_processed = df[available_cols].copy()

    # データ型最適化
    step_started = time.perf_counter()
    rows_in = len(df_processed)
    df_processed = optimize_dtypes(df_processed)
    log_step(
        "SE dtype optimization",
        rows_in=rows_in,
        rows_out=len(df_processed),
        started_at=step_started,
        prefix="[preprocess]",
    )

    save_with_parquet(df_processed, output_path)
    print(f"SE前処理完了: {df_processed.shape[0]}行, {df_processed.shape[1]}列")
    return df_processed


def preprocess_ra_data(input_path=None, output_path=None):
    """race_raデータ前処理"""
    project_root = PROJECT_ROOT
    if input_path is None:
        ra_dir = project_root / "common" / "data" / "output" / "race_ra"
        csv_files = sorted(ra_dir.glob("race_ra_*.csv"))
        df_list = []
        for f in csv_files:
            df_list.append(
                read_csv_optimized(
                    f,
                    dtype=RA_READ_DTYPES,
                    chunksize=READ_CHUNK_SIZE,
                )
            )
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = read_csv_optimized(
            input_path,
            dtype=RA_READ_DTYPES,
            chunksize=READ_CHUNK_SIZE,
        )

    if output_path is None:
        output_path = (
            project_root
            / "model_training"
            / "data"
            / "01_preprocessed"
            / "RA_preprocessed.csv"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df["race_id"] = create_race_id(df)

    condition_cols = [
        "condition_2yo",
        "condition_3yo",
        "condition_4yo",
        "condition_5yo_plus",
        "condition_min_age",
    ]
    available_cond_cols = [c for c in condition_cols if c in df.columns]
    if available_cond_cols:
        for col in available_cond_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df["race_condition_code"] = df[available_cond_cols].max(axis=1)

        def map_race_level(code):
            if code in [701, 702]:
                return 1
            elif code == 703:
                return 2
            elif code in [5, 1, 2, 3]:
                return 3
            elif code == 10:
                return 4
            elif code == 16:
                return 5
            elif code in [999, 100, 99]:
                return 6
            return 0

        df["race_level"] = df["race_condition_code"].apply(map_race_level)

        def get_age_type(row):
            if row.get("condition_2yo", 0) != 0:
                return 2
            elif row.get("condition_3yo", 0) != 0:
                return 3
            elif row.get("condition_4yo", 0) != 0:
                return 4
            elif row.get("condition_5yo_plus", 0) != 0:
                return 5
            elif row.get("condition_min_age", 0) != 0:
                return 4
            return 0

        df["race_age_type"] = df.apply(get_age_type, axis=1)

    grade_map = {"A": 9, "B": 8, "C": 7, "D": 6, "E": 5, "L": 5, "F": 4, "G": 3, "H": 2}
    if "grade_code" in df.columns:
        df["grade_code"] = df["grade_code"].astype(str).str.strip()
        df["grade_code"] = df["grade_code"].map(grade_map).fillna(1).astype(int)

    if "course_kubun" in df.columns:
        df["course_kubun"] = df["course_kubun"].astype(str).str.strip()
        ck_map = {"A": 1, "A1": 1, "A2": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        df["course_kubun"] = df["course_kubun"].map(ck_map).fillna(0).astype(int)

    rename_map = {"time_3f_before": "race_first_3f", "time_3f_after": "race_last_3f"}
    df.rename(columns=rename_map, inplace=True)
    for col in ["race_first_3f", "race_last_3f"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mask_large = df[col] > 50
            df.loc[mask_large, col] = df.loc[mask_large, col] / 10.0
    if "race_first_3f" in df.columns and "race_last_3f" in df.columns:
        df["race_pace"] = df["race_first_3f"] - df["race_last_3f"]
    if "lap_times" in df.columns:
        df["race_lap1"] = (
            df["lap_times"].astype(str).str.replace(r"\D", "", regex=True).str[:3]
        )
        df["race_lap1"] = pd.to_numeric(df["race_lap1"], errors="coerce") / 10.0
        mask_invalid = (df["race_lap1"] < 5.0) | (df["race_lap1"] > 30.0)
        df.loc[mask_invalid, "race_lap1"] = np.nan
    if "obstacle_mile_time" in df.columns:
        df["obstacle_mile_time_sec"] = df["obstacle_mile_time"].apply(
            convert_time_to_seconds
        )

    col_list = [
        "race_id",
        "year",
        "month_day",
        "course_code",
        "kai",
        "nichi",
        "race_num",
        "grade_code",
        "race_type_code",
        "weight_type",
        "race_condition_code",
        "race_level",
        "race_age_type",
        "distance",
        "track_code",
        "course_kubun",
        "registered_count",
        "running_count",
        "finish_count",
        "weather_code",
        "turf_condition",
        "dirt_condition",
        "obstacle_mile_time_sec",
        "race_first_3f",
        "race_last_3f",
        "race_pace",
        "race_lap1",
    ]
    available_cols = [col for col in col_list if col in df.columns]
    df_processed = df[available_cols].copy()

    # データ型最適化
    step_started = time.perf_counter()
    rows_in = len(df_processed)
    df_processed = optimize_dtypes(df_processed)
    log_step(
        "RA dtype optimization",
        rows_in=rows_in,
        rows_out=len(df_processed),
        started_at=step_started,
        prefix="[preprocess]",
    )

    save_with_parquet(df_processed, output_path)
    print(f"RA前処理完了: {df_processed.shape[0]}行, {df_processed.shape[1]}列")
    return df_processed


def preprocess_tm_data(input_path=None, output_path=None):
    """ming_tmデータ前処理"""
    project_root = PROJECT_ROOT
    if input_path is None:
        input_path = (
            project_root / "common" / "data" / "output" / "ming_tm" / "ming_tm.csv"
        )
    if output_path is None:
        output_path = (
            project_root
            / "model_training"
            / "data"
            / "01_preprocessed"
            / "TM_preprocessed.csv"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_tm = read_csv_optimized(input_path)
    df_tm["race_id"] = create_race_id(df_tm)

    tm_long_list = []
    for i in range(1, 19):
        num_col = f"mining_pred_{i}_horse_num"
        score_col = f"mining_pred_{i}_score"

        if num_col in df_tm.columns and score_col in df_tm.columns:
            temp = df_tm[["race_id", num_col, score_col]].copy()
            temp.columns = ["race_id", "horse_num", "tm_score"]
            temp = temp.dropna(subset=["horse_num"])
            temp["horse_num"] = temp["horse_num"].astype(int)

            temp["tm_score"] = pd.to_numeric(temp["tm_score"], errors="coerce")
            temp["tm_score"] = temp["tm_score"].replace(0, np.nan)
            temp["tm_score"] = temp["tm_score"] / 10.0

            tm_long_list.append(temp)

    if tm_long_list:
        df_long = pd.concat(tm_long_list, ignore_index=True)
        df_long = df_long.drop_duplicates(subset=["race_id", "horse_num"], keep="last")
        df_long = df_long.sort_values(by=["race_id", "horse_num"], ignore_index=True)

        # データ型最適化
        step_started = time.perf_counter()
        rows_in = len(df_long)
        df_long = optimize_dtypes(df_long)
        log_step(
            "TM dtype optimization",
            rows_in=rows_in,
            rows_out=len(df_long),
            started_at=step_started,
            prefix="[preprocess]",
        )

        save_with_parquet(df_long, output_path)
        print(f"TM前処理完了: {df_long.shape[0]}行, {df_long.shape[1]}列")
        return df_long
    else:
        return pd.DataFrame()


def preprocess_all():
    print("=" * 60)
    print("全データの前処理を開始します")
    print("=" * 60)

    print("\n[1/5] SEデータの前処理")
    df_se = preprocess_se_data()
    print("-" * 30)

    print("\n[2/5] RAデータの前処理")
    df_ra = preprocess_ra_data()
    print("-" * 30)

    print("\n[3/5] TMデータの前処理")
    preprocess_tm_data()
    print("-" * 30)

    print("\n[4/5] PEDデータの前処理")
    # ★修正：エイリアスではなく定義済みの関数を呼び出す
    preprocess_ped_data(exclude_dams=True)
    print("-" * 30)

    print("\n[5/5] HCデータの前処理")
    preprocess_hc_data()
    print("-" * 30)

    print("\n" + "=" * 60)
    print("全ての前処理が完了しました。")
    print("=" * 60)
    return df_se, df_ra


# ==========================================
# ★エイリアス関数の定義 (ImportError対策)
# ==========================================
def preprocess_ped_data_exclude_dams(sk_path=None, bt_path=None, output_path=None):
    """preprocess_ped_dataのエイリアス（exclude_dams=Trueで呼び出し）"""
    return preprocess_ped_data(
        sk_path=sk_path, bt_path=bt_path, output_path=output_path, exclude_dams=True
    )


preprocess_data = preprocess_se_data


def create_horse_data(
    se_path=None,
    ra_path=None,
    tm_path=None,
    ped_path=None,
    hc_path=None,
    output_path=None,
    apply_training_filters: bool = False,
    abnormal_exclude_codes=DEFAULT_EXCLUDE_ABNORMAL_CODES,
    min_horses: int | None = None,
    exempt_track_codes=None,
):
    project_root = Path(__file__).parent.parent.parent
    base_dir = project_root / "model_training" / "data" / "01_preprocessed"
    if se_path is None:
        se_path = base_dir / "SE_preprocessed.csv"
    if ra_path is None:
        ra_path = base_dir / "RA_preprocessed.csv"
    if tm_path is None:
        tm_path = base_dir / "TM_preprocessed.csv"
    if ped_path is None:
        ped_path = base_dir / "PED_preprocessed.csv"
    if hc_path is None:
        hc_path = base_dir / "HC_preprocessed.csv"
    if output_path is None:
        output_path = base_dir / "horse_data.csv"
    se_path, ra_path, tm_path = Path(se_path), Path(ra_path), Path(tm_path)
    if ped_path:
        ped_path = Path(ped_path)
    if hc_path:
        hc_path = Path(hc_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not se_path.exists():
        raise FileNotFoundError(f"SE data not found: {se_path}")
    df_main = read_csv_optimized(se_path)
    if ra_path.exists():
        df_ra = read_csv_optimized(ra_path)
        ra_cols = [
            c for c in df_ra.columns if c not in df_main.columns or c == "race_id"
        ]
        df_main = pd.merge(df_main, df_ra[ra_cols], on="race_id", how="left")
    if tm_path.exists():
        df_tm = read_csv_optimized(tm_path)
        df_main = pd.merge(df_main, df_tm, on=["race_id", "horse_num"], how="left")

    df_main = load_and_merge_bloodline_data(df_main)

    if ped_path and ped_path.exists():
        df_ped = read_csv_optimized(ped_path)
        df_main["ketto_num"] = normalize_ketto_num(df_main["ketto_num"])
        df_ped["ketto_num"] = normalize_ketto_num(df_ped["ketto_num"])
        ped_cols = [
            c for c in df_ped.columns if c not in df_main.columns or c == "ketto_num"
        ]
        df_main = pd.merge(df_main, df_ped[ped_cols], on="ketto_num", how="left")

    if hc_path and hc_path.exists():
        df_hc = read_csv_optimized(hc_path)
        df_main["date_temp"] = pd.to_datetime(
            df_main["year"].astype(str) + df_main["month_day"].astype(str).str.zfill(4),
            format="%Y%m%d",
            errors="coerce",
        )
        df_hc["train_date"] = pd.to_datetime(df_hc["train_date"])
        df_main["ketto_num"] = normalize_ketto_num(df_main["ketto_num"])
        df_hc["ketto_num"] = normalize_ketto_num(df_hc["ketto_num"])
        df_main = df_main.sort_values("date_temp")
        df_hc = df_hc.sort_values("train_date")
        df_main = pd.merge_asof(
            df_main,
            df_hc,
            left_on="date_temp",
            right_on="train_date",
            by="ketto_num",
            direction="backward",
            tolerance=pd.Timedelta(days=28),
        )
        df_main = df_main.drop(columns=["date_temp"])

    if "ketto_num" in df_main.columns:
        df_main["ketto_num"] = normalize_ketto_num(df_main["ketto_num"])
    if "race_id" in df_main.columns and "horse_num" in df_main.columns:
        df_main = df_main.sort_values(by=["race_id", "horse_num"], ignore_index=True)

    if apply_training_filters:
        df_main, filter_meta = apply_row_filters_for_training(
            df_main,
            abnormal_exclude_codes=abnormal_exclude_codes,
            min_horses=min_horses,
            exempt_track_codes=exempt_track_codes,
        )
        print(
            "[info] create_horse_data filters applied: "
            f"rows {filter_meta['rows_before']} -> {filter_meta['rows_after']} "
            f"(abnormal_drop={filter_meta['abnormal_rows_dropped']}, "
            f"small_field_drop={filter_meta['small_field_rows_dropped']})"
        )

    print(f"Saving to {output_path}...")

    # データ型最適化
    df_main = optimize_dtypes(df_main)

    save_with_parquet(df_main, output_path)
    return df_main


def load_and_merge_bloodline_data(df_horse, sk_path=None):
    if sk_path is None:
        sk_path = (
            Path(__file__).parent.parent.parent
            / "common/data/output/blod_sk/blod_sk.csv"
        )
    sk_path = Path(sk_path)
    if not sk_path.exists():
        return df_horse
    df_sk = read_csv_optimized(
        sk_path,
        usecols=["ketto_num", "p_sire", "p_dam_sire"],
        dtype=str,
    )
    df_sk = df_sk.rename(columns={"p_sire": "sire_id", "p_dam_sire": "bms_id"})
    df_sk["sire_id"] = df_sk["sire_id"].replace(["0", "0000000000"], np.nan)
    df_sk["bms_id"] = df_sk["bms_id"].replace(["0", "0000000000"], np.nan)
    df_sk = df_sk.drop_duplicates(subset=["ketto_num"])
    if "ketto_num" in df_horse.columns:
        df_horse["ketto_num"] = normalize_ketto_num(df_horse["ketto_num"])
        df_sk["ketto_num"] = normalize_ketto_num(df_sk["ketto_num"])
        df_merged = df_horse.merge(df_sk, on="ketto_num", how="left")
        for col in ["sire_id", "bms_id"]:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna("unknown").astype("category")
        return df_merged
    return df_horse


def preprocess_ped_data(
    sk_path=None, bt_path=None, output_path=None, exclude_dams=True
):
    from tqdm import tqdm

    project_root = Path(__file__).parent.parent.parent
    if sk_path is None:
        sk_path = project_root / "common/data/output/blod_sk/blod_sk.csv"
    if bt_path is None:
        bt_path = project_root / "common/data/output/blod_bt/blod_bt.csv"
    if output_path is None:
        output_path = (
            project_root / "model_training/data/01_preprocessed/PED_preprocessed.csv"
        )
    sk_path, bt_path, output_path = Path(sk_path), Path(bt_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    BT_COLS = {"breeding_reg_num": "id", "system_id": "system_id"}
    SK_COLS = [
        "ketto_num",
        "sex_code",
        "breed_code",
        "offspring_import_flag",
        "p_sire",
        "p_dam",
        "p_sire_sire",
        "p_sire_dam",
        "p_dam_sire",
        "p_dam_dam",
        "p_sire_sire_sire",
        "p_sire_sire_dam",
        "p_sire_dam_sire",
        "p_sire_dam_dam",
        "p_dam_sire_sire",
        "p_dam_sire_dam",
        "p_dam_dam_sire",
        "p_dam_dam_dam",
    ]

    if not sk_path.exists() or not bt_path.exists():
        return pd.DataFrame()
    df_sk = read_csv_optimized(
        sk_path, dtype=str, usecols=lambda x: x in SK_COLS, low_memory=False
    )
    for c in SK_COLS[4:]:
        df_sk[c] = (
            df_sk[c]
            .replace(["0", "0000000000", "0.0", "nan", "NaN"], np.nan)
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

    df_bt = read_csv_optimized(
        bt_path, dtype=str, usecols=list(BT_COLS.keys()), low_memory=False
    ).rename(columns=BT_COLS)
    df_bt["id"] = df_bt["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    sys_dict = df_bt.drop_duplicates("id").set_index("id")["system_id"].to_dict()

    for col in tqdm(SK_COLS[4:], desc="Mapping System IDs"):
        if exclude_dams and col.endswith("dam"):
            continue
        df_sk[col + "_sys_id"] = df_sk[col].map(sys_dict)

    df_sk["ketto_num"] = pd.to_numeric(df_sk["ketto_num"], errors="coerce")

    # データ型最適化
    df_sk = optimize_dtypes(df_sk)

    save_with_parquet(df_sk, output_path)
    return df_sk


def preprocess_hc_data(hc_dir=None, output_path=None):
    from tqdm import tqdm

    project_root = Path(__file__).parent.parent.parent
    if hc_dir is None:
        hc_dir = project_root / "common/data/output/slop_hc"
    if output_path is None:
        output_path = (
            project_root / "model_training/data/01_preprocessed/HC_preprocessed.csv"
        )
    hc_dir, output_path = Path(hc_dir), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list(hc_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    df_list = []
    for f in tqdm(csv_files):
        try:
            df_list.append(read_csv_optimized(f, dtype=str))
        except Exception as e:
            print(f"[warn] failed to read HC file {f}: {e}")
    if not df_list:
        return pd.DataFrame()
    df_hc = pd.concat(df_list, ignore_index=True)

    HC_MAP = {
        "training_date": "train_date",
        "ketto_num": "ketto_num",
        "training_center": "center_code",
        "time_4f_total": "time_4f",
        "lap_time_800_600": "lap_time_800_600",
        "time_3f_total": "time_3f",
        "lap_time_600_400": "lap_time_600_400",
        "time_2f_total": "time_2f",
        "lap_time_400_200": "lap_time_400_200",
        "lap_time_200_0": "time_1f",
    }
    # マッピングに存在する列のみを選択してリネーム
    available_cols = [k for k in HC_MAP.keys() if k in df_hc.columns]
    df_hc = df_hc.rename(
        columns={k: v for k, v in HC_MAP.items() if k in available_cols}
    )
    # リネーム後の列名で選択
    renamed_cols = [HC_MAP[k] for k in available_cols]
    df_hc = df_hc[renamed_cols]

    if "ketto_num" in df_hc.columns:
        df_hc["ketto_num"] = pd.to_numeric(df_hc["ketto_num"], errors="coerce").astype(
            "Int64"
        )
    if "train_date" in df_hc.columns:
        df_hc["train_date"] = pd.to_datetime(
            df_hc["train_date"], format="%Y%m%d", errors="coerce"
        )

    def clean_time(x):
        try:
            v = float(x)
            return np.nan if v <= 0 or v >= 9000 else v / 10.0
        except Exception:
            return np.nan

    # 全てのタイム関連列をクリーニング
    time_cols = [
        "time_4f",
        "lap_time_800_600",
        "time_3f",
        "lap_time_600_400",
        "time_2f",
        "lap_time_400_200",
        "time_1f",
    ]
    for col in time_cols:
        if col in df_hc.columns:
            df_hc[col] = df_hc[col].apply(clean_time)
    df_hc = df_hc.sort_values(["train_date", "time_4f"]).drop_duplicates(
        ["ketto_num", "train_date"], keep="first"
    )

    # データ型最適化
    df_hc = optimize_dtypes(df_hc)

    save_with_parquet(df_hc, output_path)
    return df_hc


def preprocess_wc_data(wc_dir=None, output_path=None):
    """WC（ウッドチップ調教）CSV を前処理して parquet/csv に保存する。"""
    from tqdm import tqdm

    project_root = Path(__file__).parent.parent.parent
    if wc_dir is None:
        wc_dir = project_root / "common/data/output/wood_wc"
    if output_path is None:
        output_path = (
            project_root / "model_training/data/01_preprocessed/WC_preprocessed.csv"
        )
    wc_dir, output_path = Path(wc_dir), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list(wc_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    df_list = []
    for f in tqdm(csv_files, desc="WC files"):
        try:
            df_list.append(read_csv_optimized(f, dtype=str))
        except Exception as e:
            print(f"[warn] failed to read WC file {f}: {e}")
    if not df_list:
        return pd.DataFrame()
    df_wc = pd.concat(df_list, ignore_index=True)

    WC_MAP = {
        "training_date": "train_date",
        "ketto_num": "ketto_num",
        "training_center": "center_code",
        "course": "course_code",
        "time_5f_total": "time_5f",
        "lap_time_5f_4f": "lap_time_5f_4f",
        "lap_time_4f_3f": "lap_time_4f_3f",
        "lap_time_3f_2f": "lap_time_3f_2f",
        "lap_time_2f_1f": "lap_time_2f_1f",
        "lap_time_1f_0f": "lap_time_1f_0f",
    }
    available_cols = [k for k in WC_MAP.keys() if k in df_wc.columns]
    df_wc = df_wc.rename(columns={k: v for k, v in WC_MAP.items() if k in available_cols})
    renamed_cols = [WC_MAP[k] for k in available_cols]
    df_wc = df_wc[renamed_cols]

    if "ketto_num" in df_wc.columns:
        df_wc["ketto_num"] = pd.to_numeric(df_wc["ketto_num"], errors="coerce").astype("Int64")
    if "train_date" in df_wc.columns:
        df_wc["train_date"] = pd.to_datetime(
            df_wc["train_date"], format="%Y%m%d", errors="coerce"
        )

    def clean_time(x):
        try:
            v = float(x)
            return np.nan if v <= 0 or v >= 9000 else v / 10.0
        except Exception:
            return np.nan

    time_cols = [
        "time_5f",
        "lap_time_5f_4f",
        "lap_time_4f_3f",
        "lap_time_3f_2f",
        "lap_time_2f_1f",
        "lap_time_1f_0f",
    ]
    for col in time_cols:
        if col in df_wc.columns:
            df_wc[col] = df_wc[col].apply(clean_time)

    df_wc = df_wc.sort_values(["train_date", "time_5f"]).drop_duplicates(
        ["ketto_num", "train_date"], keep="first"
    )
    df_wc = optimize_dtypes(df_wc)
    save_with_parquet(df_wc, output_path)
    return df_wc


def create_main_horse_data(
    se_path=None,
    ra_path=None,
    tm_path=None,
    ped_path=None,
    hc_path=None,
    output_path=None,
):
    """
    main 用の前処理済みデータを結合して統合データセットを作成する。

    Args:
        se_path: main_SE_preprocessed.csv のパス
        ra_path: main_RA_preprocessed.csv のパス
        tm_path: TM_preprocessed.csv のパス（学習用を参照）
        ped_path: PED_preprocessed.csv のパス（学習用を参照）
        hc_path: HC_preprocessed.csv のパス（学習用を参照）
        output_path: 出力先（main_horse_data.csv）

    Returns:
        結合済みDataFrame
    """
    project_root = Path(__file__).parent.parent.parent
    base_dir = project_root / "model_training" / "data" / "01_preprocessed"
    return create_horse_data(
        se_path=se_path or (base_dir / "main_SE_preprocessed.csv"),
        ra_path=ra_path or (base_dir / "main_RA_preprocessed.csv"),
        tm_path=tm_path or (base_dir / "TM_preprocessed.csv"),
        ped_path=ped_path or (base_dir / "PED_preprocessed.csv"),
        hc_path=hc_path or (base_dir / "HC_preprocessed.csv"),
        output_path=output_path or (base_dir / "main_horse_data.csv"),
        apply_training_filters=False,
    )


def preprocess_main_data():
    """
    main/data/race/ からSE, RAデータを読み込み、前処理を実行して保存する。
    その後、データを結合して統合データセットを作成する。

    入力:
        - main/data/race/race_ra.csv
        - main/data/race/race_se.csv

    出力:
        - model_training/data/01_preprocessed/main_SE_preprocessed.csv
        - model_training/data/01_preprocessed/main_RA_preprocessed.csv
        - model_training/data/01_preprocessed/main_horse_data.csv
    """
    project_root = Path(__file__).parent.parent.parent

    # 入力パス
    main_race_dir = project_root / "main" / "data" / "race"
    se_input_path = main_race_dir / "race_se.csv"
    ra_input_path = main_race_dir / "race_ra.csv"

    # 出力パス
    output_dir = project_root / "model_training" / "data" / "01_preprocessed"
    se_output_path = output_dir / "main_SE_preprocessed.csv"
    ra_output_path = output_dir / "main_RA_preprocessed.csv"
    horse_output_path = output_dir / "main_horse_data.csv"

    print("=" * 60)
    print("main/data/race のデータ前処理を開始します")
    print("=" * 60)

    # SEデータの前処理
    print("\n[1/3] SEデータの前処理")
    if not se_input_path.exists():
        raise FileNotFoundError(f"SEデータファイルが見つかりません: {se_input_path}")

    df_se = preprocess_se_data(input_path=se_input_path, output_path=se_output_path)
    print(f"SE前処理完了: {df_se.shape[0]}行, {df_se.shape[1]}列")
    print(f"保存先: {se_output_path}")
    print("-" * 30)

    # RAデータの前処理
    print("\n[2/3] RAデータの前処理")
    if not ra_input_path.exists():
        raise FileNotFoundError(f"RAデータファイルが見つかりません: {ra_input_path}")

    df_ra = preprocess_ra_data(input_path=ra_input_path, output_path=ra_output_path)
    print(f"RA前処理完了: {df_ra.shape[0]}行, {df_ra.shape[1]}列")
    print(f"保存先: {ra_output_path}")
    print("-" * 30)

    # データの結合
    print("\n[3/3] データの結合")
    df_horse = create_main_horse_data(
        se_path=se_output_path,
        ra_path=ra_output_path,
        output_path=horse_output_path,
    )
    print(f"保存先: {horse_output_path}")
    print("-" * 30)

    print("\n" + "=" * 60)
    print("main/data/race のデータ前処理が完了しました")
    print("=" * 60)

    return df_se, df_ra, df_horse


def update_preprocessed_data(
    *,
    mode: str = "all",
    state_path: str | None = None,
) -> dict:
    """
    前処理済みデータを「更新（作り直し）」するための統一エントリポイント。

    - mode="train": common/data/output の蓄積データから、学習用前処理（SE/RA/TM/PED/HC）を更新
    - mode="main":  main/data/race の直近データから、予測用前処理（main_SE/main_RA）を更新
    - mode="all":   上記両方

    実行後、更新日(YYYYMMDD)を state に保存する。

    Args:
        mode: "train" | "main" | "all"
        state_path: 状態ファイルのパス（省略時: model_training/data/state/preprocessed_last_update.json）

    Returns:
        state(dict): 保存した状態（last_update_date 等）
    """
    project_root = Path(__file__).parent.parent.parent
    if state_path is None:
        state_file = (
            project_root
            / "model_training"
            / "data"
            / "state"
            / "preprocessed_last_update.json"
        )
    else:
        state_file = Path(state_path)

    mode = str(mode).lower().strip()
    if mode not in {"train", "main", "all"}:
        raise ValueError('mode must be one of: "train", "main", "all"')

    out: dict = {}
    if mode in {"train", "all"}:
        # 既存の学習パイプライン（フル作り直し）
        # - 出力は model_training/data/01_preprocessed/ に揃う
        preprocess_all()
        # ★重要: horse_data.csv は preprocess_all() では作られないため、ここで必ず再生成する
        # これをしないと create_features/create_pastfeatures が古い horse_data を参照し、年が途中で止まる
        print("\n" + "=" * 60)
        print("最終データセット(horse_data.csv)を作成します")
        print("=" * 60)
        create_horse_data()
        out["train_updated"] = True
        out["horse_data_updated"] = True

    if mode in {"main", "all"}:
        # 予測用の直近データ（main/data/race）から前処理
        preprocess_main_data()
        out["main_updated"] = True

    return update_state(state_file, mode=mode, updates=out)


if __name__ == "__main__":
    preprocess_all()
    print("\n" + "=" * 60)
    print("最終データセット(horse_data.csv)を作成します")
    print("=" * 60)
    create_horse_data()

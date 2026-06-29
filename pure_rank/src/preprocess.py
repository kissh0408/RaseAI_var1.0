"""
preprocess.py — RaceAI_var1.0 前処理スクリプト

RaceAI_var2.0.0 の前処理済み Parquet を読み込み、pure_rank 用の
SE_preprocessed / RA_preprocessed / SK_preprocessed を生成する。

市場情報（odds, popularity）は一切出力しない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── パス解決 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── SE 前処理 ─────────────────────────────────────────────────────────────────

# var2.0.0 SE parquet に存在する列のうち pure_rank で使う列
_SE_SOURCE_COLS = [
    "race_id", "year", "month_day", "course_code", "kai", "nichi", "race_num",
    "wakuban", "horse_num", "ketto_num",
    "sex_code", "age",
    "trainer_code", "jockey_code",
    "burden_weight", "blinker_code",
    "horse_weight", "horse_weight_change", "abnormal_code",
    "finish_rank",
    "racetime",       # 走破タイム（秒 float）
    "time_3f_after",  # 上がり3F（秒 float）
    "corner_1", "corner_2", "corner_3", "corner_4",
    "hon_shokin", "fuka_shokin",
    "running_style_code",
    # 明示的に除外: odds, popularity（市場情報）
]

# var2.0.0 horse_data には SE+RA が結合済み。RA 由来の列も含まれる。
_RA_SOURCE_COLS_FROM_HD = [
    "race_id", "year", "month_day", "course_code", "kai", "nichi", "race_num",
    "grade_code", "race_type_code", "weight_type",
    "race_condition_code", "race_level", "race_age_type",
    "distance", "track_code", "course_kubun",
    "registered_count", "running_count", "finish_count",
    "weather_code", "turf_condition", "dirt_condition",
]

_SK_SOURCE_COLS = ["ketto_num", "sire_id", "bms_id"]


def _make_race_date(df: pd.DataFrame) -> pd.Series:
    """year + month_day から race_date (datetime) を生成する。
    例: year=2015, month_day=104 → 2015-01-04
    """
    return pd.to_datetime(
        df["year"].astype(str) + df["month_day"].astype(str).str.zfill(4),
        format="%Y%m%d",
    )


def preprocess_se(src_hd: pd.DataFrame, dst_parquet: Path) -> pd.DataFrame:
    """SE（出走成績）を pure_rank 用に前処理して保存する。

    Parameters
    ----------
    src_hd : horse_data.parquet を読み込んだ DataFrame
    dst_parquet : 保存先パス
    """
    # horse_data から SE 列のみ抽出（重複行を除去: ketto_num×race_id でユニーク化）
    available_cols = [c for c in _SE_SOURCE_COLS if c in src_hd.columns]
    df = src_hd[available_cols].drop_duplicates(subset=["race_id", "ketto_num"]).copy()

    # race_date 生成
    df["race_date"] = _make_race_date(df)

    # is_win / is_place フラグ（finish_rank > 0 の行のみ有効）
    df["is_win"] = (df["finish_rank"] == 1).astype(np.int8)
    df["is_place"] = (df["finish_rank"] <= 3).astype(np.int8)

    # horse_weight_change: var2.0.0 では符号付き float として保存済み
    # (weight_change_sign × weight_change は var2.0.0 の前処理で計算済み)

    # racetime, time_3f_after: 秒単位 float として var2.0.0 で変換済み
    # 0 値を NaN に変換（未記録を意味する）
    df.loc[df["racetime"] <= 0, "racetime"] = np.nan
    df.loc[df["time_3f_after"] <= 0, "time_3f_after"] = np.nan

    # 保存
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False, compression="snappy")
    print(f"[preprocess_se] saved: {dst_parquet} | rows={len(df):,} | cols={len(df.columns)}")
    return df


def preprocess_ra(src_hd: pd.DataFrame, dst_parquet: Path) -> pd.DataFrame:
    """RA（レース情報）を pure_rank 用に前処理して保存する。

    Parameters
    ----------
    src_hd : horse_data.parquet を読み込んだ DataFrame
    dst_parquet : 保存先パス
    """
    available_cols = [c for c in _RA_SOURCE_COLS_FROM_HD if c in src_hd.columns]
    # race_id 単位でユニーク化（SE+RA merged なので horse 数だけ重複している）
    df = src_hd[available_cols].drop_duplicates(subset=["race_id"]).copy()

    # race_date 生成
    df["race_date"] = _make_race_date(df)

    # surface_code: track_code の十の位（1=芝, 2=ダート, 5=障害）
    df["surface_code"] = (df["track_code"] // 10).astype(np.int8)

    # track_condition_code: 芝→turf_condition, ダート→dirt_condition
    # 0 = コード無し（障害・データ未記録）
    df["track_condition_code"] = np.where(
        df["surface_code"] == 1,
        df["turf_condition"],
        df["dirt_condition"],
    ).astype(np.int8)

    # surface_condition: 馬場種別 × 状態 の複合コード
    df["surface_condition"] = (df["surface_code"] * 10 + df["track_condition_code"]).astype(np.int8)

    # distance_category: 距離帯カテゴリ
    # 0=短距離(≤1400), 1=マイル(1401-1800), 2=中距離(1801-2200), 3=長距離(>2200)
    df["distance_category"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 99999],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(np.int8)

    # horse_count として running_count を使う（RA の出走頭数、前処理前の値）
    df = df.rename(columns={"running_count": "horse_count"})

    # 保存
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False, compression="snappy")
    print(f"[preprocess_ra] saved: {dst_parquet} | rows={len(df):,} | cols={len(df.columns)}")
    return df


def preprocess_sk(src_hd: pd.DataFrame, dst_parquet: Path) -> pd.DataFrame:
    """SK（血統）を pure_rank 用に前処理して保存する。

    ketto_num をキーとして sire_id（父）と bms_id（母父）を保存する。
    var2.0.0 horse_data には sire_id / bms_id が既に結合済み。

    Parameters
    ----------
    src_hd : horse_data.parquet を読み込んだ DataFrame
    dst_parquet : 保存先パス
    """
    available_cols = [c for c in _SK_SOURCE_COLS if c in src_hd.columns]
    df = src_hd[available_cols].drop_duplicates(subset=["ketto_num"]).copy()

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False, compression="snappy")
    print(f"[preprocess_sk] saved: {dst_parquet} | rows={len(df):,} | cols={len(df.columns)}")
    return df


def preprocess_hc(hc_dir: Path, dst_parquet: Path) -> pd.DataFrame:
    """HC（坂路調教）CSV全年をまとめて前処理して保存する。"""
    import glob
    files = sorted(glob.glob(str(hc_dir / "slop_hc_*.csv")))
    if not files:
        raise FileNotFoundError(f"HC CSVが見つかりません: {hc_dir}")

    USE_COLS = [
        "ketto_num", "training_date", "training_center",
        "time_4f_total", "time_3f_total", "lap_time_200_0",
    ]
    dfs = []
    for f in files:
        tmp = pd.read_csv(f, encoding="utf-8-sig", usecols=USE_COLS, dtype=str)
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    # 数値変換
    for col in ["ketto_num", "time_4f_total", "time_3f_total", "lap_time_200_0"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["training_center"] = pd.to_numeric(df["training_center"], errors="coerce").astype("Int8")

    # 日付変換
    df["training_date"] = pd.to_datetime(
        df["training_date"].astype(str).str.zfill(8), format="%Y%m%d", errors="coerce"
    )

    # 無効行除外（タイム0 or NaN）
    df = df[
        (df["time_3f_total"] > 0) &
        (df["time_4f_total"] > 0) &
        (df["lap_time_200_0"] > 0) &
        df["training_date"].notna() &
        df["ketto_num"].notna()
    ].copy()

    # タイム変換: 1/10秒単位 → 実秒 float32
    df["hc_3f_sec"]  = (df["time_3f_total"]  / 10.0).astype("float32")
    df["hc_4f_sec"]  = (df["time_4f_total"]  / 10.0).astype("float32")
    df["hc_200_sec"] = (df["lap_time_200_0"] / 10.0).astype("float32")
    df["ketto_num"]  = df["ketto_num"].astype(np.int64)

    out_cols = ["ketto_num", "training_date", "training_center", "hc_3f_sec", "hc_4f_sec", "hc_200_sec"]
    df = df[out_cols].sort_values(["ketto_num", "training_date"]).reset_index(drop=True)

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False, compression="snappy")
    print(f"[preprocess_hc] saved: {dst_parquet} | rows={len(df):,}")
    return df


def preprocess_wc(wc_dir: Path, dst_parquet: Path) -> pd.DataFrame:
    """WC（コース調教）CSV全年をまとめて前処理して保存する。"""
    import glob
    files = sorted(glob.glob(str(wc_dir / "wood_wc_*.csv")))
    if not files:
        raise FileNotFoundError(f"WC CSVが見つかりません: {wc_dir}")

    USE_COLS = [
        "ketto_num", "training_date", "training_center", "course",
        "time_4f_total", "time_3f_total", "lap_time_1f_0f",
    ]
    dfs = []
    for f in files:
        tmp = pd.read_csv(f, encoding="utf-8-sig", usecols=USE_COLS, dtype=str)
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    for col in ["ketto_num", "time_4f_total", "time_3f_total", "lap_time_1f_0f"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["training_center"] = pd.to_numeric(df["training_center"], errors="coerce").astype("Int8")
    df["course"] = pd.to_numeric(df["course"], errors="coerce").astype("Int8")

    df["training_date"] = pd.to_datetime(
        df["training_date"].astype(str).str.zfill(8), format="%Y%m%d", errors="coerce"
    )

    df = df[
        (df["time_3f_total"] > 0) &
        (df["time_4f_total"] > 0) &
        (df["lap_time_1f_0f"] > 0) &
        df["training_date"].notna() &
        df["ketto_num"].notna()
    ].copy()

    df["wc_3f_sec"] = (df["time_3f_total"]   / 10.0).astype("float32")
    df["wc_4f_sec"] = (df["time_4f_total"]   / 10.0).astype("float32")
    df["wc_1f_sec"] = (df["lap_time_1f_0f"] / 10.0).astype("float32")
    df["ketto_num"] = df["ketto_num"].astype(np.int64)

    out_cols = ["ketto_num", "training_date", "training_center", "course", "wc_3f_sec", "wc_4f_sec", "wc_1f_sec"]
    df = df[out_cols].sort_values(["ketto_num", "training_date"]).reset_index(drop=True)

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False, compression="snappy")
    print(f"[preprocess_wc] saved: {dst_parquet} | rows={len(df):,}")
    return df


def main() -> None:
    cfg = load_config()

    src_dir = Path(cfg["data"]["src_parquet_dir"])
    dst_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]

    # var2.0.0 の horse_data.parquet を読む（SE+RA+血統 の結合済みテーブル）
    src_hd_path = src_dir / "horse_data.parquet"
    print(f"Loading source: {src_hd_path}")
    hd = pd.read_parquet(src_hd_path)
    print(f"  rows={len(hd):,}, cols={len(hd.columns)}")

    # 前処理実行
    preprocess_se(hd, dst_dir / "SE_preprocessed.parquet")
    preprocess_ra(hd, dst_dir / "RA_preprocessed.parquet")
    preprocess_sk(hd, dst_dir / "SK_preprocessed.parquet")

    # HC/WC 調教データの前処理
    hc_dir = Path(cfg["data"]["hc_dir"])
    wc_dir = Path(cfg["data"]["wc_dir"])
    preprocess_hc(hc_dir, dst_dir / "HC_preprocessed.parquet")
    preprocess_wc(wc_dir, dst_dir / "WC_preprocessed.parquet")

    print("\n[preprocess] Done.")


if __name__ == "__main__":
    main()

"""
preprocess.py — RaceAI_var1.0 前処理スクリプト

RaceAI_var2.0.0 の前処理済み Parquet を読み込み、pure_rank 用の
SE_preprocessed / RA_preprocessed / SK_preprocessed を生成する。

HR（払戻）データは評価・シミュレーション専用。特徴量には merge しない。
JV-Link 取得後: common/data/output/race_hr/race_hr_YYYY.csv を配置して
  python pure_rank/src/preprocess.py --hr-only
を実行する。
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
    # JRA公式データマイニング予想（0B13経由、race_se.mining_predicted_rank）。
    # 市場オッズ・人気とは別の JRA 自身の予測アルゴリズム出力（Phase 6, v42_mining 実験）。
    # 仕様書: docs/specs/2026-07-04-phase6-jra-mining-design.md
    "mining_predicted_rank",
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

    # mining_predicted_rank: JRA公式データマイニング予想の着順予想（1=予想最速、
    # horse_count=予想最遅）。var2.0.0 側の horse_data.parquet 生成時に既に
    # 0→NaN 変換済み（preprocessing.py 316-320行目）だが、防御的に再度実施する。
    # 仕様書: docs/specs/2026-07-04-phase6-jra-mining-design.md 5-1節
    if "mining_predicted_rank" in df.columns:
        df["mining_predicted_rank"] = pd.to_numeric(
            df["mining_predicted_rank"], errors="coerce"
        )
        df.loc[df["mining_predicted_rank"] <= 0, "mining_predicted_rank"] = np.nan

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


def _make_race_id(row: pd.Series) -> str:
    """year/month_day/course_code/kai/nichi/race_num から 16 桁 race_id を生成。"""
    return (
        str(int(row["year"])).zfill(4)
        + str(int(row["month_day"])).zfill(4)
        + str(int(row["course_code"])).zfill(2)
        + str(int(row["kai"])).zfill(2)
        + str(int(row["nichi"])).zfill(2)
        + str(int(row["race_num"])).zfill(2)
    )


def _parse_kumi(kumi: str) -> tuple[int | None, int | None]:
    """JV-Link 組番文字列（例: '0305'）を馬番2頭に分解する。"""
    s = str(kumi).strip()
    if not s or s == "0" * len(s) or len(s) < 4:
        return None, None
    h1 = int(s[:2])
    h2 = int(s[2:4])
    if h1 <= 0 or h2 <= 0:
        return None, None
    return min(h1, h2), max(h1, h2)


def _parse_hr_records(hr_dir: Path) -> pd.DataFrame:
    """HR CSV を long format の払戻テーブルに変換する。

    出力列: race_id, bet_type, horse_num_1, horse_num_2, payout
    payout は 100 円あたりの払戻金額（整数）。
    """
    import glob

    files = sorted(glob.glob(str(hr_dir / "race_hr_*.csv")))
    if not files:
        raise FileNotFoundError(
            f"HR CSV が見つかりません: {hr_dir / 'race_hr_*.csv'}\n"
            "JV-Link で HR レコードを取得し common/data/output/race_hr/ に配置してください。"
        )

    quinella_cols = [
        ("quinella_1_kumi", "quinella_1_money"),
        ("quinella_2_kumi", "quinella_2_money"),
        ("quinella_3_kumi", "quinella_3_money"),
    ]
    wide_cols = [
        (f"wide_{i}_kumi", f"wide_{i}_money") for i in range(1, 8)
    ]
    win_cols = [
        (f"win_{i}_horse", f"win_{i}_money") for i in range(1, 4)
    ]

    rows: list[dict] = []
    for fpath in files:
        hr = pd.read_csv(fpath, encoding="utf-8-sig", dtype=str, low_memory=False)
        hr = hr[hr.get("record_id", "HR") == "HR"] if "record_id" in hr.columns else hr

        for _, row in hr.iterrows():
            try:
                race_id = _make_race_id(row)
            except (ValueError, TypeError, KeyError):
                continue

            for hcol, mcol in win_cols:
                if hcol not in hr.columns or mcol not in hr.columns:
                    continue
                hraw = str(row.get(hcol, "")).strip()
                if not hraw or hraw == "00":
                    continue
                try:
                    horse = int(hraw)
                    payout = int(str(row.get(mcol, "0")).strip() or "0")
                except ValueError:
                    continue
                if horse <= 0 or payout <= 0:
                    continue
                rows.append({
                    "race_id": race_id,
                    "bet_type": "win",
                    "horse_num_1": horse,
                    "horse_num_2": 0,
                    "payout": payout,
                })

            for kumi_col, money_col in quinella_cols:
                if kumi_col not in hr.columns or money_col not in hr.columns:
                    continue
                h1, h2 = _parse_kumi(row.get(kumi_col, ""))
                if h1 is None:
                    continue
                try:
                    payout = int(str(row.get(money_col, "0")).strip() or "0")
                except ValueError:
                    continue
                if payout <= 0:
                    continue
                rows.append({
                    "race_id": race_id,
                    "bet_type": "quinella",
                    "horse_num_1": h1,
                    "horse_num_2": h2,
                    "payout": payout,
                })

            for kumi_col, money_col in wide_cols:
                if kumi_col not in hr.columns or money_col not in hr.columns:
                    continue
                h1, h2 = _parse_kumi(row.get(kumi_col, ""))
                if h1 is None:
                    continue
                try:
                    payout = int(str(row.get(money_col, "0")).strip() or "0")
                except ValueError:
                    continue
                if payout <= 0:
                    continue
                rows.append({
                    "race_id": race_id,
                    "bet_type": "wide",
                    "horse_num_1": h1,
                    "horse_num_2": h2,
                    "payout": payout,
                })

    if not rows:
        raise ValueError(f"HR CSV から有効な払戻レコードを抽出できませんでした: {hr_dir}")

    out = pd.DataFrame(rows)
    out["horse_num_1"] = out["horse_num_1"].astype(np.int16)
    out["horse_num_2"] = out["horse_num_2"].astype(np.int16)
    out["payout"] = out["payout"].astype(np.int32)
    return out.drop_duplicates(
        subset=["race_id", "bet_type", "horse_num_1", "horse_num_2"]
    ).reset_index(drop=True)


def preprocess_hr(hr_dir: Path, dst_parquet: Path) -> pd.DataFrame:
    """HR（払戻）CSV を評価用 Parquet に変換する。特徴量には使用しない。"""
    df = _parse_hr_records(hr_dir)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False, compression="snappy")
    print(f"[preprocess_hr] saved: {dst_parquet} | rows={len(df):,}")
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="pure_rank preprocessing")
    parser.add_argument("--hr-only", action="store_true", help="HR payout preprocessing only")
    args = parser.parse_args()

    cfg = load_config()
    dst_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]

    if args.hr_only:
        hr_dir = Path(cfg["data"].get("hr_dir", "common/data/output/race_hr"))
        if not hr_dir.is_absolute():
            hr_dir = PROJECT_ROOT / hr_dir
        preprocess_hr(hr_dir, dst_dir / "HR_preprocessed.parquet")
        print("\n[preprocess] HR Done.")
        return

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

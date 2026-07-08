"""CSVデータからSE/RA/PED/UMテーブルをJVData.dbに構築する。

共通データソース:
  common/data/output/race_se/race_se_YYYY.csv  (2015-2026)
  common/data/output/race_ra/race_ra_YYYY.csv  (2015-2026)
  common/data/output/blod_sk/blod_sk.csv       (血統)

変換仕様:
  race_id       = YYYY MMDD CC KK NN RR (4+4+2+2+2+2 = 16桁文字列)
  finish_time   = (time // 1000)*60 + (time % 1000)/10  [秒]
  agari3f       = time_3f_after / 10  [秒]
  time_diff     = time_diff / 10  [秒、勝者からの差]
  carry_weight  = burden_weight / 10  [kg]
  odds          = odds / 10  [デシマルオッズ]
  surface_code  = 1(芝)/ 2(ダート)/ 3(障害) from track_code
  lap_times     = スペース区切り秒数文字列
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from pipeline_common import get_db_connection

CSV_DIR = ROOT / "common" / "data" / "output"
YEARS = list(range(2015, 2027))


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------

RACE_ID_COLS = ["year", "month_day", "course_code", "kai", "nichi", "race_num"]


def _make_race_id_vec(df: pd.DataFrame) -> pd.Series:
    """RACE_ID_COLS から16桁race_idをベクトル演算で生成する（NaNなし前提）。"""
    return (
        df["year"].astype(int).astype(str).str.zfill(4)
        + df["month_day"].astype(int).astype(str).str.zfill(4)
        + df["course_code"].astype(int).astype(str).str.zfill(2)
        + df["kai"].astype(int).astype(str).str.zfill(2)
        + df["nichi"].astype(int).astype(str).str.zfill(2)
        + df["race_num"].astype(int).astype(str).str.zfill(2)
    )


def _make_race_date_vec(df: pd.DataFrame) -> pd.Series:
    """year/month_day から 'YYYY-MM-DD' をベクトル演算で生成する。"""
    md = df["month_day"].astype(int).astype(str).str.zfill(4)
    return (
        df["year"].astype(int).astype(str).str.zfill(4)
        + "-" + md.str[:2] + "-" + md.str[2:]
    )


def _time_to_seconds(t: float | int) -> float:
    """JV-Link走破タイム (MMSS.S形式整数) → 秒"""
    if pd.isna(t) or t == 0:
        return np.nan
    t = int(t)
    minutes = t // 1000
    rest = t % 1000
    seconds = rest // 10
    tenths = rest % 10
    return minutes * 60 + seconds + tenths / 10.0


def _surface_from_track_code(track_code: int) -> int:
    """JV-Link track_code → surface_code (1=芝, 2=ダート, 3=障害)"""
    tc = int(track_code) if not pd.isna(track_code) else 0
    if 10 <= tc <= 19:
        return 1
    elif 20 <= tc <= 29:
        return 2
    elif tc >= 50:
        return 3
    return 2  # 不明はダートとして扱う


def _parse_lap_times(lap_str: str) -> str:
    """75文字の連続ラップタイム文字列 → スペース区切り秒数文字列"""
    if not isinstance(lap_str, str) or not lap_str.strip():
        return ""
    s = lap_str.strip().replace(" ", "")
    laps = []
    for i in range(0, len(s), 3):
        chunk = s[i:i+3]
        if len(chunk) == 3:
            try:
                val = int(chunk)
                if val > 0:
                    laps.append(f"{val / 10:.1f}")
            except ValueError:
                pass
    return " ".join(laps) if laps else ""


# ---------------------------------------------------------------------------
# RA テーブル構築
# ---------------------------------------------------------------------------

def build_ra_table(conn: sqlite3.Connection) -> None:
    print("Building RA table...")
    dfs = []
    for year in YEARS:
        path = CSV_DIR / "race_ra" / f"race_ra_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df["_year_load"] = year
        dfs.append(df)
        print(f"  {year}: {len(df)} rows")

    if not dfs:
        raise FileNotFoundError("race_ra CSVが見つかりません")

    df = pd.concat(dfs, ignore_index=True)

    # 数値変換
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num",
                "distance", "track_code", "running_count", "registered_count",
                "turf_condition", "dirt_condition", "grade_code", "weather_code"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # race_id / race_date
    df["race_id"] = _make_race_id_vec(df)
    df["race_date"] = _make_race_date_vec(df)

    # surface_code
    df["surface_code"] = df["track_code"].apply(_surface_from_track_code)

    # track_condition_code (芝はturf_condition, ダートはdirt_condition)
    df["track_condition_code"] = np.where(
        df["surface_code"] == 1,
        df["turf_condition"].replace(0, 1),
        df["dirt_condition"].replace(0, 1),
    )

    # lap_times: 文字列形式に変換
    if "lap_times" in df.columns:
        df["lap_times"] = df["lap_times"].apply(_parse_lap_times)
    else:
        df["lap_times"] = ""

    # grade_code: NaN/0 → 7 (条件戦のデフォルト)
    df["grade_code"] = df["grade_code"].replace(0, 7)

    # horse_count
    df["horse_count"] = df["running_count"]

    # base_time・standard_weight は SE から後で埋める（とりあえず0）
    df["base_time"] = 0.0
    df["standard_weight"] = 55.0

    out_cols = [
        "race_id", "race_date", "course_code", "race_num", "distance",
        "surface_code", "track_condition_code", "grade_code", "horse_count",
        "base_time", "standard_weight", "lap_times", "weather_code",
        "year", "month_day", "kai", "nichi",
    ]
    out = df[out_cols].drop_duplicates("race_id")

    out.to_sql("RA", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ra_race_id ON RA(race_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ra_race_date ON RA(race_date)")
    conn.commit()
    print(f"  RA: {len(out)} races saved")


# ---------------------------------------------------------------------------
# SE テーブル構築
# ---------------------------------------------------------------------------

def build_se_table(conn: sqlite3.Connection) -> None:
    print("Building SE table...")
    dfs = []
    for year in YEARS:
        path = CSV_DIR / "race_se" / f"race_se_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df["_year_load"] = year
        dfs.append(df)
        print(f"  {year}: {len(df)} rows")

    if not dfs:
        raise FileNotFoundError("race_se CSVが見つかりません")

    df = pd.concat(dfs, ignore_index=True)

    # 数値変換
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num",
                "horse_num", "wakuban", "age", "sex_code",
                "burden_weight", "horse_weight", "weight_change",
                "abnormal_code", "finish_rank", "final_rank",
                "time", "time_3f_after", "time_diff", "odds",
                "corner_1", "corner_2", "corner_3", "corner_4",
                "running_style_code"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ID生成
    df["race_id"] = _make_race_id_vec(df)
    df["horse_id"] = pd.to_numeric(df["ketto_num"], errors="coerce").fillna(0).astype(int)

    # 時間変換
    df["finish_time"] = df["time"].apply(_time_to_seconds)
    df["agari3f"] = df["time_3f_after"] / 10.0
    df["time_diff_sec"] = df["time_diff"] / 10.0

    # 馬体重変化の符号
    sign = df["weight_change_sign"].map({"+": 1, "-": -1}).fillna(0)
    df["horse_weight_diff"] = df["weight_change"] * sign

    # 負担重量
    df["carry_weight"] = df["burden_weight"] / 10.0

    # オッズ (0→NaN)
    df["odds_decimal"] = (df["odds"] / 10.0).replace(0, np.nan)

    # コーナー通過順位 (0→NaN)
    for c in ["corner_1", "corner_2", "corner_3", "corner_4"]:
        df[c] = df[c].replace(0, np.nan)

    # finish_rank: final_rank を優先（入線順位より確定着順）
    df["finish_rank_final"] = np.where(
        df["final_rank"] > 0, df["final_rank"], df["finish_rank"]
    )

    # jockey_id / trainer_id
    df["jockey_id"] = pd.to_numeric(df["jockey_code"], errors="coerce").fillna(0).astype(int)
    df["trainer_id"] = pd.to_numeric(df["trainer_code"], errors="coerce").fillna(0).astype(int)

    # gate_num = wakuban
    df["gate_num"] = df["wakuban"].astype(int)

    # 元のカラムを削除してからリネーム（重複防止）
    df = df.drop(columns=["finish_rank", "time_diff", "odds"], errors="ignore")
    df = df.rename(columns={
        "finish_rank_final": "finish_rank",
        "time_diff_sec": "time_diff",
        "odds_decimal": "odds",
    })

    save_cols = [
        "race_id", "horse_id", "horse_num", "gate_num",
        "finish_rank", "abnormal_code", "carry_weight",
        "horse_weight", "horse_weight_diff",
        "finish_time", "agari3f", "time_diff",
        "jockey_id", "trainer_id", "odds",
        "corner_1", "corner_2", "corner_3", "corner_4",
        "running_style_code", "sex_code", "age",
    ]
    out = df[save_cols]

    out.to_sql("SE", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_se_race_id ON SE(race_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_se_horse_id ON SE(horse_id)")
    conn.commit()
    print(f"  SE: {len(out)} horse-race rows saved")

    # base_time を SE から RA に更新（各レースの勝者タイム）
    print("  Updating RA.base_time from winner times...")
    winner_times = df[df["finish_rank"] == 1][["race_id", "finish_time"]].copy()
    winner_times = winner_times.dropna().rename(columns={"finish_time": "base_time"})
    winner_times.to_sql("_winner_times", conn, if_exists="replace", index=False)
    conn.execute("""
        UPDATE RA SET base_time = (
            SELECT base_time FROM _winner_times WHERE _winner_times.race_id = RA.race_id
        )
        WHERE EXISTS (
            SELECT 1 FROM _winner_times WHERE _winner_times.race_id = RA.race_id
        )
    """)
    conn.execute("DROP TABLE IF EXISTS _winner_times")
    conn.commit()


# ---------------------------------------------------------------------------
# PED テーブル（血統）
# ---------------------------------------------------------------------------

def build_ped_table(conn: sqlite3.Connection) -> None:
    print("Building PED table (pedigree)...")
    sk_path = CSV_DIR / "blod_sk" / "blod_sk.csv"
    if not sk_path.exists():
        print("  blod_sk.csv not found, skipping PED table")
        return

    df = pd.read_csv(sk_path, dtype=str, low_memory=False)
    df["horse_id"] = pd.to_numeric(df["ketto_num"], errors="coerce").fillna(0).astype(int)
    df["sire_id"] = pd.to_numeric(df["p_sire"], errors="coerce").fillna(0).astype(int)
    df["bms_id"] = pd.to_numeric(df["p_dam_sire"], errors="coerce").fillna(0).astype(int)  # 母父

    out = df[["horse_id", "sire_id", "bms_id"]].drop_duplicates("horse_id")
    out.to_sql("PED", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ped_horse_id ON PED(horse_id)")
    conn.commit()
    print(f"  PED: {len(out)} horses saved")


# ---------------------------------------------------------------------------
# UM テーブル（馬マスター、SEから推定）
# ---------------------------------------------------------------------------

def build_um_table(conn: sqlite3.Connection) -> None:
    """BasicFeatureBuilder._attach_horse_meta() のクエリを通すために UM テーブルを作成。

    SEテーブルの race_year - age で birth_year を近似する。
    """
    print("Building UM table (horse master from SE)...")

    # race_id先頭4文字が race_year
    df = pd.read_sql_query(
        """
        SELECT se.horse_id,
               se.sex_code,
               se.age,
               CAST(SUBSTR(se.race_id, 1, 4) AS INTEGER) AS race_year
        FROM SE se
        WHERE se.finish_rank > 0
        """,
        conn,
    )

    # 馬ごとに最初のレースの race_year - age = birth_year を推定
    df["birth_year"] = df["race_year"] - df["age"]
    horse_meta = (
        df.sort_values("race_year")
        .groupby("horse_id")
        .agg(birth_year=("birth_year", "first"), sex_code=("sex_code", "first"))
        .reset_index()
    )
    horse_meta["birth_year"] = horse_meta["birth_year"].fillna(2010).astype(int).clip(1990, 2024)
    horse_meta["birth_date"] = horse_meta["birth_year"].astype(str) + "-01-01"

    horse_meta[["horse_id", "sex_code", "birth_date"]].to_sql("UM", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_um_horse_id ON UM(horse_id)")
    conn.commit()
    print(f"  UM: {len(horse_meta)} horses saved")


# ---------------------------------------------------------------------------
# DM テーブル（JRAデータマイニング予測タイム）
# ---------------------------------------------------------------------------

def _decode_dm_time(t: pd.Series) -> pd.Series:
    """JRA DM予測タイム (MSSCC形式) → 秒。例: 11552 → 75.52s"""
    t = pd.to_numeric(t, errors="coerce")
    t = t.where(t != 0)  # 0は欠損扱い
    minutes = t // 10000
    seconds = (t % 10000) // 100
    centiseconds = t % 100
    return minutes * 60 + seconds + centiseconds / 100.0


def _load_mining_long(csv_path: Path, value_suffixes: list[str]) -> pd.DataFrame:
    """ming_*.csv のワイド形式（mining_pred_{i}_*）をhorse-level縦型に変換する。

    iterrowsを避け、予測スロットi（1..18）ごとの列スライスで処理する。
    horse_num が NaN/0 のスロットは除外する（元実装と同じ条件）。
    """
    df = pd.read_csv(csv_path, encoding="cp932", low_memory=False)
    for col in RACE_ID_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=RACE_ID_COLS)
    df["race_id"] = _make_race_id_vec(df)

    frames = []
    for i in range(1, 19):
        hn_col = f"mining_pred_{i}_horse_num"
        if hn_col not in df.columns:
            continue
        sub = pd.DataFrame({"race_id": df["race_id"]})
        sub["horse_num"] = pd.to_numeric(df[hn_col], errors="coerce")
        for suffix in value_suffixes:
            src = f"mining_pred_{i}_{suffix}"
            sub[suffix] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
        sub = sub[sub["horse_num"].notna() & (sub["horse_num"] != 0)]
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["horse_num"] = out["horse_num"].astype(int)
    return out


def build_dm_table(conn: sqlite3.Connection) -> None:
    """ming_dm.csv からDM予測タイムをhorse-level縦型テーブルに変換して保存。"""
    print("Building DM table...")
    csv_path = CSV_DIR / "ming_dm" / "ming_dm.csv"
    if not csv_path.exists():
        print("  [SKIP] ming_dm.csv が見つかりません")
        return

    long_df = _load_mining_long(csv_path, ["time", "error+", "error-"])
    if long_df.empty:
        print("  [WARN] DM データが空です")
        return

    dm_df = pd.DataFrame({
        "race_id": long_df["race_id"],
        "horse_num": long_df["horse_num"],
        "dm_pred_time_s": _decode_dm_time(long_df["time"]),
        "dm_error_plus_s": long_df["error+"] / 100,
        "dm_error_minus_s": long_df["error-"] / 100,
    })
    dm_df.to_sql("DM", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_race_horse ON DM(race_id, horse_num)")
    conn.commit()
    print(f"  DM: {len(dm_df)} rows saved")


# ---------------------------------------------------------------------------
# TM テーブル（JRA公式タイム指数）
# ---------------------------------------------------------------------------

def build_tm_table(conn: sqlite3.Connection) -> None:
    """ming_tm.csv からタイム指数をhorse-level縦型テーブルに変換して保存。"""
    print("Building TM table...")
    csv_path = CSV_DIR / "ming_tm" / "ming_tm.csv"
    if not csv_path.exists():
        print("  [SKIP] ming_tm.csv が見つかりません")
        return

    long_df = _load_mining_long(csv_path, ["score"])
    if long_df.empty:
        print("  [WARN] TM データが空です")
        return

    tm_df = pd.DataFrame({
        "race_id": long_df["race_id"],
        "horse_num": long_df["horse_num"],
        # int(score)相当の切り捨て後、INTEGER/NULLで保存するためInt64を使う
        "jra_tm_score": np.trunc(long_df["score"]).astype("Int64"),
    })
    tm_df.to_sql("TM", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_race_horse ON TM(race_id, horse_num)")
    conn.commit()
    print(f"  TM: {len(tm_df)} rows saved")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    conn = get_db_connection()
    try:
        build_ra_table(conn)
        build_se_table(conn)
        build_ped_table(conn)
        build_um_table(conn)
        build_dm_table(conn)
        build_tm_table(conn)

        # 確認
        for table in ["RA", "SE", "PED", "UM", "DM", "TM"]:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {n} rows")
            except Exception:
                print(f"  {table}: (テーブルなし)")

        print("\n=== DB準備完了 ===")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

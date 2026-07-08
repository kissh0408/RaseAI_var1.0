"""当日出馬表から本番用特徴量を生成する（train/serve skew ゼロ設計）。

NOTE: 本番 rank 系（main/main.py）は別経路。本スクリプトは binary champion 用で
      v3/v4 + v6 going + top3 past まで生成（train_config.backtest_feature_file と整合）。

設計方針:
  当日CSV（race_ra.csv / race_se.csv、JV-Link出馬表フォーマット）を
  「作業用DBコピー」に履歴と同形式でINSERTし、学習時と同一の
  FeatureCreatorビルダー群＋v3/v4特徴量関数をそのまま実行して
  当日行だけをスライスする。特徴量ロジックの二重実装を排除し、
  学習と本番の特徴量定義の乖離を構造的に防ぐ。

当日データの特性（2段階フロー）:
  朝（出馬表確定時）: オッズ=前売り or NaN、馬体重=NaN → 本スクリプト実行
  直前（約50分前） : 確定オッズ・馬体重を merge_late_info() で結合
                     （market系はstrategy_engineがリアルタイムオッズから再計算）

実行:
  python main/build_today_features.py --ra test/race_ra.csv --se test/race_se.csv
出力:
  main/work/today_features_{YYYYMMDD}.parquet
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from builders import (
    BasicFeatureBuilder,
    InteractionFeatureBuilder,
    MiningFeatureBuilder,
    PaceFeatureBuilder,
    PastPerformanceBuilder,
    RunningStyleBuilder,
)
from create_features_v3 import add_jra_tm_orthogonalized, add_weight_relative_z
from create_features_v4 import add_course_topology
from champion_features import apply_champion_feature_stack
from prepare_db import (
    _make_race_date_vec,
    _make_race_id_vec,
    _surface_from_track_code,
)

DB_PATH = ROOT / "common" / "data" / "JVData.db"
WORK_DIR = ROOT / "main" / "work"


def parse_today_ra(ra_csv: Path) -> pd.DataFrame:
    """当日 race_ra.csv → RAテーブルスキーマ（prepare_db と同一変換）。"""
    df = pd.read_csv(ra_csv)
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num",
                "distance", "track_code", "running_count", "grade_code", "weather_code",
                "turf_condition", "dirt_condition"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["race_id"] = _make_race_id_vec(df)
    df["race_date"] = _make_race_date_vec(df)
    df["surface_code"] = df["track_code"].apply(_surface_from_track_code)
    # 馬場状態: 未発表(0)は良(1)扱い（prepare_dbと同一のデフォルト）
    df["track_condition_code"] = np.where(
        df["surface_code"] == 1,
        df["turf_condition"].replace(0, 1),
        df["dirt_condition"].replace(0, 1),
    )
    # 出馬表段階では running_count=0 のため登録頭数でフォールバック
    # （build_today_features 側でSE頭数による上書きも行う）
    df["horse_count"] = df["running_count"].where(
        df["running_count"] > 0, df["registered_count"]
    )
    # レース未実施のため確定後にしか埋まらない列はNaN
    df["base_time"] = np.nan
    df["standard_weight"] = np.nan
    df["lap_times"] = None

    ra_cols = ["race_id", "race_date", "course_code", "race_num", "distance",
               "surface_code", "track_condition_code", "grade_code", "horse_count",
               "base_time", "standard_weight", "lap_times", "weather_code",
               "year", "month_day", "kai", "nichi"]
    return df[ra_cols]


def parse_today_se(se_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """当日 race_se.csv → SEテーブルスキーマ + DMテーブル行（mining列から）。"""
    df = pd.read_csv(se_csv)
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num",
                "wakuban", "horse_num", "sex_code", "age", "abnormal_code"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["race_id"] = _make_race_id_vec(df)

    se = pd.DataFrame({
        "race_id": df["race_id"],
        "horse_id": df["ketto_num"].astype(str),
        "horse_num": df["horse_num"],
        "gate_num": df["wakuban"],
        "finish_rank": 0,            # 未確定
        "abnormal_code": df["abnormal_code"],
        "carry_weight": pd.to_numeric(df["burden_weight"], errors="coerce") / 10.0,
        "horse_weight": pd.to_numeric(df["horse_weight"], errors="coerce"),  # 通常NaN（後で結合）
        "horse_weight_diff": np.nan,
        "finish_time": np.nan,
        "agari3f": np.nan,
        "time_diff": np.nan,
        "jockey_id": df["jockey_code"].astype(str),
        "trainer_id": df["trainer_code"].astype(str),
        "odds": pd.to_numeric(df["odds"], errors="coerce") / 10.0,  # 前売り（あれば）
        "corner_1": np.nan, "corner_2": np.nan, "corner_3": np.nan, "corner_4": np.nan,
        "running_style_code": pd.to_numeric(df.get("running_style_code"), errors="coerce"),
        "sex_code": df["sex_code"],
        "age": df["age"],
    })

    # DM（JRA公式マイニング予測）が出馬表に同梱されていれば抽出
    dm = pd.DataFrame()
    if "mining_predicted_time" in df.columns:
        pred = pd.to_numeric(df["mining_predicted_time"], errors="coerce")
        valid = pred.notna() & (pred > 0)
        if valid.any():
            # prepare_db._decode_dm_time と同一: MMSSf → 秒
            t = pred[valid].astype(int)
            secs = (t // 1000) * 60 + (t % 1000) / 10.0
            dm = pd.DataFrame({
                "race_id": df.loc[valid, "race_id"],
                "horse_num": df.loc[valid, "horse_num"],
                "dm_pred_time_s": secs,
                "dm_error_plus_s": pd.to_numeric(df.loc[valid, "mining_error_plus"], errors="coerce") / 10.0,
                "dm_error_minus_s": pd.to_numeric(df.loc[valid, "mining_error_minus"], errors="coerce") / 10.0,
            })
    return se, dm


def _make_race_id_from_parts(df: pd.DataFrame) -> pd.Series:
    """realtime系CSV（year/month_day/course/kai/nichi/race_num列）からrace_idを構築。"""
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return _make_race_id_vec(df)


def load_realtime_odds(path: Path) -> pd.DataFrame:
    """realtime_odds/o1_odds.csv → race_id, horse_num, odds（デシマル）。"""
    df = pd.read_csv(path)
    if "race_id" not in df.columns:
        df["race_id"] = _make_race_id_from_parts(df)
    df["race_id"] = df["race_id"].astype(str)
    df["odds"] = pd.to_numeric(df["odds_raw"], errors="coerce") / 10.0
    df["horse_num"] = pd.to_numeric(df["horse_num"], errors="coerce").astype(int)
    return df[["race_id", "horse_num", "odds"]].dropna(subset=["odds"])


def load_realtime_weights(path: Path) -> pd.DataFrame:
    """realtime_wh/wh.csv → race_id, horse_num, horse_weight, horse_weight_diff。"""
    df = pd.read_csv(path)
    df["race_id"] = _make_race_id_from_parts(df)
    df["horse_num"] = pd.to_numeric(df["horse_num"], errors="coerce").astype(int)
    df["horse_weight"] = pd.to_numeric(df["horse_weight"], errors="coerce")
    sign = df["weight_change_sign"].astype(str).map({"-": -1}).fillna(1)
    df["horse_weight_diff"] = pd.to_numeric(df["weight_change"], errors="coerce") * sign
    return df[["race_id", "horse_num", "horse_weight", "horse_weight_diff"]].dropna(
        subset=["horse_weight"]
    )


def _extract_raw_records(path: Path, prefix_hex: str) -> list[str]:
    """realtime_tm/dm CSVから生レコード文字列を復元する。

    フェッチャーのカラムパースが壊れていてもhex生データは保持されているため、
    全セルから指定プレフィックス（'544d'=TM, '444d'=DM）のhex文字列を探して復号する。
    """
    df = pd.read_csv(path, dtype=str)
    records = []
    for _, row in df.iterrows():
        for v in row.values:
            if isinstance(v, str) and len(v) > 60 and v.startswith(prefix_hex):
                try:
                    records.append(bytes.fromhex(v).decode("ascii", errors="replace"))
                except ValueError:
                    pass
                break
    return records


def _record_race_id(s: str) -> str:
    """生レコードのヘッダからrace_idを構築する。

    構造: 種別(2)+区分(1)+作成日(8)+開催日(8)+場(2)+回(2)+日(2)+R(2)+発表時刻(4)+ボディ
    """
    kaisai = s[11:19]
    return kaisai[:4] + kaisai[4:8] + s[19:21] + s[21:23] + s[23:25] + s[25:27]


def load_realtime_tm(path: Path) -> pd.DataFrame:
    """realtime_tm/tm.csv（生レコード）→ TMテーブル形式（race_id, horse_num, jra_tm_score）。

    ボディ: 18×(馬番2桁 + スコア4桁)。馬番00は空スロット。
    """
    rows = []
    for s in _extract_raw_records(path, "544d"):
        race_id, body = _record_race_id(s), s[31:]
        for i in range(18):
            seg = body[i * 6:(i + 1) * 6]
            if len(seg) < 6 or not seg[:2].strip().isdigit():
                break
            horse = int(seg[:2])
            if horse == 0:
                continue
            score = pd.to_numeric(seg[2:6], errors="coerce")
            rows.append((race_id, horse, score))
    return pd.DataFrame(rows, columns=["race_id", "horse_num", "jra_tm_score"]).dropna()


def load_realtime_dm(path: Path) -> pd.DataFrame:
    """realtime_dm/dm.csv（生レコード）→ DMテーブル形式。

    ボディ: 18×(馬番2桁 + 予測タイム5桁[M分SS.ff秒] + 誤差+4桁 + 誤差-4桁)。
    予測タイム0は予測なしとしてスキップ。
    """
    rows = []
    for s in _extract_raw_records(path, "444d"):
        race_id, body = _record_race_id(s), s[31:]
        for i in range(18):
            seg = body[i * 15:(i + 1) * 15]
            if len(seg) < 15 or not seg[:2].strip().isdigit():
                break
            horse = int(seg[:2])
            t = pd.to_numeric(seg[2:7], errors="coerce")
            if horse == 0 or not t or t <= 0:
                continue
            pred_s = (t // 10000) * 60 + (t % 10000) / 100.0
            ep = pd.to_numeric(seg[7:11], errors="coerce") / 10.0
            em = pd.to_numeric(seg[11:15], errors="coerce") / 10.0
            rows.append((race_id, horse, pred_s, ep, em))
    return pd.DataFrame(
        rows,
        columns=["race_id", "horse_num", "dm_pred_time_s", "dm_error_plus_s", "dm_error_minus_s"],
    )


def build_today_features(
    ra_csv: Path,
    se_csv: Path,
    odds_csv: Path | None = None,
    wh_csv: Path | None = None,
    tm_csv: Path | None = None,
    dm_csv: Path | None = None,
) -> pd.DataFrame:
    """作業用DBに当日行を挿入し、学習時と同一のビルダーで特徴量を生成して当日分を返す。"""
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] 当日CSVをDBスキーマへ変換...")
    ra_today = parse_today_ra(ra_csv)
    se_today, dm_today = parse_today_se(se_csv)

    # 直前データ（あれば）をSE行・DM/TMに反映してからビルダーに通す。
    # こうすることで weight系・market系・TM/DM系特徴量が学習時と同一ロジックで計算される。
    def _merge_by_horse_num(base: pd.DataFrame, late: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        late = late.drop_duplicates(["race_id", "horse_num"], keep="last")
        merged = base.merge(
            late[["race_id", "horse_num"] + cols],
            on=["race_id", "horse_num"], how="left", suffixes=("", "_late"),
        )
        for c in cols:
            if f"{c}_late" in merged.columns:
                merged[c] = merged[f"{c}_late"].combine_first(merged[c])
                merged = merged.drop(columns=[f"{c}_late"])
        return merged

    tm_today = pd.DataFrame()
    if odds_csv is not None:
        odds_rt = load_realtime_odds(odds_csv)
        se_today = _merge_by_horse_num(se_today, odds_rt, ["odds"])
        print(f"  リアルタイムオッズ反映: {len(odds_rt)}件")
    if wh_csv is not None:
        wh_rt = load_realtime_weights(wh_csv)
        se_today = _merge_by_horse_num(se_today, wh_rt, ["horse_weight", "horse_weight_diff"])
        print(f"  馬体重反映: {len(wh_rt)}件")
    if dm_csv is not None:
        dm_rt = load_realtime_dm(dm_csv)
        if len(dm_rt) > 0:
            dm_today = dm_rt  # 出馬表同梱のDMより速報を優先
        print(f"  DM反映: {len(dm_rt)}件")
    if tm_csv is not None:
        tm_today = load_realtime_tm(tm_csv)
        print(f"  TM反映: {len(tm_today)}件")
    # running_count 未確定(0)のレースのみSEの実エントリー数で補完
    n_per_race = se_today.groupby("race_id")["horse_id"].size()
    fallback = ra_today["race_id"].map(n_per_race).fillna(0).astype(int)
    ra_today["horse_count"] = ra_today["horse_count"].where(
        ra_today["horse_count"] > 0, fallback
    )
    today_race_ids = set(ra_today["race_id"])
    date_str = str(ra_today["race_date"].iloc[0]).replace("-", "")[:8]
    print(f"  対象: {len(ra_today)}レース / {len(se_today)}頭 / DM {len(dm_today)}件 ({date_str})")

    print("[2/5] 作業用DBコピー作成...")
    work_db = WORK_DIR / "JVData_today.db"
    shutil.copy2(DB_PATH, work_db)

    conn = sqlite3.connect(work_db)
    try:
        # 同一race_idの既存行（再実行時の残骸）を除去してから挿入
        ids = list(today_race_ids)
        ph = ",".join("?" * len(ids))
        for table in ["RA", "SE"]:
            conn.execute(f"DELETE FROM {table} WHERE race_id IN ({ph})", ids)
        # DM/TMは当日データがある場合のみ差し替え（既存のming由来データを温存）
        if len(dm_today) > 0:
            conn.execute(f"DELETE FROM DM WHERE race_id IN ({ph})", ids)
        if len(tm_today) > 0:
            conn.execute(f"DELETE FROM TM WHERE race_id IN ({ph})", ids)
        # 当日以外の未確定行（結果未取込のレース）は履歴集計を汚染するため作業DBから除去
        # （include_pending=True で混入し、career_win_rate 等が学習時と乖離するのを防ぐ）
        n_del = conn.execute("DELETE FROM SE WHERE finish_rank = 0").rowcount
        if n_del:
            print(f"  作業DBから未確定行 {n_del} 件を除去")
        ra_today.astype({"race_date": str}).to_sql("RA", conn, if_exists="append", index=False)
        se_today.to_sql("SE", conn, if_exists="append", index=False)
        if len(dm_today) > 0:
            dm_today.to_sql("DM", conn, if_exists="append", index=False)
        if len(tm_today) > 0:
            tm_today.to_sql("TM", conn, if_exists="append", index=False)
        conn.commit()

        print("[3/5] 学習時と同一のビルダー群を実行（履歴+当日）...")
        df = BasicFeatureBuilder(conn, include_pending=True).build()
        df = PastPerformanceBuilder(conn).enrich(df)
        df = RunningStyleBuilder(conn).enrich(df)
        df = PaceFeatureBuilder(conn).enrich(df)
        df = InteractionFeatureBuilder(conn).enrich(df)
        df = MiningFeatureBuilder(conn).enrich(df)
    finally:
        conn.close()

    print("[4/5] v3/v4 + champion (v6 going top3) 特徴量を適用...")
    df = add_jra_tm_orthogonalized(df)
    df = add_weight_relative_z(df)
    df = add_course_topology(df)
    # surface_cond_code（create_features_v4 と同一定義）
    df["surface_cond_code"] = (
        df["surface_code"].fillna(0).astype(int) * 10
        + df["track_condition_code"].fillna(0).astype(int)
    )
    df = apply_champion_feature_stack(df)

    print("[5/5] 当日行をスライスして保存...")
    today = df[df["race_id"].isin(today_race_ids)].copy()

    # UMテーブル未登録馬（新馬等）の年齢・性別を出馬表の値でフォールバック
    meta = se_today[["race_id", "horse_id", "age", "sex_code"]].rename(
        columns={"age": "_se_age", "sex_code": "_se_sex"}
    )
    # DB往復で型が変わりうるためマージキーを文字列に統一
    for frame in (today, meta):
        frame["race_id"] = frame["race_id"].astype(str)
        frame["horse_id"] = frame["horse_id"].astype(str)
    today = today.merge(meta, on=["race_id", "horse_id"], how="left")
    today["horse_age"] = today["horse_age"].fillna(today["_se_age"])
    today["horse_sex_code"] = today["horse_sex_code"].fillna(today["_se_sex"])
    today = today.drop(columns=["_se_age", "_se_sex"])
    out_path = WORK_DIR / f"today_features_{date_str}.parquet"
    today.to_parquet(out_path, index=False)
    print(f"  保存: {out_path} ({len(today)}頭 × {today.shape[1]}列)")
    return today


def merge_late_info(
    today_features: pd.DataFrame,
    odds_df: pd.DataFrame | None = None,
    weight_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """直前に確定するオッズ・馬体重を結合し、依存特徴量を更新する。

    odds_df  : race_id, horse_id, odds（market系はstrategy_engineが再計算するため結合のみ）
    weight_df: race_id, horse_id, horse_weight, horse_weight_diff
    """
    df = today_features.copy()
    if weight_df is not None and len(weight_df) > 0:
        df = df.drop(columns=["horse_weight", "horse_weight_diff"], errors="ignore").merge(
            weight_df[["race_id", "horse_id", "horse_weight", "horse_weight_diff"]],
            on=["race_id", "horse_id"], how="left",
        )
        # weight_diff は「前走比の増減」= JV発表の horse_weight_diff をそのまま使用
        df["weight_diff"] = df["horse_weight_diff"]
        # レース内相対Zスコアを再計算（v3と同一関数）
        df = add_weight_relative_z(df)
    if odds_df is not None and len(odds_df) > 0:
        df = df.drop(columns=["odds"], errors="ignore").merge(
            odds_df[["race_id", "horse_id", "odds"]],
            on=["race_id", "horse_id"], how="left",
        )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra", required=True, help="当日 race_ra.csv のパス")
    parser.add_argument("--se", required=True, help="当日 race_se.csv のパス")
    parser.add_argument("--odds", help="realtime_odds/o1_odds.csv（直前オッズ）")
    parser.add_argument("--wh", help="realtime_wh/wh.csv（馬体重）")
    parser.add_argument("--tm", help="realtime_tm/tm.csv（TM指数）")
    parser.add_argument("--dm", help="realtime_dm/dm.csv（DM予測タイム）")
    args = parser.parse_args()
    build_today_features(
        Path(args.ra), Path(args.se),
        odds_csv=Path(args.odds) if args.odds else None,
        wh_csv=Path(args.wh) if args.wh else None,
        tm_csv=Path(args.tm) if args.tm else None,
        dm_csv=Path(args.dm) if args.dm else None,
    )

"""
today_adapter.py — 当日データ変換アダプタ

main/data/race/race_ra.csv, race_se.csv（JV-Link 生データ形式。JVOpen "RACE" の
RA/SE レコードを common/data/src/jv_schemas.py の RA_SCHEMA / SE_SCHEMA に従って
固定長パースし CSV 化したもの）を、SE_preprocessed.parquet / RA_preprocessed.parquet
と同一スキーマの DataFrame に変換する。

仕様書: docs/specs/2026-07-04-today-prediction-design.md 4-C節。

重要な注意（実装時点の制約）:
    本モジュールは JV-Link 接続（実機 Windows・32bit Python・JRA-VAN 契約回線）
    なしに実装されている。RA_SCHEMA / SE_SCHEMA は本プロジェクト自身の
    common/data/src/jv_schemas.py に定義された固定長パーサーのフィールド名であり
    （var2.0.0 の別スキーマではなく、var1.0 自身が race_ra.csv/race_se.csv を
    生成する際に使うフィールド名そのもの）、preprocess.py の
    _SE_SOURCE_COLS / _RA_SOURCE_COLS_FROM_HD の列名とほぼ一致することを
    静的解析で確認済みだが、実際に run_today_se_ra_and_realtime() を実行して
    出力される CSV の実列名・実データでの動作確認はできていない。
    # TODO: 実際の race_ra.csv/race_se.csv 列名・値で要検証（仕様書 Step 3 参照）
    列名の不一致が実行時に起きても構造は再利用できるよう、列名マッピングを
    _TODAY_RA_COL_MAP / _TODAY_SE_COL_MAP の2箇所にまとめてある。

禁止事項:
- オッズ・人気 (odds, popularity) を出力 DataFrame に含めない（明示的に drop する）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# pure_rank/src 自身を sys.path に追加する（スクリプト直接実行・他モジュールからの
# import どちらでも "from common import ..." 形式の bare import が解決できるように）。
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import FORBIDDEN_MARKET_COLS  # noqa: E402
from create_features import _check_no_market_features  # noqa: E402  (既存ロジック再利用)
from preprocess import _make_race_date, _make_race_id  # noqa: E402  (既存ロジック再利用)

# ═══════════════════════════════════════════════════════════════════════════════
# 列名マッピング（実列名確定後は本辞書のみ更新すればよい設計）
# ═══════════════════════════════════════════════════════════════════════════════

# race_ra.csv の生列名 → RA_preprocessed.parquet と同一列名。
# common/data/src/jv_schemas.py の RA_SCHEMA とほぼ同名のため変換は最小限。
# TODO: 実際の race_ra.csv 列名で要検証（仕様書Step3参照）
_TODAY_RA_COL_MAP: dict[str, str] = {
    "running_count": "horse_count",  # preprocess.preprocess_ra() と同じリネーム
}

# race_se.csv の生列名 → SE_preprocessed.parquet と同一列名。
# common/data/src/jv_schemas.py の SE_SCHEMA では走破タイムのキー名が "time" であり、
# preprocess.py 側の期待列名 "racetime" と異なる（唯一の実質的なリネーム）。
# TODO: 実際の race_se.csv 列名で要検証（仕様書Step3参照）
_TODAY_SE_COL_MAP: dict[str, str] = {
    "time": "racetime",
}

# RA_preprocessed.parquet に存在するが JV-Link RA レコードに直接存在しない
# 派生・メタ列（var2.0.0 の horse_data.parquet 生成時に付与されたもの）。
# create_features.py のどの _build_* 関数からも参照されない
# （pure_rank/src/common.py の FORBIDDEN_COLS に列挙されメタ列として特徴量から
# 除外されるため、当日行では 0 埋めのプレースホルダで構わない）。
_RA_UNUSED_META_COLS: dict[str, int] = {
    "race_condition_code": 0,
    "race_level": 0,
    "race_age_type": 0,
}

# 当日未実施のため NaN 埋めする SE 列（仕様書4-C節 手順6）。
# is_win / is_place は日次集計 groupby の型安定性のため 0 で埋める（NaNではなく）。
_SE_NAN_COLS: list[str] = [
    "finish_rank", "racetime", "time_3f_after",
    "corner_1", "corner_2", "corner_3", "corner_4",
    "abnormal_code", "hon_shokin", "fuka_shokin", "running_style_code",
]
_SE_ZERO_COLS: list[str] = ["is_win", "is_place"]

# preprocess.py の _SE_SOURCE_COLS / _RA_SOURCE_COLS_FROM_HD と同じ列順序
_SE_OUTPUT_COLS: list[str] = [
    "race_id", "year", "month_day", "course_code", "kai", "nichi", "race_num",
    "wakuban", "horse_num", "ketto_num",
    "sex_code", "age",
    "trainer_code", "jockey_code",
    "burden_weight", "blinker_code",
    "horse_weight", "horse_weight_change", "abnormal_code",
    "finish_rank",
    "racetime", "time_3f_after",
    "corner_1", "corner_2", "corner_3", "corner_4",
    "hon_shokin", "fuka_shokin",
    "running_style_code",
    "mining_predicted_rank",
    "race_date", "is_win", "is_place",
]

_RA_OUTPUT_COLS: list[str] = [
    "race_id", "year", "month_day", "course_code", "kai", "nichi", "race_num",
    "grade_code", "race_type_code", "weight_type",
    "race_condition_code", "race_level", "race_age_type",
    "distance", "track_code", "course_kubun",
    "registered_count", "horse_count", "finish_count",
    "weather_code", "turf_condition", "dirt_condition",
    "race_date", "surface_code", "track_condition_code",
    "surface_condition", "distance_category",
]


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """指定列を pd.to_numeric で数値化する（in-place）。存在しない列はスキップ。"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _drop_forbidden_market_cols(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """市場情報列（odds, popularity 等）が存在すれば明示的に drop する。"""
    found = FORBIDDEN_MARKET_COLS & set(df.columns)
    if found:
        print(f"  [today_adapter:{label}] 市場情報列を drop: {sorted(found)}")
        df = df.drop(columns=list(found))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CSV 読み込み
# ═══════════════════════════════════════════════════════════════════════════════

def load_today_csv(race_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """race_ra.csv / race_se.csv を dtype=str で読み込む。

    Parameters
    ----------
    race_dir : main/data/race ディレクトリ（run_today_se_ra_and_realtime() の出力先）
    """
    ra_path = race_dir / "race_ra.csv"
    se_path = race_dir / "race_se.csv"
    if not ra_path.exists() or not se_path.exists():
        raise FileNotFoundError(
            f"当日データが見つかりません: {ra_path}, {se_path}\n"
            f"先に run_today_se_ra_and_realtime() を実行してください "
            f"(main/notebook_bootstrap.py)。"
        )
    ra_raw = pd.read_csv(ra_path, dtype=str, encoding="utf-8-sig")
    se_raw = pd.read_csv(se_path, dtype=str, encoding="utf-8-sig")
    print(f"  [today_adapter] loaded race_ra.csv: {len(ra_raw):,} rows, {len(ra_raw.columns)} cols")
    print(f"  [today_adapter] loaded race_se.csv: {len(se_raw):,} rows, {len(se_raw.columns)} cols")
    return ra_raw, se_raw


# ═══════════════════════════════════════════════════════════════════════════════
# RA 変換
# ═══════════════════════════════════════════════════════════════════════════════

def convert_today_ra(ra_raw: pd.DataFrame) -> pd.DataFrame:
    """race_ra.csv（生データ）を RA_preprocessed.parquet 相当の DataFrame に変換する。"""
    df = ra_raw.rename(columns=_TODAY_RA_COL_MAP).copy()
    df = _drop_forbidden_market_cols(df, label="RA")

    _to_numeric(df, [
        "year", "month_day", "course_code", "kai", "nichi", "race_num",
        "grade_code", "race_type_code", "weight_type", "distance", "track_code",
        "course_kubun", "registered_count", "horse_count", "finish_count",
        "weather_code", "turf_condition", "dirt_condition",
    ])

    # var2.0.0 由来のメタ列（今回の132列特徴量では未使用。プレースホルダで埋める）
    for col, default in _RA_UNUSED_META_COLS.items():
        if col not in df.columns:
            df[col] = default

    df["race_id"] = df.apply(_make_race_id, axis=1)
    df["race_date"] = _make_race_date(df)

    # surface_code / track_condition_code / surface_condition / distance_category:
    # preprocess.preprocess_ra() と全く同じ計算式（4-C節 手順4）。
    # preprocess.py 自体は変更禁止のため、ここでは同じ式を独立に再実装している
    # （preprocess_ra はファイルパスからの読み込み前提で直接呼び出せないため）。
    df["surface_code"] = (df["track_code"] // 10).astype("int8")
    df["track_condition_code"] = np.where(
        df["surface_code"] == 1,
        df["turf_condition"],
        df["dirt_condition"],
    ).astype("int8")
    df["surface_condition"] = (
        df["surface_code"] * 10 + df["track_condition_code"]
    ).astype("int8")
    df["distance_category"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 99999],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype("int8")

    n_zero_cond = int((df["track_condition_code"] == 0).sum())
    if n_zero_cond > 0:
        bad_races = df.loc[df["track_condition_code"] == 0, "race_id"].tolist()
        print(
            f"  [today_adapter:RA][WARN] track_condition_code=0 の行が {n_zero_cond} 件 "
            f"（障害 or 馬場状態未確定の可能性）。該当レースはシナリオ生成をスキップすること: "
            f"{bad_races}"
        )

    keep_cols = [c for c in _RA_OUTPUT_COLS if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["race_id"]).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SE 変換
# ═══════════════════════════════════════════════════════════════════════════════

def convert_today_se(se_raw: pd.DataFrame) -> pd.DataFrame:
    """race_se.csv（生データ）を SE_preprocessed.parquet 相当の DataFrame に変換する。"""
    df = se_raw.rename(columns=_TODAY_SE_COL_MAP).copy()
    # race_se.csv には JV SE_SCHEMA 由来の odds / popularity 列が実在する
    # （0B31 速報単勝オッズ取得後は race_se.csv 自体に反映される。仕様書1章参照）。
    # 特徴量化の最初のステップで必ず drop する。
    df = _drop_forbidden_market_cols(df, label="SE")

    _to_numeric(df, [
        "year", "month_day", "course_code", "kai", "nichi", "race_num",
        "wakuban", "horse_num", "ketto_num",
        "sex_code", "age", "trainer_code", "jockey_code",
        "burden_weight", "blinker_code", "horse_weight",
        "weight_change_sign", "weight_change",
        "abnormal_code", "finish_rank", "racetime", "time_3f_after",
        "corner_1", "corner_2", "corner_3", "corner_4",
        "hon_shokin", "fuka_shokin", "running_style_code",
        "mining_predicted_rank",
    ])

    df["race_id"] = df.apply(_make_race_id, axis=1)
    df["race_date"] = _make_race_date(df)

    # burden_weight: JV 生値は kg*10 の整数と推定（例: 550 = 55.0kg）。
    # TODO: 実データで要検証（仕様書Step3参照。現行132列特徴量では未使用のため
    # 影響はないが、将来 burden_weight を特徴量化する場合は要確認）。
    if "burden_weight" in df.columns:
        df["burden_weight"] = df["burden_weight"].astype(float) / 10.0

    # horse_weight_change: preprocess.py のコメントによれば var2.0.0 側で
    # weight_change_sign × weight_change から符号付き float に変換済み。
    # JV 生データでは符号(weight_change_sign)と差分(weight_change)が別フィールド。
    # TODO: weight_change_sign の実際のコード値（'+'/'-'/'0' か '1'/'2'/'0' か）は
    # 仕様書Step3で要検証。ここでは JRA-VAN 一般的な増減符号コード
    # (0=変化なし, 1=増, 2=減) を仮定するが、'+'/'-' 文字の可能性もあるため両対応にする。
    if "weight_change_sign" in df.columns and "weight_change" in df.columns:
        sign_raw = se_raw.get(
            "weight_change_sign",
            pd.Series(index=df.index, dtype="object"),
        ).astype(str).str.strip()
        sign_map = {"1": 1, "+": 1, "2": -1, "-": -1, "0": 0, "": 0, "nan": 0}
        sign = sign_raw.map(sign_map).fillna(0).astype(float)
        df["horse_weight_change"] = sign * df["weight_change"].fillna(0)
    else:
        df["horse_weight_change"] = np.nan

    # 当日未実施のため NaN / 0 埋め（仕様書4-C節 手順6）
    for col in _SE_NAN_COLS:
        df[col] = np.nan
    for col in _SE_ZERO_COLS:
        df[col] = 0

    if "mining_predicted_rank" not in df.columns:
        df["mining_predicted_rank"] = np.nan
    else:
        # preprocess_se() と同じ 0→NaN 変換（0=マイニング未実施/対象外）
        df.loc[df["mining_predicted_rank"] <= 0, "mining_predicted_rank"] = np.nan

    keep_cols = [c for c in _SE_OUTPUT_COLS if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["race_id", "ketto_num"]).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# スキーマ整合性検証
# ═══════════════════════════════════════════════════════════════════════════════

def validate_schema_parity(
    se_today: pd.DataFrame,
    ra_today: pd.DataFrame,
    preprocessed_dir: Path,
) -> None:
    """変換後 DataFrame の列集合が SE_preprocessed/RA_preprocessed と一致するか検証する。

    仕様書4-C節 手順8: 「出力 DataFrame の列集合が SE_preprocessed.parquet ∪
    RA_preprocessed.parquet と完全一致することをアサートする」。
    列が不足していれば ValueError、余剰があれば警告のみ（is_win/is_place のように
    今回のアダプタが明示的に追加する列は正当な一致のため問題ない）。
    """
    import pyarrow.parquet as pq

    se_path = preprocessed_dir / "SE_preprocessed.parquet"
    ra_path = preprocessed_dir / "RA_preprocessed.parquet"
    se_cols = set(pq.ParquetFile(se_path).schema.names)
    ra_cols = set(pq.ParquetFile(ra_path).schema.names)

    missing_se = se_cols - set(se_today.columns)
    missing_ra = ra_cols - set(ra_today.columns)
    if missing_se or missing_ra:
        raise ValueError(
            f"[today_adapter] スキーマ不一致（不足列）: "
            f"missing_se={sorted(missing_se)}, missing_ra={sorted(missing_ra)}\n"
            f"today_adapter.py の _TODAY_SE_COL_MAP / _TODAY_RA_COL_MAP または "
            f"変換ロジックを確認してください。"
        )

    extra_se = set(se_today.columns) - se_cols
    extra_ra = set(ra_today.columns) - ra_cols
    if extra_se or extra_ra:
        print(
            f"  [today_adapter][INFO] 想定外の余剰列（無害だが記録）: "
            f"extra_se={sorted(extra_se)}, extra_ra={sorted(extra_ra)}"
        )
    print("  [today_adapter] スキーマ整合性チェック PASS "
          f"(SE {len(se_cols)}列, RA {len(ra_cols)}列)")


# ═══════════════════════════════════════════════════════════════════════════════
# SE + RA + SK 結合（create_features._load_data() のマージ形状を再現）
# ═══════════════════════════════════════════════════════════════════════════════

def build_today_merged(race_dir: Path, preprocessed_dir: Path) -> pd.DataFrame:
    """当日データを読み込み・変換し、create_features._load_data() と同じ形状の
    単一 DataFrame（SE+RA+SK 結合済み、フィルタ適用前）を返す。

    _load_data() 自体はファイルパスから読み込む前提で当日データを注入できないため、
    同じマージロジックをここで独立に再現している（preprocess.py / create_features.py
    は変更しない方針のため）。
    """
    ra_raw, se_raw = load_today_csv(race_dir)
    ra_today = convert_today_ra(ra_raw)
    se_today = convert_today_se(se_raw)
    validate_schema_parity(se_today, ra_today, preprocessed_dir)

    sk = pd.read_parquet(preprocessed_dir / "SK_preprocessed.parquet")

    # _load_data() と同じ結合パターン: SE の race_date は drop し RA 側を使う
    ra_merge_cols = [
        "race_id", "grade_code", "distance", "track_code", "horse_count",
        "weather_code", "surface_code", "track_condition_code",
        "surface_condition", "distance_category", "race_date",
    ]
    se_today = se_today.drop(columns=["race_date"], errors="ignore")
    ra_subset = ra_today[[c for c in ra_merge_cols if c in ra_today.columns]].copy()

    df = se_today.merge(ra_subset, on="race_id", how="inner")

    sk_cols = ["ketto_num", "sire_id", "bms_id"]
    sk_subset = sk[[c for c in sk_cols if c in sk.columns]].copy()
    df["ketto_num"] = pd.to_numeric(df["ketto_num"], errors="coerce").astype(np.int64)
    sk_subset["ketto_num"] = pd.to_numeric(sk_subset["ketto_num"], errors="coerce").astype(np.int64)
    df = df.merge(sk_subset, on="ketto_num", how="left")

    _check_no_market_features(df)
    print(f"  [today_adapter] Merged today data: {len(df):,} rows, {len(df.columns)} cols, "
          f"{df['race_id'].nunique()} races")
    return df


if __name__ == "__main__":
    # 単体動作確認用（JV-Link接続なしでは main/data/race/race_ra.csv が無く FileNotFoundError になる）
    import json as _json

    PROJECT_ROOT = _THIS_DIR.parent.parent
    cfg_path = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = _json.load(f)
    preprocessed_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]
    race_dir = PROJECT_ROOT / "main" / "data" / "race"
    merged = build_today_merged(race_dir, preprocessed_dir)
    print(merged.head())

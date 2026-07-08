"""features_v3 に診断改善特徴量を追加して features_v4.parquet を生成する。

追加特徴量:
  - course_last_straight_m : 競馬場別最終直線距離（コース形態）
  - course_n_corners       : 距離帯別コーナー数（コース形態）
  - surface_cond_code      : surface_code * 10 + track_condition_code（交互作用）
  - base_time_cond_zscore  : 基準タイムの条件内 z-score（ダート荒れ馬場の物理特性）
  - lap_time_std           : ラップタイム標準偏差（ペース変動）
  - early_pace_ratio       : 前半3F比率（展開指標）

実行: python model_training/src/create_features_v4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from pipeline_common import FEATURES_DIR, get_db_connection, save_features

# ---------------------------------------------------------------------------
# コース形態ルックアップ（JRA公式情報）
# ---------------------------------------------------------------------------

# 最終直線距離 (m) by course_code
# 新潟は特殊: distance <= 1000 の場合は直線競走 → last_straight = distance
LAST_STRAIGHT_M: dict[int, int] = {
    1: 264,   # 札幌
    2: 262,   # 函館
    3: 292,   # 福島
    4: 359,   # 新潟（内回り/外回り外 代表値; 直線は別処理）
    5: 525,   # 東京
    6: 310,   # 中山
    7: 412,   # 中京
    8: 404,   # 京都（外回り代表値）
    9: 473,   # 阪神（外回り代表値）
    10: 293,  # 小倉
}

# ---------------------------------------------------------------------------
# 特徴量生成関数
# ---------------------------------------------------------------------------

def add_course_topology(df: pd.DataFrame) -> pd.DataFrame:
    """コース形態特徴量を追加する。"""
    df["course_last_straight_m"] = df["course_code"].map(LAST_STRAIGHT_M).fillna(350).astype(float)
    # 新潟直線競走の特殊処理
    niigata_straight = (df["course_code"] == 4) & (df["distance"] <= 1000)
    df.loc[niigata_straight, "course_last_straight_m"] = df.loc[niigata_straight, "distance"].astype(float)

    # コーナー数: 新潟直線競走=0、1400m以下=2、それ以外=4
    df["course_n_corners"] = np.select(
        [niigata_straight, df["distance"] <= 1400],
        [0.0, 2.0],
        default=4.0,
    )
    return df


def add_surface_cond_code(df: pd.DataFrame) -> pd.DataFrame:
    """芝×良=11, 芝×稍重=12, ダート×重=23 などの交互作用コードを追加。"""
    df["surface_cond_code"] = (
        df["surface_code"].fillna(0).astype(int) * 10
        + df["track_condition_code"].fillna(0).astype(int)
    ).astype(float)
    return df


def add_base_time_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """base_time の条件内 z-score を追加する。

    ダートが水分を含んで高速化する非線形特性を捉える。
    リーク防止: shift(1).expanding() で当該レースを除外。

    WARNING（時系列リーク・要対応）: z-scoreの分子 x は「当該レースのbase_time
    （=当該レース勝者の走破タイム）」でありレース確定後にしか得られない。
    過去統計（mean/std）はshift済みでも、分子が当日結果そのもののため
    学習時リーク + 推論時には計算不能（train/serve skew）。
    修正は特徴量値が変わるためバックテスト影響評価とセットで行うこと。
    """
    df = df.sort_values("race_date").copy()
    df["dist_bucket"] = (df["distance"] // 200 * 200).astype(int)

    group_cols = ["course_code", "dist_bucket", "surface_code", "track_condition_code"]

    def _zscore_transform(x: pd.Series) -> pd.Series:
        hist = x.shift(1).expanding()
        mean = hist.mean()
        std = hist.std().fillna(1.0).clip(lower=0.1)
        return (x - mean) / std

    df["base_time_cond_zscore"] = (
        df.groupby(group_cols)["base_time"]
        .transform(_zscore_transform)
    )
    df = df.drop(columns=["dist_bucket"])
    return df


def add_lap_features(df: pd.DataFrame, ra_df: pd.DataFrame) -> pd.DataFrame:
    """ラップタイム統計特徴量を追加する。

    lap_time_std    : ラップタイムの標準偏差（ペース変動の大きさ）
    early_pace_ratio: 前半3Fの合計 / 全体タイム（展開指標）

    WARNING（時系列リーク・要対応）: ここで使うlap_timesは「当該レースの
    実測ラップ」でありレース確定後にしか得られない（shiftなしで当該レースに直結合）。
    学習時リーク + 推論時には計算不能（train/serve skew）。
    修正は特徴量値が変わるためバックテスト影響評価とセットで行うこと。
    """
    def parse_laps(lap_str: str) -> list[float] | None:
        if not isinstance(lap_str, str) or not lap_str.strip():
            return None
        try:
            return [float(v) for v in lap_str.strip().split()]
        except ValueError:
            return None

    ra_work = ra_df[["race_id", "lap_times"]].copy()
    ra_work["laps"] = ra_work["lap_times"].apply(parse_laps)

    def lap_std(laps: list[float] | None) -> float:
        if laps is None or len(laps) < 2:
            return np.nan
        return float(np.std(laps))

    def early_pace(laps: list[float] | None) -> float:
        if laps is None or len(laps) < 3:
            return np.nan
        total = sum(laps)
        if total <= 0:
            return np.nan
        return float(sum(laps[:3]) / total)

    ra_work["lap_time_std"] = ra_work["laps"].apply(lap_std)
    ra_work["early_pace_ratio"] = ra_work["laps"].apply(early_pace)

    df = df.merge(
        ra_work[["race_id", "lap_time_std", "early_pace_ratio"]],
        on="race_id",
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def create_features_v4() -> None:
    v3_path = FEATURES_DIR / "features_v3.parquet"

    if not v3_path.exists():
        raise FileNotFoundError(f"features_v3.parquet が見つかりません: {v3_path}")

    print("features_v3.parquet 読み込み中...")
    df = pd.read_parquet(v3_path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    print(f"  {len(df):,} 行 × {df.shape[1]} 列")

    conn = get_db_connection()
    ra_df = pd.read_sql_query(
        "SELECT race_id, course_code, distance, surface_code, track_condition_code, base_time, lap_times FROM RA",
        conn,
    )
    conn.close()

    # course_code, distance 等を RA から補完（features_v3 に欠けている場合に備える）
    for col in ["course_code", "distance", "surface_code", "track_condition_code", "base_time"]:
        if col not in df.columns:
            df = df.merge(ra_df[["race_id", col]].drop_duplicates("race_id"), on="race_id", how="left")
        else:
            # RAの値で欠損を補完
            ra_col = ra_df[["race_id", col]].drop_duplicates("race_id").rename(columns={col: f"_{col}_ra"})
            df = df.merge(ra_col, on="race_id", how="left")
            df[col] = df[col].combine_first(df[f"_{col}_ra"])
            df = df.drop(columns=[f"_{col}_ra"])

    print("コース形態特徴量を追加中...")
    df = add_course_topology(df)

    print("surface_cond_code を追加中...")
    df = add_surface_cond_code(df)

    print("base_time_cond_zscore を追加中...")
    df = add_base_time_zscore(df)

    print("ラップ特徴量を追加中...")
    df = add_lap_features(df, ra_df)

    # 追加カラムの確認
    new_cols = [
        "course_last_straight_m", "course_n_corners",
        "surface_cond_code", "base_time_cond_zscore",
        "lap_time_std", "early_pace_ratio",
    ]
    print("\n--- 追加特徴量サマリ ---")
    for col in new_cols:
        if col in df.columns:
            nan_rate = df[col].isna().mean()
            print(f"  {col:<30}: nan={nan_rate:.1%}  mean={df[col].mean():.3f}  std={df[col].std():.3f}")
        else:
            print(f"  {col}: 追加失敗")

    # save_features経由でparquet + manifest（+ CSV）を更新する（CLAUDE.md規約）
    print("\nfeatures_v4 保存中...")
    save_features(df, "features_v4")
    print(f"  完了: {len(df):,} 行 × {df.shape[1]} 列")


if __name__ == "__main__":
    create_features_v4()

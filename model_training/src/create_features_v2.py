"""FeatureCreator: 全ビルダーを統合してfeatures_v2.parquetを生成する。

Composer パターン:
  BasicFeatureBuilder  → ベーステーブル + 基本特徴量
  PastPerformanceBuilder → 過去成績 + EMA + RPR
  RunningStyleBuilder  → 脚質 + コーナー + 展開
  PaceFeatureBuilder   → PCI + 真Zスコア
  InteractionFeatureBuilder → 交互作用項
  MiningFeatureBuilder → JRA公式DM予測タイム・TM指数（base_margin用）

出力: model_training/data/02_features/features_v2.parquet

使用方法:
  python model_training/src/create_features_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

import pandas as pd

from builders import (
    BasicFeatureBuilder,
    InteractionFeatureBuilder,
    MiningFeatureBuilder,
    PaceFeatureBuilder,
    PastPerformanceBuilder,
    RunningStyleBuilder,
)
from pipeline_common import get_db_connection, save_features, validate_no_leakage


class FeatureCreator:
    """全ビルダーを順番に適用し、統合特徴量を生成するオーケストレータ。"""

    V2_FEATURE_COLS = [
        # -- Basic --
        "horse_age", "carry_weight", "gate_number", "horse_count",
        "days_since_last_race", "weight_diff", "weight_diff_trend", "class_change",
        "draw_bias_score",
        "surface_code", "distance", "track_condition_code", "horse_sex_code",
        "market_prob",
        # -- PastPerformance --
        "last3_agari3f_mean", "last5_rank_mean", "last5_time_diff_mean",
        "last5_rank_std", "top3_rate_career", "top3_rate_class",
        "career_win_rate", "distance_win_rate", "course_win_rate",
        "surface_win_rate", "condition_win_rate",
        "jockey_30d_win_rate", "trainer_30d_win_rate", "jockey_course_win_rate",
        "sire_surface_win_rate", "bms_win_rate",
        "ema_rank", "ema_time_diff", "rpr_score",
        # -- RunningStyle --
        "corner_pos_mean", "corner_pos_last",
        "running_style_mode", "front_rate",
        "race_front_count", "race_front_ratio", "race_style_entropy",
        # -- Pace --
        "pci_past_mean", "agari_z_score", "race_pci",
        # -- Interaction --
        "gate_surface_cross",
        "summer_mare_sin", "summer_mare_cos",
        # -- v2 extended --
        "past_speed_index_mean",
        # -- Mining (JRA公式) --
        "jra_tm_score", "jra_tm_rank",
        "jra_dm_rank", "jra_dm_gap_to_best", "jra_dm_uncertainty",
    ]

    def __init__(self) -> None:
        self.conn = get_db_connection()

    def run(self) -> pd.DataFrame:
        print("=== FeatureCreator v2 開始 ===")

        # 1. ベーステーブル生成（BasicFeatureBuilder は build() を呼ぶ）
        print("[1/6] BasicFeatureBuilder: ベーステーブル生成...")
        basic = BasicFeatureBuilder(self.conn)
        df = basic.build()
        print(f"  ベーステーブル: {len(df)} rows, {len(df.columns)} cols")

        # 2. 過去成績 + EMA + RPR
        print("[2/6] PastPerformanceBuilder: 過去成績・EMA・RPR...")
        df = PastPerformanceBuilder(self.conn).enrich(df)

        # 3. 脚質・コーナー・レース展開
        print("[3/6] RunningStyleBuilder: 脚質・コーナー・展開...")
        df = RunningStyleBuilder(self.conn).enrich(df)

        # 4. PCI・真Zスコア
        print("[4/6] PaceFeatureBuilder: PCI・真Zスコア...")
        df = PaceFeatureBuilder(self.conn).enrich(df)

        # 5. 交互作用項
        print("[5/6] InteractionFeatureBuilder: 交互作用項...")
        df = InteractionFeatureBuilder(self.conn).enrich(df)

        # 6. JRA公式マイニングデータ（DM/TM）
        print("[6/6] MiningFeatureBuilder: JRA DM予測タイム・TM指数...")
        df = MiningFeatureBuilder(self.conn).enrich(df)

        # NaN チェック
        existing_feat_cols = [c for c in self.V2_FEATURE_COLS if c in df.columns]
        validate_no_leakage(df, existing_feat_cols)

        # 欠損している特徴量の報告
        missing = set(self.V2_FEATURE_COLS) - set(df.columns)
        if missing:
            print(f"[WARN] 以下の特徴量が生成されませんでした（DBデータ不足の可能性）: {sorted(missing)}")

        print(f"=== 完了: {len(df)} rows, {len(df.columns)} cols ===")
        return df

    def save(self, df: pd.DataFrame) -> None:
        save_features(df, "features_v2")

    def close(self) -> None:
        self.conn.close()


def main() -> None:
    creator = FeatureCreator()
    try:
        df = creator.run()
        creator.save(df)
    finally:
        creator.close()


if __name__ == "__main__":
    main()

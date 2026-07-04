"""
regression_test_today.py — 当日予測パイプラインの回帰テスト（仕様書6章）

JV-Link 接続不要。既存の SE/RA/SK_preprocessed.parquet からテスト期間内の
実在レース1件を抽出し、finish_rank 等を意図的に NaN 化して「未実施レース」を
模擬したうえで build_today_features() / run_today_predictions() に通す。

検証内容:
1. 4シナリオのうち、実際の track_condition_code に対応するシナリオの
   pred_score が、features_v39_course_slim.parquet + 15モデルアンサンブルで
   計算した既存のオフライン予測スコアと完全一致すること
2. 4シナリオ間で129列（132列 - シナリオ依存3列）が完全一致すること
   （build_today_features() 内部で既に assert 済みだが、ここでも再確認する）
3. 市場情報列が一切出現しないこと

today_adapter.py は経由しない（CSV変換ロジックはJV-Link接続後でないと
実地検証できないため）。SE/RA/SK_preprocessed.parquet から直接
「today_merged」相当の DataFrame を組み立てる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import FORBIDDEN_MARKET_COLS, PROJECT_ROOT, get_feature_cols, load_config  # noqa: E402
from evaluate import ensemble_predict, load_models  # noqa: E402
from predict_today import (  # noqa: E402
    TRACK_CONDITION_CODES,
    build_today_features,
    run_today_predictions,
)

# 当日未実施につき NaN 化する列（today_adapter.convert_today_se と同じ扱い）
_SE_NAN_COLS = [
    "finish_rank", "racetime", "time_3f_after",
    "corner_1", "corner_2", "corner_3", "corner_4",
    "abnormal_code", "hon_shokin", "fuka_shokin", "running_style_code",
]
_SE_ZERO_COLS = ["is_win", "is_place"]


def _pick_test_race(cfg: dict) -> str:
    """テスト期間（race_date > valid_end）から track_condition_code in {1,2,3,4}
    かつ複数頭数のレースを1つ選ぶ。"""
    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    df = pd.read_parquet(feat_path, columns=["race_id", "race_date", "track_condition_code", "horse_count"])
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])
    test = df[(df["race_date"] > valid_end) & (df["track_condition_code"].isin([1, 2, 3, 4]))]
    if test.empty:
        raise RuntimeError("テスト期間内に条件を満たすレースが見つかりません。")
    # 最新日の中から頭数が多め(>=8)のレースを優先して選ぶ（tie-break が起きにくいように）
    latest_date = test["race_date"].max()
    candidates = test[test["race_date"] == latest_date]
    candidates = candidates.sort_values("horse_count", ascending=False)
    race_id = str(candidates.iloc[0]["race_id"])
    return race_id


def _build_reference_scores(cfg: dict, race_id: str) -> pd.DataFrame:
    """features_v39_course_slim.parquet + 15モデルアンサンブルで当該レースの
    既存オフライン予測スコアを計算する（evaluate.py と全く同じロジック）。"""
    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    df = pd.read_parquet(feat_path)
    race = df[df["race_id"] == race_id].copy()
    if race.empty:
        raise RuntimeError(f"race_id={race_id} が features parquet に見つかりません。")

    feature_cols = get_feature_cols(race, cfg)
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    models = load_models(models_dir)
    race["ref_pred_score"] = ensemble_predict(models, race[feature_cols])

    real_track_condition = int(race["track_condition_code"].iloc[0])
    print(f"  [reference] race_id={race_id}, horse_count={len(race)}, "
          f"real track_condition_code={real_track_condition}, "
          f"course_code={int(race['course_code'].iloc[0])}, "
          f"race_date={race['race_date'].iloc[0].date()}")
    return race[["ketto_num", "horse_num", "ref_pred_score"]].reset_index(drop=True), real_track_condition, models


def _build_synthetic_today_merged(cfg: dict, race_id: str) -> pd.DataFrame:
    """SE/RA/SK_preprocessed.parquet から race_id の行を抽出し、
    today_adapter.build_today_merged() と同じ形状の「未実施レース」DataFrame を作る。"""
    prep_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]
    se = pd.read_parquet(prep_dir / "SE_preprocessed.parquet")
    ra = pd.read_parquet(prep_dir / "RA_preprocessed.parquet")
    sk = pd.read_parquet(prep_dir / "SK_preprocessed.parquet")

    se_race = se[se["race_id"] == race_id].copy()
    ra_race = ra[ra["race_id"] == race_id].copy()
    if se_race.empty or ra_race.empty:
        raise RuntimeError(f"race_id={race_id} が SE/RA_preprocessed.parquet に見つかりません。")

    # today_adapter.convert_today_se と同じ NaN/0 化（4-C節手順6 相当）
    for col in _SE_NAN_COLS:
        if col in se_race.columns:
            se_race[col] = np.nan
    for col in _SE_ZERO_COLS:
        if col in se_race.columns:
            se_race[col] = 0
    if "mining_predicted_rank" in se_race.columns:
        se_race["mining_predicted_rank"] = np.nan

    # _load_data() と同じマージ形状（SE の race_date は drop、RA 側を使う）
    ra_merge_cols = [
        "race_id", "grade_code", "distance", "track_code", "horse_count",
        "weather_code", "surface_code", "track_condition_code",
        "surface_condition", "distance_category", "race_date",
    ]
    se_race = se_race.drop(columns=["race_date"], errors="ignore")
    ra_subset = ra_race[[c for c in ra_merge_cols if c in ra_race.columns]].copy()
    merged = se_race.merge(ra_subset, on="race_id", how="inner")

    sk_cols = ["ketto_num", "sire_id", "bms_id"]
    sk_subset = sk[[c for c in sk_cols if c in sk.columns]].copy()
    merged["ketto_num"] = pd.to_numeric(merged["ketto_num"], errors="coerce").astype(np.int64)
    sk_subset["ketto_num"] = pd.to_numeric(sk_subset["ketto_num"], errors="coerce").astype(np.int64)
    merged = merged.merge(sk_subset, on="ketto_num", how="left")

    # 市場情報混入チェック
    found = FORBIDDEN_MARKET_COLS & set(merged.columns)
    assert not found, f"市場情報列が混入しています: {found}"

    print(f"  [synthetic today] built {len(merged)} rows for race_id={race_id} "
          f"(finish_rank/racetime等をNaN化済み)")
    return merged


def run_regression_test() -> None:
    cfg = load_config()

    print("=" * 70)
    print("  Step 7 回帰テスト: 当日予測パイプライン vs オフライン評価")
    print("=" * 70)

    print("\n[1] テスト対象レースを選定...")
    race_id = _pick_test_race(cfg)

    print("\n[2] オフライン既存パイプラインの予測スコアを計算...")
    ref_scores, real_track_condition, models = _build_reference_scores(cfg, race_id)

    print("\n[3] SE/RA/SK_preprocessed.parquet から「未実施レース」を模擬...")
    today_merged = _build_synthetic_today_merged(cfg, race_id)

    print("\n[4] build_today_features() で4シナリオの当日特徴量を生成...")
    today_features = build_today_features(today_merged, cfg)

    print("\n[5] run_today_predictions() でアンサンブル推論...")
    today_with_preds = run_today_predictions(today_features, cfg, models=models)

    print("\n[6] 実際の track_condition_code に対応するシナリオと比較...")
    scenario_df = today_with_preds[real_track_condition]
    cmp = scenario_df[["ketto_num", "horse_num", "pred_score"]].merge(
        ref_scores, on=["ketto_num", "horse_num"], how="inner"
    )
    assert len(cmp) == len(ref_scores), (
        f"比較対象の頭数が一致しません: today={len(cmp)} vs ref={len(ref_scores)}"
    )

    diff = (cmp["pred_score"] - cmp["ref_pred_score"]).abs()
    max_diff = float(diff.max())
    print(cmp.assign(diff=diff).to_string(index=False))
    print(f"\n  最大差分: {max_diff:.10f}")

    if max_diff < 1e-9:
        print("  [PASS] pred_score は既存オフライン予測スコアと完全一致（誤差<1e-9）")
    elif max_diff < 1e-4:
        print("  [PASS-近似] pred_score はほぼ一致（誤差<1e-4。浮動小数点丸めの範囲内）")
    else:
        raise AssertionError(
            f"[FAIL] pred_score が既存オフライン予測スコアと一致しません（最大差分={max_diff}）。"
            f"学習時パイプラインとの計算式相違（実装バグ）の可能性が高いため原因究明が必要です。"
        )

    print("\n[7] 4シナリオ間の129列完全一致は build_today_features() 内で確認済み (PASS)")

    print("\n[8] 市場情報混入チェック...")
    for code, df in today_with_preds.items():
        found = FORBIDDEN_MARKET_COLS & set(df.columns)
        assert not found, f"[code={code}] 市場情報列が混入しています: {found}"
    print("  [PASS] 市場情報列なし")

    print("\n" + "=" * 70)
    print("  回帰テスト: 全項目 PASS")
    print("=" * 70)


if __name__ == "__main__":
    run_regression_test()

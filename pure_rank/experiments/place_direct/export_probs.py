"""place_direct: export_probs.py

TEST 期間(2025-01-01〜)で 4 系列の複勝(top3)確率を算出し、1 parquet にまとめる。

  (a) Stern 逆算: 既存 fold2 OOS L1 スコア → fusion_probs (alpha=0, beta=formal)
      → prob_fusion.src.place_prob.place_prob_from_p_win(lam2=0.6018, lam3=0.6381)
  (b) Harville 逆算: 同 p_win → place_prob_from_p_win(lam2=1.0, lam3=1.0)
  (c) 直接予測 raw: 本実験 binary モデル 5 シード平均
  (d) 直接予測 normalized: (c) をレース内合計 3 に正規化（clip+再配分）

(a)(b) は既存コードパス（prob_fusion / evaluation の実装）を import して再利用する
（新規実装しない。仕様書 §5 の指示）。

出力: pure_rank/experiments/place_direct/scores/probs_place_direct_fold2_oos.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import load_config as load_prod_config  # noqa: E402
from evaluate import ensemble_predict  # noqa: E402

from place_lib import get_experiment_feature_cols, normalize_place_probs  # noqa: E402

from evaluation.odds_loader import attach_odds_from_se_parquet  # noqa: E402
from prob_fusion.src.fit_fusion import fusion_probs  # noqa: E402
from prob_fusion.src.market_prob import attach_market_q  # noqa: E402
from prob_fusion.src.oos_protocol import TEST_START  # noqa: E402
from prob_fusion.src.place_prob import place_prob_from_p_win  # noqa: E402

DATA_DIR = EXP_DIR / "data"
MODELS_DIR = EXP_DIR / "models"
SCORES_DIR = EXP_DIR / "scores"
CONFIG_PATH = EXP_DIR / "config.json"

L1_SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"
FUSION_REPORT_PATH = ROOT / "evaluation" / "reports" / "fusion_oos_fold2.json"

EXPECTED_SEEDS = 5


def load_experiment_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _compute_stern_harville(exp_cfg: dict) -> pd.DataFrame:
    """(a) Stern, (b) Harville の複勝確率を既存コードパスで算出する。"""
    fusion_report = json.loads(FUSION_REPORT_PATH.read_text(encoding="utf-8"))
    formal = fusion_report["formal"]
    alpha, beta = float(formal["alpha"]), float(formal["beta"])
    print(f"fusion params (formal, TEST事前fit): alpha={alpha}, beta={beta}")

    lam2_s = exp_cfg["place_prob"]["lam2_stern"]
    lam3_s = exp_cfg["place_prob"]["lam3_stern"]
    lam2_h = exp_cfg["place_prob"]["lam2_harville"]
    lam3_h = exp_cfg["place_prob"]["lam3_harville"]
    print(f"Stern lam2={lam2_s}, lam3={lam3_s} | Harville lam2={lam2_h}, lam3={lam3_h}")

    scores = pd.read_parquet(L1_SCORES_PATH)
    scores["race_id"] = scores["race_id"].astype(str)
    if "horse_number" not in scores.columns:
        scores["horse_number"] = scores["horse_num"]
    scores["horse_num"] = scores["horse_number"].astype(int)
    scores = scores[pd.to_datetime(scores["race_date"]) >= pd.Timestamp(TEST_START)].copy()
    print(f"L1 OOS scores (TEST only): rows={len(scores):,}, races={scores['race_id'].nunique():,}")

    scores = attach_odds_from_se_parquet(scores)
    scores = attach_market_q(scores)

    stern_list, harville_list = [], []
    n_races = 0
    for race_id, grp in scores.groupby("race_id", sort=False):
        z = grp["pure_score_z"].astype(float).to_numpy()
        ln_q = grp["ln_market_q"].astype(float).to_numpy()
        p_win = fusion_probs(z, ln_q, alpha, beta)

        p_stern = place_prob_from_p_win(p_win, lam2_s, lam3_s)
        p_harville = place_prob_from_p_win(p_win, lam2_h, lam3_h)

        out = grp[["race_id", "ketto_num", "horse_num"]].copy()
        out["p_stern"] = p_stern
        out["p_harville"] = p_harville
        stern_list.append(out)
        n_races += 1

    print(f"Stern/Harville computed for {n_races:,} races")
    return pd.concat(stern_list, ignore_index=True)


def _predict_direct(exp_cfg: dict, prod_cfg: dict) -> pd.DataFrame:
    """(c) 直接予測 raw の 5 シード平均を計算する。"""
    test_df = pd.read_parquet(DATA_DIR / "test_2025.parquet")
    test_df["race_id"] = test_df["race_id"].astype(str)
    print(f"place_direct test rows={len(test_df):,}, races={test_df['race_id'].nunique():,}")

    feature_cols = get_experiment_feature_cols(test_df, prod_cfg)
    assert "target_place" not in feature_cols

    model_paths = sorted(MODELS_DIR.glob("place_direct_seed*.txt"))
    if len(model_paths) != EXPECTED_SEEDS:
        raise ValueError(
            f"place_direct モデル数が {len(model_paths)} 本（期待 {EXPECTED_SEEDS} 本）: {MODELS_DIR}\n"
            "先に train_fold2.py を実行してください。"
        )
    models = [lgb.Booster(model_file=str(p)) for p in model_paths]
    print(f"{len(models)} モデルで TEST 期間をスコアリング")

    p_raw = ensemble_predict(models, test_df[feature_cols])
    test_df = test_df.copy()
    test_df["p_direct_raw"] = p_raw

    p_norm, clip_count = normalize_place_probs(
        test_df["p_direct_raw"].to_numpy(),
        test_df["race_id"].to_numpy(),
        max_iter=exp_cfg["normalize"]["max_iter"],
    )
    test_df["p_direct_norm"] = p_norm
    print(f"normalize: clip発生頭数={clip_count:,} / {len(test_df):,}")

    keep_cols = [
        "race_id", "ketto_num", "horse_num", "race_date",
        "horse_count", "finish_rank", "target_place",
        "p_direct_raw", "p_direct_norm",
    ]
    return test_df[keep_cols], clip_count


def export_place_direct_probs() -> Path:
    exp_cfg = load_experiment_config()
    prod_cfg = load_prod_config()

    ab_df = _compute_stern_harville(exp_cfg)
    cd_df, clip_count = _predict_direct(exp_cfg, prod_cfg)

    ab_df["race_id"] = ab_df["race_id"].astype(str)
    cd_df["race_id"] = cd_df["race_id"].astype(str)
    ab_df["ketto_num"] = ab_df["ketto_num"].astype(int)
    cd_df["ketto_num"] = cd_df["ketto_num"].astype(int)

    merged = cd_df.merge(
        ab_df[["race_id", "ketto_num", "p_stern", "p_harville"]],
        on=["race_id", "ketto_num"],
        how="inner",
    )
    print(f"Merged rows={len(merged):,} (cd_df={len(cd_df):,}, ab_df={len(ab_df):,})")

    n_dropped = len(cd_df) - len(merged)
    if n_dropped > 0:
        print(f"WARNING: {n_dropped:,} 行が merge で欠落（オッズ欠損等）")

    # リーク防止チェック（§9）: TEST レース集合が既存 OOS の 4,775 レースと一致
    n_races = merged["race_id"].nunique()
    print(f"TEST races in merged output: {n_races:,}")
    if n_races != 4775:
        print(f"WARNING: TEST race count {n_races:,} != 既存 OOS 4,775（merge 欠落の可能性）")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORES_DIR / "probs_place_direct_fold2_oos.parquet"
    merged.to_parquet(out_path, index=False, compression="snappy")
    print(f"Saved: {out_path} ({len(merged):,} rows, {n_races:,} races)")
    print(f"clip_count (normalize): {clip_count:,}")
    return out_path


if __name__ == "__main__":
    export_place_direct_probs()

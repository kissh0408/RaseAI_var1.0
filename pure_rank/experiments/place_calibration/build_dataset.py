"""place_calibration: build_dataset.py

fold2 OOS L1 スコア（`pure_rank/data/03_scores/scores_v39_course_slim_fold2_oos.parquet`）
を読み込み、前フェーズ (place_direct) (a) 系列と完全同一のコードパスで p_win を算出する:

    attach_odds_from_se_parquet -> attach_market_q
    -> fusion_probs(z, ln_q, alpha=0.0, beta=formal) から p_win

その後 y_place（複勝実績）・horse_count を付与し、
prob_fusion.src.oos_protocol.split_oos_periods で fit(2023-01-01..2024-12-31) /
TEST(2025-01-01..) に分割する。

保存列は odds・market_q・ln_market_q を含めない（較正の入力は p_win と y_place のみ。
仕様書 §2, §9）。

出力:
    pure_rank/experiments/place_calibration/data/fit_2023_2024.parquet
    pure_rank/experiments/place_calibration/data/test_2025.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.odds_loader import attach_odds_from_se_parquet  # noqa: E402
from prob_fusion.src.fit_fusion import fusion_probs  # noqa: E402
from prob_fusion.src.market_prob import attach_market_q  # noqa: E402
from prob_fusion.src.oos_protocol import split_oos_periods  # noqa: E402

L1_SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"
CONFIG_PATH = EXP_DIR / "config.json"
DATA_DIR = EXP_DIR / "data"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _compute_p_win(cfg: dict) -> pd.DataFrame:
    """前フェーズ (a) 系列と完全同一のコードパスで p_win を算出する。

    (place_direct/export_probs.py の _compute_stern_harville と同一手順。
    新規実装ではなく既存コードパスの再利用。仕様書 §3.0)
    """
    alpha = float(cfg["fusion"]["alpha"])
    beta = float(cfg["fusion"]["beta"])
    print(f"fusion params (formal): alpha={alpha}, beta={beta}")

    scores = pd.read_parquet(L1_SCORES_PATH)
    scores["race_id"] = scores["race_id"].astype(str)
    if "horse_number" not in scores.columns:
        scores["horse_number"] = scores["horse_num"]
    scores["horse_num"] = scores["horse_number"].astype(int)
    print(f"L1 OOS scores (all fold2 OOS period): rows={len(scores):,}, races={scores['race_id'].nunique():,}")

    scores = attach_odds_from_se_parquet(scores)
    scores = attach_market_q(scores)

    out_rows = []
    n_races = 0
    for race_id, grp in scores.groupby("race_id", sort=False):
        z = grp["pure_score_z"].astype(float).to_numpy()
        ln_q = grp["ln_market_q"].astype(float).to_numpy()
        p_win = fusion_probs(z, ln_q, alpha, beta)

        out = grp[["race_id", "race_date", "ketto_num", "horse_num", "finish_rank"]].copy()
        out["p_win"] = p_win
        out["horse_count"] = len(grp)
        out_rows.append(out)
        n_races += 1

    print(f"p_win computed for {n_races:,} races")
    return pd.concat(out_rows, ignore_index=True)


def build_place_calibration_dataset() -> dict[str, Path]:
    cfg = _load_config()

    df = _compute_p_win(cfg)
    df["y_place"] = (df["finish_rank"].astype(int) <= 3).astype(int)
    print(f"y_place positive rate: {df['y_place'].mean():.4f}")

    # 較正の入力は p_win と y_place のみ（odds/market_q等の生の市場列は保存しない。§2, §9）
    keep_cols = [
        "race_id", "race_date", "ketto_num", "horse_num",
        "finish_rank", "horse_count", "p_win", "y_place",
    ]
    df = df[keep_cols].copy()

    fit_start = cfg["fit_period"]["start"]
    fit_end = cfg["fit_period"]["end"]
    test_start = cfg["test_period"]["start"]
    fit_df, test_df = split_oos_periods(
        df, fit_start=fit_start, fit_end=fit_end, test_start=test_start
    )

    n_fit_races = fit_df["race_id"].nunique()
    n_test_races = test_df["race_id"].nunique()
    print(
        f"Fit({fit_start}..{fit_end}): rows={len(fit_df):,}, races={n_fit_races:,} "
        f"({fit_df['race_date'].min().date()} - {fit_df['race_date'].max().date()})"
    )
    print(
        f"Test(>={test_start}): rows={len(test_df):,}, races={n_test_races:,} "
        f"({test_df['race_date'].min().date()} - {test_df['race_date'].max().date()})"
    )

    # リーク防止チェック（§9）
    assert fit_df["race_date"].max() <= pd.Timestamp(fit_end), "fit にfit_end超過混入"
    assert fit_df["race_date"].min() >= pd.Timestamp(fit_start), "fit にfit_start未満混入"
    assert test_df["race_date"].min() >= pd.Timestamp(test_start), "test にtest_start未満混入"
    overlap = set(fit_df["race_id"]) & set(test_df["race_id"])
    assert not overlap, f"fit/test でrace_id重複: {len(overlap)}件"

    # fit 期間レース数が formal λ fit（6,786 レース）と整合することを確認しログに残す（§9）
    formal_n = cfg["s0_baseline"]["fit_n_races_formal"]
    print(f"Fit races: {n_fit_races:,} (formal λ fit races: {formal_n:,}, diff={n_fit_races - formal_n:+,})")

    # TEST レース集合が既存 OOS の 4,775 レース / 66,020 頭と一致（assert。§9）
    expected_n_races = cfg["test_period"]["expected_n_races"]
    expected_n_horses = cfg["test_period"]["expected_n_horses"]
    if n_test_races != expected_n_races or len(test_df) != expected_n_horses:
        raise AssertionError(
            f"TEST集合が既存OOSと不一致: races={n_test_races} (expected {expected_n_races}), "
            f"horses={len(test_df)} (expected {expected_n_horses})"
        )
    print(f"OK: TEST races={n_test_races:,} horses={len(test_df):,} matches existing OOS reference")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "fit": DATA_DIR / "fit_2023_2024.parquet",
        "test": DATA_DIR / "test_2025.parquet",
    }
    fit_df.to_parquet(out_paths["fit"], index=False, compression="snappy")
    test_df.to_parquet(out_paths["test"], index=False, compression="snappy")
    for name, path in out_paths.items():
        print(f"Saved {name}: {path}")

    return out_paths


if __name__ == "__main__":
    build_place_calibration_dataset()

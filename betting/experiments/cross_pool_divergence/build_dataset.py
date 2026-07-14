"""cross_pool_divergence: build_dataset.py

券種別ユニット parquet（units_place / units_wide / units_quinella）を構築する。

入力は L0 前処理層（SE/RA preprocessed）と確定オッズ・確定払戻のみ。L1（pure_rank の
特徴量生成物・学習済みスコア）は一切参照しない。

Rule 3（期間規律）: 本スクリプトは config.protocol の fit_start〜fit_end のみを対象に
ビルドする（TEST(2025+) 行は生成しない）。Stage 3 実装時に対象期間を拡張する。

出力:
    betting/experiments/cross_pool_divergence/data/units_place.parquet
    betting/experiments/cross_pool_divergence/data/units_wide.parquet
    betting/experiments/cross_pool_divergence/data/units_quinella.parquet
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import divergence_lib as dl  # noqa: E402

from betting.src.ev_filters import harville_quinella_pair_prob  # noqa: E402
from betting.src.pair_probs import stern_wide_pair_prob  # noqa: E402
from betting.src.wide_ev_core import (  # noqa: E402
    compute_race_overround,
    get_pair_odds,
    load_wide_odds_lookup,
    norm_pair,
    odds_dir_default,
)
from evaluation.odds_loader import attach_odds_from_se_parquet  # noqa: E402
from evaluation.place_payout_loader import (  # noqa: E402
    attach_place_payout,
    build_place_payout_lookup,
)
from prob_fusion.src.market_prob import attach_market_q  # noqa: E402
from prob_fusion.src.place_prob import stern_place_probs  # noqa: E402

SE_PATH = ROOT / "pure_rank" / "data" / "01_preprocessed" / "SE_preprocessed.parquet"
RA_PATH = ROOT / "pure_rank" / "data" / "01_preprocessed" / "RA_preprocessed.parquet"
HR_PARQUET_PATH = ROOT / "pure_rank" / "data" / "01_preprocessed" / "HR_preprocessed.parquet"
HR_CSV_DIR = ROOT / "common" / "data" / "output" / "race_hr"
FUSION_REPORT_PATH = ROOT / "evaluation" / "reports" / "fusion_oos_fold2.json"
CONFIG_PATH = EXP_DIR / "config.json"
DATA_DIR = EXP_DIR / "data"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_lambda_params(cfg: dict) -> tuple[float, float]:
    report = json.loads(FUSION_REPORT_PATH.read_text(encoding="utf-8"))
    formal = report[cfg["fusion_params_key"]]
    return float(formal["lam2"]), float(formal["lam3"])


def _apply_base_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ex = cfg["exclusion_filters"]
    out = df[
        (~df["grade_code"].isin(ex["grade_code_exclude"]))
        & (~df["abnormal_code"].isin(ex["abnormal_code_exclude"]))
        & (df["horse_count"] >= ex["horse_count_min"])
        & (df["finish_rank"] > 0)
    ].copy()
    return out


def _load_base_frame(cfg: dict, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """ベースフレームを構築する。start/end 未指定時は config の fit 期間
    （fit_start〜fit_end）。Stage 3 のみ start=test_start（end=None=上限なし）で
    呼び出す（fit 用データファイルには一切影響しない）。"""
    se_cols = ["race_id", "horse_num", "finish_rank", "abnormal_code", "race_date"]
    se = pd.read_parquet(SE_PATH, columns=se_cols)
    se["race_id"] = se["race_id"].astype(str)

    ra_cols = ["race_id", "horse_count", "grade_code", "race_date"]
    ra = pd.read_parquet(RA_PATH, columns=ra_cols)
    ra["race_id"] = ra["race_id"].astype(str)
    ra = ra.drop(columns=["race_date"])  # SE の race_date を正とする

    df = se.merge(ra, on="race_id", how="inner")
    print(f"SE+RA merged: rows={len(df):,}, races={df['race_id'].nunique():,}")

    period_start = pd.Timestamp(start or cfg["protocol"]["fit_start"])
    dates = pd.to_datetime(df["race_date"])
    mask = dates >= period_start
    if start is None or end is not None:
        period_end = pd.Timestamp(end or cfg["protocol"]["fit_end"])
        mask &= dates <= period_end
        end_label = str(period_end.date())
    else:
        end_label = "open"
    df = df[mask].copy()
    print(f"date filtered [{period_start.date()}..{end_label}]: rows={len(df):,}, races={df['race_id'].nunique():,}")

    n_before_filters = len(df)
    df = _apply_base_filters(df, cfg)
    print(f"base filters applied: rows={len(df):,} (dropped {n_before_filters - len(df):,})")

    n_before_odds = len(df)
    df = attach_odds_from_se_parquet(df)
    n_missing_odds = int(df["odds"].isna().sum())
    df = df.dropna(subset=["odds"]).copy()
    print(
        f"odds attached: rows={len(df):,}, missing_odds_excluded={n_missing_odds:,} "
        f"({n_missing_odds / max(n_before_odds, 1):.4%})"
    )

    # market_q はレース内で残存馬に対して再正規化される（比例法）
    df = attach_market_q(df, method=cfg["q_method"])

    # 人気順位はレース内で残存馬（オッズ既知馬）のみを対象に確定する
    df["pop_rank"] = 0
    for race_id, idx in df.groupby("race_id").groups.items():
        sub = df.loc[idx]
        ranks = dl.assign_popularity_rank(sub["odds"].to_numpy(), sub["horse_num"].to_numpy())
        df.loc[idx, "pop_rank"] = ranks
    df["pop_band_single"] = df["pop_rank"].apply(dl.assign_pop_band_single)
    df["fs_band"] = df["horse_count"].apply(dl.assign_fs_band)
    return df


def _build_place_units(df: pd.DataFrame, cfg: dict, lam2: float, lam3: float) -> pd.DataFrame:
    place_cfg = cfg["place"]
    hc_min, hc_max = place_cfg["horse_count_min"], place_cfg["horse_count_max"]
    small_max = place_cfg["small_field_horse_count_max"]

    sub = df[(df["horse_count"] >= hc_min) & (df["horse_count"] <= hc_max)].copy()

    hr_dir = HR_CSV_DIR if HR_CSV_DIR.exists() else None
    payout_lookup = build_place_payout_lookup(
        hr_dir=hr_dir,
        hr_parquet=HR_PARQUET_PATH if HR_PARQUET_PATH.is_file() else None,
    )
    sub = attach_place_payout(sub, payout_lookup)

    rows = []
    n_races_used = 0
    t0 = time.perf_counter()
    for race_id, grp in sub.groupby("race_id", sort=False):
        n = len(grp)
        horse_count = int(grp["horse_count"].iloc[0])
        p_win = grp["market_q"].astype(float).to_numpy()
        p2, p3 = stern_place_probs(p_win, lam2, lam3)
        for i in range(n):
            row = grp.iloc[i]
            m, y = dl.place_m_and_hit(horse_count, int(row["finish_rank"]), small_field_max=small_max)
            if horse_count <= small_max:
                p_theo = float(p_win[i] + p2[i])
            else:
                p_theo = float(p_win[i] + p2[i] + p3[i])
            o_val = float(row["place_multiplier"]) if bool(row["place_paid"]) else float("nan")
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": row["race_date"],
                    "horse_count": horse_count,
                    "unit_key": int(row["horse_num"]),
                    "pop_rank": int(row["pop_rank"]),
                    "pop_band": row["pop_band_single"],
                    "fs_band": row["fs_band"],
                    "m": m,
                    "y": float(y),
                    "p_theo": p_theo,
                    "O": o_val,
                }
            )
        n_races_used += 1
    elapsed = time.perf_counter() - t0
    print(f"place units: races={n_races_used:,}, units={len(rows):,}, elapsed={elapsed:.1f}s")
    return pd.DataFrame(rows)


def _build_pair_units(
    df: pd.DataFrame,
    cfg: dict,
    lam2: float,
    lam3: float,
    *,
    bet_type: str,
) -> tuple[pd.DataFrame, dict]:
    bt_cfg = cfg[bet_type]
    hc_min, hc_max = bt_cfg["horse_count_min"], bt_cfg["horse_count_max"]
    m = int(bt_cfg["m"])
    max_horses = cfg["max_horses_per_race"]

    sub = df[(df["horse_count"] >= hc_min) & (df["horse_count"] <= hc_max) & (df["horse_count"] <= max_horses)].copy()

    years = sorted(pd.to_datetime(sub["race_date"]).dt.year.unique().tolist())
    odds_type = "Wide" if bet_type == "wide" else "Quinella"
    odds_lookup = load_wide_odds_lookup(years, odds_dir_default(ROOT), odds_type=odds_type)

    rows = []
    n_races_used = 0
    n_races_excluded_missing_odds = 0
    t0 = time.perf_counter()
    for race_id, grp in sub.groupby("race_id", sort=False):
        n = len(grp)
        horse_count = int(grp["horse_count"].iloc[0])
        horse_nums = grp["horse_num"].astype(int).to_numpy()
        p_win = grp["market_q"].astype(float).to_numpy()
        pop_ranks = grp["pop_rank"].astype(int).to_numpy()
        finish = grp["finish_rank"].astype(int).to_numpy()

        # 全ペアのオッズが揃っているか事前確認（1つでも欠損なら当該レースを除外）
        odds_map: dict = {}
        odds_ok = True
        for i in range(n):
            for j in range(i + 1, n):
                pair = norm_pair(int(horse_nums[i]), int(horse_nums[j]))
                o = get_pair_odds(race_id, pair[0], pair[1], odds_lookup)
                if o is None or o <= 1.0:
                    odds_ok = False
                    break
                odds_map[pair] = float(o)
            if not odds_ok:
                break
        if not odds_ok:
            n_races_excluded_missing_odds += 1
            continue

        p_pool_map, or_r = dl.compute_p_pool(odds_map, m)
        # OR_r の定義一致検査（§4.4/§9-6要求）: compute_race_overround（本番ワイドEVコアの
        # 定義）と divergence_lib の定義が一致することをビルド時にも sanity check する。
        or_r_ref = compute_race_overround(race_id, odds_lookup)
        if np.isfinite(or_r) and np.isfinite(or_r_ref) and not np.isclose(or_r, or_r_ref, rtol=1e-6):
            print(f"  [warn] OR_r mismatch race={race_id}: lib={or_r} ref={or_r_ref}")
        t_hat = dl.effective_takeout_from_or(or_r, m)

        p_dict = {int(h): float(p) for h, p in zip(horse_nums, p_win)}

        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = int(horse_nums[i]), int(horse_nums[j])
                pair = norm_pair(hi, hj)
                if bet_type == "wide":
                    p_theo = stern_wide_pair_prob(p_win, i, j, lam2, lam3)
                    y = 1.0 if finish[i] <= 3 and finish[j] <= 3 else 0.0
                else:
                    p_theo = harville_quinella_pair_prob(p_dict[hi], p_dict[hj])
                    y = 1.0 if finish[i] <= 2 and finish[j] <= 2 else 0.0

                pair_band = dl.assign_pair_band(int(pop_ranks[i]), int(pop_ranks[j]))
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": grp["race_date"].iloc[0],
                        "horse_count": horse_count,
                        "unit_key": f"{pair[0]}-{pair[1]}",
                        "pair_band": pair_band,
                        "fs_band": grp["fs_band"].iloc[0],
                        "m": m,
                        "y": y,
                        "p_theo": float(p_theo),
                        "O": odds_map[pair],
                        "p_pool": p_pool_map[pair],
                        "OR_r": or_r,
                        "effective_takeout": t_hat,
                    }
                )
        n_races_used += 1
    elapsed = time.perf_counter() - t0
    print(
        f"{bet_type} units: races_used={n_races_used:,}, "
        f"races_excluded_missing_odds={n_races_excluded_missing_odds:,}, "
        f"units={len(rows):,}, elapsed={elapsed:.1f}s"
    )
    build_log = {
        "bet_type": bet_type,
        "n_races_used": n_races_used,
        "n_races_excluded_missing_odds": n_races_excluded_missing_odds,
        "n_units": len(rows),
        "elapsed_sec": elapsed,
    }
    return pd.DataFrame(rows), build_log


def build_cross_pool_divergence_datasets() -> dict:
    cfg = _load_config()
    lam2, lam3 = _load_lambda_params(cfg)
    print(f"fusion params: lam2={lam2}, lam3={lam3}")

    df = _load_base_frame(cfg)

    place_df = _build_place_units(df, cfg, lam2, lam3)
    wide_df, wide_log = _build_pair_units(df, cfg, lam2, lam3, bet_type="wide")
    quinella_df, quinella_log = _build_pair_units(df, cfg, lam2, lam3, bet_type="quinella")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    place_df.to_parquet(DATA_DIR / "units_place.parquet", index=False, compression="snappy")
    wide_df.to_parquet(DATA_DIR / "units_wide.parquet", index=False, compression="snappy")
    quinella_df.to_parquet(DATA_DIR / "units_quinella.parquet", index=False, compression="snappy")

    build_log = {
        "protocol": cfg["protocol"],
        "place": {"n_units": len(place_df), "n_races": int(place_df["race_id"].nunique()) if len(place_df) else 0},
        "wide": wide_log,
        "quinella": quinella_log,
    }
    log_path = DATA_DIR / "build_log.json"
    log_path.write_text(json.dumps(build_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(build_log, indent=2, ensure_ascii=False))
    return build_log


if __name__ == "__main__":
    build_cross_pool_divergence_datasets()

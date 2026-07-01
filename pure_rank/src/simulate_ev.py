"""
simulate_ev.py -- ワイド・馬連 EV シミュレーション（評価専用）

HR 払戻データと Harville 確率を使い、純粋能力モデルの期待値を検証する。
オッズ・払戻は特徴量に使わない（事後評価のみ）。

強化版:
- EV 閾値スイープ（threshold: 0.8〜1.5）
- レース条件別 ROI（surface_code / distance_category / weather_code）
- キャリブレーション確認（予測確率 vs 実的中率）
- WideOdds 事前オッズを使った真の EV 計算（2026-07-01〜）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (
    PROJECT_ROOT,
    ensemble_predict,
    get_feature_cols,
    load_config,
    load_models,
)
from predict import (
    _best_wide_pair,
    _build_wide_lookup,
    _norm_pair,
    apply_platt_to_p_win,
    compute_race_probabilities,
    compute_race_probabilities_from_p_win,
    load_calibration_models,
    run_fit_calibration,
    softmax_with_temperature,
)

PAIR_KEY = tuple[int, int]
STAKE = 100.0

# 市場乖離スコア閾値スイープ用定数
LOG_DIV_THRESHOLDS: list[float] = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def _build_hr_lookup(hr_df: pd.DataFrame, bet_type: str) -> dict[str, dict[PAIR_KEY, int]]:
    """race_id -> {(h1,h2): payout} の辞書を構築。"""
    sub = hr_df[hr_df["bet_type"] == bet_type]
    lookup: dict[str, dict[PAIR_KEY, int]] = {}
    for _, row in sub.iterrows():
        rid = str(row["race_id"])
        key = _norm_pair(int(row["horse_num_1"]), int(row["horse_num_2"]))
        lookup.setdefault(rid, {})[key] = int(row["payout"])
    return lookup


def _build_odds_lookup(
    years: list[int],
    odds_dir: Path,
    odds_type: str,
) -> dict[str, dict[PAIR_KEY, float]]:
    """{odds_type}Odds_YYYY.csv を複数年読み込み、race_id -> {(h1,h2): odds} の辞書を返す。

    Parameters
    ----------
    years     : テストセットの年リスト
    odds_dir  : Odds CSV が格納されたディレクトリ
    odds_type : "Wide" または "Quinella"

    Returns
    -------
    dict[race_id_str, dict[(h1,h2), odds]]
        - race_id_str: str 16 桁（int64 を str() 変換したもの。features_*.parquet の race_id と一致）
        - (h1, h2): _norm_pair() で正規化（小さい馬番が先頭）
        - odds: float（100円あたりの払戻倍率）

    除外条件
    --------
    - odds_status != "ok" の行（発売前取消・発売後取消を除外）
    - odds が NaN の行
    - CSV ファイルが存在しない年（警告を出してスキップ）
    """
    lookup: dict[str, dict[PAIR_KEY, float]] = {}
    for year in years:
        path = odds_dir / f"{odds_type}Odds_{year}.csv"
        if not path.exists():
            print(f"  [warn] {odds_type}Odds_{year}.csv not found, skipping")
            continue
        df = pd.read_csv(path)
        # odds_status フィルタ・NaN フィルタ（ベット可能なオッズのみ）
        df = df[(df["odds_status"] == "ok") & df["odds"].notna()].copy()
        # race_id: int64 -> str 16 桁（features の race_id と同じ形式）
        df["race_id_str"] = df["race_id"].apply(lambda x: str(int(x)))
        # ペア正規化（小さい馬番が先頭）
        df["h_min"] = df[["horse_num_1", "horse_num_2"]].min(axis=1).astype(int)
        df["h_max"] = df[["horse_num_1", "horse_num_2"]].max(axis=1).astype(int)
        df["pair_key"] = list(zip(df["h_min"], df["h_max"]))
        # レースごとに辞書を構築（同一ペアが複数スナップショット存在する場合は最後の値を採用）
        for rid, grp in df.groupby("race_id_str"):
            lookup[rid] = dict(zip(grp["pair_key"], grp["odds"].astype(float)))
    print(f"  {odds_type}Odds loaded: {len(lookup):,} races across {years}")
    return lookup


def _best_quinella_pair(quinella_matrix: np.ndarray) -> tuple[int, int]:
    n = quinella_matrix.shape[0]
    best_i, best_j = 0, 1 if n > 1 else 0
    best_p = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if quinella_matrix[i, j] > best_p:
                best_p = quinella_matrix[i, j]
                best_i, best_j = i, j
    return best_i, best_j


def _collect_bets_per_race(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]] | None = None,
    quinella_odds_lookup: dict[str, dict[PAIR_KEY, float]] | None = None,
) -> pd.DataFrame:
    """
    テストセット全レース分のベット情報を1行1レースの DataFrame として返す。

    ワイド EV は WideOdds 事前オッズを使った真の期待値で計算する:
      EV_wide = P_wide x wide_odds（odds は decimal 倍率、100円ベットで odds×100円返却）
    オッズが取得できないレースは EV_wide = NaN とする。

    馬連 EV は QuinellaOdds 事前オッズを使った真の期待値で計算する:
      EV_quin = P_quin x quinella_odds（Wide と同じ計算パターン）
    quinella_odds_lookup が None の場合は HR 払戻平均へのフォールバック（後方互換）。
    """
    if wide_odds_lookup is None:
        wide_odds_lookup = {}

    df = df_test.copy()
    df["pred_score"] = predictions

    wide_lookup = _build_hr_lookup(hr_df, "wide")
    quin_lookup = _build_hr_lookup(hr_df, "quinella")

    # 馬連フォールバック払戻平均:
    # quinella_odds_lookup=None で呼び出された場合（後方互換パス）に EV 計算の参照値として使用する。
    # quinella_odds_lookup が提供される通常パスでは参照されない。
    quin_ref_payout = float(hr_df[hr_df["bet_type"] == "quinella"]["payout"].mean())

    rows: list[dict] = []
    for race_id, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        grp = grp.sort_values("pred_score", ascending=False).reset_index(drop=True)
        horse_nums = grp["horse_num"].astype(int).values
        scores = grp["pred_score"].values
        probs = compute_race_probabilities(scores, T_opt)

        wi, wj = _best_wide_pair(probs["wide_matrix"])
        qi, qj = _best_quinella_pair(probs["quinella_matrix"])
        wide_key = _norm_pair(int(horse_nums[wi]), int(horse_nums[wj]))
        quin_key = _norm_pair(int(horse_nums[qi]), int(horse_nums[qj]))
        p_wide = float(probs["wide_matrix"][wi, wj])
        p_quin = float(probs["quinella_matrix"][qi, qj])

        wide_payout = int(wide_lookup.get(rid, {}).get(wide_key, 0))
        quin_payout = int(quin_lookup.get(rid, {}).get(quin_key, 0))

        # ワイド: WideOdds 事前オッズによる真の EV
        # EV = P_wide x odds（odds は decimal 倍率。/100 は不要）
        prior_odds_wide = wide_odds_lookup.get(rid, {}).get(wide_key, None)
        ev_wide = (p_wide * prior_odds_wide) if prior_odds_wide is not None else float("nan")

        # 馬連: QuinellaOdds 事前オッズによる真の EV
        # EV = P_quinella × odds（Wide と同じ計算パターン）
        if quinella_odds_lookup is not None:
            prior_odds_quin = quinella_odds_lookup.get(rid, {}).get(quin_key, None)
            ev_quin = (p_quin * prior_odds_quin) if prior_odds_quin is not None else float("nan")
        else:
            # フォールバック（quinella_odds_lookup 未提供時の後方互換）
            ref_q = quin_payout if quin_payout > 0 else quin_ref_payout
            ev_quin = p_quin * ref_q / STAKE

        first = grp.iloc[0]
        rows.append({
            "race_id": rid,
            "p_wide": p_wide,
            "p_quin": p_quin,
            "ev_wide": ev_wide,
            "ev_quin": ev_quin,
            "payout_wide": wide_payout,
            "payout_quin": quin_payout,
            "hit_wide": int(wide_payout > 0),
            "hit_quin": int(quin_payout > 0),
            "surface_code": int(first["surface_code"]) if "surface_code" in grp.columns else -1,
            "distance_category": first["distance_category"] if "distance_category" in grp.columns else -1,
            "weather_code": int(first["weather_code"]) if "weather_code" in grp.columns else -1,
            # 時系列順 MDD 計算のため race_date を追加
            "race_date": first["race_date"] if "race_date" in grp.columns else pd.NaT,
        })

    return pd.DataFrame(rows)


def ev_threshold_sweep(
    df_bets: pd.DataFrame,
    thresholds: list[float],
    bet_type: str = "wide",
) -> pd.DataFrame:
    """
    EV 閾値を変化させて ROI・的中率・ベット件数を計算する。

    Parameters
    ----------
    df_bets   : _collect_bets_per_race() の出力 DataFrame
    thresholds: EV 閾値リスト（例: [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]）
    bet_type  : "wide" または "quinella"

    Returns
    -------
    pd.DataFrame: threshold / n_bets / hit_rate / return_rate / total_profit
    """
    ev_col = f"ev_{bet_type}"
    hit_col = f"hit_{bet_type}"
    pay_col = f"payout_{bet_type}"

    records: list[dict] = []
    for t in thresholds:
        subset = df_bets[df_bets[ev_col] >= t]
        n = len(subset)
        if n == 0:
            records.append({
                "threshold": t,
                "n_bets": 0,
                "hit_rate": float("nan"),
                "return_rate": float("nan"),
                "total_profit": float("nan"),
            })
            continue
        hits = int(subset[hit_col].sum())
        total_payout = float(subset[pay_col].sum())
        total_stake = n * STAKE
        records.append({
            "threshold": t,
            "n_bets": n,
            "hit_rate": hits / n,
            "return_rate": total_payout / total_stake,
            "total_profit": total_payout - total_stake,
        })

    return pd.DataFrame(records)


def roi_by_condition(
    df_bets: pd.DataFrame,
    ev_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    surface_code / distance_category / weather_code ごとの ROI を集計する。

    EV > ev_threshold のベットに限定して条件別 ROI を計算する。
    EV が NaN のレースは自動的に除外される（NaN >= threshold は False）。

    Returns
    -------
    pd.DataFrame: condition_type / condition_value / n_bets / hit_rate / return_rate
    """
    subset = df_bets[df_bets["ev_wide"] >= ev_threshold].copy()
    records: list[dict] = []

    for cond_col in ["surface_code", "distance_category", "weather_code"]:
        if cond_col not in subset.columns:
            continue
        for val, grp in subset.groupby(cond_col):
            n = len(grp)
            if n == 0:
                continue
            hits = int(grp["hit_wide"].sum())
            total_payout = float(grp["payout_wide"].sum())
            records.append({
                "condition_type": cond_col,
                "condition_value": str(val),
                "n_bets": n,
                "hit_rate": hits / n,
                "return_rate": total_payout / (n * STAKE),
                "total_profit": total_payout - n * STAKE,
            })

    df_cond = pd.DataFrame(records)
    if df_cond.empty:
        return df_cond
    return df_cond.sort_values("return_rate", ascending=False).reset_index(drop=True)


def check_calibration(
    df_bets: pd.DataFrame,
    n_bins: int = 10,
) -> dict:
    """
    予測勝率（p_wide）と実際の的中率のズレを計測する。

    スコアを n_bins のビンに分割し、
    predicted_prob vs actual_hit_rate を比較する。

    Returns
    -------
    dict: bins リスト + 要約統計
    """
    df = df_bets.copy().sort_values("p_wide")
    df["bin"] = pd.qcut(df["p_wide"], q=n_bins, labels=False, duplicates="drop")

    bins: list[dict] = []
    for b, grp in df.groupby("bin"):
        if len(grp) == 0:
            continue
        predicted = float(grp["p_wide"].mean())
        actual = float(grp["hit_wide"].mean())
        bins.append({
            "bin": int(b),
            "n": len(grp),
            "predicted_prob": round(predicted, 4),
            "actual_hit_rate": round(actual, 4),
            "diff": round(actual - predicted, 4),
        })

    if not bins:
        return {"bins": [], "mean_abs_error": None, "max_abs_error": None}

    diffs = [abs(b["diff"]) for b in bins]
    return {
        "bins": bins,
        "mean_abs_error": round(float(np.mean(diffs)), 4),
        "max_abs_error": round(float(np.max(diffs)), 4),
    }


def simulate_ev(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]] | None = None,
) -> dict:
    """テストセットで Harville 最大 P_wide / P_quinella 戦略の回収率を計算。

    後方互換: 以前の呼び出しインターフェースを維持しつつ拡張結果を返す。
    wide_odds_lookup を渡さない場合、EV_wide は NaN になる（後方互換）。
    """
    df_bets = _collect_bets_per_race(df_test, predictions, hr_df, T_opt, wide_odds_lookup)

    n_races = len(df_bets)
    total_wide_payout = float(df_bets["payout_wide"].sum())
    total_quin_payout = float(df_bets["payout_quin"].sum())
    total_stake = n_races * STAKE

    return {
        "n_races": n_races,
        "wide_return_rate": total_wide_payout / total_stake if total_stake > 0 else 0.0,
        "quinella_return_rate": total_quin_payout / total_stake if total_stake > 0 else 0.0,
        "ev_positive_rate": float((df_bets["ev_wide"] > 1.0).mean()) if n_races > 0 else 0.0,
        "hit_rate_wide": float(df_bets["hit_wide"].mean()) if n_races > 0 else 0.0,
        "hit_rate_quinella": float(df_bets["hit_quin"].mean()) if n_races > 0 else 0.0,
    }


def _collect_bets_with_calibration(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    calib: dict,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]] | None = None,
) -> pd.DataFrame:
    """
    テストセット全レースについて4手法分のベット情報を1行1レースで返す。

    columns:
      race_id, hit_wide, payout_wide,           (共通)
      p_wide_base, ev_wide_base,                (ベースライン)
      p_wide_platt, ev_wide_platt, (pair_platt) (手法1 Platt)
      p_wide_roi_t, ev_wide_roi_t,              (手法2 ROI-T)
      p_wide_isotonic, ev_wide_isotonic,        (手法3 Isotonic)
      surface_code, distance_category, weather_code

    ワイド EV は各手法の選択ペアに対応する WideOdds 事前オッズで計算する。
    オッズ未取得のレースは EV = NaN。wide_ref_payout への依存を全手法から除去。
    """
    if wide_odds_lookup is None:
        wide_odds_lookup = {}

    df = df_test.copy()
    df["pred_score"] = predictions

    wide_lookup = _build_wide_lookup(hr_df)

    platt = calib.get("platt")
    isotonic = calib.get("isotonic")
    T_roi = float(calib.get("T_roi", T_opt))

    rows: list[dict] = []
    for race_id, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        grp_r = grp.sort_values("pred_score", ascending=False).reset_index(drop=True)
        horse_nums = grp_r["horse_num"].astype(int).values
        scores = grp_r["pred_score"].values
        n = len(scores)
        first = grp_r.iloc[0]

        # --- ベースライン ----------------------------------------------------
        probs_base = compute_race_probabilities(scores, T_opt)
        wi_b, wj_b = _best_wide_pair(probs_base["wide_matrix"])
        key_b = _norm_pair(int(horse_nums[wi_b]), int(horse_nums[wj_b]))
        p_wide_b = float(probs_base["wide_matrix"][wi_b, wj_b])
        payout_b = int(wide_lookup.get(rid, {}).get(key_b, 0))
        prior_b = wide_odds_lookup.get(rid, {}).get(key_b, None)
        ev_b = (p_wide_b * prior_b) if prior_b is not None else float("nan")

        # --- 手法1: Platt ---------------------------------------------------
        if platt is not None:
            p_win_raw = softmax_with_temperature(scores, T_opt)
            p_win_cal = apply_platt_to_p_win(p_win_raw, platt)
            probs_p = compute_race_probabilities_from_p_win(p_win_cal)
            wi_p, wj_p = _best_wide_pair(probs_p["wide_matrix"])
            key_p = _norm_pair(int(horse_nums[wi_p]), int(horse_nums[wj_p]))
            p_wide_p = float(probs_p["wide_matrix"][wi_p, wj_p])
            payout_p = int(wide_lookup.get(rid, {}).get(key_p, 0))
            prior_p_val = wide_odds_lookup.get(rid, {}).get(key_p, None)
            ev_p = (p_wide_p * prior_p_val) if prior_p_val is not None else float("nan")
        else:
            p_wide_p = ev_p = float("nan")
            payout_p = 0

        # --- 手法2: ROI-T ---------------------------------------------------
        probs_t = compute_race_probabilities(scores, T_roi)
        wi_t, wj_t = _best_wide_pair(probs_t["wide_matrix"])
        key_t = _norm_pair(int(horse_nums[wi_t]), int(horse_nums[wj_t]))
        p_wide_t = float(probs_t["wide_matrix"][wi_t, wj_t])
        payout_t = int(wide_lookup.get(rid, {}).get(key_t, 0))
        prior_t = wide_odds_lookup.get(rid, {}).get(key_t, None)
        ev_t = (p_wide_t * prior_t) if prior_t is not None else float("nan")

        # --- 手法3: Isotonic ------------------------------------------------
        if isotonic is not None:
            # Harville p_wide を全ペアに Isotonic 適用してから最良ペアを選ぶ
            wide_mat_iso = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    p_w = float(probs_base["wide_matrix"][i, j])
                    p_w_cal = float(isotonic.predict([p_w])[0])
                    wide_mat_iso[i, j] = p_w_cal
                    wide_mat_iso[j, i] = p_w_cal
            wi_i, wj_i = _best_wide_pair(wide_mat_iso)
            key_i = _norm_pair(int(horse_nums[wi_i]), int(horse_nums[wj_i]))
            p_wide_i = float(wide_mat_iso[wi_i, wj_i])
            payout_i = int(wide_lookup.get(rid, {}).get(key_i, 0))
            prior_i = wide_odds_lookup.get(rid, {}).get(key_i, None)
            ev_i = (p_wide_i * prior_i) if prior_i is not None else float("nan")
        else:
            p_wide_i = ev_i = float("nan")
            payout_i = 0

        # ベースラインの hit/payout を正としてレース共通情報を記録
        rows.append({
            "race_id": rid,
            # shared ground truth (ベースラインの選択ペアで判定)
            "payout_wide": payout_b,
            "hit_wide": int(payout_b > 0),
            # ベースライン
            "p_wide_base": p_wide_b,
            "ev_wide_base": ev_b,
            # 手法1 Platt（選択ペアが違う場合は payout も変わる）
            "payout_platt": payout_p,
            "hit_platt": int(payout_p > 0),
            "p_wide_platt": p_wide_p,
            "ev_wide_platt": ev_p,
            # 手法2 ROI-T
            "payout_roi_t": payout_t,
            "hit_roi_t": int(payout_t > 0),
            "p_wide_roi_t": p_wide_t,
            "ev_wide_roi_t": ev_t,
            # 手法3 Isotonic
            "payout_isotonic": payout_i,
            "hit_isotonic": int(payout_i > 0),
            "p_wide_isotonic": p_wide_i,
            "ev_wide_isotonic": ev_i,
            # 条件
            "surface_code": int(first["surface_code"]) if "surface_code" in grp_r.columns else -1,
            "distance_category": first["distance_category"] if "distance_category" in grp_r.columns else -1,
            "weather_code": int(first["weather_code"]) if "weather_code" in grp_r.columns else -1,
        })

    return pd.DataFrame(rows)


def _roi_stats(df_bets: pd.DataFrame, ev_col: str, pay_col: str, hit_col: str, threshold: float) -> dict:
    """EV 閾値フィルタ後の ROI 統計を計算する。NaN は自動除外される。"""
    sub = df_bets[df_bets[ev_col] >= threshold]
    n = len(sub)
    if n == 0:
        return {"n_bets": 0, "hit_rate": float("nan"), "roi": float("nan")}
    hits = int(sub[hit_col].sum())
    total_payout = float(sub[pay_col].sum())
    return {
        "n_bets": n,
        "hit_rate": hits / n,
        "roi": total_payout / (n * STAKE),
    }


def compare_calibration_methods(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    calib: dict,
    ev_threshold: float = 1.0,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]] | None = None,
) -> dict:
    """
    3手法のキャリブレーション結果をテストセットで比較する。

    Returns
    -------
    dict: 各手法の {n_bets, hit_rate, roi} を含む比較結果
    """
    df_bets = _collect_bets_with_calibration(
        df_test, predictions, hr_df, T_opt, calib, wide_odds_lookup
    )
    T_roi = float(calib.get("T_roi", T_opt))

    # --- ベースライン（全件）------------------------------------------------
    n_all = len(df_bets)
    roi_all = float(df_bets["payout_wide"].sum()) / (n_all * STAKE) if n_all > 0 else 0.0
    hit_all = float(df_bets["hit_wide"].mean()) if n_all > 0 else 0.0

    # --- 各手法の EV フィルタ後統計 ----------------------------------------
    base_ev = _roi_stats(df_bets, "ev_wide_base", "payout_wide", "hit_wide", ev_threshold)
    platt_ev = _roi_stats(df_bets, "ev_wide_platt", "payout_platt", "hit_platt", ev_threshold)
    roi_t_ev = _roi_stats(df_bets, "ev_wide_roi_t", "payout_roi_t", "hit_roi_t", ev_threshold)
    iso_ev = _roi_stats(df_bets, "ev_wide_isotonic", "payout_isotonic", "hit_isotonic", ev_threshold)

    result = {
        "baseline_all": {"n_bets": n_all, "hit_rate": hit_all, "roi": roi_all},
        "baseline_ev": base_ev,
        "platt_ev": platt_ev,
        "roi_t_ev": {"T_roi": T_roi, **roi_t_ev},
        "isotonic_ev": iso_ev,
    }
    return result, df_bets


# ─── リスク調整評価指標 ─────────────────────────────────────────────────────────

def compute_max_drawdown(pnl_series: np.ndarray) -> tuple[float, float]:
    """
    累積 P&L 時系列からピーク比最大ドローダウンを計算する。

    Parameters
    ----------
    pnl_series : 各ベットの P&L 配列（時系列順）

    Returns
    -------
    tuple[float, float]: (mdd_yen, mdd_pct)
        mdd_yen : 最大ドローダウン（円）、常に >= 0
        mdd_pct : mdd_yen / 累積最大値（ピーク比率）
    """
    arr = np.asarray(pnl_series, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    mdd_yen = float(drawdown.max())
    max_peak = float(running_max.max())
    # ピークがゼロ以下（全損失系列）の場合は絶対値を基準にする
    mdd_pct = mdd_yen / max(max_peak, 1.0) if max_peak > 0 else 0.0
    return mdd_yen, mdd_pct


def compute_sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    ベット単位収益率のシャープレシオを計算する。

    Parameters
    ----------
    returns  : 各ベットの収益率配列（hit: (payout-100)/100, miss: -1.0）
    risk_free: リスクフリーレート（デフォルト 0.0）

    Returns
    -------
    float: mean(returns - risk_free) / std(returns)
           std が 0 または サンプルが 1 件以下の場合は nan を返す
    """
    r = np.asarray(returns, dtype=float)
    if len(r) <= 1:
        return float("nan")
    std = float(np.std(r, ddof=1))
    if std < 1e-12:
        return float("nan")
    return float((np.mean(r) - risk_free) / std)


def compute_kelly_fractions(
    df_bets: pd.DataFrame,
    fraction: float = 0.25,
    ev_col: str = "ev_wide",
    p_col: str = "p_wide",
) -> pd.Series:
    """
    df_bets の各行に fraction Kelly ベット分率を計算して返す。

    計算式:
        prior_odds = ev / p_wide
        b = prior_odds - 1  （net odds）
        f_full = max(p - (1-p)/b, 0)
        f_quarter = f_full * fraction

    EV <= 1.0 または p <= 0 または prior_odds <= 1.0 の行は 0.0 を返す。
    """
    ev = df_bets[ev_col].values.astype(float)
    p = df_bets[p_col].values.astype(float)
    result = np.zeros(len(df_bets), dtype=float)

    for idx in range(len(df_bets)):
        ev_i = ev[idx]
        p_i = p[idx]
        if np.isnan(ev_i) or np.isnan(p_i) or p_i <= 0 or ev_i <= 1.0:
            continue
        prior_odds = ev_i / p_i
        if prior_odds <= 1.0:
            continue
        b = prior_odds - 1.0
        f_full = max((p_i - (1.0 - p_i) / b), 0.0)
        result[idx] = f_full * fraction

    return pd.Series(result, index=df_bets.index)


def simulate_kelly_quarter(
    df_bets: pd.DataFrame,
    initial_bankroll: float = 100_000.0,
    fraction: float = 0.25,
    ev_threshold: float = 1.0,
    ev_col: str = "ev_wide",
    pay_col: str = "payout_wide",
    hit_col: str = "hit_wide",
    p_col: str = "p_wide",
    min_bet: float = 10.0,
    ruin_threshold: float = 1_000.0,
) -> dict:
    """
    時系列順に 1/4 Kelly でベットし、シミュレーション結果を返す。

    Parameters
    ----------
    df_bets         : _collect_bets_per_race() の出力（race_date カラム推奨）
    initial_bankroll: 初期資金（円）
    fraction        : Kelly 分率（0.25 = 1/4 Kelly）
    ev_threshold    : ベット条件（EV >= この値のみ）
    min_bet         : 最小ベット額（円）
    ruin_threshold  : 残高がこれを下回ったらシミュレーション終了

    Returns
    -------
    dict:
        initial_capital, final_balance, total_profit_yen, final_return_pct,
        n_bets, hit_rate, mdd_yen, mdd_pct, sharpe_per_bet, ruined, balance_series
    """
    sub = df_bets[df_bets[ev_col] >= ev_threshold].copy()
    if "race_date" in sub.columns:
        sub = sub.sort_values("race_date").reset_index(drop=True)

    balance = float(initial_bankroll)
    balance_series: list[float] = [balance]
    pnl_list: list[float] = []
    returns_list: list[float] = []
    hit_count = 0
    ruined = False
    n_bets = 0

    for _, row in sub.iterrows():
        ev_i = row[ev_col]
        p_i = row[p_col]
        hit_i = int(row[hit_col])

        if pd.isna(ev_i) or pd.isna(p_i) or p_i <= 0 or ev_i <= 1.0:
            continue

        prior_odds = float(ev_i) / float(p_i)
        if prior_odds <= 1.0:
            continue

        b = prior_odds - 1.0
        f_full = max((float(p_i) - (1.0 - float(p_i)) / b), 0.0)
        f_quarter = f_full * fraction

        # 10円単位で切り捨て
        bet_size_raw = f_quarter * balance
        bet_size = max(int(bet_size_raw / 10) * 10, int(min_bet))

        n_bets += 1

        if hit_i:
            profit = (prior_odds - 1.0) * bet_size
            balance += profit
            pnl_list.append(profit)
            returns_list.append(profit / bet_size)
            hit_count += 1
        else:
            balance -= bet_size
            pnl_list.append(-float(bet_size))
            returns_list.append(-1.0)

        balance_series.append(balance)

        if balance < ruin_threshold:
            ruined = True
            break

    # MDD は残高時系列から直接計算（peak-to-trough in balance）
    if len(balance_series) > 1:
        bal_arr = np.array(balance_series, dtype=float)
        running_max_bal = np.maximum.accumulate(bal_arr)
        drawdown_bal = running_max_bal - bal_arr
        mdd_yen = float(drawdown_bal.max())
        # mdd_pct は初期資金比
        mdd_pct = mdd_yen / max(initial_bankroll, 1.0)
    else:
        mdd_yen, mdd_pct = 0.0, 0.0

    sharpe_raw = compute_sharpe_ratio(np.array(returns_list)) if returns_list else float("nan")
    sharpe = None if (isinstance(sharpe_raw, float) and np.isnan(sharpe_raw)) else round(sharpe_raw, 6)

    return {
        "initial_capital": initial_bankroll,
        "final_balance": round(balance, 2),
        "total_profit_yen": round(balance - initial_bankroll, 2),
        "final_return_pct": round((balance - initial_bankroll) / initial_bankroll, 6),
        "n_bets": n_bets,
        "hit_rate": round(hit_count / n_bets, 6) if n_bets > 0 else None,
        "mdd_yen": round(mdd_yen, 2),
        "mdd_pct": round(mdd_pct, 6),
        "sharpe_per_bet": sharpe,
        "ruined": ruined,
        "balance_series": balance_series,
    }


def compute_risk_metrics(
    df_bets: pd.DataFrame,
    ev_thresholds: list[float] | None = None,
    initial_capital: float = 100_000.0,
    kelly_fraction: float = 0.25,
    bet_type: str = "wide",
) -> dict:
    """
    複数 EV 閾値でリスク調整評価指標をまとめて計算する。

    Parameters
    ----------
    df_bets       : _collect_bets_per_race() の出力（race_date カラム必須）
    ev_thresholds : 評価する EV 閾値リスト（デフォルト [1.0, 1.3]）
    bet_type      : "wide" または "quin"

    Returns
    -------
    dict: {"ev_1.0": {"fixed_stake": {...}, "kelly_quarter": {...}}, ...}
    """
    if ev_thresholds is None:
        ev_thresholds = [1.0, 1.3]

    ev_col = f"ev_{bet_type}"
    hit_col = f"hit_{bet_type}"
    pay_col = f"payout_{bet_type}"
    p_col = f"p_{bet_type}"

    df_sorted = df_bets.copy()
    if "race_date" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("race_date").reset_index(drop=True)

    result: dict = {}

    for threshold in ev_thresholds:
        sub = df_sorted[df_sorted[ev_col] >= threshold].copy()
        n = len(sub)
        key = f"ev_{threshold}"

        if n == 0:
            result[key] = {
                "fixed_stake": {
                    "n_bets": 0, "hit_rate": None, "roi": None,
                    "mdd_yen": None, "mdd_pct": None,
                    "sharpe_per_bet": None, "total_profit_yen": None,
                },
                "kelly_quarter": {
                    "initial_capital": initial_capital,
                    "final_balance": initial_capital,
                    "n_bets": 0, "hit_rate": None,
                    "mdd_yen": None, "mdd_pct": None,
                    "sharpe_per_bet": None, "total_profit_yen": None,
                    "final_return_pct": None, "ruined": False,
                },
            }
            continue

        hits = int(sub[hit_col].sum())
        total_payout = float(sub[pay_col].sum())
        total_stake = n * STAKE

        # Fixed-stake PnL / returns
        pnl_arr = np.where(
            sub[hit_col].values == 1,
            sub[pay_col].values.astype(float) - STAKE,
            -STAKE,
        ).astype(float)
        returns_arr = np.where(
            sub[hit_col].values == 1,
            (sub[pay_col].values.astype(float) - STAKE) / STAKE,
            -1.0,
        ).astype(float)

        # fixed stake MDD: cumulative PnL starting from 0
        cumulative_pnl = np.concatenate([[0.0], np.cumsum(pnl_arr)])
        running_max_fs = np.maximum.accumulate(cumulative_pnl)
        drawdown_fs = running_max_fs - cumulative_pnl
        mdd_yen_fs = float(drawdown_fs.max())
        # mdd_pct は total_stake 比（固定ベットに初期資金概念がないため）
        mdd_pct_fs = mdd_yen_fs / max(total_stake, 1.0)
        sharpe_raw_fs = compute_sharpe_ratio(returns_arr)
        sharpe_fs = None if (isinstance(sharpe_raw_fs, float) and np.isnan(sharpe_raw_fs)) else round(sharpe_raw_fs, 6)

        fs = {
            "n_bets": n,
            "hit_rate": round(hits / n, 6),
            "roi": round(total_payout / total_stake, 6),
            "mdd_yen": round(mdd_yen_fs, 2),
            "mdd_pct": round(mdd_pct_fs, 6),
            "sharpe_per_bet": sharpe_fs,
            "total_profit_yen": round(total_payout - total_stake, 2),
        }

        # Kelly quarter simulation（全件を渡してフィルタリングは内部で行う）
        kq_result = simulate_kelly_quarter(
            df_sorted,
            initial_bankroll=initial_capital,
            fraction=kelly_fraction,
            ev_threshold=threshold,
            ev_col=ev_col,
            pay_col=pay_col,
            hit_col=hit_col,
            p_col=p_col,
        )
        kq = {
            "initial_capital": kq_result["initial_capital"],
            "final_balance": kq_result["final_balance"],
            "n_bets": kq_result["n_bets"],
            "hit_rate": kq_result["hit_rate"],
            "mdd_yen": kq_result["mdd_yen"],
            "mdd_pct": kq_result["mdd_pct"],
            "sharpe_per_bet": kq_result["sharpe_per_bet"],
            "total_profit_yen": kq_result["total_profit_yen"],
            "final_return_pct": kq_result["final_return_pct"],
            "ruined": kq_result["ruined"],
        }

        result[key] = {"fixed_stake": fs, "kelly_quarter": kq}

    return result


# ─── 市場乖離ベット戦略 ──────────────────────────────────────────────────────────

def compute_race_overround(
    race_id: str,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]],
) -> float:
    """
    1 レースの全ペアから overround（全ペア 1/odds の合計）を計算する。

    Parameters
    ----------
    race_id          : str 形式の race_id
    wide_odds_lookup : _build_odds_lookup() の出力

    Returns
    -------
    float: sum(1/odds) for all valid pairs。レース未発見の場合は 1.0 を返す。
    """
    pairs = wide_odds_lookup.get(race_id, {})
    if not pairs:
        return 1.0
    total = sum(1.0 / v for v in pairs.values() if v > 0)
    return max(total, 1.0)  # overround は常に >= 1.0


def collect_divergence_bets_per_race(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]],
) -> pd.DataFrame:
    """
    各レースで log_divergence が最大のペアを選ぶ戦略のベット DataFrame を返す。

    アルゴリズム:
    1. 全ペアの p_model（Harville wide_matrix）を計算
    2. レース全体の overround を計算
    3. log_divergence = log(p_model * odds * overround) を各ペアで計算
    4. WideOdds が取得できないペアはスキップ
    5. argmax(log_divergence) のペアを選択

    Returns
    -------
    pd.DataFrame:
        race_id, horse_num_1, horse_num_2, p_model, p_implied_raw, p_implied,
        overround, ev_raw, log_divergence, abs_divergence, odds_wide,
        payout_wide, hit_wide, race_date, surface_code, distance_category, weather_code
    """
    df = df_test.copy()
    df["pred_score"] = predictions

    wide_lookup = _build_hr_lookup(hr_df, "wide")

    rows: list[dict] = []
    for race_id, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        rid = str(race_id)
        grp = grp.sort_values("pred_score", ascending=False).reset_index(drop=True)
        horse_nums = grp["horse_num"].astype(int).values
        scores = grp["pred_score"].values
        probs = compute_race_probabilities(scores, T_opt)
        n = len(grp)
        first = grp.iloc[0]

        overround = compute_race_overround(rid, wide_odds_lookup)

        # 全ペアを走査し log_divergence 最大のペアを選ぶ
        best_log_div = float("-inf")
        best_info: tuple | None = None

        for i in range(n):
            for j in range(i + 1, n):
                p_w = float(probs["wide_matrix"][i, j])
                if p_w <= 0:
                    continue
                key = _norm_pair(int(horse_nums[i]), int(horse_nums[j]))
                odds = wide_odds_lookup.get(rid, {}).get(key, None)
                if odds is None or odds <= 0:
                    continue
                log_div = float(np.log(p_w * odds * overround))
                if log_div > best_log_div:
                    best_log_div = log_div
                    best_info = (i, j, key, p_w, odds)

        if best_info is None:
            continue  # このレースは WideOdds 未取得

        i, j, key, p_w, odds = best_info
        payout = int(wide_lookup.get(rid, {}).get(key, 0))
        p_implied_raw = 1.0 / odds
        p_implied = p_implied_raw / overround
        ev_raw = p_w * odds
        abs_div = p_w - p_implied

        rows.append({
            "race_id": rid,
            "horse_num_1": key[0],
            "horse_num_2": key[1],
            "p_model": p_w,
            "p_implied_raw": p_implied_raw,
            "p_implied": p_implied,
            "overround": overround,
            "ev_raw": ev_raw,
            "log_divergence": best_log_div,
            "abs_divergence": abs_div,
            "odds_wide": odds,
            "payout_wide": payout,
            "hit_wide": int(payout > 0),
            "race_date": first["race_date"] if "race_date" in grp.columns else pd.NaT,
            "surface_code": int(first["surface_code"]) if "surface_code" in grp.columns else -1,
            "distance_category": first["distance_category"] if "distance_category" in grp.columns else -1,
            "weather_code": int(first["weather_code"]) if "weather_code" in grp.columns else -1,
        })

    return pd.DataFrame(rows)


def sweep_divergence_threshold(
    df_div_bets: pd.DataFrame,
    thresholds: list[float],
    div_col: str = "log_divergence",
) -> pd.DataFrame:
    """
    乖離スコア閾値をスイープして ROI・的中率・ベット数を返す。

    Parameters
    ----------
    df_div_bets : collect_divergence_bets_per_race() の出力
    thresholds  : 閾値リスト
    div_col     : "log_divergence" または "abs_divergence"

    Returns
    -------
    pd.DataFrame: threshold / n_bets / hit_rate / return_rate / total_profit
    """
    records: list[dict] = []
    for t in thresholds:
        subset = df_div_bets[df_div_bets[div_col] >= t]
        n = len(subset)
        if n == 0:
            records.append({
                "threshold": t, "n_bets": 0,
                "hit_rate": float("nan"), "return_rate": float("nan"),
                "total_profit": float("nan"),
            })
            continue
        hits = int(subset["hit_wide"].sum())
        total_payout = float(subset["payout_wide"].sum())
        total_stake = n * STAKE
        records.append({
            "threshold": t, "n_bets": n,
            "hit_rate": hits / n,
            "return_rate": total_payout / total_stake,
            "total_profit": total_payout - total_stake,
        })
    return pd.DataFrame(records)


def compare_ev_vs_divergence(
    df_bets_ev: pd.DataFrame,
    df_bets_div: pd.DataFrame,
    ev_threshold: float = 1.0,
    div_threshold: float = 0.0,
) -> dict:
    """
    EV戦略 vs Divergence戦略 vs 複合戦略を比較する。

    Parameters
    ----------
    df_bets_ev  : _collect_bets_per_race() の出力（argmax p_model でペア選択）
    df_bets_div : collect_divergence_bets_per_race() の出力（argmax log_divergence）
    ev_threshold: EV フィルタ閾値
    div_threshold: log_divergence フィルタ閾値

    Returns
    -------
    dict:
        ev_only      : EV フィルタのみ（argmax p_model）
        div_only     : divergence フィルタのみ（argmax log_divergence）
        combined     : EV >= threshold AND log_divergence > threshold の積集合
    """
    def _stats(sub: pd.DataFrame, pay_col: str, hit_col: str) -> dict:
        n = len(sub)
        if n == 0:
            return {"n_bets": 0, "hit_rate": None, "roi": None}
        hits = int(sub[hit_col].sum())
        total_payout = float(sub[pay_col].sum())
        return {
            "n_bets": n,
            "hit_rate": round(hits / n, 6),
            "roi": round(total_payout / (n * STAKE), 6),
        }

    # 戦略 A: EV のみ（argmax p_model, EV >= threshold）
    ev_only = df_bets_ev[df_bets_ev["ev_wide"] >= ev_threshold]

    # 戦略 C: divergence のみ（argmax log_divergence, log_div > div_threshold）
    div_only = df_bets_div[df_bets_div["log_divergence"] > div_threshold]

    # 戦略 D: 複合（argmax log_divergence で選択、EV >= threshold AND log_div > div_threshold）
    combined = df_bets_div[
        (df_bets_div["ev_raw"] >= ev_threshold) &
        (df_bets_div["log_divergence"] > div_threshold)
    ]

    return {
        "ev_only": _stats(ev_only, "payout_wide", "hit_wide"),
        "div_only": _stats(div_only, "payout_wide", "hit_wide"),
        "combined": _stats(combined, "payout_wide", "hit_wide"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Wide/Quinella EV simulation (eval only)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--ev-thresholds",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5],
        help="EV threshold list for sweep",
    )
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="Fit calibration models on validation set before comparison",
    )
    parser.add_argument(
        "--compare-calibration",
        action="store_true",
        help="Output calibration method comparison table (uses saved models or fits if missing)",
    )
    parser.add_argument(
        "--divergence",
        action="store_true",
        help="Compute market divergence betting strategy (argmax log_divergence)",
    )
    args = parser.parse_args()

    cfg = load_config()
    T_opt = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    hr_path = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "HR_preprocessed.parquet"
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]

    if not hr_path.exists():
        raise FileNotFoundError(
            f"HR_preprocessed.parquet が見つかりません: {hr_path}\n"
            "JV-Link で HR CSV を取得後、python pure_rank/src/preprocess.py --hr-only を実行してください。"
        )

    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)
    valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
    df_test = df[df["race_date"] > valid_end_ts].copy()
    print(f"Test set: {len(df_test):,} rows, {df_test['race_id'].nunique():,} races")

    print(f"Loading HR payouts: {hr_path}")
    hr_df = pd.read_parquet(hr_path)
    print(f"  HR rows: {len(hr_df):,}")

    # --- WideOdds 事前オッズの読み込み --------------------------------------
    test_years = sorted(df_test["race_date"].dt.year.unique().tolist())
    odds_dir = PROJECT_ROOT / "common" / "data" / "output" / "odds"
    print(f"\nLoading WideOdds for years: {test_years}")
    wide_odds_lookup = _build_odds_lookup(test_years, odds_dir, "Wide")

    print(f"\nLoading QuinellaOdds for years: {test_years}")
    quinella_odds_lookup = _build_odds_lookup(test_years, odds_dir, "Quinella")

    feature_cols = get_feature_cols(df_test, cfg)
    models = load_models(models_dir)
    preds = ensemble_predict(models, df_test[feature_cols])

    print(f"\nCollecting per-race bets (T_opt={T_opt})...")
    df_bets = _collect_bets_per_race(
        df_test, preds, hr_df, T_opt,
        wide_odds_lookup=wide_odds_lookup,
        quinella_odds_lookup=quinella_odds_lookup,
    )
    print(f"  Collected {len(df_bets):,} race-bets")

    # --- EV=NaN 率の集計・報告 -----------------------------------------------
    n_ev_na = int(df_bets["ev_wide"].isna().sum())
    n_total = len(df_bets)
    print(f"  EV=NaN (no odds): {n_ev_na}/{n_total} ({n_ev_na/n_total*100:.1f}%)")
    n_quin_ev_na = int(df_bets["ev_quin"].isna().sum())
    print(f"  Quinella EV=NaN (no odds): {n_quin_ev_na}/{n_total} ({n_quin_ev_na/n_total*100:.1f}%)")

    # --- 全体統計 -------------------------------------------------------------
    n_races = len(df_bets)
    total_stake = n_races * STAKE
    overall_wide_rr = float(df_bets["payout_wide"].sum()) / total_stake
    overall_quin_rr = float(df_bets["payout_quin"].sum()) / total_stake

    print(f"\n--- Overall ---")
    print(f"  n_races            : {n_races:,}")
    print(f"  wide_return_rate   : {overall_wide_rr:.4f} ({overall_wide_rr*100:.2f}%)")
    print(f"  quinella_return_rate: {overall_quin_rr:.4f} ({overall_quin_rr*100:.2f}%)")
    print(f"  wide_hit_rate      : {df_bets['hit_wide'].mean():.4f} ({df_bets['hit_wide'].mean()*100:.2f}%)")
    print(f"  quinella_hit_rate  : {df_bets['hit_quin'].mean():.4f} ({df_bets['hit_quin'].mean()*100:.2f}%)")

    # --- EV=1.0 フィルタ後 --------------------------------------------------
    ev10_wide = df_bets[df_bets["ev_wide"] >= 1.0]
    ev10_quin = df_bets[df_bets["ev_quin"] >= 1.0]

    ev_filtered: dict = {
        "threshold": 1.0,
        "wide_n_bets": len(ev10_wide),
        "wide_hit_rate": float(ev10_wide["hit_wide"].mean()) if len(ev10_wide) > 0 else None,
        "wide_return_rate": float(ev10_wide["payout_wide"].sum() / (len(ev10_wide) * STAKE))
        if len(ev10_wide) > 0 else None,
        "quinella_n_bets": len(ev10_quin),
        "quinella_hit_rate": float(ev10_quin["hit_quin"].mean()) if len(ev10_quin) > 0 else None,
        "quinella_return_rate": float(ev10_quin["payout_quin"].sum() / (len(ev10_quin) * STAKE))
        if len(ev10_quin) > 0 else None,
    }
    def _fmt(v: float | None, fmt: str) -> str:
        return format(v, fmt) if v is not None else "N/A"

    print(f"\n--- EV >= 1.0 Filter ---")
    print(f"  wide : n={ev_filtered['wide_n_bets']:,}, "
          f"hit={_fmt(ev_filtered['wide_hit_rate'], '.3f')}, "
          f"ROI={_fmt(ev_filtered['wide_return_rate'], '.4f')}")
    print(f"  quin : n={ev_filtered['quinella_n_bets']:,}, "
          f"hit={_fmt(ev_filtered['quinella_hit_rate'], '.3f')}, "
          f"ROI={_fmt(ev_filtered['quinella_return_rate'], '.4f')}")

    # --- EV 閾値スイープ -----------------------------------------------------
    print(f"\n--- EV Threshold Sweep (wide) ---")
    sweep_wide = ev_threshold_sweep(df_bets, args.ev_thresholds, bet_type="wide")
    print(sweep_wide.to_string(index=False))

    print(f"\n--- EV Threshold Sweep (quinella) ---")
    sweep_quin = ev_threshold_sweep(df_bets, args.ev_thresholds, bet_type="quin")
    print(sweep_quin.to_string(index=False))

    # --- 条件別 ROI ----------------------------------------------------------
    print(f"\n--- ROI by Condition (EV >= 1.0) ---")
    df_cond = roi_by_condition(df_bets, ev_threshold=1.0)
    if not df_cond.empty:
        print(df_cond.head(20).to_string(index=False))
        best = df_cond.iloc[0]
        best_condition = {
            "condition_type": best["condition_type"],
            "condition_value": best["condition_value"],
            "n_bets": int(best["n_bets"]),
            "return_rate": float(best["return_rate"]),
        }
    else:
        best_condition = {}

    # --- キャリブレーション --------------------------------------------------
    print(f"\n--- Calibration Check (wide) ---")
    calib_check = check_calibration(df_bets, n_bins=10)
    if calib_check["bins"]:
        print(f"  mean_abs_error: {calib_check['mean_abs_error']:.4f}")
        print(f"  max_abs_error : {calib_check['max_abs_error']:.4f}")
        for b in calib_check["bins"]:
            print(f"  bin={b['bin']:2d} n={b['n']:5d} pred={b['predicted_prob']:.4f} actual={b['actual_hit_rate']:.4f} diff={b['diff']:+.4f}")

    # --- charts/ に calibration.json を保存 ---------------------------------
    charts_dir = PROJECT_ROOT / cfg["data"]["features_dir"] / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    calib_path = charts_dir / "calibration_wide.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_check, f, indent=2, ensure_ascii=False)
    print(f"\n  Calibration saved: {calib_path}")

    # --- キャリブレーション手法比較 ------------------------------------------
    calib_comparison: dict = {}
    if args.fit_calibration or args.compare_calibration:
        print(f"\n{'='*60}")
        print("=== キャリブレーション手法比較 ===")
        print(f"{'='*60}")

        if args.fit_calibration:
            print("\n[学習] バリデーションセット（2024）でキャリブレーションモデルを学習...")
            calib_models = run_fit_calibration(cfg)
        else:
            print("\n[読み込み] 保存済みキャリブレーションモデルを探索...")
            calib_models = load_calibration_models(models_dir)
            if not calib_models:
                print("  保存済みモデルが見つかりません。--fit-calibration で学習してください。")
                calib_models = {}

        if calib_models:
            T_roi = float(calib_models.get("T_roi", T_opt))
            print(f"\n[評価] テストセット（2025+）で3手法を比較 (T_opt={T_opt}, T_roi={T_roi:.2f})...")
            comparison, df_bets_calib = compare_calibration_methods(
                df_test, preds, hr_df, T_opt, calib_models,
                ev_threshold=1.0,
                wide_odds_lookup=wide_odds_lookup,
            )

            def _fmt_pct(v) -> str:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "N/A"
                return f"{v * 100:.2f}%"

            def _fmt_n(v) -> str:
                if v is None:
                    return "N/A"
                return f"{v:,}"

            print("\n{:<30} {:>6} {:>10} {:>8} {:>10}".format(
                "method", "EV_thr", "wide_ROI", "n_bets", "hit_rate"
            ))
            print("-" * 70)
            b_all = comparison["baseline_all"]
            print("{:<30} {:>6} {:>10} {:>8} {:>10}".format(
                "baseline_all",
                "ALL",
                _fmt_pct(b_all["roi"]),
                _fmt_n(b_all["n_bets"]),
                _fmt_pct(b_all["hit_rate"]),
            ))
            b_ev = comparison["baseline_ev"]
            print("{:<30} {:>6} {:>10} {:>8} {:>10}".format(
                "baseline_EV>=1.0",
                "1.0",
                _fmt_pct(b_ev["roi"]),
                _fmt_n(b_ev["n_bets"]),
                _fmt_pct(b_ev["hit_rate"]),
            ))
            p_ev = comparison["platt_ev"]
            print("{:<30} {:>6} {:>10} {:>8} {:>10}".format(
                "Platt_EV>=1.0",
                "1.0",
                _fmt_pct(p_ev["roi"]),
                _fmt_n(p_ev["n_bets"]),
                _fmt_pct(p_ev["hit_rate"]),
            ))
            t_ev = comparison["roi_t_ev"]
            print("{:<30} {:>6} {:>10} {:>8} {:>10}".format(
                f"ROI-T={T_roi:.2f}_EV>=1.0",
                "1.0",
                _fmt_pct(t_ev["roi"]),
                _fmt_n(t_ev["n_bets"]),
                _fmt_pct(t_ev["hit_rate"]),
            ))
            i_ev = comparison["isotonic_ev"]
            print("{:<30} {:>6} {:>10} {:>8} {:>10}".format(
                "Isotonic_EV>=1.0",
                "1.0",
                _fmt_pct(i_ev["roi"]),
                _fmt_n(i_ev["n_bets"]),
                _fmt_pct(i_ev["hit_rate"]),
            ))
            print("-" * 70)

            # 最良手法を特定
            candidates = {
                "Platt": p_ev,
                f"ROI-T={T_roi:.2f}": t_ev,
                "Isotonic": i_ev,
            }
            best_name = max(
                candidates,
                key=lambda k: candidates[k]["roi"] if not np.isnan(candidates[k].get("roi", float("nan"))) else -1,
            )
            best = candidates[best_name]
            n_total_calib = comparison["baseline_all"]["n_bets"]
            print(f"\n[Best] {best_name}")
            print(f"  EV>=1.0 wide ROI: {_fmt_pct(best['roi'])}")
            print(f"  n_bets: {_fmt_n(best['n_bets'])} / {n_total_calib:,}")
            print(f"  hit_rate: {_fmt_pct(best['hit_rate'])}")
            roi_100 = (best.get("roi") or 0) >= 1.0
            n_ok = (best.get("n_bets") or 0) >= 200
            print(f"  ROI>=100%: {'Yes' if roi_100 else 'No'}")
            print(f"  n_bets>=200: {'Yes' if n_ok else 'No'}")

            calib_comparison = comparison

            # calibration_comparison.json を保存
            comp_path = charts_dir / "calibration_comparison.json"

            def _safe(v):
                if isinstance(v, float) and np.isnan(v):
                    return None
                if isinstance(v, (np.floating,)):
                    return float(v)
                if isinstance(v, (np.integer,)):
                    return int(v)
                return v

            def _clean_dict(d: dict) -> dict:
                return {k: _safe(v) for k, v in d.items()}

            comp_json = {
                "T_opt": T_opt,
                "T_roi": T_roi,
                "ev_threshold": 1.0,
                "baseline_all": _clean_dict(comparison["baseline_all"]),
                "baseline_ev": _clean_dict(comparison["baseline_ev"]),
                "platt_ev": _clean_dict(comparison["platt_ev"]),
                "roi_t_ev": _clean_dict(comparison["roi_t_ev"]),
                "isotonic_ev": _clean_dict(comparison["isotonic_ev"]),
                "best_method": best_name,
            }
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_json, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"\n  Comparison saved: {comp_path}")

    # --- リスク調整評価指標 ---------------------------------------------------
    print(f"\n--- Risk-Adjusted Metrics (Wide) ---")
    risk_metrics_wide = compute_risk_metrics(
        df_bets,
        ev_thresholds=[1.0, 1.3],
        initial_capital=100_000.0,
        kelly_fraction=0.25,
        bet_type="wide",
    )
    for ev_key, metrics in risk_metrics_wide.items():
        fs = metrics["fixed_stake"]
        kq = metrics["kelly_quarter"]
        print(f"\n[{ev_key}]")
        if fs["n_bets"] > 0:
            print(
                f"  Fixed stake  : n={fs['n_bets']:,}, ROI={fs['roi']*100:.2f}%, "
                f"MDD={fs['mdd_yen']:.0f}yen ({fs['mdd_pct']*100:.1f}%), "
                f"Sharpe={fs['sharpe_per_bet']}"
            )
        else:
            print(f"  Fixed stake  : n=0 (no bets above threshold)")
        print(
            f"  Kelly (1/4)  : initial={kq['initial_capital']:,.0f}yen, "
            f"final={kq['final_balance']:,.0f}yen, "
            f"MDD={kq['mdd_yen']:,.0f}yen ({kq['mdd_pct']*100:.1f}%), "
            f"ruined={kq['ruined']}"
        )

    # --- 市場乖離ベット戦略 ---------------------------------------------------
    divergence_results: dict = {}
    if args.divergence:
        print(f"\n{'='*60}")
        print("=== 市場乖離ベット戦略 ===")
        print(f"{'='*60}")

        print("\n[1] 全レースの log_divergence 最大ペアを収集...")
        df_div_bets = collect_divergence_bets_per_race(
            df_test, preds, hr_df, T_opt, wide_odds_lookup
        )
        n_div = len(df_div_bets)
        print(f"  divergence bets: {n_div:,} races (WideOdds 取得済みのみ)")

        if n_div > 0:
            print(f"\n[2] log_divergence スイープ")
            sweep_div = sweep_divergence_threshold(
                df_div_bets, LOG_DIV_THRESHOLDS, div_col="log_divergence"
            )
            print(sweep_div.to_string(index=False))

            print(f"\n[3] EV vs Divergence 戦略比較 (ev_threshold=1.0, div_threshold=0.0)")
            comparison = compare_ev_vs_divergence(
                df_bets, df_div_bets, ev_threshold=1.0, div_threshold=0.0
            )
            for strat_name, stats in comparison.items():
                n_b = stats["n_bets"]
                roi = stats["roi"]
                hit = stats["hit_rate"]
                print(
                    f"  {strat_name:<15}: n={n_b:5d}, "
                    f"ROI={roi*100:.2f}% " if roi is not None else f"  {strat_name:<15}: n={n_b:5d}, ROI=N/A ",
                    end=""
                )
                print(f"hit={hit*100:.1f}%" if hit is not None else "hit=N/A")

            # EV=1.3 でも比較
            print(f"\n[4] EV vs Divergence 戦略比較 (ev_threshold=1.3, div_threshold=0.0)")
            comparison_13 = compare_ev_vs_divergence(
                df_bets, df_div_bets, ev_threshold=1.3, div_threshold=0.0
            )
            for strat_name, stats in comparison_13.items():
                n_b = stats["n_bets"]
                roi = stats["roi"]
                hit = stats["hit_rate"]
                roi_str = f"ROI={roi*100:.2f}%" if roi is not None else "ROI=N/A"
                hit_str = f"hit={hit*100:.1f}%" if hit is not None else "hit=N/A"
                print(f"  {strat_name:<15}: n={n_b:5d}, {roi_str} {hit_str}")

            divergence_results = {
                "n_races_with_odds": n_div,
                "log_divergence_sweep": sweep_div.to_dict("records"),
                "comparison_ev10": comparison,
                "comparison_ev13": comparison_13,
            }

    # --- ev_results.json を拡張形式で保存 ------------------------------------
    def _to_json(v):
        if isinstance(v, (np.floating, float)):
            return float(v) if not np.isnan(v) else None
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v

    def _df_to_records(df: pd.DataFrame) -> list[dict]:
        return [{k: _to_json(v) for k, v in row.items()} for row in df.to_dict("records")]

    # WideOdds カバレッジ統計
    n_races_with_odds = n_total - n_ev_na
    wide_odds_coverage = {
        "n_races_total": n_total,
        "n_races_with_odds": n_races_with_odds,
        "n_races_ev_na": n_ev_na,
        "coverage_rate": round(n_races_with_odds / n_total, 6) if n_total > 0 else 0.0,
    }

    # QuinellaOdds カバレッジ統計
    n_quin_races_with_odds = n_total - n_quin_ev_na
    quinella_odds_coverage = {
        "n_races_total": n_total,
        "n_races_with_odds": n_quin_races_with_odds,
        "n_races_ev_na": n_quin_ev_na,
        "coverage_rate": round(n_quin_races_with_odds / n_total, 6) if n_total > 0 else 0.0,
    }

    # risk_metrics: NaN/None を安全に変換
    def _clean_risk(d: dict) -> dict:
        out: dict = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = _clean_risk(v)
            elif isinstance(v, (np.floating, float)):
                out[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.integer, int)):
                out[k] = int(v)
            elif isinstance(v, bool):
                out[k] = v
            else:
                out[k] = v
        return out

    risk_metrics_json = {"wide": _clean_risk(risk_metrics_wide)}

    # divergence_comparison: ev_only/div_only/combined
    divergence_comparison_json: dict | None = None
    if divergence_results:
        divergence_comparison_json = {
            "ev_only": divergence_results.get("comparison_ev10", {}).get("ev_only"),
            "div_only": divergence_results.get("comparison_ev10", {}).get("div_only"),
            "combined": divergence_results.get("comparison_ev10", {}).get("combined"),
        }

    results = {
        "n_races": n_races,
        "overall": {
            "wide_return_rate": round(overall_wide_rr, 6),
            "quinella_return_rate": round(overall_quin_rr, 6),
            "wide_hit_rate": round(float(df_bets["hit_wide"].mean()), 6),
            "quinella_hit_rate": round(float(df_bets["hit_quin"].mean()), 6),
        },
        "ev_filtered": ev_filtered,
        "ev_sweep_wide": _df_to_records(sweep_wide),
        "ev_sweep_quinella": _df_to_records(sweep_quin),
        "best_condition": best_condition,
        "calibration": {
            "mean_abs_error": calib_check.get("mean_abs_error"),
            "max_abs_error": calib_check.get("max_abs_error"),
            "n_bins": len(calib_check.get("bins", [])),
        },
        "calibration_comparison": calib_comparison if calib_comparison else None,
        "wide_odds_coverage": wide_odds_coverage,
        "quinella_odds_coverage": quinella_odds_coverage,
        "risk_metrics": risk_metrics_json,
        "divergence_comparison": divergence_comparison_json,
    }

    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / cfg["data"]["features_dir"] / "ev_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")

    # divergence 結果を別ファイルにも保存
    if divergence_results and args.divergence:
        div_out_path = out_path.parent / "ev_results_divergence.json"

        def _clean_div(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, dict):
                return {kk: _clean_div(vv) for kk, vv in v.items()}
            if isinstance(v, list):
                return [_clean_div(x) for x in v]
            return v

        div_json = {
            "divergence_strategy": {
                "strategy_a_ev10": {
                    "ペア選択": "argmax_p_model",
                    **divergence_results["comparison_ev10"]["ev_only"],
                },
                "strategy_c_div0": {
                    "ペア選択": "argmax_log_divergence",
                    "log_divergence_threshold": 0.0,
                    **divergence_results["comparison_ev10"]["div_only"],
                },
                "strategy_d_combined_ev10": {
                    "ペア選択": "argmax_log_divergence",
                    "ev_threshold": 1.0,
                    "log_divergence_threshold": 0.0,
                    **divergence_results["comparison_ev10"]["combined"],
                },
                "strategy_a_ev13": {
                    "ペア選択": "argmax_p_model",
                    **divergence_results["comparison_ev13"]["ev_only"],
                },
                "strategy_d_combined_ev13": {
                    "ペア選択": "argmax_log_divergence",
                    "ev_threshold": 1.3,
                    "log_divergence_threshold": 0.0,
                    **divergence_results["comparison_ev13"]["combined"],
                },
                "log_divergence_sweep": _clean_div(divergence_results["log_divergence_sweep"]),
            }
        }

        with open(div_out_path, "w", encoding="utf-8") as f:
            json.dump(_clean_div(div_json), f, indent=2, ensure_ascii=False)
        print(f"  Divergence results saved: {div_out_path}")


if __name__ == "__main__":
    main()

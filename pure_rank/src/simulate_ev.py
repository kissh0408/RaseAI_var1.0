"""
simulate_ev.py — ワイド・馬連 EV シミュレーション（評価専用）

HR 払戻データと Harville 確率を使い、純粋能力モデルの期待値を検証する。
オッズ・払戻は特徴量に使わない（事後評価のみ）。

強化版:
- EV 閾値スイープ（threshold: 0.8〜1.5）
- レース条件別 ROI（surface_code / distance_category / weather_code）
- キャリブレーション確認（予測確率 vs 実的中率）
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
from predict import _best_wide_pair, compute_race_probabilities

PAIR_KEY = tuple[int, int]
STAKE = 100.0


def _normalize_pair(h1: int, h2: int) -> PAIR_KEY:
    return (min(h1, h2), max(h1, h2))


def _build_wide_odds_lookup(
    years: list[int],
    odds_dir: Path,
) -> dict[str, dict[PAIR_KEY, float]]:
    """WideOdds_YYYY.csv を複数年読み込み、race_id -> {(h1,h2): odds} の辞書を返す。

    Parameters
    ----------
    years : 対象年リスト
    odds_dir : WideOdds CSV が格納されたディレクトリ

    Returns
    -------
    dict[race_id_str, dict[(h1,h2), odds]]
        - race_id_str: str 16桁（int64 を str() 変換したもの）
        - (h1, h2): _normalize_pair() で正規化（小さい馬番が先頭）
        - odds: float（事前オッズ）

    除外条件
    --------
    - odds_status != "ok" の行
    - odds が NaN の行
    - CSV ファイルが存在しない年（警告を出してスキップ）
    """
    lookup: dict[str, dict[PAIR_KEY, float]] = {}
    for year in years:
        path = odds_dir / f"WideOdds_{year}.csv"
        if not path.exists():
            print(f"  [warn] WideOdds_{year}.csv not found, skipping")
            continue
        df = pd.read_csv(path)
        df = df[(df["odds_status"] == "ok") & df["odds"].notna()].copy()
        df["race_id_str"] = df["race_id"].apply(lambda x: str(int(x)))
        df["h_min"] = df[["horse_num_1", "horse_num_2"]].min(axis=1).astype(int)
        df["h_max"] = df[["horse_num_1", "horse_num_2"]].max(axis=1).astype(int)
        df["pair_key"] = list(zip(df["h_min"], df["h_max"]))
        for rid, grp in df.groupby("race_id_str"):
            lookup[rid] = dict(zip(grp["pair_key"], grp["odds"].astype(float)))
    print(f"  WideOdds loaded: {len(lookup):,} races across {years}")
    return lookup


def _build_hr_lookup(hr_df: pd.DataFrame, bet_type: str) -> dict[str, dict[PAIR_KEY, int]]:
    """race_id -> {(h1,h2): payout} の辞書を構築。"""
    sub = hr_df[hr_df["bet_type"] == bet_type]
    lookup: dict[str, dict[PAIR_KEY, int]] = {}
    for _, row in sub.iterrows():
        rid = str(row["race_id"])
        key = _normalize_pair(int(row["horse_num_1"]), int(row["horse_num_2"]))
        lookup.setdefault(rid, {})[key] = int(row["payout"])
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
) -> pd.DataFrame:
    """
    テストセット全レース分のベット情報を1行1レースの DataFrame として返す。

    ワイド EV は WideOdds 事前オッズを使った真の期待値で計算する:
      EV_wide = P_wide x wide_odds（wide_odds_lookup が None なら HR 払戻フォールバック）
    オッズが取得できないレースは EV_wide = NaN とする。
    """
    if wide_odds_lookup is None:
        wide_odds_lookup = {}

    df = df_test.copy()
    df["pred_score"] = predictions

    wide_hr_lookup = _build_hr_lookup(hr_df, "wide")
    quin_lookup = _build_hr_lookup(hr_df, "quinella")

    # HR フォールバック用
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
        wide_key = _normalize_pair(int(horse_nums[wi]), int(horse_nums[wj]))
        quin_key = _normalize_pair(int(horse_nums[qi]), int(horse_nums[qj]))
        p_wide = float(probs["wide_matrix"][wi, wj])
        p_quin = float(probs["quinella_matrix"][qi, qj])

        wide_payout = int(wide_hr_lookup.get(rid, {}).get(wide_key, 0))
        quin_payout = int(quin_lookup.get(rid, {}).get(quin_key, 0))

        # ワイド: WideOdds 事前オッズによる真の EV（取得できない場合は NaN）
        prior_odds_wide = wide_odds_lookup.get(rid, {}).get(wide_key, None)
        ev_wide = (p_wide * prior_odds_wide) if prior_odds_wide is not None else float("nan")

        # 馬連: HR 払戻フォールバック
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
) -> dict:
    """テストセットで Harville 最大 P_wide / P_quinella 戦略の回収率を計算。

    後方互換: 以前の呼び出しインターフェースを維持しつつ拡張結果を返す。
    """
    df_bets = _collect_bets_per_race(df_test, predictions, hr_df, T_opt)

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
        "--models-dir",
        type=str,
        default=None,
        help="モデルディレクトリのパス（省略時は train_config.json の models_dir を使用）",
    )
    args = parser.parse_args()

    cfg = load_config()
    T_opt = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    hr_path = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "HR_preprocessed.parquet"
    if args.models_dir:
        models_dir = PROJECT_ROOT / args.models_dir
    else:
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

    # --- WideOdds 事前オッズの読み込み
    test_years = sorted(df_test["race_date"].dt.year.unique().tolist())
    odds_dir = PROJECT_ROOT / "common" / "data" / "output" / "odds"
    print(f"\nLoading WideOdds for years: {test_years}")
    wide_odds_lookup = _build_wide_odds_lookup(test_years, odds_dir)

    feature_cols = get_feature_cols(df_test, cfg)
    models = load_models(models_dir)
    preds = ensemble_predict(models, df_test[feature_cols])

    print(f"\nCollecting per-race bets (T_opt={T_opt})...")
    df_bets = _collect_bets_per_race(df_test, preds, hr_df, T_opt, wide_odds_lookup)
    print(f"  Collected {len(df_bets):,} race-bets")

    # --- EV=NaN 率の集計・報告
    n_ev_na = int(df_bets["ev_wide"].isna().sum())
    n_total = len(df_bets)
    print(f"  EV=NaN (no odds): {n_ev_na}/{n_total} ({n_ev_na/n_total*100:.1f}%)")

    # ─── 全体統計 ─────────────────────────────────────────────────────────────
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

    # ─── EV=1.0 フィルタ後 ────────────────────────────────────────────────────
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

    # ─── EV 閾値スイープ ────────────────────────────────────────────────────────
    print(f"\n--- EV Threshold Sweep (wide) ---")
    sweep_wide = ev_threshold_sweep(df_bets, args.ev_thresholds, bet_type="wide")
    print(sweep_wide.to_string(index=False))

    print(f"\n--- EV Threshold Sweep (quinella) ---")
    sweep_quin = ev_threshold_sweep(df_bets, args.ev_thresholds, bet_type="quin")
    print(sweep_quin.to_string(index=False))

    # ─── 条件別 ROI ────────────────────────────────────────────────────────────
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

    # ─── キャリブレーション ───────────────────────────────────────────────────
    print(f"\n--- Calibration Check (wide) ---")
    calib = check_calibration(df_bets, n_bins=10)
    if calib["bins"]:
        print(f"  mean_abs_error: {calib['mean_abs_error']:.4f}")
        print(f"  max_abs_error : {calib['max_abs_error']:.4f}")
        for b in calib["bins"]:
            print(f"  bin={b['bin']:2d} n={b['n']:5d} pred={b['predicted_prob']:.4f} actual={b['actual_hit_rate']:.4f} diff={b['diff']:+.4f}")

    # ─── charts/ に calibration.json を保存 ───────────────────────────────────
    charts_dir = PROJECT_ROOT / cfg["data"]["features_dir"] / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    calib_path = charts_dir / "calibration_wide.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"\n  Calibration saved: {calib_path}")

    # ─── ev_results.json を拡張形式で保存 ────────────────────────────────────
    def _to_json(v):
        if isinstance(v, (np.floating, float)):
            return float(v) if not np.isnan(v) else None
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v

    def _df_to_records(df: pd.DataFrame) -> list[dict]:
        return [{k: _to_json(v) for k, v in row.items()} for row in df.to_dict("records")]

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
            "mean_abs_error": calib.get("mean_abs_error"),
            "max_abs_error": calib.get("max_abs_error"),
            "n_bins": len(calib.get("bins", [])),
        },
    }

    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / cfg["data"]["features_dir"] / "ev_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()

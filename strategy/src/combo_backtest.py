"""
combo_backtest.py — 馬連・ワイドのバックテスト実装

NOTE: v24 rank バックテスト実験用に実装。現行 strategy/src/backtest.py では未使用。
      参照元スクリプト（run_rank1_v24_backtest.py）は実験完了につき削除済み。

evaluation.csv（モデル予測スコア）と QuinellaOdds_*.csv / WideOdds_*.csv を組み合わせ、
Harville式ペア確率を計算し、Kelly基準でベットサイジングを行う。

設計上の注意:
- オッズ列は evaluation.csv の odds（単勝）を使わず、専用の組み合わせオッズファイルを参照する。
  これにより「予測時点で確定していないオッズを説明変数に使う」禁止を遵守する。
- 的中判定はfinish_rankのみで行い、モデルスコアを的中条件に組み込まない。
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from strategy.src.ev_filters import harville_wide_pair_prob
except ModuleNotFoundError:
    from ev_filters import harville_wide_pair_prob


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _softmax(arr: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """数値安定なsoftmax。temperatureで分布の鋭さを調整する。"""
    shifted = arr - arr.max()
    exp_vals = np.exp(shifted / max(temperature, 1e-12))
    total = exp_vals.sum()
    if total < 1e-30:
        # 全要素が同一の縮退ケース: 一様分布にフォールバック
        return np.ones_like(arr, dtype=float) / max(len(arr), 1)
    return exp_vals / total


def _single_kelly_fraction(prob: float, odds: float) -> float:
    """
    単純Kelly比率: f = (b*p - q) / b
    b = odds - 1.0（純利益倍率）
    負になる場合は0を返す（ベットしない）。
    """
    b = max(odds - 1.0, 1e-12)
    q = 1.0 - prob
    return max((b * prob - q) / b, 0.0)


def _detect_years_in_eval(eval_df: pd.DataFrame) -> List[int]:
    """evaluation.csv に含まれる年を valid_year 列から取得する。"""
    if "valid_year" in eval_df.columns:
        return sorted(eval_df["valid_year"].dropna().unique().astype(int).tolist())
    # フォールバック: race_id の先頭4桁から推定
    race_ids = eval_df["race_id"].astype(str)
    years = sorted(set(int(rid[:4]) for rid in race_ids if len(rid) >= 4))
    return years


# ---------------------------------------------------------------------------
# load_combo_odds
# ---------------------------------------------------------------------------

def load_combo_odds(
    odds_dir: Union[str, Path],
    years: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    QuinellaOdds_*.csv / WideOdds_*.csv を読み込み、正規化済みDataFrameを返す。

    Parameters
    ----------
    odds_dir : str | Path
        common/data/output/odds/ のパス
    years : list[int] | None
        対象年リスト。None のときディレクトリ内の全年を自動検出する。

    Returns
    -------
    quinella_df : pd.DataFrame
        columns = [race_id, horse_num_1, horse_num_2, quinella_odds]
        horse_num_1 < horse_num_2 に正規化済み
    wide_df : pd.DataFrame
        columns = [race_id, horse_num_1, horse_num_2, wide_odds]
        同上
    """
    odds_dir = Path(odds_dir)

    # 自動年検出: QuinellaOdds_YYYY.csv のYYYY部分を収集
    if years is None:
        detected: List[int] = []
        for f in odds_dir.glob("QuinellaOdds_*.csv"):
            stem = f.stem  # "QuinellaOdds_2023"
            try:
                year = int(stem.split("_")[-1])
                detected.append(year)
            except ValueError:
                pass
        years = sorted(detected)
        if not years:
            raise FileNotFoundError(
                f"QuinellaOdds_*.csv が見つかりません: {odds_dir}"
            )

    quinella_frames: List[pd.DataFrame] = []
    wide_frames: List[pd.DataFrame] = []

    read_cols_q = ["race_id", "horse_num_1", "horse_num_2", "odds", "odds_status"]
    read_cols_w = ["race_id", "horse_num_1", "horse_num_2", "odds", "odds_status"]

    for year in sorted(years):
        q_path = odds_dir / f"QuinellaOdds_{year}.csv"
        w_path = odds_dir / f"WideOdds_{year}.csv"

        # --- 馬連 ---
        if q_path.exists():
            try:
                raw_q = pd.read_csv(
                    q_path,
                    usecols=lambda c: c in read_cols_q,
                    dtype={"race_id": "int64", "horse_num_1": "int64", "horse_num_2": "int64"},
                    low_memory=False,
                    encoding="utf-8-sig",
                )
                # odds_status フィルタ
                if "odds_status" in raw_q.columns:
                    raw_q = raw_q[raw_q["odds_status"] == "ok"].copy()
                raw_q = raw_q.rename(columns={"odds": "quinella_odds"})
                raw_q = raw_q[["race_id", "horse_num_1", "horse_num_2", "quinella_odds"]]
                # 馬番正規化: 常に horse_num_1 < horse_num_2
                swap_mask = raw_q["horse_num_1"] > raw_q["horse_num_2"]
                raw_q.loc[swap_mask, ["horse_num_1", "horse_num_2"]] = (
                    raw_q.loc[swap_mask, ["horse_num_2", "horse_num_1"]].values
                )
                quinella_frames.append(raw_q)
            except Exception as e:
                warnings.warn(f"[load_combo_odds] {q_path} 読み込み失敗: {e}")
        else:
            warnings.warn(f"[load_combo_odds] ファイルが存在しません（スキップ）: {q_path}")

        # --- ワイド ---
        if w_path.exists():
            try:
                raw_w = pd.read_csv(
                    w_path,
                    usecols=lambda c: c in read_cols_w,
                    dtype={"race_id": "int64", "horse_num_1": "int64", "horse_num_2": "int64"},
                    low_memory=False,
                    encoding="utf-8-sig",
                )
                if "odds_status" in raw_w.columns:
                    raw_w = raw_w[raw_w["odds_status"] == "ok"].copy()
                raw_w = raw_w.rename(columns={"odds": "wide_odds"})
                raw_w = raw_w[["race_id", "horse_num_1", "horse_num_2", "wide_odds"]]
                swap_mask = raw_w["horse_num_1"] > raw_w["horse_num_2"]
                raw_w.loc[swap_mask, ["horse_num_1", "horse_num_2"]] = (
                    raw_w.loc[swap_mask, ["horse_num_2", "horse_num_1"]].values
                )
                wide_frames.append(raw_w)
            except Exception as e:
                warnings.warn(f"[load_combo_odds] {w_path} 読み込み失敗: {e}")
        else:
            warnings.warn(f"[load_combo_odds] ファイルが存在しません（スキップ）: {w_path}")

    if not quinella_frames:
        raise RuntimeError(
            f"有効な QuinellaOdds ファイルが見つかりませんでした。years={years}"
        )
    if not wide_frames:
        raise RuntimeError(
            f"有効な WideOdds ファイルが見つかりませんでした。years={years}"
        )

    quinella_df = pd.concat(quinella_frames, ignore_index=True)
    wide_df = pd.concat(wide_frames, ignore_index=True)

    return quinella_df, wide_df


# ---------------------------------------------------------------------------
# _build_pair_candidates
# ---------------------------------------------------------------------------

def _build_pair_candidates(
    race_df: pd.DataFrame,
    *,
    pair_top_n: int,
    wide_top_n: int,
    rank2_blend: float,
) -> List[Dict[str, Any]]:
    """
    1レース分のDataFrameからHarvilleベースの馬連・ワイド候補ペアを生成する。

    Parameters
    ----------
    race_df : pd.DataFrame
        columns に horse_num, pred_rank1, pred_rank2 が必要
    pair_top_n : int
        馬連用パートナー候補数
    wide_top_n : int
        ワイド用パートナー候補数（pair_top_n との大きい方を採用）
    rank2_blend : float
        Harville確率と Direct確率のブレンド比率（0=完全Harville, 1=完全Direct）

    Returns
    -------
    list of dict
        {horse_num_1, horse_num_2, quinella_prob, wide_prob, rank}
    """
    if len(race_df) < 2:
        return []

    scores_r1 = race_df["pred_rank1"].to_numpy(dtype=float)
    scores_r2 = race_df["pred_rank2"].to_numpy(dtype=float)
    horse_nums = race_df["horse_num"].to_numpy(dtype=int)

    scores_r1 = np.nan_to_num(scores_r1, nan=0.0).clip(min=0.0)
    scores_r2 = np.nan_to_num(scores_r2, nan=0.0)

    # pred_rank1 はキャリブレーション済み確率 → sum正規化のみ（softmax不要）
    r1_sum = scores_r1.sum()
    harville_prob = scores_r1 / r1_sum if r1_sum > 1e-12 else np.ones(len(scores_r1)) / len(scores_r1)

    # pred_rank2 は raw スコア → softmaxで確率化
    rank2_prob = _softmax(scores_r2)

    # アンカー馬: harville_prob が最大の馬
    anchor_idx = int(np.argmax(harville_prob))

    # パートナー候補: アンカー以外を rank2_prob 降順でソート
    partner_indices = [i for i in range(len(horse_nums)) if i != anchor_idx]
    partner_indices_sorted = sorted(
        partner_indices, key=lambda i: rank2_prob[i], reverse=True
    )

    top_n = max(pair_top_n, wide_top_n)
    partner_indices_sorted = partner_indices_sorted[:top_n]

    p1 = float(harville_prob[anchor_idx])
    h1_num = int(horse_nums[anchor_idx])

    # 全頭の単勝Harville確率辞書（harville_wide_pair_prob が全頭を参照するため必要）
    p_dict_all: Dict[int, float] = {
        int(horse_nums[i]): float(harville_prob[i]) for i in range(len(horse_nums))
    }

    results: List[Dict[str, Any]] = []
    for rank, pidx in enumerate(partner_indices_sorted, start=1):
        p2 = float(harville_prob[pidx])
        r2 = float(rank2_prob[pidx])
        h2_num = int(horse_nums[pidx])

        # ---- Harville馬連確率 ----
        # P(A1st, B2nd) + P(B1st, A2nd)
        # = p1*p2/(1-p1) + p2*p1/(1-p2)  ただし分母がゼロの場合はフォールバック
        denom1 = max(1.0 - p1, 1e-12)
        denom2 = max(1.0 - p2, 1e-12)
        harville_quinella = p1 * p2 / denom1 + p2 * p1 / denom2

        # ---- Direct確率（rank2スコアを使った直接推定） ----
        # h1がアンカーなので: P(h1→h2) + P(h2→h1)
        p_h1_win_h2_place = p1 * r2          # h1が1着、h2が2着
        r1 = float(rank2_prob[anchor_idx])
        p_h2_win_h1_place = p2 * r1          # h2が1着、h1が2着
        direct_quinella = p_h1_win_h2_place + p_h2_win_h1_place

        # ---- ブレンド ----
        blended_quinella = float(
            (1.0 - rank2_blend) * harville_quinella + rank2_blend * direct_quinella
        )
        blended_quinella = float(np.clip(blended_quinella, 0.0, 1.0))

        # ---- ワイド確率（ev_filters の正確なHarville式を使用） ----
        # 固定係数近似（quinella × 2.5）は「3着以内の確率が馬連の2.5倍」という
        # 荒い経験則であり、頭数・オッズ構造によって大きく外れる。
        # 全頭の単勝確率を使うHarville展開の方がモデルの確率構造と整合する。
        wide_prob = harville_wide_pair_prob(p_dict_all, h1_num, h2_num)

        results.append(
            {
                "horse_num_1": min(h1_num, h2_num),
                "horse_num_2": max(h1_num, h2_num),
                "quinella_prob": blended_quinella,
                "wide_prob": wide_prob,
                "rank": rank,
            }
        )

    return results


# ---------------------------------------------------------------------------
# run_combo_backtest
# ---------------------------------------------------------------------------

def run_combo_backtest(
    eval_df: pd.DataFrame,
    config: Any,
    odds_dir: Union[str, Path],
    *,
    pair_top_n: int = 2,
    wide_top_n: int = 2,
    rank2_blend: float = 0.35,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    馬連・ワイドのバックテストを実行する。

    Parameters
    ----------
    eval_df : pd.DataFrame
        evaluation.csv の内容。必須列:
        race_id, horse_num, finish_rank, pred_rank1, pred_rank2, valid_year, race_num
    config : StrategyConfig or dict-like
        StrategyConfig インスタンス、または以下のキーを持つ dict:
        min_edge, fractional_kelly, max_stake_per_bet, bet_unit,
        base_slippage, initial_bankroll
    odds_dir : str | Path
        QuinellaOdds_*.csv / WideOdds_*.csv が格納されたディレクトリ

    Returns
    -------
    bet_df : pd.DataFrame
        各ベット行の詳細
    summary : dict
        馬連・ワイド別の集計結果
    """
    odds_dir = Path(odds_dir)

    # ---- config 正規化（dict / dataclass 両対応） ----
    def _get(attr: str, default: Any) -> Any:
        if isinstance(config, dict):
            return config.get(attr, default)
        return getattr(config, attr, default)

    min_edge: float = float(_get("min_edge", 0.10))
    max_expected_value: float = float(_get("max_expected_value", 2.0))
    fractional_kelly: float = float(_get("fractional_kelly", 0.08))
    max_stake_per_bet: int = int(_get("max_stake_per_bet", 3_000))
    max_invest_per_race: int = int(_get("max_invest_per_race", 50_000))
    bet_unit: int = int(_get("bet_unit", 100))
    base_slippage: float = float(_get("base_slippage", 0.01))
    initial_bankroll: float = float(_get("initial_bankroll", 100_000))
    # ワイド専用パラメータ（フォールバック: グローバル値）
    # wide_min_edge: ワイドのMDD改善のため馬連とは独立に設定可能
    # wide_max_stake_per_bet: ワイドの1ベット上限を馬連より低く設定してドローダウンを抑制
    wide_min_edge: float = float(_get("wide_min_edge", min_edge))
    wide_max_stake_per_bet: int = int(_get("wide_max_stake_per_bet", max_stake_per_bet))
    use_fixed_stake: bool = bool(_get("use_fixed_stake", False))

    # ---- 必須列チェック ----
    required_cols = {"race_id", "horse_num", "finish_rank", "pred_rank1", "pred_rank2"}
    missing = required_cols - set(eval_df.columns)
    if missing:
        raise ValueError(
            f"[run_combo_backtest] eval_df に必須列がありません: {sorted(missing)}"
        )

    # ---- オッズファイル読み込み（評価対象年のみ） ----
    years = _detect_years_in_eval(eval_df)
    quinella_df, wide_df = load_combo_odds(odds_dir, years=years)

    # ---- ルックアップ辞書を構築: {(race_id_str, h1, h2): odds}  h1 < h2 ----
    # race_id はstr統一。eval_df の race_id が int64 / str どちらでも正しくヒットさせるため
    # オッズ辞書側のキーを str に正規化する（int64化で型不一致が生じる問題の根本対処）。
    quinella_dict: Dict[Tuple[str, int, int], float] = {
        (str(int(r.race_id)), int(r.horse_num_1), int(r.horse_num_2)): float(r.quinella_odds)
        for r in quinella_df.itertuples(index=False)
    }
    wide_dict: Dict[Tuple[str, int, int], float] = {
        (str(int(r.race_id)), int(r.horse_num_1), int(r.horse_num_2)): float(r.wide_odds)
        for r in wide_df.itertuples(index=False)
    }

    # ---- レース単位処理 ----
    bet_rows: List[Dict[str, Any]] = []
    eval_df = eval_df.copy()
    # race_id を str に統一（辞書キーと型を合わせる）。
    # 数値として解釈できる場合はint経由でstr化（末尾".0"を除去）、
    # それ以外はそのままstr化する。これにより float64/int64/str いずれでも
    # オッズ辞書のキー（str）と正しくマッチする。
    _rid_numeric = pd.to_numeric(eval_df["race_id"], errors="coerce")
    eval_df["race_id"] = np.where(
        _rid_numeric.notna(),
        _rid_numeric.fillna(0).astype("int64").astype(str),
        eval_df["race_id"].astype(str),
    )

    for race_id, race in eval_df.groupby("race_id", sort=True):
        race_id_int = int(race_id)
        valid_year = int(race["valid_year"].iloc[0]) if "valid_year" in race.columns else 0
        race_num_val = int(race["race_num"].iloc[0]) if "race_num" in race.columns else 0

        # 的中判定用: finish_rank → set
        rank_map: Dict[int, int] = {
            int(r.horse_num): int(r.finish_rank)
            for r in race.itertuples(index=False)
        }

        # ペア候補生成
        candidates = _build_pair_candidates(
            race,
            pair_top_n=pair_top_n,
            wide_top_n=wide_top_n,
            rank2_blend=rank2_blend,
        )

        race_invest = 0  # レース単位の累積投資額（max_invest_per_race の追跡用）
        for cand in candidates:
            h1: int = cand["horse_num_1"]
            h2: int = cand["horse_num_2"]
            rank_order: int = cand["rank"]
            q_prob: float = cand["quinella_prob"]
            w_prob: float = cand["wide_prob"]

            key = (str(race_id_int), h1, h2)

            # ======= 馬連 =======
            q_odds_raw = quinella_dict.get(key, float("nan"))
            if not math.isnan(q_odds_raw) and rank_order <= pair_top_n:
                q_odds_eff = max(q_odds_raw * (1.0 - base_slippage), 1.01)
                q_ev = q_prob * q_odds_eff
                q_edge = q_ev - 1.0

                if use_fixed_stake or (q_edge >= min_edge and q_ev <= max_expected_value):
                    if use_fixed_stake:
                        kf_adj = 0.0
                        target_stake = float(bet_unit)
                    else:
                        kf = _single_kelly_fraction(q_prob, q_odds_eff)
                        kf_adj = kf * fractional_kelly
                        target_stake = initial_bankroll * kf_adj
                    remaining = max(max_invest_per_race - race_invest, 0)
                    capped_stake = min(target_stake, max_stake_per_bet, remaining)
                    actual_stake = int(capped_stake // bet_unit) * bet_unit

                    if actual_stake >= bet_unit:
                        race_invest += actual_stake
                        # 的中: h1, h2 の両方が2着以内かつ異なる着順（同着順は理論上不可）
                        r_h1 = rank_map.get(h1, 99)
                        r_h2 = rank_map.get(h2, 99)
                        is_hit = int(r_h1 in {1, 2} and r_h2 in {1, 2} and r_h1 != r_h2)
                        payout = float(q_odds_eff * actual_stake) if is_hit else 0.0
                        profit = payout - actual_stake

                        bet_rows.append(
                            {
                                "ticket_type": "quinella",
                                "race_id": race_id_int,
                                "horse_num_1": h1,
                                "horse_num_2": h2,
                                "ticket": f"{h1}-{h2}",
                                "pred_prob": q_prob,
                                "quinella_prob": q_prob,
                                "wide_prob": w_prob,
                                "odds_raw": q_odds_raw,
                                "odds_effective": q_odds_eff,
                                "expected_value": q_ev,
                                "edge": q_edge,
                                "kelly_fraction": kf_adj,
                                "suggested_stake": float(target_stake),
                                "actual_stake": float(actual_stake),
                                "is_hit": is_hit,
                                "payout": payout,
                                "profit": profit,
                                "valid_year": valid_year,
                                "race_num": race_num_val,
                            }
                        )

            # ======= ワイド =======
            w_odds_raw = wide_dict.get(key, float("nan"))
            if not math.isnan(w_odds_raw) and rank_order <= wide_top_n:
                w_odds_eff = max(w_odds_raw * (1.0 - base_slippage), 1.01)
                w_ev = w_prob * w_odds_eff
                w_edge = w_ev - 1.0

                if use_fixed_stake or (w_edge >= wide_min_edge and w_ev <= max_expected_value):
                    if use_fixed_stake:
                        kf_adj = 0.0
                        target_stake = float(bet_unit)
                    else:
                        kf = _single_kelly_fraction(w_prob, w_odds_eff)
                        kf_adj = kf * fractional_kelly
                        target_stake = initial_bankroll * kf_adj
                    remaining = max(max_invest_per_race - race_invest, 0)
                    capped_stake = min(target_stake, wide_max_stake_per_bet, remaining)
                    actual_stake = int(capped_stake // bet_unit) * bet_unit

                    if actual_stake >= bet_unit:
                        race_invest += actual_stake
                        # 的中: h1, h2 の両方が3着以内
                        r_h1 = rank_map.get(h1, 99)
                        r_h2 = rank_map.get(h2, 99)
                        is_hit = int(r_h1 <= 3 and r_h2 <= 3)
                        payout = float(w_odds_eff * actual_stake) if is_hit else 0.0
                        profit = payout - actual_stake

                        bet_rows.append(
                            {
                                "ticket_type": "wide",
                                "race_id": race_id_int,
                                "horse_num_1": h1,
                                "horse_num_2": h2,
                                "ticket": f"{h1}-{h2}",
                                "pred_prob": w_prob,
                                "quinella_prob": q_prob,
                                "wide_prob": w_prob,
                                "odds_raw": w_odds_raw,
                                "odds_effective": w_odds_eff,
                                "expected_value": w_ev,
                                "edge": w_edge,
                                "kelly_fraction": kf_adj,
                                "suggested_stake": float(target_stake),
                                "actual_stake": float(actual_stake),
                                "is_hit": is_hit,
                                "payout": payout,
                                "profit": profit,
                                "valid_year": valid_year,
                                "race_num": race_num_val,
                            }
                        )

    # ---- bet_df 構築 ----
    bet_df = pd.DataFrame(bet_rows) if bet_rows else pd.DataFrame(
        columns=[
            "ticket_type", "race_id", "horse_num_1", "horse_num_2", "ticket",
            "pred_prob", "quinella_prob", "wide_prob",
            "odds_raw", "odds_effective", "expected_value", "edge",
            "kelly_fraction", "suggested_stake", "actual_stake",
            "is_hit", "payout", "profit",
            "valid_year", "race_num",
        ]
    )

    # ---- サマリー集計 ----
    def _calc_type_summary(sub: pd.DataFrame) -> Dict[str, Any]:
        """馬連 or ワイドのサブセットに対して基本指標を計算する。"""
        if sub.empty:
            return {
                "n_bets": 0, "n_hits": 0, "hit_rate": 0.0,
                "invest": 0.0, "return": 0.0, "roi": 0.0,
            }
        n_bets = len(sub)
        n_hits = int(sub["is_hit"].sum())
        invest = float(sub["actual_stake"].sum())
        returned = float(sub["payout"].sum())
        roi = returned / invest if invest > 0 else 0.0
        return {
            "n_bets": n_bets,
            "n_hits": n_hits,
            "hit_rate": n_hits / n_bets if n_bets > 0 else 0.0,
            "invest": invest,
            "return": returned,
            "roi": roi,
        }

    def _yearly_roi(sub: pd.DataFrame) -> Dict[int, float]:
        """年別ROIを計算する。"""
        result: Dict[int, float] = {}
        if sub.empty or "valid_year" not in sub.columns:
            return result
        for yr, grp in sub.groupby("valid_year"):
            invest = float(grp["actual_stake"].sum())
            returned = float(grp["payout"].sum())
            result[int(yr)] = returned / invest if invest > 0 else 0.0
        return result

    q_sub = bet_df[bet_df["ticket_type"] == "quinella"] if not bet_df.empty else pd.DataFrame()
    w_sub = bet_df[bet_df["ticket_type"] == "wide"] if not bet_df.empty else pd.DataFrame()

    q_stats = _calc_type_summary(q_sub)
    w_stats = _calc_type_summary(w_sub)

    summary: Dict[str, Any] = {
        "n_bets_quinella": q_stats["n_bets"],
        "n_hits_quinella": q_stats["n_hits"],
        "hit_rate_quinella": q_stats["hit_rate"],
        "invest_quinella": q_stats["invest"],
        "return_quinella": q_stats["return"],
        "roi_quinella": q_stats["roi"],
        "n_bets_wide": w_stats["n_bets"],
        "n_hits_wide": w_stats["n_hits"],
        "hit_rate_wide": w_stats["hit_rate"],
        "invest_wide": w_stats["invest"],
        "return_wide": w_stats["return"],
        "roi_wide": w_stats["roi"],
        "yearly_quinella": _yearly_roi(q_sub),
        "yearly_wide": _yearly_roi(w_sub),
    }

    return bet_df, summary


def compare_l1_l2_wide_divergence(
    eval_df: pd.DataFrame,
    odds_dir: Union[str, Path],
    *,
    valid_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    score_col: str = "pred_rank1",
    model_prob_col: str = "model_prob",
    ev_threshold: float = 1.05,
    div_threshold: float = 0.0,
    finish_rank_col: str = "finish_rank",
) -> Dict[str, Any]:
    """
    L1 (pred_rank1 Harville) vs L2 (model_prob Harville) wide Strategy D backtest.

    Uses WideOdds CSV + finish_rank for hit detection (3rd place or better both horses).
    Intended for evaluation.csv walk-forward reports; not used in pure_rank features.
    """
    import sys

    root = Path(__file__).resolve().parents[2]
    for sub in ("strategy/src", "pure_rank/src"):
        p = root / sub
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from wide_ev_core import collect_divergence_bets_per_race, load_wide_odds_lookup
    from wide_probability import wide_probs_from_model_prob_frame

    odds_dir = Path(odds_dir)
    years = _detect_years_in_eval(eval_df)
    wide_lookup = load_wide_odds_lookup(years, odds_dir, odds_type="Wide")
    if not wide_lookup:
        return {"status": "skipped", "reason": "no_wide_odds_csv"}

    df = eval_df.copy()
    df["race_id"] = df["race_id"].astype(str)
    if "valid_year" not in df.columns:
        df["valid_year"] = df["race_id"].str[:4].astype(int)

    if valid_years is None:
        valid_years = [int(y) for y in sorted(df["valid_year"].unique()) if int(y) <= 2024]
    if test_years is None:
        test_years = [int(y) for y in sorted(df["valid_year"].unique()) if int(y) >= 2025]

    def _hit_wide(race_grp: pd.DataFrame, pair: Tuple[int, int]) -> int:
        ranks = {
            int(r["horse_num"]): int(r[finish_rank_col])
            for _, r in race_grp.iterrows()
            if pd.notna(r.get(finish_rank_col))
        }
        h1, h2 = pair
        if h1 not in ranks or h2 not in ranks:
            return 0
        return int(ranks[h1] <= 3 and ranks[h2] <= 3)

    def _run_source(sub: pd.DataFrame, source: str) -> Dict[str, Any]:
        rows: List[dict] = []
        for race_id, grp in sub.groupby("race_id"):
            if len(grp) < 2:
                continue
            horses = [int(h) for h in grp["horse_num"].astype(int)]
            if source == "L2" and model_prob_col in grp.columns:
                mp = pd.to_numeric(grp[model_prob_col], errors="coerce").fillna(0.0).values
                p_map = wide_probs_from_model_prob_frame(horses, mp)
            else:
                if score_col not in grp.columns:
                    continue
                s = pd.to_numeric(grp[score_col], errors="coerce").fillna(0.0).values
                total = s.sum()
                p_win = s / total if total > 1e-12 else np.ones(len(s)) / len(s)
                p_dict = {horses[i]: float(p_win[i]) for i in range(len(horses))}
                p_map = {}
                for i in range(len(horses)):
                    for j in range(i + 1, len(horses)):
                        h1, h2 = horses[i], horses[j]
                        key = (min(h1, h2), max(h1, h2))
                        p_map[key] = harville_wide_pair_prob(p_dict, h1, h2)
            pick = collect_divergence_bets_per_race(
                str(race_id),
                p_map,
                wide_lookup,
                strategy="D",
                ev_threshold=ev_threshold,
                div_threshold=div_threshold,
            )
            if pick is None or not pick.get("bet"):
                continue
            pair = pick["pair"]
            hit = _hit_wide(grp, pair)
            payout_mult = float(pick["wide_odds"]) if hit else 0.0
            rows.append({"hit": hit, "payout_mult": payout_mult, "ev_wide": pick["ev_wide"]})

        if not rows:
            return {"n_bets": 0, "roi": None, "hit_rate": None}
        n = len(rows)
        ret = sum(r["payout_mult"] for r in rows)
        hits = sum(r["hit"] for r in rows)
        return {"n_bets": n, "roi": ret / n, "hit_rate": hits / n}

    out: Dict[str, Any] = {"status": "ok", "ev_threshold": ev_threshold, "div_threshold": div_threshold}
    for label, years_list in (("valid", valid_years), ("test", test_years)):
        sub = df[df["valid_year"].isin(years_list)]
        out[label] = {
            "L1": _run_source(sub, "L1"),
            "L2": _run_source(sub, "L2") if model_prob_col in df.columns else {"status": "skipped"},
        }
    v1 = out.get("valid", {}).get("L1", {})
    v2 = out.get("valid", {}).get("L2", {})
    s1 = (v1.get("roi") or 0) * (v1.get("n_bets") or 0) ** 0.5
    s2 = (v2.get("roi") or 0) * (v2.get("n_bets") or 0) ** 0.5 if isinstance(v2, dict) and v2.get("n_bets") else -1
    out["recommended_prob_source"] = "L1" if s1 >= s2 else "L2"
    return out

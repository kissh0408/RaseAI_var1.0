"""
betting_strategy.py — 予測スコアからの買い目決定ロジック

入力:
    scores: [{"horse_num": int, "rank_score": float, "ev_score": float}]
    quinella_odds: {(h1, h2): float}  # 馬連オッズ (h1 < h2)
    wide_odds: {(h1, h2): float}      # ワイドオッズ (h1 < h2)

出力:
    {
        "tansho": [horse_num],          # 単勝
        "umaren": [(h1, h2)],           # 馬連
        "wide": [(h1, h2), ...],        # ワイド
    }
"""

from __future__ import annotations

import itertools
from typing import Any


def _normalize_key(h1: int, h2: int) -> tuple[int, int]:
    """オッズ辞書のキーを正規化 (小さい馬番が先)。"""
    return (min(h1, h2), max(h1, h2))


def decide_bets(
    scores: list[dict],
    quinella_odds: dict[tuple[int, int], float],
    wide_odds: dict[tuple[int, int], float],
    tansho_ev_threshold: float = 0.15,   # 正規化スケールに合わせた閾値
    umaren_odds_threshold: float = 3.0,
) -> dict[str, Any]:
    """
    予測スコアと実際のオッズから買い目を決定する。

    Args:
        scores: [{"horse_num": int, "rank_score": float, "ev_score": float}]
        quinella_odds: 馬連オッズ辞書 {(h1, h2): float} (h1 < h2)
        wide_odds: ワイドオッズ辞書 {(h1, h2): float} (h1 < h2)
        tansho_ev_threshold: 単勝購入の ev_score 最低閾値 (デフォルト: 0.15)
        umaren_odds_threshold: 馬連購入のオッズ最低閾値 (デフォルト: 3.0)

    Returns:
        {
            "tansho": [horse_num],
            "umaren": [(h1, h2)],
            "wide": [(h1, h2), ...],
        }
    """
    if not scores:
        return {"tansho": [], "umaren": [], "wide": []}

    # ev_score 降順でソート
    sorted_by_ev = sorted(scores, key=lambda x: x["ev_score"], reverse=True)
    # rank_score 降順でソート
    sorted_by_rank = sorted(scores, key=lambda x: x["rank_score"], reverse=True)

    # --- 単勝: ev_score最上位 かつ ev_score > threshold ---
    tansho: list[int] = []
    if sorted_by_ev and sorted_by_ev[0]["ev_score"] > tansho_ev_threshold:
        tansho.append(sorted_by_ev[0]["horse_num"])

    # --- 馬連: rank_score上位2頭 かつ 馬連オッズ > threshold ---
    umaren: list[tuple[int, int]] = []
    if len(sorted_by_rank) >= 2:
        h1 = sorted_by_rank[0]["horse_num"]
        h2 = sorted_by_rank[1]["horse_num"]
        key = _normalize_key(h1, h2)
        q_odds = quinella_odds.get(key, 0.0)
        if q_odds > umaren_odds_threshold:
            umaren.append(key)

    # --- ワイド: rank_score上位3頭のフォーメーション ---
    wide: list[tuple[int, int]] = []
    top3 = [h["horse_num"] for h in sorted_by_rank[:3]]
    if len(top3) >= 2:
        for h1, h2 in itertools.combinations(top3, 2):
            key = _normalize_key(h1, h2)
            # ワイドオッズが存在する組み合わせのみ追加
            if key in wide_odds:
                wide.append(key)

    return {
        "tansho": tansho,
        "umaren": umaren,
        "wide": wide,
    }


def calculate_payout(
    bets: dict[str, Any],
    actual_results: dict[str, Any],
    unit: int = 100,
) -> dict[str, Any]:
    """
    購入した買い目の払い戻しを計算する。

    Args:
        bets: decide_bets() の出力
        actual_results: {
            "winner": int,                      # 1着馬番
            "second": int,                      # 2着馬番
            "third": int,                       # 3着馬番
            "tansho_payout": {horse_num: float},# 単勝払い戻し
            "quinella_payout": {(h1,h2): float},# 馬連払い戻し
            "wide_payout": {(h1,h2): float},   # ワイド払い戻し
        }
        unit: 1票あたりの購入金額 (デフォルト: 100円)

    Returns:
        {
            "spent": int,        # 総購入金額
            "returned": float,   # 総払い戻し
            "profit": float,     # 損益
            "roi": float,        # 払い戻し率
            "detail": {
                "tansho": {"spent": int, "returned": float},
                "umaren": {"spent": int, "returned": float},
                "wide": {"spent": int, "returned": float},
            }
        }
    """
    winner = actual_results.get("winner")
    second = actual_results.get("second")
    third = actual_results.get("third")
    tansho_payout = actual_results.get("tansho_payout", {})
    quinella_payout = actual_results.get("quinella_payout", {})
    wide_payout = actual_results.get("wide_payout", {})

    detail: dict[str, dict] = {
        "tansho": {"spent": 0, "returned": 0.0},
        "umaren": {"spent": 0, "returned": 0.0},
        "wide": {"spent": 0, "returned": 0.0},
    }

    # 単勝
    for hnum in bets.get("tansho", []):
        detail["tansho"]["spent"] += unit
        if hnum == winner:
            detail["tansho"]["returned"] += tansho_payout.get(hnum, 0.0) * unit / 100

    # 馬連
    for combo in bets.get("umaren", []):
        detail["umaren"]["spent"] += unit
        key = _normalize_key(*combo)
        if winner is not None and second is not None:
            actual_key = _normalize_key(winner, second)
            if key == actual_key:
                detail["umaren"]["returned"] += quinella_payout.get(key, 0.0) * unit / 100

    # ワイド (上位3着以内の2頭の組み合わせが的中)
    top3_set = {winner, second, third} - {None}
    for combo in bets.get("wide", []):
        detail["wide"]["spent"] += unit
        key = _normalize_key(*combo)
        h1, h2 = key
        if h1 in top3_set and h2 in top3_set:
            detail["wide"]["returned"] += wide_payout.get(key, 0.0) * unit / 100

    total_spent = sum(v["spent"] for v in detail.values())
    total_returned = sum(v["returned"] for v in detail.values())
    roi = total_returned / total_spent if total_spent > 0 else 0.0

    return {
        "spent": total_spent,
        "returned": total_returned,
        "profit": total_returned - total_spent,
        "roi": roi,
        "detail": detail,
    }

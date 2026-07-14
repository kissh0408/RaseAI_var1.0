"""
fast_eval.py — 推論結果をキャッシュして閾値を高速スイープするスクリプト

使い方:
    # Step1: 推論実行＋キャッシュ保存（約2時間、初回のみ）
    python fast_eval.py --build_cache --year 2026

    # Step2: 閾値スイープ（数秒）
    python fast_eval.py --sweep --year 2026

    # Step3: 特定閾値で詳細確認
    python fast_eval.py --sweep --year 2026 --tansho_ev 0.20 --tansho_rank 0.70 --umaren_odds 10.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

DATA_OUTPUT_DIR = r"C:\Users\syugo\AI\RaceAI\common\data\output"
CACHE_DIR = os.path.join(_THIS_DIR, "eval_cache")


def _cache_path(year: int) -> str:
    return os.path.join(CACHE_DIR, f"scores_{year}.json")


# ------------------------------------------------------------------ #
#  キャッシュ構築                                                      #
# ------------------------------------------------------------------ #

def build_cache(year: int) -> None:
    """推論を実行してスコアと実際の結果をキャッシュに保存する。"""
    from evaluate import _load_quinella_odds, _load_wide_odds, _get_actual_results
    from race_to_text import make_race_id, race_to_prompt
    from inference import load_model, predict_batch

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[INFO] モデルロード中...", flush=True)
    tokenizer, model = load_model()

    se_path = os.path.join(DATA_OUTPUT_DIR, "race_se", f"race_se_{year}.csv")
    ra_path = os.path.join(DATA_OUTPUT_DIR, "race_ra", f"race_ra_{year}.csv")
    df_se = pd.read_csv(se_path, low_memory=False)
    df_ra = pd.read_csv(ra_path, low_memory=False)
    df_se["race_id"] = df_se.apply(make_race_id, axis=1)
    df_ra["race_id"] = df_ra.apply(make_race_id, axis=1)
    ra_indexed = df_ra.set_index("race_id")

    print("[INFO] オッズデータ読み込み中...", flush=True)
    q_odds_map = _load_quinella_odds(year)
    w_odds_map = _load_wide_odds(year)

    valid_races = []
    for race_id in df_se["race_id"].unique():
        if race_id not in ra_indexed.index:
            continue
        horses = df_se[df_se["race_id"] == race_id]
        valid = horses[(horses["abnormal_code"] == 0) & (horses["finish_rank"] > 0)]
        if len(valid) < 2:
            continue
        ra_row = ra_indexed.loc[race_id]
        valid_races.append((race_id, horses, ra_row))

    print(f"[INFO] 有効レース数: {len(valid_races)}", flush=True)

    race_texts = [race_to_prompt(rd[2], rd[1]) for rd in valid_races]
    print("[INFO] バッチ推論開始 (batch_size=8)...", flush=True)
    all_scores = predict_batch(race_texts, model, tokenizer, batch_size=8)

    cache = []
    for (race_id, horses, ra_row), scores in zip(valid_races, all_scores):
        if not scores:
            continue
        q_odds = q_odds_map.get(race_id, {})
        w_odds = w_odds_map.get(race_id, {})
        actual = _get_actual_results(horses, q_odds, w_odds)
        if not actual:
            continue
        # キーをJSON互換に変換
        q_odds_serial = {f"{k[0]},{k[1]}": v for k, v in q_odds.items()}
        w_odds_serial = {f"{k[0]},{k[1]}": v for k, v in w_odds.items()}
        actual_serial = dict(actual)
        for field in ("tansho_payout", "quinella_payout", "wide_payout"):
            actual_serial[field] = {
                f"{k[0]},{k[1]}" if isinstance(k, tuple) else str(k): v
                for k, v in actual.get(field, {}).items()
            }
        cache.append({
            "race_id": race_id,
            "scores": scores,
            "quinella_odds": q_odds_serial,
            "wide_odds": w_odds_serial,
            "actual": actual_serial,
        })

    out_path = _cache_path(year)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"[INFO] キャッシュ保存: {out_path} ({len(cache)}件)", flush=True)


# ------------------------------------------------------------------ #
#  キャッシュ読み込み                                                  #
# ------------------------------------------------------------------ #

def _load_cache(year: int) -> list[dict]:
    path = _cache_path(year)
    if not os.path.exists(path):
        raise FileNotFoundError(f"キャッシュが見つかりません: {path}\n先に --build_cache を実行してください。")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # キーを tuple に戻す
    for entry in data:
        entry["quinella_odds"] = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in entry["quinella_odds"].items()
        }
        entry["wide_odds"] = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in entry["wide_odds"].items()
        }
        for field in ("tansho_payout",):
            entry["actual"][field] = {
                int(k): v for k, v in entry["actual"].get(field, {}).items()
            }
        for field in ("quinella_payout", "wide_payout"):
            entry["actual"][field] = {
                tuple(int(x) for x in k.split(",")): v
                for k, v in entry["actual"].get(field, {}).items()
            }
    return data


# ------------------------------------------------------------------ #
#  単一パラメータでバックテスト                                         #
# ------------------------------------------------------------------ #

def run_backtest_fast(
    cache: list[dict],
    tansho_ev_threshold: float = 0.15,
    tansho_rank_threshold: float = 0.0,
    umaren_odds_threshold: float = 3.0,
    wide_min_odds: float = 0.0,
    unit: int = 100,
) -> dict:
    from betting_strategy import _normalize_key, calculate_payout

    stats = {k: {"bets": 0, "hits": 0, "spent": 0, "returned": 0.0}
             for k in ["tansho", "umaren", "wide"]}

    for entry in cache:
        scores = entry["scores"]
        q_odds = entry["quinella_odds"]
        w_odds = entry["wide_odds"]
        actual = entry["actual"]
        winner = actual.get("winner")
        second = actual.get("second")
        third = actual.get("third")

        sorted_by_ev = sorted(scores, key=lambda x: x["ev_score"], reverse=True)
        sorted_by_rank = sorted(scores, key=lambda x: x["rank_score"], reverse=True)

        # 単勝
        tansho = []
        if sorted_by_ev:
            top_ev = sorted_by_ev[0]
            if (top_ev["ev_score"] > tansho_ev_threshold and
                    top_ev["rank_score"] > tansho_rank_threshold):
                tansho.append(top_ev["horse_num"])

        # 馬連
        umaren = []
        if len(sorted_by_rank) >= 2:
            h1 = sorted_by_rank[0]["horse_num"]
            h2 = sorted_by_rank[1]["horse_num"]
            key = _normalize_key(h1, h2)
            if q_odds.get(key, 0.0) > umaren_odds_threshold:
                umaren.append(key)

        # ワイド
        import itertools
        wide = []
        top3 = [h["horse_num"] for h in sorted_by_rank[:3]]
        if len(top3) >= 2:
            for a, b in itertools.combinations(top3, 2):
                key = _normalize_key(a, b)
                if key in w_odds and w_odds[key] >= wide_min_odds:
                    wide.append(key)

        bets = {"tansho": tansho, "umaren": umaren, "wide": wide}
        payout = calculate_payout(bets, actual, unit=unit)

        top3_set = {winner, second, third} - {None}

        for hnum in tansho:
            stats["tansho"]["bets"] += 1
            stats["tansho"]["spent"] += unit
            if hnum == winner:
                stats["tansho"]["hits"] += 1
        stats["tansho"]["returned"] += payout["detail"]["tansho"]["returned"]

        for combo in umaren:
            stats["umaren"]["bets"] += 1
            stats["umaren"]["spent"] += unit
            if winner and second and set(combo) == {winner, second}:
                stats["umaren"]["hits"] += 1
        stats["umaren"]["returned"] += payout["detail"]["umaren"]["returned"]

        for combo in wide:
            stats["wide"]["bets"] += 1
            stats["wide"]["spent"] += unit
            if set(combo) <= top3_set:
                stats["wide"]["hits"] += 1
        stats["wide"]["returned"] += payout["detail"]["wide"]["returned"]

    def _fmt(s):
        bets = s["bets"]
        hits = s["hits"]
        roi = s["returned"] / s["spent"] if s["spent"] > 0 else 0.0
        hit_rate = hits / bets if bets > 0 else 0.0
        return {"bets": bets, "hits": hits, "hit_rate": hit_rate, "roi": roi}

    return {k: _fmt(stats[k]) for k in stats}


# ------------------------------------------------------------------ #
#  閾値スイープ                                                        #
# ------------------------------------------------------------------ #

def run_sweep(year: int) -> None:
    cache = _load_cache(year)
    print(f"[INFO] キャッシュ読み込み: {len(cache)}件\n")

    # パラメータグリッド
    tansho_ev_list    = [0.12, 0.15, 0.18, 0.20, 0.25]
    tansho_rank_list  = [0.0, 0.50, 0.60, 0.70, 0.75]
    umaren_odds_list  = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    print("=" * 75)
    print(f"{'tansho_ev':>10} {'rank_min':>9} {'umaren_o':>9} | "
          f"{'T-ROI':>7} {'T-hit%':>7} {'T-bets':>7} | "
          f"{'U-ROI':>7} {'U-hit%':>7} | "
          f"{'W-ROI':>7} {'W-hit%':>7}")
    print("=" * 75)

    best = {"tansho": None, "umaren": None, "wide": None}

    for tev in tansho_ev_list:
        for trk in tansho_rank_list:
            for uod in umaren_odds_list:
                r = run_backtest_fast(cache,
                    tansho_ev_threshold=tev,
                    tansho_rank_threshold=trk,
                    umaren_odds_threshold=uod)
                t = r["tansho"]; u = r["umaren"]; w = r["wide"]
                print(f"{tev:>10.2f} {trk:>9.2f} {uod:>9.1f} | "
                      f"{t['roi']:>7.3f} {t['hit_rate']:>7.1%} {t['bets']:>7} | "
                      f"{u['roi']:>7.3f} {u['hit_rate']:>7.1%} | "
                      f"{w['roi']:>7.3f} {w['hit_rate']:>7.1%}")

                if best["tansho"] is None or t["roi"] > best["tansho"]["roi"]:
                    best["tansho"] = {"params": (tev, trk, uod), **t}
                if best["umaren"] is None or u["roi"] > best["umaren"]["roi"]:
                    best["umaren"] = {"params": (tev, trk, uod), **u}

    print("\n===== BEST =====")
    for k, v in best.items():
        if v:
            print(f"  {k}: ROI={v['roi']:.3f}  hit={v['hit_rate']:.1%}  "
                  f"bets={v.get('bets','-')}  params={v['params']}")


# ------------------------------------------------------------------ #
#  main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--build_cache", action="store_true", help="推論実行＋キャッシュ保存")
    parser.add_argument("--sweep", action="store_true", help="閾値スイープ")
    parser.add_argument("--tansho_ev",   type=float, default=0.15)
    parser.add_argument("--tansho_rank", type=float, default=0.0)
    parser.add_argument("--umaren_odds", type=float, default=3.0)
    args = parser.parse_args()

    if args.build_cache:
        build_cache(args.year)
    elif args.sweep:
        run_sweep(args.year)
    else:
        # 単一パラメータで評価
        cache = _load_cache(args.year)
        r = run_backtest_fast(cache,
            tansho_ev_threshold=args.tansho_ev,
            tansho_rank_threshold=args.tansho_rank,
            umaren_odds_threshold=args.umaren_odds)
        print(f"\n===== 評価結果 (ev≥{args.tansho_ev}, rank≥{args.tansho_rank}, umaren_o≥{args.umaren_odds}) =====")
        for k, v in r.items():
            print(f"  {k:8s}: {v['bets']:5d}件  的中{v['hits']:4d}件  "
                  f"的中率={v['hit_rate']:.1%}  ROI={v['roi']:.3f}")


if __name__ == "__main__":
    main()

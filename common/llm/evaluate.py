"""
evaluate.py — 2026年データでのバックテスト評価スクリプト

使用法:
    python evaluate.py
    python evaluate.py --year 2025 --limit 100
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


def _load_quinella_odds(year: int) -> dict[str, dict[tuple[int, int], float]]:
    """
    馬連オッズを race_id → {(h1,h2): odds} 形式で返す。
    CSVには race_id カラムが既にある。odds_status=='ok' のみ使用。
    """
    path = os.path.join(DATA_OUTPUT_DIR, "odds", f"QuinellaOdds_{year}.csv")
    df = pd.read_csv(path, low_memory=False)

    # odds_status が存在する場合は 'ok' のみ使用
    if "odds_status" in df.columns:
        df = df[df["odds_status"] == "ok"]

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        rid = str(row["race_id"])
        h1 = int(row["horse_num_1"])
        h2 = int(row["horse_num_2"])
        key = (min(h1, h2), max(h1, h2))
        if rid not in result:
            result[rid] = {}
        result[rid][key] = float(row["odds"])
    return result


def _load_wide_odds(year: int) -> dict[str, dict[tuple[int, int], float]]:
    """
    ワイドオッズを race_id → {(h1,h2): odds} 形式で返す。
    CSVには race_id カラムが既にある。odds_status=='ok' のみ使用。
    """
    path = os.path.join(DATA_OUTPUT_DIR, "odds", f"WideOdds_{year}.csv")
    df = pd.read_csv(path, low_memory=False)

    # odds_status が存在する場合は 'ok' のみ使用
    if "odds_status" in df.columns:
        df = df[df["odds_status"] == "ok"]

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        rid = str(row["race_id"])
        h1 = int(row["horse_num_1"])
        h2 = int(row["horse_num_2"])
        key = (min(h1, h2), max(h1, h2))
        if rid not in result:
            result[rid] = {}
        result[rid][key] = float(row["odds"])
    return result


def _get_actual_results(
    horses_df: pd.DataFrame,
    quinella_odds_for_race: dict,
    wide_odds_for_race: dict,
) -> dict:
    """
    レース結果と払い戻し情報を整形する。
    払い戻し = オッズ × 100 (100円購入基準)
    """
    valid = horses_df[
        (horses_df["abnormal_code"] == 0) &
        (horses_df["finish_rank"] > 0)
    ].sort_values("finish_rank")

    if valid.empty:
        return {}

    ranks = valid.set_index("finish_rank")["horse_num"].to_dict()
    winner = int(ranks.get(1, -1))
    second = int(ranks.get(2, -1))
    third = int(ranks.get(3, -1))

    # 単勝払い戻し: winner のオッズ × 100
    tansho_row = valid[valid["horse_num"] == winner]
    tansho_payout: dict[int, float] = {}
    if not tansho_row.empty:
        odds_val = int(tansho_row.iloc[0]["odds"]) / 10.0
        tansho_payout[winner] = odds_val * 100

    # 馬連払い戻し
    quinella_payout: dict[tuple, float] = {}
    if winner > 0 and second > 0:
        key = (min(winner, second), max(winner, second))
        if key in quinella_odds_for_race:
            quinella_payout[key] = quinella_odds_for_race[key] * 100

    # ワイド払い戻し
    wide_payout: dict[tuple, float] = {}
    top3 = [h for h in [winner, second, third] if h > 0]
    import itertools
    for h1, h2 in itertools.combinations(top3, 2):
        key = (min(h1, h2), max(h1, h2))
        if key in wide_odds_for_race:
            wide_payout[key] = wide_odds_for_race[key] * 100

    return {
        "winner": winner if winner > 0 else None,
        "second": second if second > 0 else None,
        "third": third if third > 0 else None,
        "tansho_payout": tansho_payout,
        "quinella_payout": quinella_payout,
        "wide_payout": wide_payout,
    }


def run_backtest(
    model,
    tokenizer,
    test_se_path: str,
    test_ra_path: str,
    quinella_odds_path: str,
    wide_odds_path: str,
    unit: int = 100,
    race_limit: int | None = None,
) -> dict:
    """
    バックテストを実行する。

    Args:
        model: LoRAアダプタ適用済みモデル
        tokenizer: トークナイザ
        test_se_path: race_se_{year}.csv のパス
        test_ra_path: race_ra_{year}.csv のパス
        quinella_odds_path: QuinellaOdds_{year}.csv のパス
        wide_odds_path: WideOdds_{year}.csv のパス
        unit: 1票あたりの購入金額 (デフォルト: 100円)
        race_limit: テストするレース数の上限 (None=全て)

    Returns:
        {
            "total_races": int,
            "tansho": {"bets": int, "hits": int, "hit_rate": float, "roi": float},
            "umaren": {"bets": int, "hits": int, "hit_rate": float, "roi": float},
            "wide": {"bets": int, "hits": int, "hit_rate": float, "roi": float},
        }
    """
    from race_to_text import make_race_id, race_to_prompt
    from inference import predict_batch
    from betting_strategy import decide_bets, calculate_payout

    df_se = pd.read_csv(test_se_path, low_memory=False)
    df_ra = pd.read_csv(test_ra_path, low_memory=False)
    df_se["race_id"] = df_se.apply(make_race_id, axis=1)
    df_ra["race_id"] = df_ra.apply(make_race_id, axis=1)
    ra_indexed = df_ra.set_index("race_id")

    print("[INFO] オッズデータ読み込み中...", flush=True)
    year = int(df_se["year"].iloc[0])
    quinella_odds_map = _load_quinella_odds(year)
    wide_odds_map = _load_wide_odds(year)

    race_ids = df_se["race_id"].unique()
    if race_limit:
        race_ids = race_ids[:race_limit]

    # 有効なレースだけ先に収集
    valid_races: list[tuple] = []  # (race_id, horses_df, ra_row)
    for race_id in race_ids:
        if race_id not in ra_indexed.index:
            continue
        horses = df_se[df_se["race_id"] == race_id]
        valid = horses[(horses["abnormal_code"] == 0) & (horses["finish_rank"] > 0)]
        if len(valid) < 2:
            continue
        ra_row = ra_indexed.loc[race_id]
        valid_races.append((race_id, horses, ra_row))

    print(f"[INFO] 有効レース数: {len(valid_races)}", flush=True)

    # バッチ推論
    race_texts = [race_to_prompt(rd[2], rd[1]) for rd in valid_races]
    print("[INFO] バッチ推論開始 (batch_size=8)...", flush=True)
    all_scores = predict_batch(race_texts, model, tokenizer, batch_size=8)

    stats = {
        "tansho": {"bets": 0, "hits": 0, "spent": 0, "returned": 0.0},
        "umaren": {"bets": 0, "hits": 0, "spent": 0, "returned": 0.0},
        "wide": {"bets": 0, "hits": 0, "spent": 0, "returned": 0.0},
    }
    total_races = 0
    skipped = 0

    for (race_id, horses, ra_row), scores in tqdm(
        zip(valid_races, all_scores), total=len(valid_races), desc="集計"
    ):
        if not scores:
            skipped += 1
            continue

        # 買い目決定
        q_odds = quinella_odds_map.get(race_id, {})
        w_odds = wide_odds_map.get(race_id, {})
        bets = decide_bets(scores, q_odds, w_odds)

        # 実際の結果
        actual = _get_actual_results(horses, q_odds, w_odds)
        if not actual:
            skipped += 1
            continue

        # 払い戻し計算
        payout = calculate_payout(bets, actual, unit=unit)

        # 集計
        winner = actual.get("winner")
        second = actual.get("second")
        third = actual.get("third")

        # 単勝的中判定
        for hnum in bets["tansho"]:
            stats["tansho"]["bets"] += 1
            stats["tansho"]["spent"] += unit
            if hnum == winner:
                stats["tansho"]["hits"] += 1
        stats["tansho"]["returned"] += payout["detail"]["tansho"]["returned"]

        # 馬連的中判定
        for combo in bets["umaren"]:
            stats["umaren"]["bets"] += 1
            stats["umaren"]["spent"] += unit
            if winner and second:
                if set(combo) == {winner, second}:
                    stats["umaren"]["hits"] += 1
        stats["umaren"]["returned"] += payout["detail"]["umaren"]["returned"]

        # ワイド的中判定
        top3 = {winner, second, third} - {None}
        for combo in bets["wide"]:
            stats["wide"]["bets"] += 1
            stats["wide"]["spent"] += unit
            if set(combo) <= top3:
                stats["wide"]["hits"] += 1
        stats["wide"]["returned"] += payout["detail"]["wide"]["returned"]

        total_races += 1

    print(f"[INFO] 処理済み: {total_races}レース, スキップ: {skipped}レース")

    def _summarize(s: dict) -> dict:
        bets = s["bets"]
        hits = s["hits"]
        spent = s["spent"]
        returned = s["returned"]
        return {
            "bets": bets,
            "hits": hits,
            "hit_rate": hits / bets if bets > 0 else 0.0,
            "roi": returned / spent if spent > 0 else 0.0,
        }

    return {
        "total_races": total_races,
        "tansho": _summarize(stats["tansho"]),
        "umaren": _summarize(stats["umaren"]),
        "wide": _summarize(stats["wide"]),
    }


def main():
    parser = argparse.ArgumentParser(description="LFM-8B バックテスト評価")
    parser.add_argument("--year", type=int, default=2026, help="対象年")
    parser.add_argument(
        "--adapter_path",
        default=os.path.join(_THIS_DIR, "models", "lora_adapters"),
        help="LoRAアダプタパス",
    )
    parser.add_argument("--limit", type=int, default=None, help="レース数上限")
    args = parser.parse_args()

    from inference import load_model

    tokenizer, model = load_model(lora_adapter_path=args.adapter_path)

    year = args.year
    result = run_backtest(
        model=model,
        tokenizer=tokenizer,
        test_se_path=os.path.join(DATA_OUTPUT_DIR, "race_se", f"race_se_{year}.csv"),
        test_ra_path=os.path.join(DATA_OUTPUT_DIR, "race_ra", f"race_ra_{year}.csv"),
        quinella_odds_path=os.path.join(DATA_OUTPUT_DIR, "odds", f"QuinellaOdds_{year}.csv"),
        wide_odds_path=os.path.join(DATA_OUTPUT_DIR, "odds", f"WideOdds_{year}.csv"),
        race_limit=args.limit,
    )

    print("\n===== バックテスト結果 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n===== サマリー =====")
    print(f"  総レース数: {result['total_races']:,}")
    for bet_type in ["tansho", "umaren", "wide"]:
        s = result[bet_type]
        print(
            f"  {bet_type:8s}: {s['bets']:5d}件購入, "
            f"{s['hits']:4d}件的中, "
            f"的中率={s['hit_rate']:.1%}, "
            f"ROI={s['roi']:.3f}"
        )


if __name__ == "__main__":
    main()

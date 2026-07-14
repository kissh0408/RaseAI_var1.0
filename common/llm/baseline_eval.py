"""
baseline_eval.py — ファインチューニングなしのベースモデルで2026年評価
100レースのみ実行してROI/的中率の速報値を取得する。
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys

import pandas as pd

logger = logging.getLogger(__name__)
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from race_to_text import load_race_data, race_to_prompt, make_race_id
from betting_strategy import decide_bets, calculate_payout

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = r"C:\Users\syugo\AI\RaceAI\common\data\output"
RACE_LIMIT = 100

SYSTEM_INSTRUCTION = (
    "競馬レースを分析し、各馬のrank_scoreとev_scoreをJSON形式で出力してください。"
    '出力形式: {"scores":[{"horse_num":int,"rank_score":float,"ev_score":float}]}'
)


def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model


def predict_race(race_text: str, tokenizer, model) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": race_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    try:
        m = re.search(r"\{.*\}", generated, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return data.get("scores", [])
    except Exception as e:
        logger.warning("JSONパース失敗 (生成文字列先頭50文字: %s): %s", generated[:50], e)
    return []


def load_odds(year: int, ticket: str) -> dict[str, dict]:
    path = os.path.join(DATA_DIR, "odds", f"{ticket}Odds_{year}.csv")
    df = pd.read_csv(path, low_memory=False)
    if "odds_status" in df.columns:
        df = df[df["odds_status"] == "ok"]
    result: dict = {}
    for _, row in df.iterrows():
        rid = str(row["race_id"])
        h1, h2 = int(row["horse_num_1"]), int(row["horse_num_2"])
        key = (min(h1, h2), max(h1, h2))
        result.setdefault(rid, {})[key] = float(row["odds"])
    return result


def main():
    print("[INFO] ベースモデルロード中...")
    tokenizer, model = load_base_model()
    print("[INFO] 2026年データロード...")
    df_se, df_ra = load_race_data(2026)
    q_odds = load_odds(2026, "Quinella")
    w_odds = load_odds(2026, "Wide")
    ra_idx = df_ra.set_index("race_id")

    race_ids = df_se["race_id"].unique()[:RACE_LIMIT]
    stats = {k: {"bets": 0, "hits": 0, "spent": 0, "returned": 0.0}
             for k in ["tansho", "umaren", "wide"]}
    total = 0

    for race_id in tqdm(race_ids, desc="評価"):
        if race_id not in ra_idx.index:
            continue
        horses = df_se[df_se["race_id"] == race_id]
        valid = horses[(horses["abnormal_code"] == 0) & (horses["finish_rank"] > 0)]
        if len(valid) < 2:
            continue

        ra_row = ra_idx.loc[race_id]
        race_text = race_to_prompt(ra_row, horses)
        scores = predict_race(race_text, tokenizer, model)
        if not scores:
            continue

        qo = q_odds.get(race_id, {})
        wo = w_odds.get(race_id, {})
        bets = decide_bets(scores, qo, wo)

        ranked = valid.sort_values("finish_rank")
        ranks = ranked.set_index("finish_rank")["horse_num"].to_dict()
        winner = int(ranks.get(1, -1))
        second = int(ranks.get(2, -1)) if 2 in ranks else None
        third = int(ranks.get(3, -1)) if 3 in ranks else None

        winner_row = valid[valid["horse_num"] == winner] if winner > 0 else pd.DataFrame()
        tansho_odds = int(winner_row["odds"].iloc[0]) / 10.0 if not winner_row.empty else 0.0
        actual = {
            "winner": winner if winner > 0 else None,
            "second": second,
            "third": third,
            "tansho_payout": {winner: tansho_odds * 100} if winner > 0 else {},
            "quinella_payout": qo,
            "wide_payout": wo,
        }
        payout = calculate_payout(bets, actual)

        for t in ["tansho", "umaren", "wide"]:
            d = payout["detail"][t]
            stats[t]["spent"] += d["spent"]
            stats[t]["returned"] += d["returned"]
            if d["spent"] > 0:
                stats[t]["bets"] += d["spent"] // 100

        top3 = {winner, second, third} - {None}
        for h in bets["tansho"]:
            if h == winner:
                stats["tansho"]["hits"] += 1
        for combo in bets["umaren"]:
            if winner and second and set(combo) == {winner, second}:
                stats["umaren"]["hits"] += 1
        for combo in bets["wide"]:
            if set(combo) <= top3:
                stats["wide"]["hits"] += 1
        total += 1

    print(f"\n===== ベースライン評価結果 ({total}レース) =====")
    for t in ["tansho", "umaren", "wide"]:
        s = stats[t]
        roi = s["returned"] / s["spent"] if s["spent"] > 0 else 0.0
        hr = s["hits"] / s["bets"] if s["bets"] > 0 else 0.0
        print(f"  {t:8s}: {s['bets']:4d}件 的中{s['hits']:3d}件 "
              f"的中率={hr:.1%} ROI={roi:.3f} ({roi*100:.1f}%)")

    result = {
        "model": BASE_MODEL_ID + " (no fine-tuning)",
        "races": total,
        "tansho": {
            "bets": stats["tansho"]["bets"],
            "hit_rate": stats["tansho"]["hits"] / max(stats["tansho"]["bets"], 1),
            "roi": stats["tansho"]["returned"] / max(stats["tansho"]["spent"], 1),
        },
        "umaren": {
            "bets": stats["umaren"]["bets"],
            "hit_rate": stats["umaren"]["hits"] / max(stats["umaren"]["bets"], 1),
            "roi": stats["umaren"]["returned"] / max(stats["umaren"]["spent"], 1),
        },
        "wide": {
            "bets": stats["wide"]["bets"],
            "hit_rate": stats["wide"]["hits"] / max(stats["wide"]["bets"], 1),
            "roi": stats["wide"]["returned"] / max(stats["wide"]["spent"], 1),
        },
    }
    out_path = os.path.join(_THIS_DIR, "baseline_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")


if __name__ == "__main__":
    main()

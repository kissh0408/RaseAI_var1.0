"""
dataset_builder.py — 競馬予測LLM学習用データセット構築スクリプト

JSONL形式:
{
    "instruction": "...",
    "input": "レース情報テキスト",
    "output": "{\"reasoning\": \"...\", \"scores\": [{\"horse_num\": 1, \"rank_score\": 0.85, \"ev_score\": 0.12}, ...]}"
}

使用法:
    python dataset_builder.py                   # デフォルト (train=2015-2024, val=2025, test=2026)
    python dataset_builder.py --years 2023 2024 --out data/custom.jsonl
    python dataset_builder.py --split test       # 2026年のみ
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

# パッケージが実行パスに入るよう調整
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from race_to_text import (
    load_race_data,
    race_to_prompt,
    build_label,
    SEX_CODE_MAP,
)

DATA_DIR = os.path.join(_THIS_DIR, "data")

SYSTEM_INSTRUCTION = (
    "競馬レースを分析し、各馬のrank_scoreとev_scoreをJSON形式で出力してください。"
    '出力形式: {"scores":[{"horse_num":int,"rank_score":float,"ev_score":float}]}'
)


def _build_reasoning(horses_df: pd.DataFrame, scores: list[dict]) -> str:
    """
    スコアリング根拠の定型文を自動生成する。
    モデルにとって推論の足がかりになる情報を提供する。
    """
    score_map = {s["horse_num"]: s for s in scores}
    lines = []

    valid = horses_df[horses_df["abnormal_code"] == 0].sort_values(
        "horse_num"
    )

    for _, h in valid.iterrows():
        hnum = int(h["horse_num"])
        if hnum not in score_map:
            continue
        sc = score_map[hnum]

        pop_raw = h.get("popularity", 0)
        popularity = int(pop_raw) if pd.notna(pop_raw) else 0
        burden_raw = h.get("burden_weight", 0)
        burden_kg = (int(burden_raw) if pd.notna(burden_raw) else 0) / 10.0
        hw_raw = h.get("horse_weight", 0)
        horse_weight = int(hw_raw) if pd.notna(hw_raw) else 0
        sign_raw = h.get("weight_change_sign", "+")
        sign = str(sign_raw).strip() if pd.notna(sign_raw) else "+"
        change_raw = h.get("weight_change", 0)
        change = int(change_raw) if pd.notna(change_raw) else 0
        weight_diff = f"{sign}{change}"
        age_raw = h.get("age", 0)
        age = int(age_raw) if pd.notna(age_raw) else 0
        sex_raw = h.get("sex_code", 1)
        sex = SEX_CODE_MAP.get(int(sex_raw) if pd.notna(sex_raw) else 1, "不明")

        lines.append(
            f"馬番{hnum}は{popularity}番人気、"
            f"斤量{burden_kg:.1f}kg、"
            f"{sex}{age}歳、"
            f"馬体重{horse_weight}kg({weight_diff})、"
            f"rank_score={sc['rank_score']:.4f}、ev_score={sc['ev_score']:.4f}と予測。"
        )

    return " ".join(lines)


def _build_test_output_placeholder(horses_df: pd.DataFrame) -> str:
    """
    テストデータ用: ラベルなしのプレースホルダーを返す。
    出走馬リストのみ含む。
    """
    horse_nums = (
        horses_df[horses_df["abnormal_code"] == 0]
        .sort_values("horse_num")["horse_num"]
        .astype(int)
        .tolist()
    )
    return json.dumps(
        {
            "reasoning": "",
            "scores": [
                {"horse_num": h, "rank_score": None, "ev_score": None}
                for h in horse_nums
            ],
        },
        ensure_ascii=False,
    )


def build_dataset(
    years: list[int],
    output_path: str,
    is_test: bool = False,
) -> int:
    """
    指定年のデータからJSONLを生成し output_path に書き出す。

    Args:
        years: 対象年のリスト
        output_path: 出力先ファイルパス
        is_test: True の場合はラベルなし (testデータ)

    Returns:
        書き出したサンプル数
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for year in years:
            print(f"  [{year}] データ読み込み中...")
            try:
                df_se, df_ra = load_race_data(year)
            except FileNotFoundError as e:
                print(f"  [{year}] スキップ: {e}")
                continue

            # race_id でグループ化
            ra_indexed = df_ra.set_index("race_id")

            for race_id, horses in tqdm(
                df_se.groupby("race_id"),
                desc=f"  {year}",
                leave=False,
            ):
                if race_id not in ra_indexed.index:
                    skipped += 1
                    continue

                ra_row = ra_indexed.loc[race_id]

                # 有効な出走馬が存在するか確認
                valid_horses = horses[horses["abnormal_code"] == 0]
                if len(valid_horses) < 2:
                    skipped += 1
                    continue

                # プロンプト生成
                prompt_text = race_to_prompt(ra_row, horses)

                if is_test:
                    # テストデータ: ラベルなし
                    output_str = _build_test_output_placeholder(horses)
                else:
                    # ラベル生成
                    scores = build_label(horses, race_id)
                    if not scores:
                        skipped += 1
                        continue

                    output_str = json.dumps(
                        {"scores": scores},
                        ensure_ascii=False,
                    )

                record = {
                    "instruction": SYSTEM_INSTRUCTION,
                    "input": prompt_text,
                    "output": output_str,
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"  完了: {count}件書き出し, {skipped}件スキップ → {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="競馬予測LLMデータセット構築")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="生成するスプリット (all=train+val+test)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="対象年 (--split と併用不可。指定した場合は --out が必要)",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="出力先ファイルパス (--years 使用時に必要)",
    )
    args = parser.parse_args()

    if args.years:
        if not args.out:
            parser.error("--years を指定する場合は --out も必要です")
        build_dataset(args.years, args.out, is_test=False)
        return

    # デフォルト: train/val/test
    splits_to_run = (
        ["train", "val", "test"] if args.split == "all" else [args.split]
    )

    split_config = {
        "train": {
            "years": [2023, 2024],  # 直近2年のみ（高速化のため）
            "path": os.path.join(DATA_DIR, "train_dataset.jsonl"),
            "is_test": False,
        },
        "val": {
            "years": [2025],
            "path": os.path.join(DATA_DIR, "val_dataset.jsonl"),
            "is_test": False,
        },
        "test": {
            "years": [2026],
            "path": os.path.join(DATA_DIR, "test_dataset.jsonl"),
            "is_test": True,
        },
    }

    totals = {}
    for split in splits_to_run:
        cfg = split_config[split]
        print(f"\n=== {split.upper()} スプリット: {cfg['years']} ===")
        n = build_dataset(cfg["years"], cfg["path"], is_test=cfg["is_test"])
        totals[split] = n

    print("\n===== 完了サマリー =====")
    for split, n in totals.items():
        cfg = split_config[split]
        print(f"  {split:5s}: {n:6d}件  → {cfg['path']}")

    # サンプル表示
    for split in splits_to_run:
        path = split_config[split]["path"]
        if os.path.exists(path):
            print(f"\n--- {split} 先頭1サンプル ---")
            with open(path, encoding="utf-8") as f:
                first = f.readline()
            obj = json.loads(first)
            print(f"instruction: {obj['instruction'][:60]}...")
            print(f"input (先頭200文字):\n{obj['input'][:200]}")
            print(f"output (先頭300文字):\n{obj['output'][:300]}")


if __name__ == "__main__":
    main()

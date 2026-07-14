"""
run_pipeline.py — 全体オーケストレーター

使用法:
    python run_pipeline.py --mode build_dataset
    python run_pipeline.py --mode train
    python run_pipeline.py --mode evaluate
    python run_pipeline.py --mode evaluate --year 2025 --limit 50

モード:
    build_dataset : train/val/test のJSONLデータセットを構築
    train         : Q-LoRAファインチューニングを実行
    evaluate      : バックテスト評価を実行
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

DATA_DIR = os.path.join(_THIS_DIR, "data")
ADAPTER_DIR = os.path.join(_THIS_DIR, "models", "lora_adapters")
DATA_OUTPUT_DIR = r"C:\Users\syugo\AI\RaceAI\common\data\output"


def mode_build_dataset(args):
    """データセット構築モード。"""
    from dataset_builder import build_dataset

    splits = {
        "train": {
            "years": list(range(2015, 2025)),
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

    target_splits = [args.split] if args.split != "all" else ["train", "val", "test"]

    for split_name in target_splits:
        cfg = splits[split_name]
        print(f"\n=== {split_name.upper()} スプリット ===")
        build_dataset(cfg["years"], cfg["path"], is_test=cfg["is_test"])

    print("\nデータセット構築完了!")


def mode_train(args):
    """ファインチューニングモード。"""
    from trainer import train, _check_dependencies

    if not _check_dependencies():
        sys.exit(1)

    train_file = args.train_file or os.path.join(DATA_DIR, "train_dataset.jsonl")
    val_file = args.val_file or os.path.join(DATA_DIR, "val_dataset.jsonl")

    if not os.path.exists(train_file):
        print(f"[ERROR] 学習データが見つかりません: {train_file}")
        print("先に --mode build_dataset を実行してください。")
        sys.exit(1)

    train(
        train_file=train_file,
        val_file=val_file,
        output_dir=ADAPTER_DIR,
        num_epochs=args.epochs,
        dry_run=args.dry_run,
    )


def mode_evaluate(args):
    """バックテスト評価モード。"""
    from inference import load_model
    from evaluate import run_backtest
    import json

    year = args.year

    if not os.path.exists(ADAPTER_DIR):
        print(f"[ERROR] LoRAアダプタが見つかりません: {ADAPTER_DIR}")
        print("先に --mode train を実行してください。")
        sys.exit(1)

    tokenizer, model = load_model(lora_adapter_path=ADAPTER_DIR)

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

    # 結果をファイルに保存
    os.makedirs(os.path.join(_THIS_DIR, "results"), exist_ok=True)
    result_path = os.path.join(_THIS_DIR, "results", f"backtest_{year}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RaceAI LLMパイプライン オーケストレーター"
    )
    parser.add_argument(
        "--mode",
        choices=["build_dataset", "train", "evaluate"],
        required=True,
        help="実行モード",
    )

    # build_dataset オプション
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="build_dataset モードで生成するスプリット",
    )

    # train オプション
    parser.add_argument("--train_file", type=str, help="学習データJSONLパス")
    parser.add_argument("--val_file", type=str, help="検証データJSONLパス")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument(
        "--dry_run", action="store_true", help="モデルロード確認のみ"
    )

    # evaluate オプション
    parser.add_argument("--year", type=int, default=2026, help="評価対象年")
    parser.add_argument("--limit", type=int, default=None, help="評価レース数上限")

    args = parser.parse_args()

    if args.mode == "build_dataset":
        mode_build_dataset(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "evaluate":
        mode_evaluate(args)


if __name__ == "__main__":
    main()

"""
trainer.py — LFM-8B Q-LoRAファインチューニングスクリプト

使用法:
    python trainer.py --train_file data/train_dataset.jsonl --val_file data/val_dataset.jsonl
    python trainer.py --train_file data/train_dataset.jsonl --val_file data/val_dataset.jsonl --epochs 1 --dry_run

ハードウェア想定:
    GPU: RTX 5070 Ti (VRAM 16GB)
    CUDA: 12.x
    Python: 3.10+
    bf16: True (RTX 5070 Ti はbfloat16サポート)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_OUTPUT_DIR = os.path.join(_THIS_DIR, "models", "lora_adapters")

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def _check_dependencies() -> bool:
    """必要なライブラリが揃っているか確認する。"""
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] 以下のパッケージが未インストールです: {', '.join(missing)}")
        print("インストール: pip install " + " ".join(missing))
        return False
    return True


def load_jsonl_as_dataset(path: str):
    """JSONL ファイルを HuggingFace Dataset として読み込む。"""
    from datasets import Dataset

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    """
    Instruction-Input-Output を 1 つのテキストに結合する。
    SFTTrainer は 'text' フィールドを使う。
    """
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}


def get_bnb_config():
    """4bit量子化設定 (BitsAndBytes)。"""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_lora_config():
    """LFM-8B向け LoRA 設定。"""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def get_sft_config(
    output_dir: str,
    num_epochs: int,
    run_name: str = "lfm8b_race",
):
    """RTX 5070 Ti (16GB VRAM) 向けの SFTConfig (trl 1.x 対応)。"""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",
        run_name=run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # SFT固有設定 (trl 1.x)
        max_length=1300,
        packing=False,
        dataset_text_field="text",
    )


def train(
    train_file: str,
    val_file: str,
    output_dir: str,
    num_epochs: int = 3,
    dry_run: bool = False,
):
    """
    メイン学習関数。

    Args:
        train_file: 学習データ JSONL パス
        val_file: 検証データ JSONL パス
        output_dir: LoRA アダプタ保存先
        num_epochs: エポック数
        dry_run: True の場合モデルロードのみ確認して終了
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    print(f"[INFO] ベースモデル: {BASE_MODEL_ID}")
    print(f"[INFO] 学習データ: {train_file}")
    print(f"[INFO] 検証データ: {val_file}")
    print(f"[INFO] 出力先: {output_dir}")
    print(f"[INFO] エポック数: {num_epochs}")
    print(f"[INFO] CUDA 利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # トークナイザ
    print("[INFO] トークナイザをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # モデル (4bit量子化)
    print("[INFO] モデルをロード中 (4bit QLoRA)...")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="sdpa",   # PyTorch SDPA で高速化
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    if dry_run:
        print("[DRY RUN] モデルロード確認完了。学習はスキップ。")
        return

    # データセット
    print("[INFO] データセットを読み込み中...")
    train_dataset = load_jsonl_as_dataset(train_file).map(format_prompt)
    val_dataset = load_jsonl_as_dataset(val_file).map(format_prompt)
    print(f"[INFO] train: {len(train_dataset)}件, val: {len(val_dataset)}件")

    # 学習
    os.makedirs(output_dir, exist_ok=True)
    sft_config = get_sft_config(output_dir, num_epochs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("[INFO] 学習開始...")
    trainer.train()

    print(f"[INFO] LoRAアダプタを保存中: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] 学習完了!")


def estimate_training_time(train_samples: int, epochs: int = 3) -> None:
    """
    推定学習時間を計算して表示する。
    RTX 5070 Ti での実効スループットは約 500 tokens/sec を想定。
    平均プロンプト長は約 512 tokens、出力長は約 256 tokens と仮定。
    """
    avg_tokens = 768  # input + output 合計
    batch_size = 4
    grad_accum = 4
    steps_per_epoch = train_samples / (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    # RTX 5070 Ti での推定スループット (tokens/sec)
    throughput = 500.0
    total_tokens = train_samples * avg_tokens * epochs
    est_seconds = total_tokens / throughput
    est_hours = est_seconds / 3600

    print(f"\n===== 推定学習時間 =====")
    print(f"  学習サンプル数    : {train_samples:,}件")
    print(f"  エポック数        : {epochs}")
    print(f"  総ステップ数      : {total_steps:,.0f}")
    print(f"  平均トークン/サンプル: {avg_tokens}")
    print(f"  推定スループット  : {throughput:.0f} tokens/sec")
    print(f"  推定学習時間      : {est_hours:.1f}時間 ({est_seconds/60:.0f}分)")
    print(f"  ※ RTX 5070 Ti (16GB VRAM) + bf16 + 4bit QLoRA での概算値")


def main():
    parser = argparse.ArgumentParser(description="LFM-8B Q-LoRAファインチューニング")
    parser.add_argument(
        "--train_file",
        default=os.path.join(_THIS_DIR, "data", "train_dataset.jsonl"),
        help="学習データJSONLパス",
    )
    parser.add_argument(
        "--val_file",
        default=os.path.join(_THIS_DIR, "data", "val_dataset.jsonl"),
        help="検証データJSONLパス",
    )
    parser.add_argument(
        "--output_dir",
        default=ADAPTER_OUTPUT_DIR,
        help="LoRAアダプタ保存先",
    )
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="モデルロード確認のみ (学習はしない)",
    )
    parser.add_argument(
        "--estimate_time",
        action="store_true",
        help="推定学習時間を表示して終了",
    )
    args = parser.parse_args()

    if args.estimate_time:
        # JSONL の行数をカウント
        path = args.train_file
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        else:
            n = 100_000  # デフォルト想定
            print(f"[WARN] {path} が見つからないため {n:,}件と仮定して計算")
        estimate_training_time(n, args.epochs)
        return

    if not _check_dependencies():
        sys.exit(1)

    train(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

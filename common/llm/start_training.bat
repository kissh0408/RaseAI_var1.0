@echo off
echo ========================================
echo LFM2.5-8B-A1B Q-LoRA ファインチューニング
echo 開始: %DATE% %TIME%
echo ========================================
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"C:\Users\syugo\anaconda3\envs\keiba-ml\python.exe" ^
    "C:\Users\syugo\AI\RaceAI\common\llm\trainer.py" ^
    --train_file "C:\Users\syugo\AI\RaceAI\common\llm\data\train_dataset.jsonl" ^
    --val_file "C:\Users\syugo\AI\RaceAI\common\llm\data\val_dataset.jsonl" ^
    --epochs 3 ^
    >> "C:\Users\syugo\AI\RaceAI\common\llm\train_log.txt" 2>&1
echo 終了: %DATE% %TIME%
echo 終了: %DATE% %TIME% >> "C:\Users\syugo\AI\RaceAI\common\llm\train_log.txt"

@echo off
echo ========================================
echo 2026年バックテスト評価
echo 開始: %DATE% %TIME%
echo ========================================
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"C:\Users\syugo\anaconda3\envs\keiba-ml\python.exe" ^
    "C:\Users\syugo\AI\RaceAI\common\llm\evaluate.py" ^
    --year 2026 ^
    --adapter_path "C:\Users\syugo\AI\RaceAI\common\llm\models\lora_adapters" ^
    >> "C:\Users\syugo\AI\RaceAI\common\llm\eval_log.txt" 2>&1
echo 終了: %DATE% %TIME%

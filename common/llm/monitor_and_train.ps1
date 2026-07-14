# monitor_and_train.ps1
# dry_run完了を待って本学習を開始し、完了後に評価を実行するスクリプト
$pyExe = "C:\Users\syugo\anaconda3\envs\keiba-ml\python.exe"
$llmDir = "C:\Users\syugo\AI\RaceAI\common\llm"
$logFile = "$llmDir\train_log.txt"
$errFile = "$llmDir\train_err.txt"
$evalLog = "$llmDir\eval_log.txt"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 学習ログ確認: $logFile"

# dry_run完了チェック
$dryRunDone = (Get-Content $logFile -ErrorAction SilentlyContinue) -join "" -match "DRY RUN.*完了"
if ($dryRunDone) {
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] dry_run完了確認。本学習を開始します..."
    & $pyExe "$llmDir\trainer.py" `
        --train_file "$llmDir\data\train_dataset.jsonl" `
        --val_file "$llmDir\data\val_dataset.jsonl" `
        --epochs 3 2>&1 | Tee-Object -FilePath $logFile -Append
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 学習完了。評価を開始します..."
    & $pyExe "$llmDir\evaluate.py" `
        --year 2026 `
        --adapter_path "$llmDir\models\lora_adapters" 2>&1 | Tee-Object -FilePath $evalLog -Append
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 評価完了。$evalLog を確認してください。"
} else {
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] dry_runはまだ実行中またはダウンロード中です..."
    Get-Content $logFile -ErrorAction SilentlyContinue | Select-Object -Last 5
    Get-Content $errFile -ErrorAction SilentlyContinue | Select-Object -Last 3
}

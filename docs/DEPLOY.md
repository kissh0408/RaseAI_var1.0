# デプロイ・モデル復元手順

`model_training/models/` と `pure_rank/models/` は `.gitignore` 対象のため、clone 直後は空です。

## 本番モデル（必須）

| 層 | ファイル | 本番 ID |
|----|---------|---------|
| Layer 1 | `pure_rank/models/lambdarank_fold{1,2,3}_seed{42..46}.txt`（15本） | v39_course_slim |
| Layer 2 | `model_training/models/binary_fold{1,2,3}_seed{42..46}.txt`（15本） | Phase 1a-A2 |

## 退避スナップショット

統合作業前の退避: `backup_before_unified_integration_20260705/`

```powershell
Copy-Item backup_before_unified_integration_20260705/pure_rank_models/*.txt pure_rank/models/
Copy-Item backup_before_unified_integration_20260705/model_training_models_full/binary_*.txt model_training/models/
```

（実際のファイル名は退避ディレクトリ内の一覧を確認すること）

## 検証ゲート

```bash
python pure_rank/src/evaluate.py          # Top-1 ≈ 30.2%
python strategy/src/backtest.py           # Phase 1a-A2 ROI 退行なし
python -c "from main.notebook_bootstrap import run_unified_today"
```

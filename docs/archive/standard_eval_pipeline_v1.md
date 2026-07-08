# 標準評価パイプライン v1

**ステータス**: 確定（2026-06-17）  
**目的**: 本番モデル差し替え判断と切り離し、**同一条件での再現可能な評価**を固定する。

---

## 位置づけ

| 項目 | 内容 |
|------|------|
| specv2 モデル | **標準評価の参照モデル**（WF OOF 2018–2025）。本番 `ensemble_v5` への自動昇格は行わない |
| v5 OOF | 比較・回帰用（`evaluation_v5_oof.csv`） |
| 旧 baseline 127.5% | 条件不一致のため **参照のみ**。新 baseline は本パイプラインで再計測 |

---

## ファイル契約

| パス | 役割 |
|------|------|
| `model_training/data/03_train/evaluation_specv2_oof.csv` | specv2 WF OOF（**評価の正**） |
| `model_training/data/03_train/evaluation_v5_oof.csv` | v5 WF OOF（3シード平均） |
| `model_training/data/03_train/baseline_standard_eval.json` | 同一条件 baseline |
| `model_training/data/03_train/compare_v5_specv2_eval.json` | モデル×プロファイル比較 |
| `model_training/data/03_train/wide_production_gates_report.json` | C1 ワイド OOF ゲート結果 |
| `strategy/models/calibration_isotonic_specv2.json` | 本番 calibrator（OOF 2018–24 fit） |

---

## 評価プロファイル

| ID | 条件 |
|----|------|
| `production_live` | `strategy_config.json` + **specv2 cal** + race_num 8–12（**本番 baseline**） |
| `production` | 同上（legacy cal 比較行あり。本番では specv2 を使用） |
| `v5_meta` | specv2 cal + **race_num なし**（旧記録比較用） |

両プロファイルとも `strategy_config_from_runtime()` 経由。minimal StrategyConfig は **使用禁止**。

---

## 標準コマンド

```bash
# 1. v5 OOF 再生成（必要時のみ。specv2 OOF は退避・復元される）
python model_training/scripts/regenerate_v5_ensemble_oof.py --force

# 2. モデル×戦略プロファイル比較
python model_training/scripts/compare_production_ensemble_eval.py --years 2025 all

# 3. MDD 要因切り分け（specv2 OOF 固定）
python model_training/scripts/diagnose_win_mdd.py --years 2025 all

# 4. 本番接続後ゲート検証（2025 テストフォールド）
python model_training/scripts/run_standard_eval_gates.py

# 5. baseline 更新
python model_training/scripts/update_standard_baseline.py

# 6. specv2 calibrator 学習（2018–2024 OOF fit）
python model_training/scripts/fit_specv2_calibrator.py

# 7. E2E（2025 ゲート合格後）
python main/tests/e2e_test.py

# 8. Track C1 ワイド本番ゲート（OOF 2025）
python model_training/scripts/run_wide_production_gates.py
```

---

## 合格ゲート（参照）

### 単勝（2025 テストフォールド）— CLAUDE.md / standard_eval

| 指標 | 合格 |
|------|------|
| ROI | ≥ 105% |
| MDD | ≥ -20% |
| Sharpe | ≥ 0.10 |
| n_bets | ≥ 500 |

> **北極星（ポートフォリオ合算）**: Floor ROI ≥ **115%**、Target ≥ **130%** — `post_deploy_roadmap_v2.md`

### ワイド anchor（Track C1・2025 OOF）

| 指標 | 合格 |
|------|------|
| ROI | ≥ 115% |
| n_bets | ≥ 200 |

---

## ワークフロー（5エージェント）

```
domain-planner  → 戦略/calibrator 改善仕様
model-strategy-generator → calibrator 再学習・戦略パラメータ実装
backtest-evaluator → 本パイプラインで再評価
（モデル変更時のみ data-generator + 再学習）
```

本番 `ensemble_v5` 差し替えは **MDD ゲート通過 + deployment-evaluator E2E** 後のみ。

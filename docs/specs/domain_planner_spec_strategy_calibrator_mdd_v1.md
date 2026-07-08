# Domain Planner 仕様: 戦略層・Calibrator 改善（MDD 是正）

**版**: v1  
**日付**: 2026-06-17  
**前提**: [standard_eval_pipeline_v1.md](./standard_eval_pipeline_v1.md) + `mdd_diagnosis_report.json`

---

## 1. 問題定義

標準評価（specv2 OOF 2018–2025）では **ROI・Sharpe は合格圏** だが **MDD が -20% を大幅に下回る**。

| プロファイル | 2025 ROI | 2025 MDD | 判定 |
|-------------|----------|----------|------|
| production (8–12R) | ~147% | ~-76% | MDD NG |
| v5_meta | ~146% | ~-52% | MDD NG |

**モデル差（specv2 vs v5 OOF）は ±2pp で無視できる。** 改善余地は **戦略層・calibrator・サイジング** に集中する。

---

## 診断結果サマリー（2026-06-17 実行）

### 2025 単勝 WIN

| 要因 | 結果 |
|------|------|
| **race_num 8–12** | MDD **-76.4%** vs なし **-52.1%** → **24pp 悪化**（bet 数は 5,793→2,428 に減） |
| **legacy calibrator** | no_calibrator と **同一** → v4 由来 calibrator は実質無効 |
| **OOF calibrator（2018–2024 fit）** | MDD **-19.9%** / ROI **151.2%** / Sharpe **0.124** → **MDD ゲート合格** |
| Kelly 0.05/0.04 | **変化なし**（max_stake キャップ支配の可能性） |
| dynamic_edge off | MDD **悪化**（-78.7%） |

### 全期間（2018–2025）

| 要因 | 結果 |
|------|------|
| race_num 8–12 | MDD **改善**（-50.6% vs -69.1%）だが ROI **-12pp** |
| OOF calibrator | MDD **-31.5%**（production）— 全期間では依然 NG |

**採用方針（model-strategy-generator）**: `calibration_isotonic_specv2.json` を次イテレーションの calibrator 候補とする。本番切替前に 2024 以前フォールドでも MDD を確認すること。

---

`diagnose_win_mdd.py` で以下を **1 変数ずつ** 検証する:

| 要因 | 検証シナリオ | 期待 |
|------|-------------|------|
| race_num 8–12 | production vs v5_meta | MDD Δ の符号で悪化/改善を判定 |
| legacy calibrator (v4 由来) | no_calibrator / calibrator_oof_fit | OOF 再 fit で MDD 改善 |
| Kelly 0.08 | kelly_0.05 / kelly_0.04 / flat_stake | サイズ縮小で MDD 改善 |
| dynamic_edge | no_dynamic_edge | エッジ厳格化で bet 数↓・MDD↓ |
| monthly_dd_limit | 本番停止のみ（-8%） | BT MDD とは別。sim 列で参考 |

**禁止**: テスト年（2025）の診断結果から閾値を後付け（後出しじゃんけん）。

---

## 3. model-strategy-generator 実装要件

### 3-1. Calibrator 再学習（最優先）

| 項目 | 仕様 |
|------|------|
| 入力 | `evaluation_specv2_oof.csv` の `pred_rank1` |
| fit 期間 | `valid_year < 2025`（2018–2024 OOF のみ） |
| 手法 | Isotonic（rank1 勝率） |
| 出力 | `strategy/models/calibration_isotonic_specv2.json`（**既存 v4 ファイルは上書きしない**） |
| 評価 | 標準パイプライン `production` プロファイルで MDD/ROI 再計測 |
| 採用条件 | MDD 改善かつ ROI ≥ 105%、学習期間 ECE/Brier 劣化なし |

### 3-2. Kelly / サイジング

| 候補 | 根拠 |
|------|------|
| `kelly_fraction=0.05` | diagnose で MDD 改善幅を確認後 |
| `dynamic_kelly_enabled=true` | 高オッズ帯の tail risk 縮小 |
| flat stake 比較 | MDD 下限の参考（本番採用は EV/Kelly 方針と要協議） |

**変更は 1 パラメータずつ。** `train_config.json` / `strategy_config.json` に集約。

### 3-3. dynamic_edge

- `no_dynamic_edge` シナリオで MDD が大幅改善する場合 → バンド設計見直し（**学習期間分析のみ**）
- production 本番は `dynamic_edge_enabled=true` 維持を既定とし、改善案は A/B 比較

### 3-4. race_num 8–12（2026-06-17 確定）

**legacy cal 下**: 2025 MDD 24pp 悪化 → 一時的に production から除外を検討。  
**specv2 cal 接続後**（再診断）:

| 条件 | 2025 MDD | 全期間 MDD |
|------|----------|------------|
| 8–12R + specv2 cal | **-19.9%** ✅ | **-31.5%** |
| R制限なし + specv2 cal | -25.9% | -50.6% |

→ **race_num 8–12 は specv2 cal とセットで維持**（`strategy_config.json` に反映済み）。  
全期間 MDD -31.5% は **参考指標**。本番移行ゲートは **2025 テストフォールド** を優先。

---

## 4. 評価・ゲート

- 評価は **必ず** `compare_production_ensemble_eval.py` + `diagnose_win_mdd.py`
- 合格: CLAUDE.md 表（ROI / MDD / Sharpe / n_bets）
- baseline 更新: `update_standard_baseline.py`

---

## 5. スコープ外（本イテレーション）

- 特徴量追加・モデル再学習（specv2 ≈ v5 のため優先度低）
- binary 残差系（`inference_common`）との統合
- 本番 `ensemble_v5` 差し替え

---

## 6. エージェント割当

| フェーズ | エージェント | 成果物 |
|---------|-------------|--------|
| 1 | domain-planner | 本仕様書 |
| 2 | — | （データ変更なし） |
| 3 | model-strategy-generator | calibrator JSON + config 1 変数変更 |
| 3 | backtest-evaluator | mdd_diagnosis + baseline 更新 |
| 4 | deployment-evaluator | MDD ゲート通過後のみ E2E |

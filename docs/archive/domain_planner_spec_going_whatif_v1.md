# Domain Planner Spec: 馬場 What-if シナリオ改善 v1

**日付**: 2026-06-17  
**ステータス**: 確定（実装 Phase 0）  
**担当**: domain-planner → data-generator → model-strategy-generator → backtest-evaluator → deployment-evaluator

---

## 1. 背景

本番 Resulut（`main/Resulut/{競馬場}/馬場_{良|稍重|重|不良}/*.csv`）は **JV 馬場コード 1〜4 の what-if シナリオ別予測** を出力する。WE 速報馬場取得は不安定なため **現状維持**。運用判断は「重・不良になったら順位がどう動くか」を 4 シナリオ比較で行う。

**根本原因（2026-06 診断）:**

- 推論時 `apply_uniform_baba_jv_code` が `going_match_score_*_imputed` 等を再計算していない
- 本番 v5 モデルは `turf_condition` gain=0（木未使用）。`going_match_score_turf_imputed` のみ微弱に効く

---

## 2. シナリオ定義

| JV コード | ラベル | Resulut フォルダ |
|-----------|--------|------------------|
| 1 | 良 | `馬場_良` |
| 2 | 稍重 | `馬場_稍重` |
| 3 | 重 | `馬場_重` |
| 4 | 不良 | `馬場_不良` |

- 全レースを **同一 JV コード** に上書き（芝: `turf_condition=jv`, `dirt_condition=0` / ダート: 逆）
- 主列 `pred_rank1` は `strategy_baba_scenario_jv_code` のシナリオを複製

---

## 3. 特徴量要件

### 3-1. 推論時再計算必須列（シナリオ依存）

| 列 | 優先度 |
|----|--------|
| `going_match_score_turf_imputed` | P0 |
| `going_match_score_dirt_imputed` | P0 |
| `going_change_lag1`, `going_worsening_flag` | P1 |
| `going_x_turf_heavy_winrate`, `going_x_dirt_heavy_winrate` | P2（学習式と一致） |
| `going_match_score_turf/dirt`, `current_going_win_rate_*`, one-hot | 既存 |

### 3-2. 将来追加（Phase 2 以降）

**`is_baba_match`**: `horse_going_preference` の argmax カテゴリ（1=良,2=稍重,3=重,4=不良）と `jv_code` が一致 → 1、否则 0。

---

## 4. 実験ラダー（1 変数ルール）

| Exp | 変更 | feature file | output dir |
|-----|------|--------------|------------|
| A | interaction_constraints | v25_odds | ensemble_v6_expA |
| B | A + rank1 baba weight multiplier | v25_odds | ensemble_v6_expB |
| C | B + v26 going delta | v26_going_delta | ensemble_v6_expC |
| D | C + v27 track variant | v27_track_variant | ensemble_v6（**leak ブロック — 別 Phase**） |

**Exp-D:** `daily_track_variant` / `tm_score_surface_adj` は `BASE_LEAK_COLS` 登録済み。**v6 第一候補は Exp-C（Option A）**。

---

## 5. 合格基準

### 5-1. 全体（standard_eval_pipeline_v1）

| 指標 | 合格 |
|------|------|
| ROI | ≥ 105% |
| MDD | ≥ -20% |
| Sharpe | ≥ 0.10 |
| n_bets | ≥ 500 |

### 5-2. 感度（evaluate_going_experiment_gate.py）

| 指標 | 閾値 |
|------|------|
| top1 flip rate（良 vs 不良） | ≥ 8% |
| max_diff mean（良 vs 不良） | ≥ 0.013 |
| 稍重 vs 重 完全一致率 | ≤ 47% |
| going gain share | ≥ 2% |

### 5-3. セグメント（ノイズ防止）

| セグメント | 条件 | n_bets | 判定 |
|------------|------|--------|------|
| heavy | track_condition_code ∈ {3,4} | ≥ 200 | 未満 → **判定保留** |
| heavy ROI | n ≥ 200 時 | ≥ 100% | 未満 → リジェクト |
| soft | code = 2 | ≥ 200 | 同上 |

**Exp-B 追加:** heavy ROI が Exp-A 比 +2pp 以上（n ≥ 200 時のみ）

### 5-4. Phase 1（v5 モデル変更なし）

- baba1 vs baba4 max_diff > 0 の頭数 > 100/486
- 推論 4 シナリオ wall time ≤ 現行 2 倍

---

## 6. 検証コマンド

```bash
python -m pytest tests/test_inference_baba_sync.py tests/test_going_improvement.py -q
python model_training/scripts/diagnostics_going_sensitivity.py --models-dir model_training/models/ensemble_v5_specv2
python model_training/scripts/run_going_experiment.py --experiment C
python model_training/scripts/evaluate_going_experiment_gate.py --experiment C
python main/tests/e2e_test.py
```

---

## 7. ロールバック

- 推論: `inference_pipeline.py` の recompute 呼び出しを revert
- モデル: `production_training` / `strategy_config.json` を `ensemble_v5_specv2` に戻す

---

## 8. Exp A→C 実行手順（ローカル parquet 必須）

`model_training/data/02_features/features_past_v25_odds.parquet` が存在すること。

```bash
# v26/v27 派生（任意）
python model_training/scripts/build_going_feature_parquets.py

# 1変数ずつ
python model_training/scripts/run_going_eval_pipeline.py --experiment A --fast
python model_training/scripts/run_going_eval_pipeline.py --experiment B --fast
python model_training/scripts/run_going_eval_pipeline.py --experiment C --fast

# 昇格（ゲート PASS 時）
python model_training/scripts/run_going_v6_promotion.py --models-dir ensemble_v6_expC
python model_training/scripts/fit_specv2_calibrator.py
python main/tests/e2e_test.py
```

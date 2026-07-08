# RaceAI 仕様書インデックス

**最終更新**: 2026-06-17  
**最高意思決定**: [post_deploy_roadmap_v2.md](post_deploy_roadmap_v2.md)  
**北極星 KPI**: Floor ROI ≥ **115%** / Target ≥ **130%**、MDD ≤ -20%、合算的中 ≥ 20%

---

## 確定・運用中

| 仕様 | ファイル | ステータス |
|------|---------|-----------|
| **デプロイ後ロードマップ v2** | [post_deploy_roadmap_v2.md](post_deploy_roadmap_v2.md) | **最高意思決定** |
| 標準評価パイプライン | [standard_eval_pipeline_v1.md](standard_eval_pipeline_v1.md) | 確定 |
| ワイド本番（C1） | [domain_planner_spec_wide_production_v1.md](domain_planner_spec_wide_production_v1.md) | 本番反映済み |
| 戦略・calibrator MDD | [domain_planner_spec_strategy_calibrator_mdd_v1.md](domain_planner_spec_strategy_calibrator_mdd_v1.md) | specv2 採用済み |

---

## Phase 1 以降（仕様確定・未実装）

| Track | 仕様 | 着手 |
|-------|------|------|
| **C2** | [domain_planner_spec_c2_multivariate_kelly_v1.md](domain_planner_spec_c2_multivariate_kelly_v1.md) | 2026-07-15〜 |
| **D** | [domain_planner_spec_d_regime_detection_v1.md](domain_planner_spec_d_regime_detection_v1.md) | D-1 並行可 |
| **E** | [domain_planner_spec_e_mlops_monitoring_v1.md](domain_planner_spec_e_mlops_monitoring_v1.md) | 凍結中 MVP |
| **B** | [domain_planner_spec_hit_rate_priority_v1.md](domain_planner_spec_hit_rate_priority_v1.md) | 並行 R&D 20% |
| **B'** | [domain_planner_spec_b_duration_forecasting_v1.md](domain_planner_spec_b_duration_forecasting_v1.md) | B 打切り後 |

---

## 凍結（〜2026-07-15）

モデル・HP・特徴量・calibrator・戦略閾値の変更 **禁止**。例外: 致命バグ。Track E はアラートのみ。

---

## 本番構成

| レイヤ | 内容 |
|--------|------|
| モデル | `ensemble_v5`（3シード） |
| calibrator | `calibration_isotonic_specv2.json` |
| 単勝 | EV 1.05 / Kelly 0.08 / race_num 8–12 |
| ワイド | phase1_5 / O3 / wide_min_edge 0.05 |

---

## 検証コマンド

```bash
python model_training/scripts/run_standard_eval_gates.py
python model_training/scripts/run_wide_production_gates.py
python main/tests/e2e_test.py
```

---

## 旧版

| ファイル | 備考 |
|---------|------|
| [post_deploy_roadmap_v1.md](post_deploy_roadmap_v1.md) | v2 に supersede |

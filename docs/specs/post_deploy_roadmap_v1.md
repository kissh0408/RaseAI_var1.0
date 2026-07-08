# デプロイ後ロードマップ v1

**確定日**: 2026-06-17  
**ステータス**: ⚠️ **superseded by [post_deploy_roadmap_v2.md](post_deploy_roadmap_v2.md)**  
**方針**: A（本番 ROI/MDD 安定）+ C1（ワイド本番統合）+ B（的中率25% R&D は別ライン）

---

## 1. 本番凍結（Track A — 0〜4週）

| 項目 | 値 | 変更 |
|------|-----|------|
| モデル | `ensemble_v5` | 差し替え禁止 |
| calibrator | `calibration_isotonic_specv2.json` | 固定 |
| 単勝戦略 | EV 1.05 / Kelly 0.08 / race_num 8–12 | 固定 |
| 安全弁 | monthly_dd -8% | 運用必須 |
| 評価 | `standard_eval_pipeline_v1` | 全実験の土俵 |

**週次モニタリング**: ROI / 的中率 / n_bets vs `baseline_standard_eval.json`（±10pp 以内）

**4週後判断**: 実績が baseline から大きく乖離 → calibrator 再 fit または condition_ev（学習期間のみ）を検討

---

## 2. 収益拡大（Track C1 — 進行中）

**目的**: 単勝に加え **O3 速報オッズ + Harville ワイド EV** を本番推奨に統合する。

| ステップ | 内容 | 状態 |
|----------|------|------|
| C1-1 | `online_phase=phase1_5` + `wide_bets_enabled=true` | ✅ |
| C1-2 | `wide_min_edge` を combo_backtest と同一（0.05） | ✅ |
| C1-3 | 馬連・複勝は本番 OFF（1変数ルール） | ✅ |
| C1-4 | OOF ワイド anchor ゲート (`run_wide_production_gates.py`) | ✅ |
| C1-5 | E2E ワイド smoke | ✅ |
| C1-6 | 4週実績モニタ（win + wide 合算 P&L） | **運用中** |

**合格ゲート（2025 OOF）**

| 指標 | 下限 |
|------|------|
| ROI（wide anchor） | ≥ 115% |
| 的中率（wide anchor） | ≥ 20%（参考） |
| n_bets | ≥ 200 |

単勝ゲートは Track A で既合格。C1 は **ワイド単体** を追加監視。

詳細: `docs/specs/domain_planner_spec_wide_production_v1.md`

---

## 3. R&D 別ライン（Track B — 本番非連動）

**主 KPI**: 的中率 ≥ 25%、ROI ≥ 115%（3 fold 同時）

| 順 | 施策 | 担当 |
|----|------|------|
| 1 | valid 期間 grid（max_odds 3–8） | model-strategy-generator |
| 2 | 特徴量 1 本ずつ | data-generator |
| 3 | Rank v24 再実験 | model-strategy-generator |

仕様: `docs/specs/domain_planner_spec_hit_rate_priority_v1.md`  
**本番 merge は別ゲート**（Track A/C1 と混在禁止）

---

## 4. やらないこと

- specv2 モデルの本番 promotion
- LightGBM HP 大量 sweep
- テスト fold から condition_ev 導出
- 旧 baseline 127.5% / MDD -9.4% との直接比較

---

## 5. 関連ファイル

| ファイル | 役割 |
|---------|------|
| `strategy/config/strategy_config.json` | 本番戦略 + C1 フラグ |
| `model_training/scripts/run_wide_production_gates.py` | ワイド OOF ゲート |
| `baseline_standard_eval.json` | Track A baseline |

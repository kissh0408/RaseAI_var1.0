# Domain Planner Spec: 的中率優先戦略 v1

**日付**: 2026-06-17  
**ステータス**: 🟡 Track B（R&D 20% — 本番 merge 禁止）  
**北極星 KPI**: [post_deploy_roadmap_v2.md](post_deploy_roadmap_v2.md) 案1（ポートフォリオ ROI/MDD）が本番。本書の単勝25%は R&D 専用。  
**打切り**: 3 実験連続で valid 的中 +3pt 未満 → [domain_planner_spec_b_duration_forecasting_v1.md](domain_planner_spec_b_duration_forecasting_v1.md) へ移行  
**前提 champion**: mc80 + `features_v6_going_v1`（Phase4 DA修正版）

---

## 1. 方針転換

| 項目 | 旧方針（ROI優先） | 新方針 |
|------|------------------|--------|
| 主 KPI | 回収率 ≥ 105% | **的中率 ≥ 25%** |
| 副 KPI / 下限 | Sharpe ≥ 0.10 | **回収率 ≥ 115%** |
| ベット哲学 | 市場の見落とし（中〜長穴） | **勝率の高い馬を EV 条件付きで選抜** |

理論目安: ROI 115% × 的中率 25% → 的中時平均配当 ≈ **4.6 倍**（オッズ帯 3〜5 倍中心）。

---

## 2. 合格基準

| 指標 | 合格 | 要改善 | リジェクト |
|------|------|--------|-----------|
| 的中率 | **≥ 25%** | 20〜25% | < 20% |
| ROI | **≥ 115%** | 105〜115% | < 105% |
| 最大 DD | ≤ -20% | -20〜-30% | > -30% |
| ベット数 | ≥ 100 / fold | 50〜100 | < 50（判定保留） |

Sharpe ≥ 0.10 は参考（`evaluation.py` PASS_CRITERIA に hit_rate 追加済み）。

---

## 3. 現状ギャップ（2026-06-17 champion）

| Fold | 的中率 | ROI | 平均オッズ | ギャップ |
|------|--------|-----|-----------|---------|
| F1 | 17.9% | 213% | 12.1x | -7.1pt |
| F2 | 8.8% | 135% | 13.2x | -16.2pt |
| F3 | 13.0% | 142% | 13.2x | -12.0pt |

**構造的原因**: 現行 binary モデルは 12x 前後の高オッズ帯を選定。的中率25%と両立する平均オッズ（≈4.6x）と乖離。

---

## 4. Grid 探索結果（1440 通り・テスト期間）

スクリプト: `model_training/scripts/strategy_hit_rate_sweep.py`  
結果: `model_training/models/strategy_hit_rate_sweep.json`

| ゲート | 結果 |
|--------|------|
| 3F 同時: 的中≥25% かつ ROI≥115% かつ n≥50 | **0 / 1440** |

**結論**: 戦略パラメータ調整のみでは不可。**モデル・特徴量変更が必要**。

参考（件数犠牲）: max_odds=5, min_model_prob 高め → 的中 45%超も n≈17（統計的無意味）。

---

## 5. 次フェーズ（進行中）

### 5-1. 第一レバー: 戦略（valid 期間のみで決定）

| パラメータ | 現行 | 探索方向 |
|-----------|------|----------|
| `max_odds` | 50.0 | **3.0〜8.0** |
| `min_model_prob` | 0.05 | **0.12〜0.25** |
| `max_picks_per_race` | 2 | **1** |
| `ev_threshold` | 1.05 | 1.02〜1.15 |

→ valid 期間 grid のみ。テストは最終 1 回確認。

### 5-2. 第二レバー: 特徴量（data-generator）

| 特徴量 | 目的 |
|--------|------|
| `last5_rank_std` | 着順安定性 |
| `top3_rate_career` | 複勝率 |
| `top3_rate_class` | 同クラス連対率 |

### 5-3. 第三レバー: モデル

- rank1 勝率キャリブレーション強化
- binary Isotonic 再評価（Step2 REJECT 済み。条件変更で再試行）
- `max_model_rank=1` フィルタ

**推奨順序**: valid 期間戦略探索 → 特徴量 1 件ずつ → モデル変更

---

## 6. ワークフロー

```
domain-planner（本書）
  → model-strategy-generator: valid 期間 grid + 特徴量実装
  → backtest-evaluator: テスト 1 回確認
  → deployment-evaluator: strategy_config 反映 + E2E
```

---

## 7. 禁止事項

1. テストフォールド結果から閾値を後付け調整
2. 的中率のみ達成し ROI < 105% の設定をリリース
3. 200 件未満で「25% 達成」と主張

---

## 関連

- `docs/issues/known_issues_20260617.md` — N-1
- `docs/experiments/2026-06-17-roadmap-final-summary.md` — 前フェーズ完了記録

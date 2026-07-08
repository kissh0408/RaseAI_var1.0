# Phase 1b: EV 閾値再チューニング（Sharpe + min_n）仕様書

**作成日**: 2026-07-05  
**状態**: Approved  
**前提**: Phase 1a-A/A2 完了（β*=0.15 init_score、全 Fold 再学習済み）

---

## 1. 背景

Phase 1a 以降、出力分布が健全化し固定閾値（EV=1.05, max_picks=2）は旧 8-tree モデル向けに過剰/過少フィルタとなっている。

旧 `bet_tuning`（VALID ROI 最大）は P-34 の件数崩壊・silent fallback を招いたため、**Sharpe Ratio + min_n ゲート** に刷新する。

---

## 2. 目的関数（確定）

$$
\text{Objective}(h) = \begin{cases}
\text{Sharpe}(\text{returns}_h) & \text{if } n_h \ge \text{min\_n\_fold} \\
-1 & \text{otherwise}
\end{cases}
$$

- $h$ = グリッド点 $(ev\_threshold, max\_picks)$
- $\text{returns}_h$ = VALID 期間のベット単位 `return_on_invest`（`evaluation.calculate_roi_metrics` と同一系列）
- Sharpe = `mean(returns) / std(returns)`（等重み、`calculate_sharpe` 既存実装）
- **Tie-breaker**: Sharpe 差 < 0.01 なら **より大きい n** を優先（安定性）。さらに同点なら **より低い ev_threshold**（保守的）

---

## 3. min_n（Fold 別最小ベット数）— 確定

Phase 1a-A2 診断（Arm A, ev=1.05, max_picks=2, isotonic OFF）の **VALID ベット数** を基準:

| Fold | VALID 期間 | VALID n (baseline) | **min_n** | 根拠 |
|------|-----------|-------------------|-----------|------|
| 1 | 2022 | **66** | **50** | 旧 min_n=100 では全 grid 不合格。50 ≈ baseline×0.75。Sharpe 安定下限 ~30。 |
| 2 | 2023 | **104** | **80** | baseline×0.77。厳しめ閾値でも候補を確保。 |
| 3 | 2024 | **146** | **100** | プロジェクト従来基準。十分な headroom。 |

**絶対下限（全 Fold）**: `min_n_absolute = 30` — これ未満は Sharpe 推定不能として常に -1。

**採用値**: `min_valid_bets_by_fold = { "1": 50, "2": 80, "3": 100 }`

---

## 4. グリッド（Rule 3: 探索空間のみ。モデル HP 変更なし）

```json
"ev_threshold_grid": [0.95, 1.0, 1.03, 1.05, 1.08, 1.1, 1.15],
"max_picks_grid": [1, 2, 3, 4]
```

- **0.95 / 1.0 追加理由**: Fold 1 VALID で ev≥1.0 通過 ~194（診断 D2）、ev=1.05 では ~75。低閾値を探索しないと min_n 達成 grid が不足。
- max_picks **4** 追加: 引き締まった分布でレース内複数エッジを拾う余地。

---

## 5. プロトコル

1. **Fold 別独立チューニング** — VALID のみ。TEST は最終報告のみ。
2. `bet_tuning.enabled=true` で backtest 実行。
3. 選定パラメータを TEST に適用し ROI / Sharpe / n を報告。
4. **退行チェック**: TEST ROI が Phase 1a-A2（チューニング前）比 **-5pp 以内**、または Sharpe 改善。
5. 本番 `binary_recommendation.py` へ Fold 別固定値を配線（P-36 解消）。

---

## 6. 合否

| 段階 | 合格 |
|------|------|
| チューニング | 全 Fold で fallback なし（min_n 達成 grid が存在） |
| TEST | Sharpe ≥ Phase 1a-A2 ベース、または ROI 退行なし |
| 本番 | evaluator 独立検証 + P-36 配線確認 |

---

## 7. 参照

- `strategy/src/bet_tuning.py`
- `model_training/models/phase1aA2_regression_comparison.json`
- `ev_distribution_diagnosis_iso_off.json` — VALID arm_a_bets

# Domain Planner Spec: C2 多変量 Kelly（win+wide 非排他）v1

**日付**: 2026-06-17  
**ステータス**: 確定（Phase 1: 2026-07-15 着手）  
**担当**: domain-planner → model-strategy-generator → backtest-evaluator

---

## 1. 問題定義

単勝とワイドは **非排他**（単勝的中 → 該当ワイドも的中しうる）。券種ごとに独立 Kelly を適用して合算すると **オーバーベット** となり、MDD が悪化する。

**禁止**: 現行のレース内 `max_invest_per_race` 比例縮小のみに依存した合算（応急処置。C2 本実装で置換）。

---

## 2. 数学的定式化

- 結果 $i = 1..n$: レースの排他結果（着順組合せまたは簡約状態空間）
- ベット $j = 1..m$: 単勝・ワイド券（$m > n$ ありうる）
- $p_i$: 結果 $i$ の確率（Harville + calibrator 由来）
- $a_j \ge 0$: ベット $j$ への資金比率、$\sum_j a_j \le 1$
- $r_{i,j}$: 結果 $i$ におけるベット $j$ のリターン（オッズ・的中行列）

**目的関数（期待対数成長）**:

$$G_\infty(a) = \sum_i p_i \log\left( 1 + \sum_j r_{i,j} a_j \right)$$

**制約**:

- $a_j \ge 0$
- $\sum_j a_j \le 1$
- 全 $i$ で $1 + \sum_j r_{i,j} a_j > 0$（全損回避）

**解法**: 勾配 + 信頼領域法（Trust-Region）または既存 `simultaneous_kelly_fractions` / `simultaneous_kelly_fractions_scipy` の拡張。

---

## 3. フラクショナル Kelly（MDD 制約）

フル Kelly $a^*$ は MDD KPI（≤ -20%）と両立しない。

**推奨制約最適化**:

$$\min_a \mathbf{1}^\top a \quad \text{s.t.} \quad \frac{G_\infty(a)}{G_\infty(a^*)} \ge k$$

- $k \in [0.5, 0.8]$ を valid 期間のみ grid（初期値 **0.65**）
- 本番は `kelly_fraction=0.08` と併用（二重縮小を OOF で確認）

---

## 4. 状態空間の簡約（実装 v1）

全着順列は計算爆発のため **v1 は以下に簡約**:

1. 各馬 $h$ の単勝ベット（最大 2 点）
2. 軸馬 × パートナーのワイド（最大 `wide_top_n` 点）
3. 結果状態: 各ベットの的中/不的中の $2^m$ ではなく、**シミュレーション 500 サンプル**で $p_i$ を近似

→ v2 で完全列挙または MC 統合（Track B duration モデルと接続可能）。

---

## 5. ベイズ CLV（Phase 1b・任意）

| 入力 | 5分前/3分前/1分前 O3・単勝オッズ |
| 出力 | 確定オッズの事後分布 |
| ルール | 事後 EV が `ev_threshold` 未満 → **ベット棄却** |

**データ**: JV-Link 0B 速報（既存パイプライン）。学習は **学習期間のみ**。

---

## 6. 合格ゲート（OOF 2025）

| 指標 | 合格 |
|------|------|
| win+wide 合算 MDD | ≤ -20% |
| 単勝単体 ROI | 151% ± 2pp（現 baseline 維持） |
| 合算 ROI | Floor ≥ **115%** / Target ≥ **130%** |
| n_bets（合算） | ≥ 500 |

**評価**: `standard_eval_pipeline_v1` + 新スクリプト `run_portfolio_eval_gates.py`（未実装・C2-1 で追加）。

---

## 7. 本番 merge 条件

1. C2-1 OOF 合格（Floor ROI ≥115%, MDD ≤-20%, 単勝 151%±2pp）
2. E2E 拡張 PASS
3. **ゲート駆動**: 上記合格時は日付凍結より OOF ゲートを優先して本番反映可
4. **1 変数**: 配分アルゴリズムのみ変更（閾値・モデルは触らない）

### 7-1. 実装ノート（2026-06-17 追記）

- **SLSQP 勾配**: $\nabla_a G_\infty = \sum_i \frac{p_i}{1 + r_i^\top a} r_i$ を `jac` に明示（有限差分禁止）
- **フラクショナル Kelly**: 2 段階 — (1) フル Kelly で $G_\infty(a^*)$、(2) $\min_a \mathbf{1}^\top a$ s.t. $G_\infty(a)/G_\infty(a^*) \ge k$
- **CRN**: baseline / independent / portfolio で同一 `race_id` に同一 MC 乱数（比較の分散除去）

---

## 8. ロールバック

`strategy_config.json` で `portfolio_kelly_enabled: false` → 現行 per-bet Kelly + レース内キャップに復帰。

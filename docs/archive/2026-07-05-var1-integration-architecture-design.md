# var1 統合アーキテクチャ仕様書（P1）

**作成日**: 2026-07-05  
**状態**: **Approved（Phase 1a-A 着手可）**  
**前提実験**: P0 var1 ablation（因果確定）、P-43 A/B/C（HP 凍結）、Arm A 本番凍結

---

## 1. 背景と確定事項

### 1.1 P0 ablation 結論（再掲）

| 条件 | best_iter (seed42) | top-10 集中度 | TEST edge>0 | TEST n (Arm A) |
|------|-------------------|---------------|-------------|----------------|
| var1 **あり**（特徴量） | 8 | 87% | 0.30% | 32 |
| var1 **除外** | 96 | 36% | 0.94% | 104 |
| champion（var1 なし） | 130 | 38% | — | — |

**確定**: `var1_pure_score_z` を LightGBM 特徴量としてフラットに混ぜると gain 支配により学習が iter 3–8 で飽和する。  
**確定**: var1 除外で学習深度・重要度分散・edge 候補数は回復する。  
**確定**: 8-tree var1 モデルの高 ROI（97.9%, n=32）は「歪んだ出力分布への EV 閾値の偶然フィット」の可能性が高い。

### 1.2 設計原則（憲法）

1. **var1 は特徴量に入れない** — P0 で因果証明済み。`get_feature_cols` / parquet 列は残してよいが **学習・推論の feature_cols から除外**。
2. **単一経路原則（P-35 解消）** — var1 信号は **init_score 合成** または **EV 後段融合** の **どちらか一方のみ**。両方同時は禁止。
3. **HP チューニング凍結** — `reg_lambda` / `min_child_samples` 等の P-43 系列は凍結。本 Phase は **構造変更のみ**。
4. **Rule 3** — 実装 Arm ごとに **1 構造パラメータ** のみ変更（例: `beta_var1` のみ）。
5. **後出しじゃんけん禁止** — キャリブレーション係数は TRAIN/VALID 期間のみで決定。TEST は最終報告のみ。

---

## 2. 第一候補: 方針 A — 複合 `base_margin`（init_score 統合）

### 2.1 位置づけ

**主軸（Primary）として設計・実装・評価を進める。**

理由:

| 観点 | 方針 A | 方針 B |
|------|--------|--------|
| 既存アーキテクチャ整合 | `market_log_odds` が既に init_score → **自然な拡張** | EV 層に新融合 → R-6 / P-35 リスク |
| P-35 解消 | var1 は init_score のみ → **構造的に単一経路** | 融合設計を誤ると二重計上再発 |
| 学習対象の明確化 | 木は「市場+純能力で説明できない残差」のみ | 木と var1 が推論時まで分離 |
| P0 との接続 | ablation モデル（var1 なし深木）+ init_score に var1 追加 | ablation モデルをそのまま EV 融合 |

### 2.2 数式定義

レース $r$、馬 $i$ に対し:

$$
\text{init\_score}_{i} = \underbrace{\log\frac{p^{\text{mkt}}_{i}}{1-p^{\text{mkt}}_{i}}}_{\text{market\_log\_odds（既存）}} + \beta \cdot \underbrace{f(z_{i})}_{\text{var1 寄与（要キャリブレーション）}}
$$

$$
P(\text{win}_{i} \mid x) = \sigma\left(\text{init\_score}_{i} + \sum_{t=1}^{T} \Delta_t(x)\right)
$$

- $z_i$ = `var1_pure_score_z`（レース内 z-score、既存 merge スクリプト出力）
- $\beta$ = **唯一の Phase 1 構造ハイパーパラメータ**（確定 sweep: $\{0.0, 0.15, 0.25, 0.35, 0.50\}$ — TRAIN/VALID で選択）
- $f(z)$ = 第 1 実装では **恒等写像** $f(z)=z$。スケール問題が残る場合のみ $f(z)=\tanh(z)$ または winsorized $z$ を Arm A2 として追加（別 Phase）。

**木が学習するもの**: 馬場・展開・血統・調教等の **条件付き残差** のみ。

### 2.3 実装タッチポイント

| ファイル | 変更 |
|----------|------|
| `model_training/src/train.py` | `compute_composite_base_margin(df, cfg)` 新規。`train_fold` の init_score 生成を置換 |
| `model_training/config/train_config.json` | `var1_init_score: { enabled, beta, z_col, market_col }` 追加 |
| `strategy/src/inference_common.py` | `predict_model_probs` と同一式で init_score 合成（学習・推論一致） |
| `get_feature_cols` | `var1_pure_score_z` を **常時除外**（P0 恒久化） |
| `apply_var1_market_blend_probs` | **Arm A では OFF 維持**（init_score に吸収済みのため EV 層 blend 不要） |

### 2.4 キャリブレーション手順（Phase 1a）

1. **固定**: var1 特徴量除外、HP = 現行 `backtest_conservative_params`（P-43 凍結値）
2. **Sweep**: $\beta \in \{0.0, 0.15, 0.25, 0.35, 0.50\}$ — Fold 3 VALID の `binary_logloss` 最小（ROI 最大化は **禁止**）
3. **Tie-breaker（平坦曲面）**: 最良 logloss との差が **0.001 未満** の候補が複数ある場合、**$\beta > 0$ のうち最小の $\beta$** を選択。該当なしなら $\beta=0$。
4. **成功ゲート（学習健全性）**:
   - `best_iteration >= 50`（全シード）
   - `top10_concentration < 0.50`
   - var1 が feature_importance に **出現しない**
4. **成功ゲート（ベッティング前）**:
   - TEST `edge_positive_rate` > P0 ablation 単独（0.94%）または同等
   - その後 **Phase 1b**: EV 閾値チューニング（Sharpe + min_n 目的、別仕様）

### 2.5 方針 A のリスクと緩和

| リスク | 緩和 |
|--------|------|
| $\beta$ 過大 → init_score 飽和 → 再び浅木化 | VALID logloss + iter ゲート。$\beta$ upper bound = 0.5 |
| market と var1 の情報重複 | init_score は **加算 logit** に留め、木側で相殺可能。相関診断を VALID で報告 |
| NaN var1 | `fillna(0)`（z=0 はレース内平均能力）— 既存 merge 統計を manifest に記録 |
| 当日 var1 未着 | **推論フェイルセーフ**: `var1_pure_score_z` 欠落・全 NaN 時は **$\beta=0$**（`market_log_odds` のみ）で推論継続 |

### 2.6 var1.0 スコア鮮度（本番要件）

- `var1_pure_score_z` の生成は var2 推論の **ハード依存（必須アップストリーム）**。
- `var1_scores_path` / `merge_var1_pure_scores.attach_var1_score_z` で当日 scores を join。
- 最新 export が間に合わない場合: `compute_composite_base_margin` が自動的に $\beta=0$ フォールバック（ベット機会喪失を回避）。
- `inference_fallback_beta_zero_on_missing_z: true`（config 既定）。

---

## 3. 第二候補: 方針 B — 2 段アンサンブル（Fallback Arm）

### 3.1 位置づけ

**方針 A が以下のいずれかで不合格の場合のみ起動**:

- $\beta$ sweep 後も `best_iteration < 50`
- VALID logloss が var1 なし ablation より **+0.001 以上悪化**
- TEST edge 回復なし **かつ** ROI も ablation 未満

### 3.2 定義

**Stage 1（学習）**: P0 ablation と同一 — var1 なし深木、`init_score = market_log_odds` のみ。

**Stage 2（推論・EV 前）**:

$$
\text{logit}(p^{\text{final}}_{i}) = \log\frac{p^{\text{model}}_{i}}{1-p^{\text{model}}_{i}} + \gamma \cdot z_i
$$

- $p^{\text{model}}$ = Stage 1 出力（**model_prob**。blend 前）
- $\gamma$ = 構造パラメータ（$\beta$ と独立、VALID で選択）
- **禁止**: `var1_pure_score_z` を feature_cols に含めること（P-35）
- **禁止**: 現行 `apply_var1_market_blend_probs`（market + var1 同時合成）を **そのまま** 使うこと — model が既に market init_score で学習済みのため **market 二重計上** になる

### 3.3 方針 B 独自の融合関数（R-6 教訓）

現行 blend（`logit(p_market) + β*z`）は **Stage 1 が market 残差学習済み** のため不適。  
B では **model_prob 基点**:

```python
# 新規: apply_var1_model_residual_blend(model_prob, z, gamma)
logit_p = log(p_model / (1 - p_model))
logit_final = logit_p + gamma * z
p_final = sigmoid(logit_final)  # → normalize_within_race
```

EV 計算は `p_final` を使用。`model_prob` との P-37 乖離は **仕様上許容**（metrics レポートに両方記載）。

---

## 4. 評価プロトコル（共通）

### 4.1 実験 Arm

| Arm | 説明 | 変更点数 |
|-----|------|----------|
| **A0** | P0 ablation 再現（参照線） | 0 |
| **A1** | 方針 A、$\beta$ sweep 勝者 | 1（$\beta$） |
| **B1** | 方針 B、$\gamma$ sweep 勝者（A 不合格時のみ） | 1（$\gamma$） |

### 4.2 必須メトリクス

**学習健全性（最優先）**

- `best_iteration`（5 シード）
- `top10_concentration`
- var1 の feature gain（= 0 であること）

**EV 診断（`diagnose_ev_distribution.py`）**

- TEST `edge_positive_rate`
- `model_edge` p50
- `n_pass_ev_only` / `n_pass_final_arm_a`

**バックテスト（Arm A 固定: blend OFF, tuning OFF, ev=1.05）**

- Fold 3 TEST ROI, n, hit_rate
- Fold 1/2 **退行チェック**（±5pp ROI）

### 4.3 合否

| 段階 | 合格 | 不合格 |
|------|------|--------|
| 学習 | iter≥50, top10<50% | iter<50 または var1 gain>0 |
| エッジ | edge>0 ≥ ablation | edge 回復なし |
| デプロイ判断 | Phase 1b チューニングへ | Planner 差し戻し → B1 または設計見直し |

---

## 5. 実装順序（implementer 向け）

```
Phase 1a-A: compute_composite_base_margin 実装 + feature から var1 恒久除外
    ↓
Phase 1a-A: Fold 3 のみ β sweep（seed 42 × 5 β → β* 決定）
    ↓
Phase 1a-A: β* で Fold 3 を 5-seed 再学習 → 成功ゲート検証
    ↓ [Fold 3 合格]
Phase 1a-A2: 同一 β* で Fold 1, 2 一括再学習 → 全 Fold 退行チェック
    ↓ [A 合格]
Phase 1b: EV 閾値・max_picks 再チューニング（Sharpe+min_n 仕様、別 doc）
    ↓ [A 不合格]
Phase 1a-B: apply_var1_model_residual_blend 新規 + B1 sweep
```

**Fold 展開（確定）**: **直列展開**。Fold 3 で $\beta^*$ を決定し成功ゲート通過後にのみ、同一 $\beta^*$ で Fold 1/2 を一括再学習する。

---

## 6. 本番移行条件

以下を **すべて** 満たすまで champion / 8-tree var1 本番モデルは維持:

1. A1（または B1）が学習健全性ゲート合格
2. Fold 3 TEST edge>0 が ablation 以上
3. evaluator 独立検証（市場情報は init_score/EV のみ、feature 混入 grep クリーン）
4. refactorer: `var1_pure_score_z` が `get_feature_cols` に含まれないことを CI grep で確認

---

## 7. 設計決定（2026-07-05 確定 — var1 init_score）

| # | 項目 | 決定 |
|---|------|------|
| 1 | **β sweep 粒度** | $\{0.0, 0.15, 0.25, 0.35, 0.50\}$ |
| 1b | **Tie-breaker** | VALID logloss 差 **< 0.001** → **$\beta>0$ の最小値** |
| 2 | **Fold 展開** | 直列: Fold 3 → 成功ゲート → Fold 1/2 一括 |
| 3 | **var1 鮮度** | ハード依存；欠落時 **β=0 フォールバック** |

**Phase 1b**: [`2026-07-05-phase1b-ev-tuning-sharpe-design.md`](2026-07-05-phase1b-ev-tuning-sharpe-design.md)

---

## 8. 参照

- `model_training/models/p0_var1_ablation_fold3_comparison.json`
- `model_training/models/p43_*_comparison.json`
- `docs/2026-07-05-current-problems-detailed.md` — P-34, P-35
- `strategy/src/inference_common.py` — `apply_var1_market_blend_probs`（B では不使用）

---

**第一候補: 方針 A（複合 init_score）**  
**Fallback: 方針 B（model_prob 残差融合、現行 market blend 禁止）**

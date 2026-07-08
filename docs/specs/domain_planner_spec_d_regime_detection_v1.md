# Domain Planner Spec: D レジーム検知（D-1 + HMM）v1

**日付**: 2026-06-17  
**ステータス**: 確定（D-1: 凍結中に分析開始可、HMM: Phase 2）  
**担当**: domain-planner → data-generator → model-strategy-generator → backtest-evaluator

---

## 1. 背景

年別 OOF MDD: 2025 **-19.9%** ✅ / 2019 **-91.6%** / 2020 **-98.6%** / 2022 **-86.4%** ❌

共変量シフト・レジームシフトへの脆弱性が主因。カレンダー静的除外は **禁止**（後出しじゃんけん + 未来レジーム非対応）。

---

## 2. D-1: 静的ヒートマップ（分析のみ）

**データ**: 学習期間 **のみ**（2018–2024 OOF fit 以前の train 領域）

**次元**: 年 × 月 × 馬場状態（`track_condition_code`）× 必要なら `course_code`

**出力**: ROI / MDD / n_bets ヒートマップ JSON + 可視化

**禁止**: テスト fold（2025）から閾値導出  
**禁止**: ヒートマップセル単位の **自動ベット除外** を本番直接適用

**用途**: HMM 観測変数設計・condition_ev **候補**リスト化のみ。

---

## 3. D-2: HMM 動的レジーム検知

### 隠れ状態（v1）

| 状態 | 意味 |
|------|------|
| $S_1$ Risk-On | 予測可能・安定 |
| $S_2$ Risk-Off | 高ボラ・予測困難 |
| $S_3$ Crash | 構造崩壊（緊急停止） |

### 観測変数 $O_t$（レース単位または日次集計）

| 変数 | 定義 |
|------|------|
| Brier MA | 直近 100 レースの Brier score 移動平均 |
| Brier 分散 | 同上の rolling variance |
| 市場効率 | 1 番人気勝率 vs モデル implied prob の乖離 |
| オッズ乖離 | \|model_odds - market_odds\| の MA |
| 環境ノイズ | 当日全レース走破タイム分散（**結果由来・学習リーク注意**: 日次集計は **前日まで**で更新） |

**学習**: 学習期間の時系列のみ。推論: フォワード・フィルタで $P(S_t | O_{1:t})$。

**実装候補**: `hmmlearn` GaussianHMM（2〜3 状態）。GMM は v1.1。

---

## 4. D-3: レジーム連動 Kelly 乗数

| $P(S_2)+P(S_3)$ | 乗数 |
|-----------------|------|
| Risk-On  dominant | 1.0 × C2 の $a_f$ |
| Risk-Off  dominant | 0.25 × $a_f$ |
| Crash（$P(S_3)>0.5$ または月次 DD 兆候） | **0**（推奨停止） |

`monthly_dd_tracker`（-8%）と **OR 条件**で停止。HMM は事前警告層。

---

## 5. D-4: condition_ev_overrides（任意）

D-1 で **学習期間内** 3 年以上同方向の劣位セルのみ候補。  
テスト fold 診断からの導出は **禁止**（CLAUDE.md）。

---

## 6. 合格ゲート

| 指標 | 現状 | 目標 |
|------|------|------|
| 全期間 OOF MDD | -31.5% | **≤ -25%** |
| 2025 MDD | -19.9% | **維持（≤ -20%）** |
| 2025 ROI | 151% | Floor **≥ 115%** / Target **≥ 130%** |

---

## 7. エージェント境界

- **domain-planner**: 状態定義・観測変数・乗数表（本書）
- **data-generator**: 観測変数パイプライン（リーク防止 shift）
- **model-strategy-generator**: HMM fit / 推論 API / Kelly 乗数接続
- **backtest-evaluator**: 年別・全期間再評価
- **deployment-evaluator**: 本番日次レジームログ

# Domain Planner Spec: B 走破時間回帰 + MC（パラダイム移行）v1

**日付**: 2026-06-17  
**ステータス**: 草案 — **Track B 打切り後に着手**  
**前提**: binary 単勝 25% が 3 実験連続で +3pt 未満

---

## 1. 動機

1440 sweep 0 件は、離散分類（着順/勝敗）における **構造限界** を示す。走破時間は dense signal を保持する。

---

## 2. ターゲット

- **連続値**: 正規化走破時間（距離・馬場補正済み `speed_index` 逆算または `finish_time` 正規化）
- **禁止**: 当該レース確定タイムを特徴量に使用（本番計算不能 + リーク）

---

## 3. モデル

| 項目 | 仕様 |
|------|------|
| アルゴリズム | LightGBM Regressor（Huber / Pseudo-Huber loss） |
| 外れ値 | RANSAC は探索のみ、本番は Huber 固定 |
| 特徴量 | `features_v6_going_v1` ベース + 1 変数追加ルール |
| 学習分割 | `train_config.json` WF 厳守 |

---

## 4. 勝率への変換（Monte Carlo）

各馬 $i$: 予測 $\hat{t}_i$、不確実性 $\sigma_i$（残差分散または quantile）

1. レースごとに **N=10,000** サンプル
2. $t_i^{(s)} \sim \mathcal{N}(\hat{t}_i, \sigma_i^2)$（下限クリップ）
3. 各サンプルで最速馬 = 勝者
4. $\hat{p}_i = \frac{1}{N}\sum_s \mathbb{1}[\text{win}_i^{(s)}]$
5. レース内正規化 → EV / Kelly（`inference_common` 相当）

---

## 5. 計算コスト対策（Surrogate）

| 段階 | 手法 |
|------|------|
| HP / 特徴量探索 | 軽量 NN が MC 入出力を近似（N=500 で学習） |
| OOF 最終評価 | フル MC N=10,000 |

---

## 6. 合格ゲート（Track B 専用・本番 merge 別）

| 指標 | 合格 |
|------|------|
| 3 fold 同時 的中 | ≥ 25% |
| ROI | ≥ 115% |
| n_bets / fold | ≥ 100 |

本番 merge には **standard_eval 2025 ゲート** を別途クリア。

---

## 7. エージェント

- **domain-planner**: 本書
- **data-generator**: 正規化時間特徴量（shift(1) 必須）
- **model-strategy-generator**: 回帰 + MC パイプライン
- **backtest-evaluator**: バックテスト（surrogate 禁止）

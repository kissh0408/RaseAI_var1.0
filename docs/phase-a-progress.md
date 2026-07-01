# Phase-A 精度向上サイクル 進捗ログ

## ベースライン（v29_fixed, 124列）

| 指標 | 値 |
|------|-----|
| Top-1 的中率 | 30.18% |
| NDCG@3 | 0.5377 |
| Spearman | 0.5063 |
| テストレース数 | 4,775 |
| 特徴量数 | 124列 |

ブランチ: `feature/phase-a-ranking`
目標: Top-1 > 30.48%（+0.3pp 以上）

---

## 候補一覧と優先順位

| 優先順 | 候補 | 特徴量 | 状態 |
|--------|------|--------|------|
| 1 | jt_ext | hist_jockey_surface_win_rate_ts + hist_trainer_course_win_rate_ts | 採用 (+0.19pp) |
| 2 | final3f | hist_final3f_fastest_rate_ts + hist_final3f_top2_rate_ts | 不採用 (-0.25pp、重複) |
| 3 | course_dist | hist_course_dist_win_rate_ts | スキップ（既に実装済み） |
| 4 | career_prize | career_prize_cumsum | スキップ（既に実装済み） |

---

## 2026-07-02 候補1: jt_ext（騎手×馬場種別 + 調教師×競馬場）

### 追加特徴量
- `hist_jockey_surface_win_rate_ts`: 騎手 × surface_code の通算勝率（日次集計 + cumsum + shift(1)）
- `hist_trainer_course_win_rate_ts`: 調教師 × course_code の通算勝率（日次集計 + cumsum + shift(1)）

### 既存との関係
- `hist_jockey_course_win_rate`（騎手×競馬場）が既存（v29 上位特徴量11位、gain=9764）
- `hist_trainer_surface_win_rate`（調教師×馬場種別）が既存（v29 上位特徴量17位、gain=6905）
- 今回は「逆の次元」を追加：騎手×馬場種別 + 調教師×競馬場

### 結果

| 指標 | ベースライン(v29) | v33_jt_ext | 差分 |
|------|-----------------|------------|------|
| Top-1 | 30.18% | **30.37%** | +0.19pp |
| NDCG@3 | 0.5377 | 0.5385 | +0.0008 |
| Spearman | 0.5063 | 0.5071 | +0.0008 |
| Top-3 | - | 62.05% | - |

### 判定: 採用

### 採用の根拠
- 全指標でベースライン超え
- `hist_jockey_surface_win_rate_ts` が特徴量重要度9位（gain=14,863）で有効な信号（v29 の `hist_jockey_course_win_rate` 11位/gain=9764 を上回る貢献）
- `hist_trainer_course_win_rate_ts` はトップ20外だが悪化なし
- リーク兆候なし（Top-1=30.37% < 40%, Spearman=0.5071 < 0.6）

### NaN率
- `hist_jockey_surface_win_rate_ts`: 1.24%（健全）
- `hist_trainer_course_win_rate_ts`: 5.14%（健全）

---

## 2026-07-02 候補2: final3f（上がり3F最速・Top2フラグ）

### 追加特徴量
- `hist_final3f_fastest_rate_ts`: 過去レースで上がり3F最速だった割合（shift+expanding）
- `hist_final3f_top2_rate_ts`: 上がり3Fで1位or2位だった割合（同上）

### 結果

| 指標 | ベースライン(v33) | v34_final3f | 差分 |
|------|-----------------|-------------|------|
| Top-1 | 30.37% | 30.12% | -0.25pp |
| NDCG@3 | 0.5385 | 0.5380 | -0.0005 |
| Spearman | 0.5071 | 0.5087 | +0.0016 |

### 判定: 不採用（ロールバック済み）

### 不採用の原因
- Spearman(hist_final3f_fastest_rate_ts, finish_rank): 中位（特徴量重要度37位、gain=2,270）
- 最相関の既存特徴量: `field_z_last3f`（6位）、`hist_last_last3f` 系 — 上がり3Fの速さ情報が重複
- NaN率: 11.22%（time_3f_after の欠損率に相当）
- 推定原因: 既存特徴量（絶対時計＋レース内z-score）が同じ情報を既に捉えている

---

## 候補3: course_dist（コース×距離別勝率）

### 判定: スキップ（既に実装済み）

`hist_same_course_dist_win_rate`（ketto_num × course_code × distance_category, shift+expanding）として
既に create_features.py の _build_hist_features に実装済み（v33_jt_ext の特徴量リストに存在）。

---

## 候補4: career_prize（通算賞金累積）

### 判定: スキップ（既に実装済み）

`hist_total_prize`（hon_shokin の shift(1).expanding().sum()）として既に実装済み。
さらに `prize_vs_field`（レース内相対賞金差）・`field_z_prize`（レース内z-score）も存在。

---

## 第1サイクル完了サマリー（初期4候補）

| 候補 | 結果 | Top-1 変化 | 備考 |
|------|------|-----------|------|
| 1 jt_ext | 採用 | +0.19pp | hist_jockey_surface_win_rate_ts が9位 |
| 2 final3f | 不採用 | -0.25pp | field_z_last3f と重複 |
| 3 course_dist | スキップ | - | hist_same_course_dist_win_rate として既存 |
| 4 career_prize | スキップ | - | hist_total_prize として既存 |

---

## 第2サイクル（planner 新候補）

### 2026-07-02 新候補1: jd_ext（騎手×距離カテゴリ）

| 指標 | v33 | v34_jd_ext | 差分 |
|------|-----|-----------|------|
| Top-1 | 30.37% | 30.20% | -0.17pp |
| NDCG@3 | 0.5385 | 0.5378 | -0.0007 |
| Spearman | 0.5071 | 0.5069 | -0.0002 |

**判定: 不採用（ロールバック済み）**
- 特徴量重要度: 27位（gain=64,049）
- `hist_jockey_surface_win_rate_ts`（7位）・`hist_jockey_course_win_rate`（17位）と重複
- 騎手次元は surface が有効だったが distance_category は course_code に近すぎる

### 2026-07-02 新候補2: rank_trend（直近3走着順トレンド）

| 指標 | v33 | v34_rank_trend | 差分 |
|------|-----|---------------|------|
| Top-1 | 30.37% | 30.28% | -0.09pp |
| NDCG@3 | 0.5385 | 0.5384 | -0.0001 |
| Spearman | 0.5071 | 0.5072 | +0.0001 |

**判定: 不採用（ロールバック済み）**
- 特徴量重要度: 31位（gain=4,005）、NaN率 21.7%
- `hist_last_rank`（3位）・`hist_avg_rank_3/5`（7位）がすでに方向性を間接的に捉えている
- 線形傾き3走では窓が短くノイズが大きい

---

## Phase-A 全サイクル完了サマリー

| 候補 | 結果 | Top-1 変化 | 備考 |
|------|------|-----------|------|
| jt_ext | 採用 | +0.19pp | v33_jt_ext として確定 |
| final3f | 不採用 | -0.25pp | field_z_last3f と重複 |
| course_dist | スキップ | - | 既存（hist_same_course_dist_win_rate） |
| career_prize | スキップ | - | 既存（hist_total_prize） |
| jd_ext | 不採用 | -0.17pp | 騎手×距離は重複 |
| rank_trend | 不採用 | -0.09pp | 着順水準と重複、ノイズ大 |

**最終状態**: Top-1=30.37%（元ベースライン 30.18% から +0.19pp）
**停止条件（+0.3pp = 30.48%）未達のためサイクル終了**
**残り候補**: td_ext（調教師×距離）/ hist_best_speed_idx_3 は根本原因分析により試験省略

---

## 根本原因分析（Orchestrator）

### なぜ+0.19ppで頭打ちになっているか

**特徴量飽和の兆候:**
- 採用できた jt_ext は「騎手×馬場種別」という未カバーの直交次元
- 不採用候補はすべて既存特徴量のサブセットまたは変換（重複情報）
- v33_jt_ext の特徴量重要度を見ると上位20件が極めて高い gain を持ち、下位候補への余地が小さい

**モデル側の飽和:**
- num_leaves=63, n_estimators=800 の現設定は 131列に対して十分複雑
- 既存の交互作用（prize_vs_field × field_z_* 系）が強力で新特徴量の補完余地が狭い
- 正則化（reg_alpha=1.0, reg_lambda=2.0）が弱信号の特徴量を抑制している可能性

### 推奨アクション（planner への依頼内容）

単純な特徴量追加では目標（30.48%）に届く可能性が低い。以下の方向性を検討すること:

1. **モデルアーキテクチャ変更**: label_gain の調整、または `num_leaves` の縮小で bias/variance を調整
2. **アンサンブル多様性の増加**: 異なる objective（binary）のモデルと平均化
3. **エラー分析**: テストセットの「外れ予測」レース（pred_top1_avg_actual_rank が高い例）を診断して弱点を特定
4. **コーナー通過順位特徴量**: corner_4（4角通過順）は is_win との相関 r=0.209 が高い。過去走の4角平均順位は未実装

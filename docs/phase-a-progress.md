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
| 2 | final3f | hist_final3f_fastest_rate_ts + hist_final3f_top2_rate_ts | 実施中 |
| 3 | course_dist | hist_course_dist_win_rate_ts | 待機 |
| 4 | career_prize | career_prize_cumsum | 待機 |

---

## 2026-07-02 候補1: jt_ext（騎手×馬場種別 + 調教師×競馬場）

### 追加特徴量
- `hist_jockey_surface_win_rate_ts`: 騎手 × surface_code の通算勝率（日次集計 + cumsum + shift(1)）
- `hist_trainer_course_win_rate_ts`: 調教師 × course_code の通算勝率（日次集計 + cumsum + shift(1)）

### 既存との関係
- `hist_jockey_course_win_rate`（騎手×競馬場）が既存（上位特徴量11位、gain=9764）
- `hist_trainer_surface_win_rate`（調教師×馬場種別）が既存（上位特徴量17位、gain=6905）
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
- `hist_jockey_surface_win_rate_ts` が特徴量重要度9位（gain=14,863）と有効な信号（v29では `hist_jockey_course_win_rate` が11位/gain=9764 だったが、これを上回る貢献）
- `hist_trainer_course_win_rate_ts` はトップ20外だが悪化なし
- リーク兆候なし（Top-1=30.37% < 40%, Spearman=0.5071 < 0.6）

### NaN率
- `hist_jockey_surface_win_rate_ts`: 1.24%（健全）
- `hist_trainer_course_win_rate_ts`: 5.14%（健全）

### 停止条件との差分
- 元ベースライン 30.18% からの累積改善: +0.19pp
- 目標（+0.3pp = 30.48%）まで残り: **+0.11pp**

---

## 2026-07-02 候補2: final3f（上がり3F最速・Top2フラグ）

### 追加特徴量（予定）
- `hist_final3f_fastest_rate_ts`: 過去レースで上がり3F最速だった割合（shift+expanding）
- `hist_final3f_top2_rate_ts`: 上がり3Fで1位or2位だった割合（同上）

### ベースライン（v33_jt_ext 採用後）

| 指標 | 値 |
|------|-----|
| Top-1 | 30.37% |
| NDCG@3 | 0.5385 |
| Spearman | 0.5071 |

### 結果

| 指標 | ベースライン(v33) | v34_final3f | 差分 |
|------|-----------------|-------------|------|
| Top-1 | 30.37% | - | - |
| NDCG@3 | 0.5385 | - | - |
| Spearman | 0.5071 | - | - |

### 判定: 評価中

---

<!-- 以降のエントリはサイクル完了後に追記 -->

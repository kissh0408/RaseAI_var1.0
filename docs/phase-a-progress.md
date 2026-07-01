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

## サイクル完了サマリー

| 候補 | 結果 | Top-1 変化 | 備考 |
|------|------|-----------|------|
| 1 jt_ext | 採用 | +0.19pp | hist_jockey_surface_win_rate_ts が9位 |
| 2 final3f | 不採用 | -0.25pp | field_z_last3f と重複 |
| 3 course_dist | スキップ | - | hist_same_course_dist_win_rate として既存 |
| 4 career_prize | スキップ | - | hist_total_prize として既存 |

**最終状態**: Top-1=30.37%（ベースライン 30.18% から +0.19pp）  
**停止条件（+0.3pp = 30.48%）未達**  
**次のアクション**: planner に新しい特徴量候補の策定を依頼

---

## planner への推奨アクション（未実施新特徴量の候補）

以下は現在の v33_jt_ext に存在しない可能性がある方向性:

1. **騎手×距離カテゴリ勝率**: `hist_jockey_dist_win_rate_ts`（jockey × distance_category）- 中距離・長距離専門騎手の適性
2. **調教師×距離カテゴリ勝率**: `hist_trainer_dist_win_rate_ts`（trainer × distance_category）
3. **馬齢×クラス交互作用**: 年齢と格の組み合わせ（古馬重賞 vs 3歳限定）
4. **最近N走の着順トレンド**: 改善中 vs 下降中の傾向（rolling差分）
5. **連闘・間隔特徴量の精緻化**: 前走間隔 × 休養後の勝率変化
6. **ペース戦略特徴量**: コーナー通過順位の変化（追い込みパターン）

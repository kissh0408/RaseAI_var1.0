# 実装仕様書: v6 改善計画 — 28.5% 達成のための優先改善案
# 日付: 2026-06-30

---

## 1. 目的と現状

**現在**: v6（116列）, Top-1 = 27.8%
**目標**: Top-1 >= 28.5%（Phase 7 基準。市場ベンチマーク ≈ 31%）
**差分**: +0.7 pp 以上が必要

全改善案は CLAUDE.md 第1条（市場情報排除）および Rule 2（時系列リーク防止）に準拠する。
テストデータの結果を見て後付け調整することは禁止。

---

## 2. 禁止事項確認

- [x] オッズ・人気・market_log_odds・init_score を特徴量に使わない
- [x] 全改善案は shift(1) / closed='left' でリーク防止済み設計になっている
- [x] テストデータの結果を見て閾値・特徴量を後付け選択しない
- [x] Rule 2 違反（全期間集計）を含む改善案を提案しない

---

## 3. 改善案一覧と優先順位

| 優先 | 改善案 | 期待効果 | 実装コスト | CLAUDE.md 違反 |
|------|--------|--------|----------|--------------|
| 1 | A-4: n_estimators 500→800 | +0.1〜+0.2 pp | 低 | なし |
| 2 | A-1: num_leaves 31→63 | +0.2〜+0.4 pp | 低 | なし |
| 3 | C-1: 距離差特徴量の追加 | +0.1〜+0.2 pp | 低 | なし |
| 4 | B-3: running_style_code の過去傾向 | +0.2〜+0.3 pp | 中 | なし |
| 5 | B-1: 騎手×血統の交互作用 | +0.2〜+0.3 pp | 中 | なし |
| 6 | B-2: 斤量負担率（burden_weight_ratio） | +0.1〜+0.2 pp | 低 | なし |
| 7 | C-2: 間隔カテゴリ別勝率 | +0.1〜+0.2 pp | 低 | なし |
| 8 | B-4: コース×距離×馬場3次交互作用 | +0.1〜+0.2 pp | 低 | なし |
| 後回し | A-2: min_child_samples 50→30 | +0.1〜+0.2 pp | 低 | なし |
| 後回し | A-3: 正則化緩和 | +0.2〜+0.3 pp | 低 | なし |

**実施原則**: 1変更ずつ評価する。複数同時変更は効果の分離を不可能にするため禁止。
ただしモデルパラメータのグループ（A-4 → A-1 → A-2 の順）は小変更なので
段階的に実施してよい。

---

## 4. 各改善案の詳細

---

### A. モデルパラメータ調整（実施順: A-4 → A-1 → A-2 → A-3）

パラメータ調整は特徴量追加より先に実施する。理由: コストが低く、
既存 116 列の情報を最大限に活用した状態でのベースラインを確認するため。

#### A-4: n_estimators 500 → 800（推奨: 最初に実施）

| 項目 | 内容 |
|------|------|
| 変更箇所 | `train_config.json` の `n_estimators` |
| 期待改善 | +0.1〜+0.2 pp |
| 根拠 | early_stopping_rounds=50 が有効なため、実際に 800 本まで使わない場合もある。現在 500 本が制限になっている可能性がある |
| リスク | 低。学習時間が増加するが early_stopping が最適点を自動探索する |
| 過学習リスク | 低（early_stopping が train/valid ギャップを監視） |

```json
// train_config.json の変更
"n_estimators": 800
```

#### A-1: num_leaves 31 → 63

| 項目 | 内容 |
|------|------|
| 変更箇所 | `train_config.json` の `num_leaves` |
| 期待改善 | +0.2〜+0.4 pp |
| 根拠 | 現在の 31 は保守的。特徴量が 116 列に増えた現在、より複雑な分割パターンを許容することで精度向上の余地がある |
| リスク | 中。過学習傾向が出た場合、train Top-1 >> test Top-1 になる |
| 過学習判定 | train/valid/test の Top-1 乖離が 5 pp 以上の場合は 31 に戻す |
| 実施条件 | A-4 の評価完了後に実施 |

```json
// train_config.json の変更
"num_leaves": 63
```

#### A-2: min_child_samples 50 → 30（後回し）

| 項目 | 内容 |
|------|------|
| 期待改善 | +0.1〜+0.2 pp |
| 根拠 | 騎手×コース、調教師×馬場等の希少組み合わせでも分割を許容できる |
| リスク | 中。NaN が多い希少特徴量グループでの過学習リスクがある |
| 実施条件 | A-1 の評価完了後。A-1 で過学習が起きた場合は実施しない |

#### A-3: 正則化緩和（後回し）

| 項目 | 内容 |
|------|------|
| 変更案 | reg_alpha: 1.0 → 0.5, reg_lambda: 2.0 → 1.0 |
| 期待改善 | +0.1〜+0.3 pp |
| リスク | 中（特に A-1/A-2 と組み合わせると過学習リスク増大） |
| 実施条件 | A-1/A-2 完了後の最後のチューニングとして実施 |

---

### B. 特徴量の追加

#### B-3: running_style_code の過去傾向（優先度: 4位）

データ確認: `running_style_code` は SE に存在（NaN=0.0%）。

| 特徴量名 | 計算方法 | リーク防止 | 期待効果 |
|---------|---------|-----------|--------|
| hist_running_style_mode | 過去走の最頻脚質コード（shift(1)+expanding で最頻値） | ○ shift済 | 脚質適性の把握 |
| hist_front_rate | 先行（逃げ・先行）脚質の過去出走比率（shift(1)+expanding） | ○ shift済 | 前走脚質傾向 |
| field_running_style_ratio | フィールド内の先行馬比率（当該レースの逃げ/先行割合） | ○（当日情報だが観測可能） | 展開予測 |

```python
# 実装例: hist_front_rate
# running_style_code: 1=逃げ, 2=先行, 3=差し, 4=追込
df["_is_front"] = df["running_style_code"].isin([1, 2]).astype(np.int8)
df["hist_front_rate"] = (
    df.groupby("ketto_num")["_is_front"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# hist_running_style_mode（最頻脚質）
# Pandas の transform で mode を取るのはコストが高いため
# 最頻脚質の近似値として「直近3走の平均」を使う
df["hist_running_style_avg3"] = (
    df.groupby("ketto_num")["running_style_code"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# field_running_style_ratio（逃げ・先行馬の比率）
df["field_front_ratio"] = df.groupby("race_id")["_is_front"].transform("mean")
```

| 実装コスト | 期待改善 | リスク |
|----------|--------|------|
| 中 | +0.2〜+0.3 pp | 低（NaN なし） |

#### B-1: 騎手×血統の交互作用（優先度: 5位）

| 特徴量名 | 計算方法 | リーク防止 | 期待効果 |
|---------|---------|-----------|--------|
| hist_jockey_sire_win_rate | jockey_code × sire_id の組み合わせ通算勝率（日次集計+shift(1)） | ○ 日次集計+shift済 | 特定騎手×種牡馬の相性 |

```python
# 実装例: 日次集計パターン（既存 _build_jockey_trainer_features に倣う）
js_daily = (
    df.groupby(["jockey_code", "sire_id", "race_date"], observed=True)
    .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
    .reset_index()
    .sort_values(["jockey_code", "sire_id", "race_date"])
)
grp_js = js_daily.groupby(["jockey_code", "sire_id"], observed=True)
js_daily["cum_wins"]  = grp_js["d_wins"].cumsum()
js_daily["cum_races"] = grp_js["d_races"].cumsum()
js_daily["cum_wins_prev"]  = grp_js["cum_wins"].shift(1)
js_daily["cum_races_prev"] = grp_js["cum_races"].shift(1)

# 最低レース数未満は NaN（希少組み合わせのノイズ抑制）
MIN_JS_RACES = 5  # ジョッキー×種牡馬は組み合わせが多く MIN を緩める
js_daily["hist_jockey_sire_win_rate"] = (
    js_daily["cum_wins_prev"] / js_daily["cum_races_prev"]
)
js_daily.loc[js_daily["cum_races_prev"] < MIN_JS_RACES, "hist_jockey_sire_win_rate"] = np.nan
```

| 実装コスト | 期待改善 | NaN率見通し | リスク |
|----------|--------|----------|------|
| 中 | +0.2〜+0.3 pp | 60〜75%（希少組み合わせが多い） | 低（NaN は LightGBM が処理） |

`sire_id` が SE にないため `_build_sire_features` が実行済みの df（sire_id 列が存在する状態）で実装すること。

#### B-2: 斤量負担率（優先度: 6位）

データ確認: `burden_weight`（NaN=0.0%）、`horse_weight`（NaN=0.6%）は SE に存在。

| 特徴量名 | 計算方法 | リーク防止 | 期待効果 |
|---------|---------|-----------|--------|
| burden_weight_ratio | burden_weight / horse_weight（当該レース情報、観測可能） | リーク対象外（レース前に確定） | 斤量負担の相対的な重さ |
| hist_weight_change_trend | 直近3走の馬体重変化の平均（shift(1)+rolling(3)） | ○ shift済 | 増減傾向の把握 |

```python
# 斤量負担率（当日情報だがレース前に確定→リーク対象外）
df["burden_weight_ratio"] = df["burden_weight"] / df["horse_weight"].replace(0, np.nan)

# 馬体重変化トレンド（hist_weight_change は前走分、trended 版を追加）
df["hist_weight_change_trend"] = (
    df.groupby("ketto_num")["horse_weight_change"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
```

| 実装コスト | 期待改善 | リスク |
|----------|--------|------|
| 低 | +0.1〜+0.2 pp | 低（NaN 0.6% のみ） |

#### B-4: コース×距離×馬場状態の3次交互作用（優先度: 8位）

既存の `hist_same_course_dist_win_rate`（コース×距離帯）に馬場状態を加えた拡張。

| 特徴量名 | 計算方法 | リーク防止 | 期待効果 |
|---------|---------|-----------|--------|
| hist_course_dist_cond_win_rate | ketto_num × course_code × distance_category × track_condition_code の勝率 | ○ shift済 | 細かい適性マッチング |

```python
df["hist_course_dist_cond_win_rate"] = (
    df.groupby(
        ["ketto_num", "course_code", "distance_category", "track_condition_code"],
        observed=True,
    )["is_win"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
```

| 実装コスト | 期待改善 | NaN率見通し | リスク |
|----------|--------|----------|------|
| 低 | +0.1〜+0.2 pp | 80〜90%（4次元交互作用は希少） | 低（NaN は LightGBM が処理） |

---

### C. 特徴量エンジニアリングの改良

#### C-1: 距離差特徴量（優先度: 3位）

| 特徴量名 | 計算方法 | リーク防止 | 期待効果 |
|---------|---------|-----------|--------|
| hist_last_dist_diff | 前走距離 - 今回距離（負=距離短縮, 正=距離延長） | ○（前走距離 = shift(1)済み情報） | 距離変化の影響把握 |
| hist_avg_dist_3_diff | 直近3走の平均距離 - 今回距離 | ○ shift済 | 距離適性ゾーンとの乖離 |

```python
# 前走距離の取得（shift(1)で当該レース除外）
df["_hist_last_dist"] = df.groupby("ketto_num")["distance"].transform(
    lambda x: x.shift(1)
)
df["hist_last_dist_diff"] = df["_hist_last_dist"] - df["distance"]

# 直近3走の平均距離
df["_hist_avg_dist_3"] = df.groupby("ketto_num")["distance"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df["hist_avg_dist_3_diff"] = df["_hist_avg_dist_3"] - df["distance"]

df = df.drop(columns=["_hist_last_dist", "_hist_avg_dist_3"])
```

| 実装コスト | 期待改善 | NaN率見通し | リスク |
|----------|--------|----------|------|
| 低 | +0.1〜+0.2 pp | 15〜20%（初出走分） | 低 |

#### C-2: 間隔カテゴリ別勝率（優先度: 7位）

既存の `hist_days_since_last`（日数）を離散化し、間隔カテゴリ別の勝率を追加する。

| 特徴量名 | 計算方法 | リーク防止 | 期待効果 |
|---------|---------|-----------|--------|
| hist_interval_category | hist_days_since_last をビン化（連闘/標準/中間/長期） | ○（hist_days_since_lastから派生、リーク不要） | 間隔効果の非線形把握 |
| hist_interval_win_rate | interval_category × ketto_num の勝率（shift(1)+expanding） | ○ shift済 | 馬固有の間隔適性 |

```python
# ビン定義: 連闘(0-8日), 短期(9-21日), 標準(22-42日), 中間(43-84日), 長期(85日以上)
def categorize_interval(days):
    if pd.isna(days):
        return np.nan
    if days <= 8:
        return 0  # 連闘
    elif days <= 21:
        return 1  # 短期
    elif days <= 42:
        return 2  # 標準
    elif days <= 84:
        return 3  # 中間
    else:
        return 4  # 長期

df["hist_interval_category"] = df["hist_days_since_last"].map(categorize_interval)

# 間隔カテゴリ別の各馬の勝率
df["hist_interval_win_rate"] = (
    df.groupby(["ketto_num", "hist_interval_category"], observed=True)["is_win"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
```

| 実装コスト | 期待改善 | リスク |
|----------|--------|------|
| 低 | +0.1〜+0.2 pp | 低 |

---

## 5. 実施スケジュール（推奨順）

### ステージ 1: パラメータ調整（コスト最小・即効性あり）

以下を1変更ずつ評価する。評価基準: valid Top-1 が前バージョンを上回ること。

```
変更 1: n_estimators 500 → 800 → evaluator に評価依頼
変更 2: num_leaves 31 → 63 → evaluator に評価依頼
（A-2, A-3 は過学習確認後に判断）
```

### ステージ 2: 特徴量追加（順番は独立しているため任意だが以下を推奨）

```
追加 1: C-1（距離差特徴量）→ 実装コスト低・効果明確
追加 2: B-3（running_style_code 過去傾向）→ NaN なし・効果中
追加 3: B-1（騎手×血統交互作用）→ 既存パターンの拡張
追加 4: B-2（斤量負担率）→ 低コスト
追加 5: C-2（間隔カテゴリ別勝率）→ 低コスト
追加 6: B-4（3次交互作用）→ NaN 率が高い場合は後回し
```

各追加後に evaluator が評価し、改善が確認されたものだけを次のバージョンに組み込む。
改善が見られない（Top-1 変化 0.1 pp 未満）場合は、その特徴量を削除して次に進む。

---

## 6. CLAUDE.md 違反チェック

| 改善案 | 市場情報混入 | Rule 2 違反 | 後出しじゃんけん |
|--------|-----------|-----------|--------------|
| A-1〜A-4（パラメータ） | なし | なし | なし |
| B-1（騎手×血統） | なし | なし（日次集計+shift） | なし |
| B-2（斤量負担率） | なし | なし（現走情報） | なし |
| B-3（running_style） | なし | なし（shift(1)） | なし |
| B-4（3次交互作用） | なし | なし（shift(1)） | なし |
| C-1（距離差） | なし | なし（shift(1)） | なし |
| C-2（間隔カテゴリ） | なし | なし（shift(1)） | なし |

全改善案で市場情報・Rule 2 違反は**なし**。

---

## 7. implementer への引き渡し事項

### ステージ 1（パラメータ調整）

1. `train_config.json` の `n_estimators` を 500 → 800 に変更してトレーニングを実行
2. evaluator による評価後、有効なら `num_leaves` を 31 → 63 に変更
3. 各変更のバージョンを `features_version` ではなく `train_config.json` のコメントで管理（設定変更のみ）

### ステージ 2（特徴量追加）

各特徴量追加時の手順:
1. `create_features.py` の適切な SECTION に追加（SECTION 番号は既存コードに従う）
2. `features_version` をインクリメント（v3cj → v7, v8... のように連番）
3. `features_*.parquet` の上書き前に自動バックアップが動作することを確認
4. 市場情報混入チェックを実行:
   ```
   grep -rn "odds\|popularity\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
   ```
5. evaluator に評価を依頼

### リーク停止閾値（再掲）

```
Top-1 > 40% または Spearman > 0.6 → 即座に実装停止し evaluator に報告
この閾値を超える精度は合格ではなく、データリークの危険信号
```

---

## 8. 付録: v6 の既存特徴量（参考）

v6 で既に実装済みの主要特徴量カテゴリ（重複実装を防ぐための確認用）:

- 過去走成績: hist_last_rank, hist_avg_rank_3/5, hist_win_rate, hist_place_rate
- タイム系: hist_last_last3f, hist_avg_last3f_3/5, hist_last_time_dev, hist_avg_time_dev_3/5
- 馬場条件別: hist_same_surface/condition/course/dist_win_rate, hist_same_weather_win_rate
- 状態系: hist_days_since_last, hist_weight_change, hist_total_prize, hist_avg_prize_3
- 血統系: hist_sire_win_rate_ts, hist_sire_surface/dist_win_rate_ts, hist_bms_win_rate_ts
- クラス移動: hist_best_grade_ever, hist_grade_diff, hist_avg_rank_top_grade
- 騎手: hist_jockey_win_rate_cum/30d/60d, hist_jockey_course_win_rate
- 調教師: hist_trainer_win_rate_cum/30d/60d, hist_trainer_surface_win_rate
- 速度指数: hist_speed_idx_last/best/avg3/cond_best
- 調教: trn_hc_*/trn_wc_*（HC/WC ベース）
- レース内相対: field_avg_win_rate, win_rate_vs_field, prize_vs_field
- その他: season_sex_score, wakuban_surface

**本仕様書で追加する特徴量はこのリストに存在しないことを確認済み。**

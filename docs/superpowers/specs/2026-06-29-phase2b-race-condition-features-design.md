# 実装仕様書: Phase 2b — レース条件特徴量の強化

**日付**: 2026-06-29
**対象**: RaceAI_var1.0 — 市場情報なし純粋能力LambdaRank
**フェーズ**: Phase 2b（天候・コース複合・フィールド強度）
**前提**: Phase 1 完了（Top-1=25.5%）+ 調教特徴量追加済み（Top-1=26.5%、features_v2）

---

## 目的

`weather_code`（天候）・コース×距離複合・フィールド強度を追加し、
純粋能力モデルの精度を Top-1 > 27% へ引き上げる。

市場ベンチマーク（1番人気 Top-1 ≈ 30〜33%）との差を縮めることが目標。
本 Phase で追加する特徴量はすべて「レース前に観測可能な客観情報」であり、
オッズ・人気は一切含まない。

---

## 禁止特徴量の確認

- [x] オッズ系データを一切含まないことを確認した（weather_code はレース前観測可能な気象情報）
- [x] 人気順位を含まないことを確認した
- [x] market_log_odds / init_score を含まないことを確認した

実装後の確認コマンド:
```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

---

## 追加する特徴量（11 列）

### SECTION 4 追加: CURRENT FEATURES

#### weather_code（追加コスト: ゼロ）

| 列名 | ソース | 計算方法 | リーク防止 | 期待効果 |
|------|--------|---------|-----------|---------|
| `weather_code` | RA | そのまま使用 | 不要（レース前観測可能） | 雨天馬場での得意・不得意は馬固有の特性。categorical として渡すだけで LightGBM が活用できる |

**重要**: `weather_code` は現在すでに `ra_merge_cols` 経由で df に存在している。
`create_features.py` の SECTION 4 と `train_config.json` の `categorical` リストへ追記するだけで使用可能。
preprocess.py の変更は不要。

```python
# SECTION 4 では特別な計算不要。categorical 指定のみ。
# weather_code: 1=晴, 2=曇, 3=雨, 4=小雨, 5=雪, 6=小雪
```

#### フィールド強度特徴量（hist_ 計算後に計算する）

フィールド強度特徴量は SECTION 3（HISTORICAL）の計算が完了した後に、
同レース出走馬の過去成績を集約して生成する。
「今回のメンバー構成の強さ」という文脈情報であり、リーク防止不要（レース前確定情報）。

| 列名 | ソース | 計算方法 | リーク防止 | 期待効果 |
|------|--------|---------|-----------|---------|
| `field_avg_win_rate` | hist_win_rate の race_id グループ平均 | `groupby('race_id')['hist_win_rate'].transform('mean')` | 不要 | フィールド強度の代理指標。強豪揃いか否かを数値化 |
| `field_avg_prize` | hist_avg_prize_3 の race_id グループ平均 | `groupby('race_id')['hist_avg_prize_3'].transform('mean')` | 不要 | 賞金ベースの競走レベル指標 |
| `win_rate_vs_field` | hist_win_rate - field_avg_win_rate | 差分計算 | 不要 | 自馬の相対的強さ。LambdaRank の相対評価を補強する |
| `prize_vs_field` | hist_avg_prize_3 - field_avg_prize | 差分計算 | 不要 | 賞金ベースの相対強度 |

NaN 処理の方針: `hist_win_rate` が NaN（新馬・出走歴なし）の場合は 0 として扱い、
`field_avg_win_rate` の計算に含める。ただし `win_rate_vs_field` 自体は NaN のままとする。

---

### SECTION 3 追加: HISTORICAL FEATURES（全て shift(1) でリーク防止）

#### 天候適性

| 列名 | ソース | 計算方法 | リーク防止 | 期待効果 |
|------|--------|---------|-----------|---------|
| `hist_same_weather_win_rate` | SE + RA(weather_code) | `groupby(['ketto_num', 'weather_code'])['is_win'].transform(lambda x: x.shift(1).expanding().mean())` | shift(1) | 晴れ得意・雨得意の馬固有傾向を定量化 |
| `hist_same_weather_avg_rank` | SE + RA(weather_code) | `groupby(['ketto_num', 'weather_code'])['finish_rank'].transform(lambda x: x.shift(1).expanding().mean())` | shift(1) | 同天候での平均着順（勝率が疎な天候コードでも機能） |

注意: `weather_code` は現在 df に結合済み（ra_merge_cols に含まれる）。
ただし SECTION 3 の groupby を実行する時点で `sort_values(['ketto_num', 'race_date'])` が必要なため、
既存の _build_hist_features() 内のソート順を継承すること。

#### コース×距離帯複合適性

| 列名 | ソース | 計算方法 | リーク防止 | 期待効果 |
|------|--------|---------|-----------|---------|
| `hist_same_course_dist_win_rate` | SE + RA | `groupby(['ketto_num', 'course_code', 'distance_category'])['is_win'].transform(lambda x: x.shift(1).expanding().mean())` | shift(1) | 「東京×マイル」「阪神×長距離」等の具体的な得意条件。`hist_same_course_win_rate` + `hist_same_dist_win_rate` より特異的 |

既存の `hist_same_course_win_rate`（距離不問）と `hist_same_dist_win_rate`（競馬場不問）の
組み合わせで表現できない複合適性をとらえる。

#### グレード適性

| 列名 | ソース | 計算方法 | リーク防止 | 期待効果 |
|------|--------|---------|-----------|---------|
| `hist_same_grade_win_rate` | SE + RA(grade_code) | `groupby(['ketto_num', 'grade_code'])['is_win'].transform(lambda x: x.shift(1).expanding().mean())` | shift(1) | 同クラス・同グレードでの勝率。クラス違いの強さ評価に必要 |
| `hist_top_grade_exp_count` | SE + RA(grade_code) | grade_code ∈ {1,2,3}（G1/G2/G3）の過去出走数の expanding sum | shift(1) | 重賞経験の有無。上位クラスで初出走 vs 場慣れの違いを定量化 |

`hist_top_grade_exp_count` の実装:
```python
df['_is_top_grade'] = df['grade_code'].isin([1, 2, 3]).astype(np.int8)
df['hist_top_grade_exp_count'] = df.groupby('ketto_num')['_is_top_grade'].transform(
    lambda x: x.shift(1).expanding().sum()
)
df = df.drop(columns=['_is_top_grade'])
```

#### 精細距離適性

| 列名 | ソース | 計算方法 | リーク防止 | 期待効果 |
|------|--------|---------|-----------|---------|
| `hist_exact_dist_win_rate` | SE + RA(distance) | 距離を 100m 単位でビニング後に groupby。既存 `hist_same_dist_win_rate`（距離帯 ±400m）より精細 | shift(1) | 1400m 得意だが 1600m では苦手、等の精細な距離適性を捉える |

`hist_exact_dist_win_rate` の実装:
```python
df['_dist_bin_100'] = (df['distance'] // 100) * 100  # 1200m, 1300m, 1400m, ...
df['hist_exact_dist_win_rate'] = df.groupby(['ketto_num', '_dist_bin_100'])['is_win'].transform(
    lambda x: x.shift(1).expanding().mean()
)
df = df.drop(columns=['_dist_bin_100'])
```

---

## 実装見送り（Phase 2b 対象外）

### B: レースペース特徴量（lap_times）

`lap_times`・`time_3f_before`（前3F）は RA の生フィールドだが、
現在 `_RA_SOURCE_COLS_FROM_HD`（preprocess.py）に含まれておらず、
var2.0.0 の horse_data.parquet にこれらのフィールドが存在するかが未確認。

**Phase 2b では実装しない。** 以下の手順で Phase 3 以降に持ち越す:
1. var2.0.0 の horse_data.parquet の列一覧を確認する
2. `time_3f_before`（前半タイム）が存在すれば preprocess.py の `_RA_SOURCE_COLS_FROM_HD` に追加
3. ペースカテゴリ（前半/全体タイム比率で HA/MS/HS 分類）を SECTION 4 に追加

---

## preprocess.py の変更

**変更不要。**

`weather_code` はすでに `_RA_SOURCE_COLS_FROM_HD` に含まれており、
`RA_preprocessed.parquet` に保存済み。`create_features.py` の `ra_merge_cols` にも
すでに含まれているため、df には結合済みの状態にある。

追加すべき変更は `create_features.py` のみ。

---

## create_features.py への追加方法

### 変更箇所 1: SECTION 3 末尾（_build_hist_features 内）

`hist_total_prize` の計算ブロック（賞金系）の後に以下を追記する:

```
# ─── 天候適性系 ───────────────────────────────────────────────────────────────
hist_same_weather_win_rate
hist_same_weather_avg_rank

# ─── コース×距離帯複合 ──────────────────────────────────────────────────────
hist_same_course_dist_win_rate

# ─── グレード適性系 ───────────────────────────────────────────────────────────
hist_same_grade_win_rate
hist_top_grade_exp_count

# ─── 精細距離適性 ─────────────────────────────────────────────────────────────
hist_exact_dist_win_rate
```

### 変更箇所 2: SECTION 4（_build_current_features 内）

既存の `season_sex_score` / `wakuban_surface` の後に以下を追記する:

```
# ─── フィールド強度（SECTION 3 計算後に依存） ────────────────────────────────
field_avg_win_rate    # hist_win_rate.fillna(0) の race_id グループ平均
field_avg_prize       # hist_avg_prize_3 の race_id グループ平均
win_rate_vs_field     # hist_win_rate - field_avg_win_rate
prize_vs_field        # hist_avg_prize_3 - field_avg_prize
```

**注意**: `_build_current_features()` は `_build_hist_features()` の後に呼ばれるため、
`hist_win_rate` と `hist_avg_prize_3` は既に df に存在している。

### 変更箇所 3: main() の呼び出し順序

現在の呼び出し順序は変更不要:
```
[3] _build_hist_features    # SECTION 3 追記
[4] _build_current_features # SECTION 4 追記（hist_ 計算後なので依存関係OK）
[5] _build_sire_features    # 変更なし
[6] _add_training_features  # 変更なし
```

---

## features_version の扱い

`features_v2.parquet`（調教特徴量あり）から `features_v3.parquet` へアップする。

理由: 特徴量セットが変わるため学習モデルとの対応関係を保持する必要がある。
v2 を上書きすると調教特徴量のみのモデルが再現不可能になる。

`train_config.json` の変更事項:
```json
"features_version": "v3"
```

バックアップ: create_features.py の既存ロジックが `features_v2.bak.parquet` を自動作成するため、
implementer は追加のバックアップ作業は不要。

---

## train_config.json の変更事項

### 変更箇所

```json
"data": {
  "features_version": "v3"   // "v2" → "v3" に変更
},
"features": {
  "categorical": [
    "surface_code",
    "track_condition_code",
    "surface_condition",
    "course_code",
    "grade_code",
    "distance_category",
    "sex_code",
    "weather_code"             // 追加
  ]
}
```

### モデルパラメータ変更なし

特徴量を追加するだけなので LambdaRank パラメータ（num_leaves=31, n_estimators=500 等）は
変更しない。特徴量効果を純粋に測定するため、1変更ずつ行う原則を守る。

---

## 評価基準

| 指標 | Phase 2b 合格 | ベースライン（features_v2） |
|------|-------------|--------------------------|
| Top-1 的中率 | > 27% | 26.5% |
| Top-3 的中率 | > 52% | — |
| NDCG@3 | > 0.50 | — |
| Spearman相関 | > 0.48 | — |
| テスト件数 | 500レース以上 | — |

リーク停止閾値: Top-1 > 40% または Spearman > 0.6 → 即座に実装停止・evaluator へ報告

---

## implementer への引き渡し事項

以下を順番に実施すること:

1. `pure_rank/config/train_config.json` を変更する
   - `features_version`: "v2" → "v3"
   - `features.categorical` リストに `"weather_code"` を追加

2. `pure_rank/src/create_features.py` の `_build_hist_features()` 末尾（賞金系の後）に
   天候適性・コース×距離帯・グレード適性・精細距離の 6 列を追加する

3. `pure_rank/src/create_features.py` の `_build_current_features()` に
   フィールド強度の 4 列（field_avg_win_rate / field_avg_prize / win_rate_vs_field / prize_vs_field）を追加する
   - `hist_win_rate` と `hist_avg_prize_3` を使うため、SECTION 3 完了後に計算すること

4. `python pure_rank/src/create_features.py` を実行して `features_v3.parquet` を生成する

5. `python pure_rank/src/train.py --ensemble` を実行して 5 シードアンサンブルで学習する

6. `python pure_rank/src/evaluate.py` を実行して evaluator へ結果を渡す

7. 市場情報混入チェックを実行する:
   ```bash
   grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
   ```

8. NaN 率が 30% を超える列があれば evaluator へ報告する（新馬・初コースは許容）

---

## NaN 率の見込み

| 列名 | 予想 NaN 率 | 理由 |
|------|-----------|------|
| `hist_same_weather_win_rate` | ～20% | 初出走・当該天候での経験なし |
| `hist_same_weather_avg_rank` | ～20% | 同上 |
| `hist_same_course_dist_win_rate` | ～30% | 競馬場×距離帯の組み合わせが細かい |
| `hist_same_grade_win_rate` | ～15% | 同グレード初出走 |
| `hist_top_grade_exp_count` | ～5% | 0（非重賞馬）は NaN ではなく 0 になる |
| `hist_exact_dist_win_rate` | ～30% | 同距離100m帯での初出走 |
| `field_avg_win_rate` | ～5% | hist_win_rate が全馬 NaN の場合のみ |
| `win_rate_vs_field` | ～20% | hist_win_rate が NaN の馬 |

LightGBM は NaN を自動で欠損値分岐として扱うため、30% 以下の NaN は精度上の問題なし。

---

## 禁止事項の確認

- [x] 単勝オッズ・人気を特徴量に含めない
- [x] market_log_odds / init_score を使わない
- [x] ROI・回収率で合否を判定しない
- [x] テストデータの結果で特徴量を後付け選択しない
- [x] features_v2.parquet をバックアップなしに上書きしない（v3 として新規生成）
- [x] weather_code はレース前観測可能な情報（リーク防止不要）

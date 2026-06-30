# Phase 1 特徴量設計書

**日付**: 2026-06-29  
**対象**: RaceAI_var1.0 — 市場情報なし純粋能力LambdaRank  
**フェーズ**: Phase 1（過去走成績ベースライン + 現状基本情報）

---

## 目的

市場情報（オッズ・人気）を一切使わず、過去走成績と現状レース条件のみで着順を予測するベースラインモデルを構築する。  
目標: Top-1 > 25%（Phase 1 基準）、Phase 7 実績（Top-1=28.5%）を将来的に超えること。

---

## アーキテクチャ方針

**アプローチ**: C（節分離型1ファイル）  
`create_features.py` 内を `HISTORICAL FEATURES` セクションと `CURRENT FEATURES` セクションに明確分離。  
列名プレフィックスで識別: 過去系=`hist_`、現状系はそのまま。

---

## ディレクトリ構造

```
pure_rank/
├── config/
│   └── train_config.json
├── data/
│   ├── 01_preprocessed/
│   │   ├── SE_preprocessed.parquet
│   │   ├── RA_preprocessed.parquet
│   │   └── SK_preprocessed.parquet
│   └── 02_features/
│       └── features_v1.parquet
├── models/
│   └── lambdarank_fold*_seed*.txt
└── src/
    ├── preprocess.py
    ├── create_features.py
    ├── train.py
    └── evaluate.py
```

---

## データフロー

```
common/data/src/ の JV-Link CSV (SE, RA, SK)
    ↓  pure_rank/src/preprocess.py
pure_rank/data/01_preprocessed/  [SE / RA / SK parquet]
    ↓  pure_rank/src/create_features.py
pure_rank/data/02_features/features_v1.parquet
    ↓  pure_rank/src/train.py
pure_rank/models/
```

---

## 前処理スクリプト（preprocess.py）

var2.0.0 の preprocessing.py パターンを参考に独自実装。  
出力: SE_preprocessed.parquet / RA_preprocessed.parquet / SK_preprocessed.parquet

**SE 前処理**:
- `race_id` = year + month_day + course_code + kai + nichi + race_num（ゼロ埋め結合）
- `race_date` = year + month_day を datetime 変換
- `finish_rank` = final_rank（int、0は除外フィルタ対象）
- `time_sec` = time フィールドを秒換算 float（例: "2023" → 202.3秒）
- `time_3f_sec` = time_3f_after を秒換算 float
- `weight_change_signed` = weight_change_sign × weight_change（符号付き体重変化）
- `is_win` = finish_rank == 1
- `is_place` = finish_rank <= 3

**除外フィルタ（必須）**:
```python
df = df[
    (~df['grade_code'].isin([8, 9])) &
    (~df['abnormal_code'].isin([1, 3, 4])) &
    (df['horse_count'] >= 5) &
    (df['finish_rank'] > 0)
]
```

**RA 前処理**:
- `race_id` 生成（SEと同一ロジック）
- `surface_code` = track_code の先頭1文字（1=芝, 2=ダート）
- `track_condition_code` = surface_code に応じて turf_condition / dirt_condition を選択
- `surface_condition` = surface_code * 10 + track_condition_code（複合コード）
- `distance_category` = 距離帯カテゴリ化（〜1400=短距離, 〜1800=マイル, 〜2200=中距離, 2201〜=長距離）

**SK 前処理**:
- `ketto_num` をキーとして p_sire（父登録番号）、p_dam_sire（母父登録番号）を抽出

---

## 特徴量リスト

### HISTORICAL FEATURES（hist_ プレフィックス、全て shift(1) でリーク防止）

| 列名 | ソース | 計算方法 |
|------|--------|---------|
| `hist_last_rank` | SE | 前1走着順（shift(1)） |
| `hist_avg_rank_3` | SE | 前3走着順平均（shift(1)+rolling(3)） |
| `hist_avg_rank_5` | SE | 前5走着順平均（shift(1)+rolling(5)） |
| `hist_win_rate` | SE | 通算勝率（shift(1)+expanding） |
| `hist_place_rate` | SE | 通算複勝率・3着以内率（shift(1)+expanding） |
| `hist_last_last3f` | SE | 前1走上がり3F（秒float） |
| `hist_avg_last3f_3` | SE | 前3走上がり3F平均 |
| `hist_avg_last3f_5` | SE | 前5走上がり3F平均 |
| `hist_last_time_dev` | SE | 前1走走破タイム偏差（同距離帯平均との差） |
| `hist_avg_time_dev_3` | SE | 前3走タイム偏差平均 |
| `hist_avg_time_dev_5` | SE | 前5走タイム偏差平均 |
| `hist_best_time_same_cond` | SE+RA | 同距離帯×同馬場種別×同馬場状態での最速タイム |
| `hist_same_course_win_rate` | SE+RA | 同競馬場での勝率 |
| `hist_same_surface_win_rate` | SE+RA | 同馬場種別（芝/ダート）勝率 |
| `hist_same_condition_win_rate` | SE+RA | 同馬場状態（良/稍重/重/不良）勝率 |
| `hist_surface_condition_win_rate` | SE+RA | 同馬場種別×同状態 複合勝率 |
| `hist_same_dist_win_rate` | SE | 同距離帯（±200m）勝率 |
| `hist_days_since_last` | SE | 前走からの休養日数 |
| `hist_weight_change` | SE | 馬体重変化（符号付き） |
| `hist_total_prize` | SE | 通算獲得賞金（expanding sum） |
| `hist_avg_prize_3` | SE | 前3走賞金平均 |
| `hist_sire_surface_win_rate` | SE+SK | 父馬産駒の同馬場種別勝率 |
| `hist_sire_dist_diff` | SE+SK | 父馬産駒の平均勝ち距離 vs 今回距離の差 |
| `hist_bms_win_rate` | SE+SK | 母父（BMS）産駒の通算勝率 |

### CURRENT FEATURES（当該レース固定情報、リーク防止不要）

| 列名 | ソース | 備考 |
|------|--------|------|
| `distance` | RA | 距離（数値） |
| `distance_category` | RA | 短距離/マイル/中距離/長距離（categorical） |
| `surface_code` | RA | 1=芝, 2=ダート（categorical） |
| `track_condition_code` | RA | 1=良〜4=不良（categorical） |
| `surface_condition` | RA | 馬場種別×状態 複合コード（categorical） |
| `course_code` | RA | 競馬場コード（categorical） |
| `grade_code` | RA | グレード（categorical） |
| `horse_count` | RA | 出走頭数 |
| `age` | SE | 馬齢 |
| `sex_code` | SE | 性別コード（categorical） |
| `burden_weight` | SE | 負担重量 |
| `wakuban` | SE | 枠番 |
| `wakuban_surface` | SE+RA | 枠番×馬場交互作用（芝=+1, ダート=−1）×wakuban |
| `season_sex_score` | SE+RA | 季節×性別スコア: cos(2π×day_of_year/365)×sex_sign |

### ラベル列

| 列名 | 内容 |
|------|------|
| `finish_rank` | 確定着順（LambdaRankラベル変換元） |
| `is_win` | 1着フラグ（Binaryベースライン用） |
| `lr_label` | LambdaRank用ラベル（レース内着順逆転: 1着=頭数-1, 最下位=0） |

---

## 時系列分割

```python
TRAIN_END = '2023-12-31'
VALID_END = '2024-12-31'
# TEST: 2024-01-01以降
```

---

## カテゴリ特徴量（lgb.Dataset に必ず指定）

```python
CATEGORICAL_FEATURES = [
    'surface_code', 'track_condition_code', 'surface_condition',
    'course_code', 'grade_code', 'distance_category', 'sex_code',
]
```

---

## リーク防止パターン（全 hist_ 系に適用）

```python
df.sort_values('race_date').groupby('horse_id')[col].transform(
    lambda x: x.shift(1).rolling(N, min_periods=1).mean()
)
```

---

## リーク停止閾値

Top-1 > 40% または Spearman > 0.6 → 即座に実装停止・evaluator へ報告

---

## 禁止事項確認

- [ ] オッズ・人気を含まない
- [ ] market_log_odds / init_score を使わない
- [ ] shift(1) で当該レース除外
- [ ] grade_code=8,9 / abnormal_code=1,3,4 を除外
- [ ] features_*.parquet バックアップ後上書き

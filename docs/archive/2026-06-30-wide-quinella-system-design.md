# 実装仕様書: ワイド・馬連最適化システム — 2026-06-30

## 禁止特徴量の確認

- [x] オッズ系データを一切含まないことを確認した
- [x] 人気順位を含まないことを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した
- [x] HR 払戻データは評価・シミュレーション用のみ（特徴量に使わない）

---

## 1. 設計概要

### 目的

現在の LambdaRank モデル（Top-1=30.2%）は1着馬を高精度で特定することに最適化されている。
ワイド・馬連への拡張には「2頭が共に3着以内に入る確率」の推定が必要であり、これは
Plackett-Luce 確率モデルとレース内相対特徴量によって実現する。

市場情報（オッズ・人気）を一切使わない制約を維持しながら、以下を達成する。

| 目標 | Phase | 基準値（Phase A 測定後に確定） |
|------|-------|-------------------------------|
| Wide Pair Coverage > 22% | Phase B 完了後 | Phase A で測定 |
| Top-3 Coverage > 55% | Phase A 完了後 | — |
| Top-1 維持 (>29%) | Phase B 完了後 | 現在 30.2% |

### 現在のシステム状態（基点）

```
モデル: LightGBM LambdaRank（5 seeds × 3 folds = 15 モデル）
特徴量バージョン: v29_fixed（120 列）
テストセット: 4,775 レース / 66,020 サンプル（2025-01-01 以降）
```

**現在の評価結果（eval_results.json）:**

| 指標 | 値 |
|------|-----|
| Top-1 的中率 | 30.2% |
| Top-3 的中率 | 62.1% |
| NDCG@3 | 0.538 |
| Spearman | 0.506 |
| pred_top1_avg_actual_rank | 3.78 |
| top2_coverage（pred-1st が実際 1or2 着） | 49.5% |

**Phase A 完了後に新たに測定する指標（現在値未計測）:**
- top3_coverage（pred-1st が実際 1〜3 着）
- wide_pair_coverage（pred-1st と pred-2nd が共に実際 1〜3 着）
- quinella_pair_coverage（pred-1st と pred-2nd が実際 1・2 着を占める）

### Phase 構成と優先順位

```
Phase A: Plackett-Luce モジュール（スコア→確率変換）
          → 実装工数: 小。モデル・特徴量変更なし。即実装可能。
          → 目的: 現状の pair coverage ベースラインを計測する。
          
Phase B: 相対特徴量の追加（within-race z-score + ペース指数）
          → 実装工数: 中。create_features.py 変更 + 再学習必要。
          → 目的: ペア的中率の向上。
          
Phase C: EV シミュレーション（HR 払戻データを使った事後検証）
          → 実装工数: 中。HR レコードのパースが必要。
          → 目的: 純粋能力モデルが実際に収益性を持つか検証。
          → 実投資には使わない（シミュレーションのみ）。
```

---

## 2. Phase A 詳細仕様: predict.py（新規ファイル）

### ファイル配置

```
pure_rank/src/predict.py   ← 新規作成（train.py / evaluate.py は変更しない）
```

### 設計原則

- **モデルは変更しない**: 既存の `lambdarank_fold*_seed*.txt` をそのまま使用する
- **特徴量は変更しない**: `features_v29_fixed.parquet` をそのまま使用する
- **温度パラメータ T はバリデーションセットで最適化する**（テストセットを使わない）

### 2.1 Softmax 温度キャリブレーション

**目的**: LambdaRank のスコア（任意の実数値）を「レース内の勝利確率」に変換する。
スケールが恣意的なため、温度パラメータ T で勝率分布の尖鋭度を調整する。

**変換式:**

```
p_i(T) = exp(score_i / T) / Σ_j exp(score_j / T)
```

- T < 1: 分布が尖鋭化（上位馬の確率が高まる）
- T = 1: 通常の Softmax
- T > 1: 分布が平滑化（下位馬の確率が上がる）

**最適化目的関数（Log-Loss）:**

バリデーションセット全レースを使って T を最適化する。

```
L(T) = -(1/N_race) × Σ_race Σ_{i in race} is_win_i × log(p_i(T))
```

- `is_win_i`: 馬 i が実際に 1 着なら 1、それ以外は 0
- 各レースで勝ち馬が 1 頭なので、各レースの寄与は `-log(p_winner(T))`

**探索戦略:**

1. 粗探索: T ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}（11点）
2. 細探索: 粗探索で最良だった T を中心に ±0.1 の範囲を 0.02 刻みで探索
3. 最終値: `T_opt`（小数点第2位）として `train_config.json` に保存する

**バリデーションセットの定義:**

`train_config.json` の `training.fold_valid_years` の最終年（2024年）を使用する。

```python
# valid: 2024-01-01 〜 2024-12-31
# test:  2025-01-01 以降（温度最適化には使わない）
```

### 2.2 Harville 公式による2着・3着確率

**数学的定義 (Harville, 1973):**

着順は各位置で残った馬から確率的に選ばれると仮定する。

```
P(i が 1 着) = p_i

P(i が 2 着) = Σ_{j≠i} P(j が 1 着) × p_i / (1 - p_j)

P(i が 3 着) = Σ_{j≠i} Σ_{k≠i, k≠j} P(j が 1 着) × P(k が 2 着 | j が 1 着) × p_i / (1 - p_j - p_k)
```

計算ノート:
- P(k が 2 着 | j が 1 着) = p_k / (1 - p_j)
- P(i が 3 着) の分母 `(1 - p_j - p_k)` が 1e-8 未満の場合は 0 として扱う（数値安定性）
- N 頭レースで O(N^2) / O(N^3) の計算量。16〜18 頭レースで十分高速。

### 2.3 ワイド・馬連確率の計算

**馬連確率（対象: 馬 i と馬 j）:**

```
P_quinella(i, j) = P(i=1着, j=2着) + P(j=1着, i=2着)
                 = p_i × p_j / (1-p_i) + p_j × p_i / (1-p_j)
```

**ワイド確率（対象: 馬 i と馬 j）:**

i と j の両方が 1〜3 着に入る確率。全ての有効な順列の和として計算する。

```
P_wide(i, j) = P(i=1, j=2) + P(j=1, i=2)                         # 1-2着パターン
             + Σ_{k≠i,j} [P(i=1, k=2, j=3) + P(j=1, k=2, i=3)]  # 1-3着パターン
             + Σ_{k≠i,j} [P(k=1, i=2, j=3) + P(k=1, j=2, i=3)]  # 2-3着パターン
```

各項の計算（Harville 展開）:
```
P(a=1, b=2, c=3) = p_a × p_b/(1-p_a) × p_c/(1-p_a-p_b)
```

実装上の注意: N 頭レース全ての pair (i,j) を計算すると O(N^3) になる。
レース内で最大 P_wide を持つ pair を「推奨ワイド」として返す。

### 2.4 新評価指標の追加（evaluate.py の修正）

Phase A では `evaluate.py` の `compute_supplementary_metrics()` を拡張する。
既存の `top2_coverage` に加えて以下を計算する。

| 指標名 | 定義 | 計算元 |
|--------|------|--------|
| `top3_coverage_rate` | P(pred-1st が実際 1〜3 着) | pred-1st の `finish_rank ≤ 3` |
| `wide_pair_coverage_rate` | P(pred-1st と pred-2nd が共に実際 1〜3 着) | score 順位 1位・2位の両方が `finish_rank ≤ 3` |
| `quinella_pair_coverage_rate` | P(pred-1st と pred-2nd が実際 1位・2位を独占) | 2頭の `finish_rank` が {1, 2} に一致 |
| `wide_harville_coverage_rate` | P(Harville最大 P_wide 組が実際共に 1〜3 着) | Harville 推奨 pair が両方 `finish_rank ≤ 3` |

**`wide_pair_coverage_rate` vs `wide_harville_coverage_rate` の違い:**
- `wide_pair_coverage_rate`: スコア上位 2 頭を機械的に選ぶ（スコアベース）
- `wide_harville_coverage_rate`: P_wide が最大になる組み合わせを選ぶ（確率ベース）

これらが同じ結果に近ければ、Harville 変換は近似的に正確。乖離が大きければ
Harville がスコア順とは異なる組み合わせを推奨していることを意味する。

### 2.5 predict.py の関数設計

```
pure_rank/src/predict.py

関数一覧（implementer への設計指示）:

1. calibrate_temperature(
       df_valid: pd.DataFrame,
       models: list[lgb.Booster],
       feature_cols: list[str],
       T_range: np.ndarray,
   ) -> float
   # バリデーションセットで log-loss を最小化する T を返す
   # df_valid: race_id, is_win, [feature_cols] を含む DataFrame
   # 戻り値: T_opt（float）

2. softmax_with_temperature(
       scores: np.ndarray,
       T: float,
   ) -> np.ndarray
   # scores を温度 T でスケーリングして Softmax を適用する
   # 数値安定化: scores -= scores.max() を事前に適用する
   # 戻り値: sum=1 の確率ベクトル

3. harville_place_probs(
       p_win: np.ndarray,
   ) -> tuple[np.ndarray, np.ndarray]
   # Harville 公式で 2着・3着確率を計算する
   # 戻り値: (p2: np.ndarray, p3: np.ndarray)、いずれも長さ N
   # 内部で 0 除算ガード (denom < 1e-8 → skip)

4. compute_race_probabilities(
       race_scores: np.ndarray,
       T: float,
   ) -> dict
   # 1レース分のスコアを受け取り、全確率を返す
   # 戻り値: {
   #     "p_win": np.ndarray,
   #     "p2": np.ndarray,
   #     "p3": np.ndarray,
   #     "p_top3": np.ndarray,  # p_win + p2 + p3
   #     "wide_matrix": np.ndarray,  # shape (N, N), [i,j] = P_wide(i,j), i<j は計算済み
   #     "quinella_matrix": np.ndarray,  # shape (N, N)
   # }

5. compute_pair_coverage_metrics(
       df_test: pd.DataFrame,
       predictions: np.ndarray,
       T_opt: float,
   ) -> dict
   # テストセット全体でペア指標を計算する
   # 戻り値:
   #   top3_coverage_rate, wide_pair_coverage_rate,
   #   quinella_pair_coverage_rate, wide_harville_coverage_rate

6. main() -> None
   # CLI エントリポイント
   # 引数: --calibrate（温度最適化のみ実行）
   #        --eval（全指標の評価）
   #        --race-id RACE_ID（単一レースの確率出力）
```

### 2.6 train_config.json への追加

Phase A 完了後、以下を追加する（implementer が測定後に記入）:

```json
"plackett_luce": {
    "T_opt": null,
    "calibration_valid_year": "2024",
    "T_search_coarse": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    "T_search_fine_step": 0.02
}
```

---

## 3. Phase B 特徴量仕様: 相対特徴量の追加

### 目的

現在の特徴量は馬固有の絶対値が主体（例: `hist_last_time_dev` は同距離平均との偏差）。
しかし「このレースのメンバー内での相対的な強さ」が馬連・ワイドには重要である。
例えば `hist_total_prize=3000万円` の馬でも、相手全員が `5000万円超` なら劣位になる。

### 実装方針

- 既存の `SECTION 4: CURRENT FEATURES` (`_build_current_features()`) に追加する
- 全て `hist_` 系特徴量から `groupby("race_id").transform()` で派生する
- 基底の `hist_` 特徴量は shift(1) 済みのため、二次派生もリーク防止済み
- 新特徴量バージョン: `v30_relative`（`train_config.json` の `features_version` を更新）
- 既存の `features_v29_fixed.parquet` はバックアップしてから上書き

### 3.1 within-race z-score 特徴量（6列）

以下の `hist_` 特徴量について、同レース内（同 `race_id`）での z-score を計算する。

**計算式:**

```python
# z_col = (original_col - race_mean) / (race_std + 1e-6)
race_mean = df.groupby("race_id")[original_col].transform("mean")
race_std  = df.groupby("race_id")[original_col].transform("std")
df[z_col] = (df[original_col] - race_mean) / (race_std + 1e-6)
```

**NaN の扱い**: `transform("mean/std")` は NaN 値を除いて計算する（pandas デフォルト）。
NaN が多い列（例: `hist_same_course_dist_win_rate` は 53.6% が NaN）は
有効値が 2 頭以下のレースで z-score が不安定になる可能性がある。
`hist_std < 0.01` のレース（全馬が同一値）では z-score=0 を設定する
（上記計算式の 1e-6 で自動的に対応される）。

| 新特徴量名 | 基底特徴量 | 備考 |
|------------|------------|------|
| `field_z_time_dev` | `hist_last_time_dev` | 値が小さい = 速い（z 負 = 有利） |
| `field_z_prize` | `hist_total_prize` | 値が大きい = 強い（z 正 = 有利） |
| `field_z_last3f` | `hist_last_last3f` | 値が小さい = 速い（z 負 = 有利） |
| `field_z_win_rate` | `hist_win_rate` | 値が大きい = 強い（z 正 = 有利） |
| `field_z_speed_idx` | `hist_speed_idx_avg3` | 値が大きい = 速い（z 正 = 有利） |
| `field_z_place_rate` | `hist_place_rate` | 値が大きい = 複勝率高（z 正 = 有利） |

**期待効果**: 絶対値特徴量が支配的な現状のモデルに、相対的な強さの情報を追加する。
特に `field_z_speed_idx` は「このメンバー内でのタイム指数の相対位置」を表し、
2着・3着予測に直接寄与すると期待される。

### 3.2 先行馬密度（ペース予測指数）（2列）

**設計根拠**: `running_style_code` はレース後判定のため特徴量として直接使用不可。
しかし各馬の「過去走での先行傾向」は shift(1) で安全に計算できる。
この傾向値のレース内平均が「そのレースに逃げ・先行馬が何頭いるか」の代理指標になる。
先行馬が多いレース（ハイペース）では差し・追込馬が有利になり、
先行馬が少ないレース（スローペース）では逃げ馬が有利になる傾向がある。

**実装手順（implementer への指示）:**

SECTION 3 の `_build_hist_features()` 内に追加する:

```python
# 先行傾向（running_style_code は過去走の値。shift(1) で当該レース除外）
# running_style_code: 1=逃げ, 2=先行, 3=差し, 4=追込
# 先行系 = {1, 2}
df["_is_front_runner"] = df["running_style_code"].isin([1, 2]).astype(np.int8)
df["hist_front_running_pref"] = df.groupby("ketto_num")["_is_front_runner"].transform(
    lambda x: x.shift(1).expanding().mean()
)
df = df.drop(columns=["_is_front_runner"])
```

SECTION 4 の `_build_current_features()` 内に追加する:

```python
# レース内の先行馬密度（全馬の hist_front_running_pref の平均）
# NaN は 0 として fillna して平均（初出走馬は先行傾向不明 → 中立）
df["_front_pref_filled"] = df["hist_front_running_pref"].fillna(0)
df["field_front_runner_density"] = df.groupby("race_id")["_front_pref_filled"].transform("mean")
df = df.drop(columns=["_front_pref_filled"])
```

| 新特徴量名 | 計算元 | 期待効果 |
|------------|--------|---------|
| `hist_front_running_pref` | `running_style_code` shift(1) expanding mean | 当該馬の先行傾向（NaN = 初出走） |
| `field_front_runner_density` | `hist_front_running_pref` の race_id 内 mean | レース内ペース指数（高 = ハイペース想定） |

**市場情報との区別**: `running_style_code` は馬の走行結果（能力・個性）であり、
市場のオッズや人気とは無関係な客観データである。使用可能。

### 3.3 相対枠順（1列）

現在の `wakuban_surface`（枠番 × 馬場符号）は方向性の交互作用特徴量であり、
「このレースの頭数に対する相対的な枠位置」を捉えていない。

| 新特徴量名 | 計算式 | 備考 |
|------------|--------|------|
| `relative_post_position` | `wakuban / horse_count` | 0 < 値 ≤ 1。内枠ほど小さい |

**実装場所**: SECTION 4 の `_build_current_features()` に追加。

```python
df["relative_post_position"] = df["wakuban"].astype(float) / df["horse_count"].astype(float)
```

### 3.4 市場情報除外確認

Phase B 実装後、以下のチェックを実行すること:

```bash
grep -rn "odds\|popularity\|market_log_odds\|init_score\|ninki" \
    C:/Users/syugo/AI/RaceAI_var1.0/pure_rank/src/ --include="*.py"
```

追加した特徴量に以下が含まれていないことを確認:
- オッズ系（単勝・複勝・馬連・ワイドオッズ）
- 人気順位
- 市場補正値

### 3.5 特徴量バージョン管理

| 項目 | 値 |
|------|-----|
| 新バージョン | `v30_relative` |
| 追加列数 | 9 列（field_z_* × 6 + hist_front_running_pref + field_front_runner_density + relative_post_position） |
| 合計列数（目安） | 129 列 |
| バックアップ要件 | `features_v29_fixed.parquet` を `features_v29_fixed_backup.parquet` にコピーしてから上書き |

---

## 4. Phase C 投資シミュレーション仕様

### 前提条件

- 実投資には使わない（シミュレーションのみ）
- オッズは特徴量として使わない（評価目的でのみ払戻データを参照）
- JV-Link HR レコード（払戻情報）から実績払戻倍率を取得する

### 4.1 必要データ

JV-Link の HR レコードから以下のフィールドを抽出する:

| フィールド | 内容 |
|-----------|------|
| `race_id` | レース識別子（SE/RA と結合キー） |
| `wide_horse_num_1`, `wide_horse_num_2` | ワイド 1 組目の馬番 |
| `wide_payout_1` | ワイド 1 組目の払戻金額（100円あたり） |
| `quinella_horse_num_1`, `quinella_horse_num_2` | 馬連 1 組目の馬番 |
| `quinella_payout_1` | 馬連 1 組目の払戻金額 |

HR レコードは複数の組み合わせ（1〜3 組）を持つ場合がある。
ワイド/馬連の的中馬番は 3 組まで存在しうる（同着の場合を除き、通常 3 組）。

### 4.2 EV（期待値）計算

**ワイド EV（レース i、組み合わせ (a,b) の期待値）:**

```
EV_wide(i, a, b) = P_wide(i, a, b) × payout_wide(i, a, b) / 100
```

- `P_wide(i, a, b)`: Harville 公式による馬 a・b の共 3 着以内確率（Phase A の出力）
- `payout_wide(i, a, b)`: 実際の払戻倍率（HR レコードから取得）
- EV > 1.0 なら期待値プラス。EV < 1.0 なら期待値マイナス。

**回収率シミュレーション（N レース）:**

```
strategy = 各レースで Harville 最大 P_wide 組を購入
return_rate = Σ_{i: 的中} payout_wide(i) / (N × 100)
```

- `的中` = 購入した組み合わせが実際のワイド払戻に含まれる
- `return_rate > 1.0` = 純利益

### 4.3 実装場所

```
pure_rank/src/simulate_ev.py   ← 新規作成（Phase A/B 完了後）

入力:
    features_v30_relative.parquet（Phase B 完了後）
    HR_preprocessed.parquet（新規パース必要）
    
出力（JSON）:
    wide_return_rate
    quinella_return_rate
    ev_positive_rate  # EV > 1.0 のレース割合
    hit_rate_wide     # 購入したワイドが的中した割合
```

### 4.4 HR レコードのパース

HR レコードは JV-Link から取得済みの CSVまたはバイナリから変換する。
前処理スクリプト: `pure_rank/src/preprocess.py` に `_parse_hr_records()` を追加する。

出力: `pure_rank/data/01_preprocessed/HR_preprocessed.parquet`

**必須確認事項:**
- HR の払戻データは特徴量として使わない（evaluate のみ）
- HR データを `features_v*.parquet` に merge しない

---

## 5. 実装順序と担当エージェント

### 実装ステップ一覧

| Step | 内容 | 担当 | 依存 |
|------|------|------|------|
| A-1 | `predict.py` 新規作成（`calibrate_temperature`, `softmax_with_temperature`） | implementer | なし |
| A-2 | `harville_place_probs`, `compute_race_probabilities` の実装 | implementer | A-1 |
| A-3 | `evaluate.py` の `compute_supplementary_metrics` を拡張（4新指標追加） | implementer | A-2 |
| A-4 | 温度最適化の実行・T_opt を `train_config.json` に記録 | implementer | A-3 |
| A-5 | ペア指標（top3_coverage, wide_pair_coverage 等）の測定 | evaluator | A-4 |
| B-1 | `create_features.py` にSection 3 への `hist_front_running_pref` 追加 | implementer | A-5 |
| B-2 | `_build_current_features` に z-score 6列 + `field_front_runner_density` + `relative_post_position` 追加 | implementer | B-1 |
| B-3 | `features_v29_fixed.parquet` をバックアップ後、`v30_relative` を生成 | implementer | B-2 |
| B-4 | `train_config.json` の `features_version` を `v30_relative` に更新 | implementer | B-3 |
| B-5 | アンサンブル再学習（`python pure_rank/src/train.py --ensemble`） | implementer | B-4 |
| B-6 | Phase B 完了後の精度評価（Top-1/NDCG@3/Spearman + 4新指標） | evaluator | B-5 |
| C-1 | HR レコードのパース・`HR_preprocessed.parquet` 生成 | implementer | B-6 |
| C-2 | `simulate_ev.py` の実装・EV シミュレーション実行 | implementer | C-1 |
| C-3 | EV シミュレーション結果の評価 | evaluator | C-2 |

### Phase A implementer への引き渡し事項

以下のタスクを実行してください。

1. `C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\src\predict.py` を新規作成する。

2. 関数設計は Section 2.5 の仕様に従う。
   `calibrate_temperature()` では T を探索するために既存の `load_models()`, `get_feature_cols()`,
   `ensemble_predict()` を `evaluate.py` からインポートして再利用すること（コード重複禁止）。

3. 温度最適化に使うバリデーションデータは `train_config.json` の
   `training.fold_valid_years` の最終年（2024年）を使う。
   テストデータ（2025-01-01 以降）は温度最適化に使わない。

4. `evaluate.py` の `compute_supplementary_metrics()` を拡張して
   `top3_coverage_rate`, `wide_pair_coverage_rate`,
   `quinella_pair_coverage_rate`, `wide_harville_coverage_rate` を返すように修正する。
   **ただし既存の `top2_coverage` は残す**（後退比較のため）。

5. T_opt を決定したら `train_config.json` の `plackett_luce.T_opt` に記入する。

6. 市場情報混入チェックを実行する:
   ```bash
   grep -rn "odds\|popularity\|market_log_odds\|init_score\|ninki" \
       C:/Users/syugo/AI/RaceAI_var1.0/pure_rank/src/predict.py
   ```

### Phase B implementer への引き渡し事項

1. `create_features.py` の `_build_hist_features()` に
   `hist_front_running_pref` の計算を追加する（Section 3.2 参照）。

2. `_build_current_features()` に以下を追加する（Section 3.1〜3.3 参照）:
   - `field_z_time_dev`, `field_z_prize`, `field_z_last3f`,
     `field_z_win_rate`, `field_z_speed_idx`, `field_z_place_rate`
   - `field_front_runner_density`
   - `relative_post_position`

3. バックアップを取ってから実行する:
   ```bash
   # バックアップ
   cp pure_rank/data/02_features/features_v29_fixed.parquet \
      pure_rank/data/02_features/features_v29_fixed_backup.parquet
   # 新バージョン生成
   python pure_rank/src/create_features.py
   # v30_relative のバージョン確認
   python -c "import pandas as pd; df=pd.read_parquet('pure_rank/data/02_features/features_v30_relative.parquet'); print(df.shape)"
   ```

4. `train_config.json` の `features_version` を `v30_relative` に変更してから
   アンサンブル学習を実行する:
   ```bash
   python pure_rank/src/train.py --ensemble
   ```

5. 市場情報混入チェックを実行する。

---

## 6. 評価基準

### Phase A 合否基準（モデル変更なし）

Phase A は新規メトリクスの計測が主目的。合否基準ではなく**ベースライン確立フェーズ**とする。

| 確認事項 | 期待値 | 備考 |
|---------|--------|------|
| T_opt が 0.5〜1.5 の範囲内 | 必須 | 範囲外なら探索幅を拡大 |
| T=1.0 の log-loss と T_opt の log-loss の差 | > 0.001 | 差がなければ T_opt=1.0 のまま |
| top3_coverage_rate | 計測して記録 | Phase B の改善ベースとなる |
| wide_pair_coverage_rate | 計測して記録 | Phase B の改善ベースとなる |
| Top-1 は変化しない（predict.py はスコアを変えない） | 30.2% のまま | 変化したら実装バグ |

**リーク停止閾値は適用しない（Phase A はスコア変換のみ）。**

### Phase B 合否基準（モデル再学習あり）

| 指標 | 合格 | 要改善 | 不合格 |
|------|------|--------|--------|
| Top-1 的中率 | > 29.0% | 28.0〜29.0% | < 28.0%（Phase 7 基準割れ） |
| Top-3 的中率 | > 62.0% | 60.0〜62.0% | < 60.0% |
| NDCG@3 | > 0.535 | 0.525〜0.535 | < 0.525（現状割れ） |
| Spearman | > 0.50 | 0.49〜0.50 | < 0.49 |
| wide_pair_coverage_rate | > Phase A 測定値 + 2pp | +0〜2pp | マイナス |

**Top-1 は下げない原則**: Phase B の目的はワイド的中率の向上であり、1着予測を犠牲にしない。
Top-1 が 29% 未満になった場合は evaluator が「Phase B の相対特徴量を削除して Phase A の
状態に戻す」差し戻し指示を出す。

**リーク停止閾値**: Top-1 > 40% または Spearman > 0.6 → 即座に実装停止して evaluator へ報告。

### Phase C 合否基準（EV シミュレーション）

Phase C は収益性の検証であり、「合否」ではなく「事実の計測」として扱う。
以下を計測して報告する:

| 指標 | 報告内容 |
|------|---------|
| `wide_return_rate` | 全テストレースでの払い戻し合計 / 購入合計 |
| `quinella_return_rate` | 同上（馬連版） |
| `ev_positive_rate` | EV > 1.0 のレースの割合 |
| `hit_rate_wide` | 購入したワイドの実際の的中率 |

> 注意: `wide_return_rate < 0.75` は購入戦略として成立しない目安。
> ただし RaceAI_var1.0 は市場残差学習（RaceAI_var2.0.0）との組み合わせを前提としているため、
> 単体での回収率より「1番人気を選んだ場合の回収率との差」が重要な評価軸となる。

---

## 7. 設計判断の記録

### Harville vs Plackett-Luce-Luce（深い理由）

Plackett-Luce モデルは「着順全体の確率」を定義するが、
Harville 公式はその中でも最もシンプルな「1着確率から機械的に2着・3着を導出する」手法である。
より精度の高い方法（Luce-Plackett の完全推定や neural ranking model）もあるが、
このプロジェクトでは以下の理由で Harville を選択する:

1. **実装の透明性**: 数式が閉形式で、バグの検出が容易
2. **既存モデルとの連続性**: LambdaRank スコアを直接入力できる
3. **Phase B で特徴量改善の効果を分離しやすい**

### 市場情報との境界線

Phase C で HR 払戻データを参照することは「市場情報を特徴量に使う」ことと区別する。

| 分類 | 内容 | 判断 |
|------|------|------|
| 禁止 | オッズ・人気を特徴量として学習に使う | 禁止（プロジェクト憲法） |
| 許可 | 過去の払戻データを事後評価（EV 計算）に使う | 許可（評価目的） |
| 許可 | 払戻データを features_*.parquet に含めない | 維持必須 |

### `running_style_code` の扱い

`running_style_code` は `evaluate.py` / `train.py` の `forbidden` 集合に含まれる（レース後判定）。
Phase B の `hist_front_running_pref` 計算では、中間変数として `running_style_code` の
**過去走の値**（shift(1) 済み）を使う。これは `is_win` を中間変数として `hist_win_rate` を
計算するのと同じパターンであり、適切なリーク防止が保たれている。

最終的に特徴量として Parquet に出力されるのは `hist_front_running_pref`（派生列）のみであり、
`running_style_code` 自体は出力しない（forbidden 設定により train.py 側でも除外される）。

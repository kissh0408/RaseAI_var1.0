# 実装仕様書: Phase 5 — タイム速度指数（Speed Index）特徴量

**日付**: 2026-06-29
**対象**: RaceAI_var1.0 — 市場情報なし純粋能力LambdaRank
**フェーズ**: Phase 5（歴史的条件別基準による標準化タイム指数）
**前提**: features_v6.parquet（116列）、Top-1=27.8%（Phase 7基準28.5%に-0.7pp）
**出力**: features_v7.parquet（120列）

---

## 目的

### 既存 time_dev 特徴量の限界

features_v6 では走破タイムの偏差として `_time_dev = racetime - race_avg_time` を使用している。
`race_avg_time` は「当該レースの出走全馬の平均」であり、スケールが固定されていない。

**問題**: 能力の低い馬ばかりのレース（地方転入直後・低クラス混合）では
`race_avg_time` 自体が遅くなる。結果として、遅い馬のレースに出走した馬が
相対的に速く見え、真の絶対能力が評価できない。

例:
- レースA（高クラス）: 全馬の平均 = 70.0秒、ある馬の racetime = 70.5秒 → time_dev = +0.5
- レースB（低クラス）: 全馬の平均 = 72.0秒、ある馬の racetime = 71.5秒 → time_dev = −0.5

time_dev だけ見るとレースBの馬の方が速く見えるが、
絶対的な走破タイムはレースAの馬（70.5秒）の方が速い。

### Phase 5 の解決策: 速度指数（Speed Index）

速度指数 = `(cond_historical_avg_time - racetime) / cond_historical_std_time`

- 同じ条件グループ（`distance_category × surface_code × track_condition_code`）の
  **過去レース全体の平均・標準偏差**を基準とする
- 正の値 = 歴史的平均より速い = 高能力
- **異なるレース間での馬の能力を同一スケールで比較可能**

Phase 5 ではこの速度指数の馬別過去走集計（最後・最高・3走平均・同条件最高）を
4列追加する。

**目標**: Top-1 >= 28.5%（Phase 7 基準到達）
**市場ベンチマーク**: 1番人気 Top-1 ≈ 30〜33%

---

## 禁止特徴量の確認

- [x] オッズ系データを一切含まないことを確認した（走破タイム・条件コードのみ使用）
- [x] 人気順位を含まないことを確認した
- [x] market_log_odds / init_score を含まないことを確認した
- [x] `racetime`・`distance_category`・`surface_code`・`track_condition_code` は
  能力の客観的指標であり、市場情報ではない

実装後の確認コマンド（実装後に必ず実行）:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

---

## 設計上の検討事項

### a. 速度指数の基準値計算における temporal leakage 防止

速度指数の計算には「条件別の歴史的平均・標準偏差」が必要だが、
当該レース当日のデータを含めてはならない。

**採用方式**: 日次集計 → 累積 cumsum → shift(1) でリーク防止

```
cond_daily[当日] に shift(1) を適用
→ cond_daily_prev[当日] = 前日以前の全累積値
→ 当日レースの _speed_idx 計算に "前日以前" の平均・標準偏差を使用
```

これは Phase 3（血統特徴量）・Phase 4（騎手・調教師特徴量）と同じパターンである。

**なぜエントリ単位の shift(1) では不十分か**: 同じ条件で同日複数レースが開催される場合
（例: 東京・芝・1600m・良 で同日に5レース実施）、エントリ単位の shift(1) では
同日の他レース結果が混入する。日次集計後に shift(1) することで当日全結果を完全除外する。

### b. 分散計算の数値安定性

分散を `E[X^2] - (E[X])^2` の形で計算するため（Welford の公式）、
浮動小数点誤差により分散が微小な負値（例: −1e-12）になることがある。

`.clip(lower=0)` を必ず適用して `cond_std_time = sqrt(max(0, var))` とする。

### c. MIN_COND_RACES = 20 の選択

20 レース分のデータがない条件（例: データセット初期の新設距離・悪天候が続く時期）では
標準偏差が不安定になりノイズを増やす。
MIN_COND_RACES = 20 でマスクし LightGBM の欠損値分岐に処理を委ねる。

- MIN が小さすぎる（< 10）: 標準偏差が不安定、ノイズ増大
- MIN が大きすぎる（> 50）: NaN率が増大し、序盤のレースで特徴量が使えない
- 20 はこのバランスの推奨値

### d. `hist_speed_idx_avg3` vs 既存 `hist_avg_time_dev_3`（rank 8）との違い

| 特徴量 | 基準 | スケール |
|--------|------|---------|
| `hist_avg_time_dev_3` | 同レース内の他馬の平均タイム | レースごとに変動。低クラスレースの結果は高く見える |
| `hist_speed_idx_avg3` | 歴史的条件別の平均・標準偏差 | 固定スケール。絶対的な速さを表現 |

両者は同じ情報を測定しているわけではないため、LightGBM は両方の情報を利用できる。
`hist_avg_time_dev_3` を削除せず、`hist_speed_idx_avg3` を「追加」する。

### e. `_speed_idx` は一時列として削除する

`_speed_idx` は当該レースの走破タイム（結果情報）を含む。
これを特徴量として直接残すと、当該レースの結果を当該レースの特徴量に使う
データリーク（training time leakage）となる。
必ず馬別 shift(1) を適用してから horse-level 特徴量とし、元の `_speed_idx` は削除する。

---

## 追加する特徴量一覧（4列）

| 列名 | 計算グループ | 計算方法 | リーク防止 | 期待NaN率 | 期待importanceオーダー |
|------|-----------|---------|-----------|---------|---------------------|
| `hist_speed_idx_last` | ketto_num | `_speed_idx.shift(1)` | shift(1) | 12〜20% | 上位15位以内 |
| `hist_speed_idx_best` | ketto_num | `_speed_idx.shift(1).expanding().max()` | shift(1) | 12〜20% | 上位20位以内 |
| `hist_speed_idx_avg3` | ketto_num | `_speed_idx.shift(1).rolling(3, min_periods=1).mean()` | shift(1) | 12〜20% | 上位25位以内 |
| `hist_speed_idx_cond_best` | ketto_num × distance_category × surface_code | `_speed_idx.shift(1).expanding().max()` | shift(1) | 30〜45% | 上位30位以内 |

**NaN率の主な原因**:
- 初出走の馬（horse_id の最初のレース）→ shift(1) が NaN
- 速度指数の基準値が存在しない条件（cond_count_prev < 20）→ `_speed_idx` 自体が NaN → 全派生列が NaN
- `hist_speed_idx_cond_best` の高NaN率: 当該条件への初出走のため（設計上の特性、許容）

**列数変化**: features_v6.parquet（116列）→ features_v7.parquet（120列）

---

## 時系列リーク防止の詳細説明

```
[Step 1] 条件別・日次集計
  cond_daily: (distance_category, surface_code, track_condition_code, race_date)
  各行 = ある条件でその日に出走した全馬の racetime の sum / sum_of_squares / count

[Step 2] 条件グループ内での cumsum
  cum_sum[i] = Σ d_sum_time[0..i] （i日目まで含む累積）
  cum_sq[i]  = Σ d_sum_sq_time[0..i]
  cum_count[i] = Σ d_count[0..i]

[Step 3] shift(1) で "当日を含まない" 前日以前の累積を取得
  cum_sum_prev[i]   = cum_sum[i-1]   （i日目のエントリを除外）
  cum_sq_prev[i]    = cum_sq[i-1]
  cum_count_prev[i] = cum_count[i-1]

[Step 4] 平均・分散・標準偏差を計算
  cond_avg_time[i] = cum_sum_prev[i] / cum_count_prev[i]
  cond_var_time[i] = cum_sq_prev[i] / cum_count_prev[i] - cond_avg_time[i]^2
  cond_std_time[i] = sqrt(clip(cond_var_time[i], lower=0))

[Step 5] df にマージ
  各レースに (distance_category, surface_code, track_condition_code, race_date) でマージ
  → cond_avg_time, cond_std_time が各行に付く

[Step 6] 速度指数の計算（当該レースの実タイムと比較）
  _speed_idx = (cond_avg_time - racetime) / cond_std_time
  ※ racetime は当該レースの結果。_speed_idx 自体はリークだが、
     shift(1) で過去の _speed_idx を参照することでリーク解消

[Step 7] 馬別 shift(1) で horse-level 特徴量を生成
  hist_speed_idx_last = shift(1)(_speed_idx)  ← 前走の速度指数

[Step 8] 一時列を削除
  drop: ["_speed_idx", "cond_avg_time", "cond_std_time"]
```

---

## 実装コード: `_build_speed_index_features(df)`

以下の関数を `create_features.py` の SECTION 5.5（`_build_jockey_trainer_features`）と
SECTION 6（`_add_training_features`）の間に SECTION 5.6 として追加する。

```python
# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.6: SPEED INDEX FEATURES
# タイム速度指数（Phase 5: 歴史的条件別基準による標準化）。
# 日次集計 → cumsum → shift(1) で当日を除外したリーク防止済み計算。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_speed_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """歴史的条件別基準による速度指数特徴量を生成する。

    アプローチ:
    - 条件グループ (distance_category, surface_code, track_condition_code) 別に
      日次集計 → cumsum → shift(1) で当日を除外した平均・標準偏差を計算する
    - _speed_idx = (cond_avg_time - racetime) / cond_std_time を計算し、
      馬別に shift(1) を適用して horse-level 特徴量を生成する
    - _speed_idx 自体（当該レースの結果情報を含む）は最後に削除する

    Notes
    -----
    - df には racetime, distance_category, surface_code, track_condition_code,
      ketto_num, race_date が必要（_build_hist_features 後の df に全て存在する）
    - _build_hist_features 内で計算・削除済みの _time_dev は再計算しない
    - この関数は _build_hist_features の後・_build_current_features の前に呼ぶこと
    """
    # 速度指数の基準値を計算するのに必要な最低レース数
    # この値未満の条件では標準偏差が不安定なため NaN マスクを適用する
    MIN_COND_RACES = 20

    # ─── Step 1: 条件別・日次集計 ──────────────────────────────────────────────
    # 同じ条件（distance_category × surface_code × track_condition_code）で
    # 同日に複数レースが開催される場合があるため、日次で先に集約する。
    cond_daily = (
        df.groupby(
            ["distance_category", "surface_code", "track_condition_code", "race_date"],
            observed=True,
        )
        .agg(
            d_sum_time=("racetime", "sum"),
            d_sum_sq_time=("racetime", lambda x: (x ** 2).sum()),
            d_count=("racetime", "count"),
        )
        .reset_index()
        .sort_values(
            ["distance_category", "surface_code", "track_condition_code", "race_date"]
        )
        .reset_index(drop=True)
    )

    # ─── Step 2: 条件グループ内での cumsum ────────────────────────────────────
    grp_cond = cond_daily.groupby(
        ["distance_category", "surface_code", "track_condition_code"],
        observed=True,
    )
    cond_daily["cum_sum"]   = grp_cond["d_sum_time"].cumsum()
    cond_daily["cum_sq"]    = grp_cond["d_sum_sq_time"].cumsum()
    cond_daily["cum_count"] = grp_cond["d_count"].cumsum()

    # ─── Step 3: shift(1) で当日を除いた前日以前の累積を取得 ──────────────────
    cond_daily["cum_sum_prev"]   = grp_cond["cum_sum"].shift(1)
    cond_daily["cum_sq_prev"]    = grp_cond["cum_sq"].shift(1)
    cond_daily["cum_count_prev"] = grp_cond["cum_count"].shift(1)

    # ─── Step 4: 平均・分散・標準偏差の計算 ────────────────────────────────────
    # Welford 公式: Var(X) = E[X^2] - (E[X])^2
    # 浮動小数点誤差で分散が微小な負値になることがあるため clip(lower=0) が必須
    cond_daily["cond_avg_time"] = (
        cond_daily["cum_sum_prev"] / cond_daily["cum_count_prev"]
    )
    cond_daily["cond_var_time"] = (
        cond_daily["cum_sq_prev"] / cond_daily["cum_count_prev"]
        - cond_daily["cond_avg_time"] ** 2
    )
    cond_daily["cond_std_time"] = np.sqrt(
        cond_daily["cond_var_time"].clip(lower=0)
    )

    # 最低レース数未満の条件は NaN マスク（標準偏差が不安定なためノイズ抑制）
    low_count_mask = cond_daily["cum_count_prev"] < MIN_COND_RACES
    cond_daily.loc[low_count_mask, "cond_avg_time"] = np.nan
    cond_daily.loc[low_count_mask, "cond_std_time"] = np.nan

    # ─── Step 5: df にマージ ──────────────────────────────────────────────────
    df = df.merge(
        cond_daily[
            [
                "distance_category", "surface_code", "track_condition_code",
                "race_date", "cond_avg_time", "cond_std_time",
            ]
        ],
        on=["distance_category", "surface_code", "track_condition_code", "race_date"],
        how="left",
    )

    # ─── Step 6: 速度指数の計算 ────────────────────────────────────────────────
    # 正の値 = 歴史的平均より速い = 高能力
    # cond_std_time == 0 の場合（全馬同タイム）は NaN を設定する
    df["_speed_idx"] = np.where(
        df["cond_std_time"] > 0,
        (df["cond_avg_time"] - df["racetime"]) / df["cond_std_time"],
        np.nan,
    )

    # ─── Step 7: 馬別 shift(1) で horse-level 特徴量を生成 ───────────────────
    # _build_hist_features の sort_values が継続している前提だが、念のため保証する
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
    grp_horse = df.groupby("ketto_num")

    # 前走の速度指数（最もリークから遠い、最重要候補）
    df["hist_speed_idx_last"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1)
    )

    # 過去最高速度指数（能力の上限値）
    df["hist_speed_idx_best"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1).expanding().max()
    )

    # 直近3走の速度指数平均（安定した能力推定。hist_avg_time_dev_3 の絶対スケール版）
    df["hist_speed_idx_avg3"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # 同条件（距離帯×馬場種別）での過去最高速度指数（条件適性の絶対評価）
    df["hist_speed_idx_cond_best"] = (
        df.groupby(["ketto_num", "distance_category", "surface_code"])["_speed_idx"]
        .transform(lambda x: x.shift(1).expanding().max())
    )

    # ─── Step 8: 一時列を削除 ─────────────────────────────────────────────────
    # _speed_idx は当該レースの結果情報を含むため特徴量として残してはならない
    # cond_avg_time / cond_std_time も中間計算値であり不要
    df = df.drop(
        columns=["_speed_idx", "cond_avg_time", "cond_std_time"],
        errors="ignore",
    )

    return df
```

---

## main() への組み込み

`create_features.py` の `main()` 関数内、
`_build_jockey_trainer_features(df)` の呼び出しと `_load_hc(cfg)` の間に
以下を挿入する。

```python
    print("\n[5.5] Building jockey/trainer features...")
    df = _build_jockey_trainer_features(df)

    # ↓ ここに追加 ↓
    print("\n[5.6] Building speed index features (hist_speed_idx_*)...")
    df = _build_speed_index_features(df)
    # ↑ ここまで ↑

    print("\n[6] Building training features (HC/WC)...")
    hc = _load_hc(cfg)
```

**配置理由**:
- `_build_speed_index_features()` は `racetime`・`distance_category`・`surface_code`・
  `track_condition_code` が必要だが、これらはフィルタ後の df に存在する
- `_build_hist_features()` 内で `_time_dev` を計算して削除しているが、
  `_speed_idx` の計算はそれとは独立した別の計算であるため問題なし
- `_build_current_features()` の前に配置することで、
  将来的に速度指数を `field_avg_speed_idx` 等の集計特徴量に使う拡張が可能になる

---

## train_config.json の変更事項

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\config\train_config.json`

```json
"data": {
  "features_version": "v7"   // "v6" → "v7" に変更
}
```

### 変更不要な項目

- `model`: LambdaRank パラメータは変更しない（特徴量効果を純粋に測定するため1変更ずつの原則を守る）
- `features.categorical`: 追加4列はすべて数値特徴量（categorical への追加不要）
- `filters`: 変更なし
- `training`: 変更なし

---

## NaN率の見込みと対処方針

| 列名 | 予想NaN率 | 主な原因 | 対処 |
|------|---------|---------|------|
| `hist_speed_idx_last` | 12〜20% | 初出走 + MIN_COND_RACES 未満の条件 | LightGBM欠損値分岐 |
| `hist_speed_idx_best` | 12〜20% | 初出走 + MIN_COND_RACES 未満の条件 | LightGBM欠損値分岐 |
| `hist_speed_idx_avg3` | 12〜20% | 初出走 + MIN_COND_RACES 未満の条件 | LightGBM欠損値分岐 |
| `hist_speed_idx_cond_best` | 30〜45% | 当該条件への初出走 | 許容（設計上の特性） |

`hist_speed_idx_cond_best` の 30〜45% は設計上の特性であり許容範囲。
「当該条件（距離帯×馬場）で初出走」というNaN自体が有用な情報として
LightGBM の欠損値分岐（missing direction）に学習される。

`hist_last_time_dev`（importance rank 2, gain=38,519）との類似性から、
`hist_speed_idx_last` も上位 15 位以内に入ることを期待するが、
主成分が既に `hist_last_time_dev` で捉えられているため「追加的な」貢献分が
importance として計測される点に注意する。
gain の絶対値ではなく、Top-1 の向上量で評価すること。

---

## 列数の変化

| 変化 | 列名 | 数 |
|------|------|---|
| 追加 | `hist_speed_idx_last`, `hist_speed_idx_best`, `hist_speed_idx_avg3`, `hist_speed_idx_cond_best` | +4 |
| **合計** | | **+4** |

`features_v6.parquet`（116列）→ `features_v7.parquet`（120列）

---

## CLAUDE.md 規則の遵守確認

### 市場情報排除（最重要）

- [x] 速度指数の計算に使う変数は `racetime`・`distance_category`・`surface_code`・`track_condition_code` のみ
- [x] `racetime` は走破タイムであり、市場情報（オッズ・人気）ではない
- [x] `odds`・`popularity`・`market_log_odds`・`init_score` は一切使用しない

### 時系列リーク防止

- [x] `cond_avg_time`・`cond_std_time` は日次集計 → cumsum → shift(1) で当日を除外
- [x] `hist_speed_idx_*` は全て `_speed_idx` に shift(1) を適用して計算
- [x] `_speed_idx`（当該レース結果を含む中間列）は最後に削除

### 後出しじゃんけん禁止

- [x] テストデータ（2025年以降）の結果を見て MIN_COND_RACES や特徴量を後付け調整しない
- [x] MIN_COND_RACES = 20 はデータリーク防止の観点から事前に設定した値

### 実験は1変更ずつ

- [x] Phase 5 では速度指数特徴量4列のみを追加する
- [x] モデルパラメータ（num_leaves・reg_alpha 等）は変更しない
- [x] 他の特徴量カテゴリ（血統・調教）は同時変更しない

---

## 評価基準

### Phase 5 合否判定

| 指標 | Phase 5 合格 | v6 現在値 | 市場ベンチマーク |
|------|------------|----------|--------------|
| Top-1 的中率 | >= 28.5% | 27.8% | 30〜33% |
| Top-3 的中率 | >= 53% | — | 60〜65% |
| NDCG@3 | >= 0.52 | — | — |
| Spearman相関 | >= 0.50 | — | — |

**最低条件**: Phase 7 ベースライン（Top-1=28.5%）以上
**リーク停止閾値**: Top-1 > 40% または Spearman > 0.6 → 即座に実装停止・evaluator へ報告

---

## 禁止事項の確認

- [x] 単勝オッズ・人気を特徴量に含めない
- [x] market_log_odds / init_score を使わない
- [x] ROI・回収率で合否を判定しない
- [x] テストデータの結果で特徴量を後付け選択しない
- [x] features_v6.parquet をバックアップなしに上書きしない（v7 として新規生成）
- [x] `hist_speed_idx_cond_best` の高NaN率（30〜45%）を「不合格」と誤判断しない
- [x] Top-1 > 40% または Spearman > 0.6 の場合は即座に実装停止・evaluator へ報告
- [x] `_speed_idx`（中間列）を特徴量として残さない

---

## implementer への引き渡し事項

以下を順番に実施すること。

### 1. create_features.py への関数追加

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\src\create_features.py`

SECTION 5.5（`_build_jockey_trainer_features`）と SECTION 6（`_add_training_features`）の間に
本仕様書の「実装コード」セクションに記載した `_build_speed_index_features()` 関数を
新規追加する（SECTION 5.6 として）。

既存コードは一切変更しない。追加のみ。

### 2. main() への呼び出し追加

同ファイルの `main()` 関数内で
`df = _build_jockey_trainer_features(df)` の直後・`hc = _load_hc(cfg)` の直前に
以下を挿入する:

```python
    print("\n[5.6] Building speed index features (hist_speed_idx_*)...")
    df = _build_speed_index_features(df)
```

### 3. train_config.json の変更

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\config\train_config.json`

- `data.features_version`: `"v6"` → `"v7"`

### 4. 特徴量生成の実行

```bash
python pure_rank/src/create_features.py
```

実行後に確認:
- `features_v7.parquet` が生成されていること
- `features_v6.bak.parquet` が自動生成されていること（バックアップ）
- 列数が120列前後であること（`manifest.json` で確認）
- NaN率レポートで `hist_speed_idx_cond_best` が 30〜45% 前後であること（設計通り）
- `hist_speed_idx_last`・`hist_speed_idx_best`・`hist_speed_idx_avg3` が 20% 以下であること
- `_speed_idx`・`cond_avg_time`・`cond_std_time` が列として残っていないこと（削除確認）

### 5. 市場情報混入チェック

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

ヒットがないことを確認してから次のステップへ進む。

### 6. アンサンブル学習の実行

```bash
python pure_rank/src/train.py --ensemble
```

### 7. 精度評価の実行

```bash
python pure_rank/src/evaluate.py
```

結果を evaluator へ渡す際に以下を報告すること:
- Top-1 / Top-3 / NDCG@3 / Spearman
- 追加4列の実際の NaN 率
- 追加4列の feature importance（gain）を報告
- `hist_last_time_dev`（rank 2）と `hist_speed_idx_last` の importance 差を確認
- Top-1 > 40% または Spearman > 0.6 の場合は即座に実装停止してevaluatorへ連絡

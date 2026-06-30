# 実装仕様書: Phase 3 — 血統特徴量の時系列正確版への置き換え + クラス移動特徴量

**日付**: 2026-06-29
**対象**: RaceAI_var1.0 — 市場情報なし純粋能力LambdaRank
**フェーズ**: Phase 3（血統特徴量の時系列正確版 + クラス移動特徴量）
**前提**: Phase 2b 完了（Top-1=28.9%, features_v3.parquet, 102列）
**出力**: features_v4.parquet

---

## 目的

Phase 2b で Top-1=28.9%（Phase 7 ベースライン 28.5% 超え）を達成した。
本 Phase では以下の 2 点を改善し、Top-1 > 29% を目指す。

1. SECTION 5 (`_build_sire_features`) の血統特徴量が**全期間統計**（前方参照バイアスあり）
   になっている問題を修正し、正確な時系列版に完全置き換えする。
2. クラス移動（降級・昇級）の強さを数値化するクラス移動特徴量 3 列を追加する。

市場ベンチマーク（1番人気 Top-1 ≈ 30〜33%）との差を 1% 以内まで詰めることが
本 Phase の現実的な目標ライン。

---

## 禁止特徴量の確認

- [x] オッズ系データを一切含まないことを確認した（血統・グレード情報はオッズと無関係）
- [x] 人気順位を含まないことを確認した
- [x] market_log_odds / init_score を含まないことを確認した

実装後の確認コマンド（実装後に必ず実行）:
```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

---

## A. 血統特徴量の時系列正確版への置き換え（最優先）

### A-1. 現状の問題と正しいアプローチの根拠

**問題**: `_build_sire_features()` の以下 2 列が全期間統計で前方参照バイアスを持つ。

| 旧列名 | 問題のある計算 |
|-------|-------------|
| `hist_sire_surface_win_rate` | `groupby(['sire_id', 'surface_code'])['is_win'].mean()` — 全期間平均 |
| `hist_bms_win_rate` | `groupby('bms_id')['is_win'].mean()` — 全期間平均 |

**なぜ馬固有 shift(1) では解決しないか**:
馬固有特徴量（`hist_win_rate` 等）は `groupby('ketto_num').shift(1)` で十分だが、
血統特徴量は異なる。同じ `sire_id` を持つ産駒が**同日に複数レース**に出走しうるため、
`sire_id` でそのまま `shift(1)` を適用すると同日の他産駒レース結果が混入する。

**正しいアプローチ**: 日次集計 → 累積 → 日次 shift(1) → メイン df へマージ。

- Step 1: `groupby(['sire_id', 'race_date'])` で当日の wins/races を日次集計する
- Step 2: `cumsum()` で累積値を計算する
- Step 3: `groupby('sire_id')['cum_wins'].shift(1)` で当日を除外する（「前日までの累計」を当日にシフト）
- Step 4: 勝率 = `cum_wins_prev / cum_races_prev` を計算する
- Step 5: `race_date × sire_id` でメイン df にマージする

この方法では「当該レース当日のすべての産駒レース結果が完全に除外」されることが保証される。

### A-2. 時系列血統特徴量リスト（4 列）

| 列名 | 計算グループ | 計算内容 | 旧列との関係 |
|------|-----------|---------|------------|
| `hist_sire_win_rate_ts` | sire_id × race_date | 父産駒の通算勝率（当日除外） | 新規追加 |
| `hist_sire_surface_win_rate_ts` | sire_id × surface_code × race_date | 父産駒の同馬場種別勝率（当日除外） | `hist_sire_surface_win_rate` を置き換え（旧列は削除） |
| `hist_sire_dist_win_rate_ts` | sire_id × distance_category × race_date | 父産駒の同距離帯勝率（当日除外） | 新規追加 |
| `hist_bms_win_rate_ts` | bms_id × race_date | 母父産駒の通算勝率（当日除外） | `hist_bms_win_rate` を置き換え（旧列は削除） |

**既存列 `hist_sire_dist_diff` の扱い**: 「父産駒の平均勝ち距離と今回距離の差」を表す
別概念の特徴量であり、削除対象ではない。ただし現実装は全期間統計を使用しており軽微なバイアスを持つ。
父の距離適性は時間的に安定しているため Phase 3 では許容し、将来フェーズで時系列化を検討する。

### A-3. 実装手順（_build_sire_features の完全置き換え）

旧 `_build_sire_features()` 関数の内容を以下の構造で**完全に置き換える**。
関数シグネチャと戻り値の型は変更しない。

```python
def _build_sire_features(df: pd.DataFrame) -> pd.DataFrame:
    """父馬・母父産駒の成績を時系列正確版で計算する。

    アプローチ: 日次集計 → 累積 → shift(1) → メイン df にマージ
    理由: 同一 sire_id の産駒が同日複数レースに出走しうるため、
         ketto_num 単位の shift(1) では同日他産駒の結果が混入する。
         日次集計後に shift(1) することで「当日を含まない累計」を保証する。
    """
    # ─── Step 1a: sire_id × race_date の日次集計（通算勝率用）─────────────────
    if "sire_id" not in df.columns or df["sire_id"].isna().all():
        for col in ["hist_sire_win_rate_ts", "hist_sire_surface_win_rate_ts",
                    "hist_sire_dist_win_rate_ts"]:
            df[col] = np.nan
    else:
        sire_daily = (
            df.groupby(["sire_id", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "race_date"])
        )
        grp_s = sire_daily.groupby("sire_id")
        sire_daily["cum_wins"]  = grp_s["d_wins"].cumsum()
        sire_daily["cum_races"] = grp_s["d_races"].cumsum()
        sire_daily["cum_wins_prev"]  = grp_s["cum_wins"].shift(1)
        sire_daily["cum_races_prev"] = grp_s["cum_races"].shift(1)
        sire_daily["hist_sire_win_rate_ts"] = (
            sire_daily["cum_wins_prev"] / sire_daily["cum_races_prev"]
        )
        df = df.merge(
            sire_daily[["sire_id", "race_date", "hist_sire_win_rate_ts"]],
            on=["sire_id", "race_date"], how="left"
        )

        # ─── Step 1b: sire_id × surface_code × race_date の日次集計 ────────────
        sire_surf = (
            df.groupby(["sire_id", "surface_code", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "surface_code", "race_date"])
        )
        grp_ss = sire_surf.groupby(["sire_id", "surface_code"], observed=True)
        sire_surf["cum_wins"]  = grp_ss["d_wins"].cumsum()
        sire_surf["cum_races"] = grp_ss["d_races"].cumsum()
        sire_surf["cum_wins_prev"]  = grp_ss["cum_wins"].shift(1)
        sire_surf["cum_races_prev"] = grp_ss["cum_races"].shift(1)
        sire_surf["hist_sire_surface_win_rate_ts"] = (
            sire_surf["cum_wins_prev"] / sire_surf["cum_races_prev"]
        )
        df = df.merge(
            sire_surf[["sire_id", "surface_code", "race_date",
                        "hist_sire_surface_win_rate_ts"]],
            on=["sire_id", "surface_code", "race_date"], how="left"
        )

        # ─── Step 1c: sire_id × distance_category × race_date の日次集計 ────────
        sire_dist = (
            df.groupby(["sire_id", "distance_category", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "distance_category", "race_date"])
        )
        grp_sd = sire_dist.groupby(["sire_id", "distance_category"], observed=True)
        sire_dist["cum_wins"]  = grp_sd["d_wins"].cumsum()
        sire_dist["cum_races"] = grp_sd["d_races"].cumsum()
        sire_dist["cum_wins_prev"]  = grp_sd["cum_wins"].shift(1)
        sire_dist["cum_races_prev"] = grp_sd["cum_races"].shift(1)
        sire_dist["hist_sire_dist_win_rate_ts"] = (
            sire_dist["cum_wins_prev"] / sire_dist["cum_races_prev"]
        )
        df = df.merge(
            sire_dist[["sire_id", "distance_category", "race_date",
                        "hist_sire_dist_win_rate_ts"]],
            on=["sire_id", "distance_category", "race_date"], how="left"
        )

    # ─── Step 2: bms_id × race_date の日次集計 ─────────────────────────────────
    if "bms_id" not in df.columns or df["bms_id"].isna().all():
        df["hist_bms_win_rate_ts"] = np.nan
    else:
        bms_daily = (
            df.groupby(["bms_id", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["bms_id", "race_date"])
        )
        grp_b = bms_daily.groupby("bms_id")
        bms_daily["cum_wins"]  = grp_b["d_wins"].cumsum()
        bms_daily["cum_races"] = grp_b["d_races"].cumsum()
        bms_daily["cum_wins_prev"]  = grp_b["cum_wins"].shift(1)
        bms_daily["cum_races_prev"] = grp_b["cum_races"].shift(1)
        bms_daily["hist_bms_win_rate_ts"] = (
            bms_daily["cum_wins_prev"] / bms_daily["cum_races_prev"]
        )
        df = df.merge(
            bms_daily[["bms_id", "race_date", "hist_bms_win_rate_ts"]],
            on=["bms_id", "race_date"], how="left"
        )

    # ─── 既存の hist_sire_dist_diff は保持（異なる概念: 距離差、全期間統計は許容）────
    if "sire_id" in df.columns and df["sire_id"].notna().any():
        sire_wins = df[df["is_win"] == 1]
        if len(sire_wins) > 0:
            sire_avg_dist = (
                sire_wins.groupby("sire_id", observed=True)["distance"]
                .mean()
                .reset_index()
                .rename(columns={"distance": "_sire_avg_win_dist"})
            )
            df = df.merge(sire_avg_dist, on="sire_id", how="left")
            df["hist_sire_dist_diff"] = (df["distance"] - df["_sire_avg_win_dist"]).abs()
            df = df.drop(columns=["_sire_avg_win_dist"])
        else:
            df["hist_sire_dist_diff"] = np.nan
    else:
        df["hist_sire_dist_diff"] = np.nan

    return df
```

### A-4. マージ時の重複キー注意事項

`sire_surf` の日次集計後に `df.merge(...)` する際、`surface_code` がカテゴリ型になっている
場合は `observed=True` を付けないと全組み合わせが展開されてメモリが増大する。
`groupby(..., observed=True)` を全グループ集計で必ず指定すること。

---

## B. クラス移動特徴量（SECTION 3 末尾に追加）

### B-1. 設計思想

G1で4着の馬がG2に出走する「降級馬」は、条件戦ではなく格上での実績を持っており
通常の `hist_same_grade_win_rate` では評価できない。
「この馬はどの格で走ってきたか」「今回は格下か格上か」を直接数値化する。

**grade_code の意味（JRA）**: 小さいほど格上（G1=1, G2=2, G3=3, 条件戦=4〜7）。
フィルタで grade_code=8, 9 は除外済みなので、有効値は 1〜7。

### B-2. クラス移動特徴量リスト（3 列）

| 列名 | 計算方法 | 期待効果 | リーク防止 |
|------|---------|---------|-----------|
| `hist_best_grade_ever` | `groupby('ketto_num')['grade_code'].transform(lambda x: x.shift(1).expanding().min())` | 過去に出走した最高格（最小 grade_code）。キャリア全体の質を示す | shift(1) |
| `hist_grade_diff` | `df['grade_code'] - df['hist_best_grade_ever']` | 正値=降級（格下出走=有利）、負値=昇級（格上出走=不利）、0=自己最高格での出走 | hist_best_grade_ever に依存 |
| `hist_avg_rank_top_grade` | G1/G2/G3（grade_code <= 3）での過去平均着順（案1採用） | 上位グレードでどのくらい走れたかを定量化。 | shift(1) |

### B-3. hist_grade_diff の解釈

- `hist_grade_diff = grade_code (今回) - hist_best_grade_ever (過去最高格)`
- 正値（例: +3）: 今回は grade 5、過去最高は grade 2 → 大幅降級、格下での出走 → 有利方向
- 0: 今回が過去最高格と同レベル → 実力の天井付近での出走
- 負値（例: -2）: 今回は grade 1、過去最高は grade 3 → 格上挑戦 → 不利方向（馬の成長段階）
- 初出走（hist_best_grade_ever = NaN）: NaN → LightGBM が欠損値分岐で処理

### B-4. hist_avg_rank_top_grade: 案1（固定閾値）の採用理由

**採用: 案1 — G3以上（grade_code <= 3）での通算平均着順（固定閾値）**

```python
df['_rank_top_grade'] = df['finish_rank'].where(df['grade_code'] <= 3, other=np.nan)
df['hist_avg_rank_top_grade'] = df.groupby('ketto_num')['_rank_top_grade'].transform(
    lambda x: x.shift(1).expanding().mean()
)
df = df.drop(columns=['_rank_top_grade'])
```

**案2（行ごとの動的比較）を不採用とする理由**:
- 案2「grade_code < 今回 grade_code の過去レースの平均着順」は各行の today_grade が
  変わるたびに異なるフィルタが必要で、pandas の transform では直接実装できない
- apply を使えば実装可能だが、数十万行の apply はメモリと処理時間の両面でリスクが高い
- 解釈の明確さ: 案1 は「重賞（G1/G2/G3）での能力」という単一概念で解釈が容易

**NaN 率の考慮**: 条件戦馬（grade 4〜7）の多くは重賞出走経験がないため、
`hist_avg_rank_top_grade` のNaN率は 70〜80% になる可能性がある。
LightGBM は NaN を欠損値分岐として自動処理するため精度上の問題はないが、
NaN 率が高い列として evaluator へ報告すること。

### B-5. 実装箇所（_build_hist_features 末尾に追加）

`_build_hist_features()` 内の既存最終ブロック（`hist_exact_dist_win_rate` の後）に追記:

```python
# ─── クラス移動特徴量 ──────────────────────────────────────────────────────────
# 過去の最高格（最小 grade_code = 格上）
df['hist_best_grade_ever'] = grp_horse['grade_code'].transform(
    lambda x: x.shift(1).expanding().min()
)
# 今回 grade_code と過去最高格の差（正=降級=有利, 負=昇級=不利）
df['hist_grade_diff'] = df['grade_code'] - df['hist_best_grade_ever']

# G1/G2/G3（grade_code <= 3）での過去平均着順（固定閾値・案1）
# grade_code<=3 以外の出走は NaN としてマスクし expanding mean で集計
df['_rank_top_grade'] = df['finish_rank'].where(df['grade_code'] <= 3, other=np.nan)
df['hist_avg_rank_top_grade'] = df.groupby('ketto_num')['_rank_top_grade'].transform(
    lambda x: x.shift(1).expanding().mean()
)
df = df.drop(columns=['_rank_top_grade'])
```

**注意**: `grp_horse = df.groupby('ketto_num')` は `_build_hist_features()` の先頭で
定義済みだが、`_rank_top_grade` は後から追加された列のため、
`df.groupby('ketto_num')['_rank_top_grade']` と直接記述すること
（`grp_horse['_rank_top_grade']` では列が見つからない場合がある）。

---

## C. 削除する旧列と train.py への影響

### C-1. 削除対象列

| 削除列名 | 削除理由 | 代替列 |
|---------|---------|--------|
| `hist_sire_surface_win_rate` | 全期間統計による前方参照バイアス | `hist_sire_surface_win_rate_ts` |
| `hist_bms_win_rate` | 全期間統計による前方参照バイアス | `hist_bms_win_rate_ts` |

旧列は `_build_sire_features()` の完全置き換えにより `features_v4.parquet` には生成されない。
既存の `features_v3.parquet`（旧列あり）のバックアップは `create_features.py` の自動バックアップ
機能（`features_v3.bak.parquet`）で保全されるため、追加のバックアップ不要。

### C-2. train.py の FORBIDDEN セットへの追加是非

`features_v4.parquet` に旧列が含まれない以上、`train.py` 側での明示的な除外は不要。
ただし、実装完了後に以下を確認すること:

```python
# train.py がカラムのホワイトリスト方式の場合: v4 の新列が feature_cols に含まれているか確認
# train.py が id_cols 除外方式の場合: 変更不要
```

`train_config.json` の `id_cols` には新列は含まれていないため、
`train.py` が `id_cols` 除外方式を採用している場合はコード変更不要。

---

## D. メモリ効率の評価

### D-1. 中間テーブルのサイズ概算

JRA データセット（2015〜2026 年、フィルタ後）の推定行数: 400,000〜600,000 行

**sire_daily（sire_id × race_date）**:
- ユニーク sire_id 数: 約 2,000〜3,000（活躍中の父馬）
- ユニーク race_date 数: 約 3,500（JRA 開催日）
- 実際の組み合わせ: 1頭の父馬が毎日産駒を走らせるわけではないため、
  最大値より大幅に少ない → 推定 **50,000〜150,000 行**
- メモリ: 64bit float 4〜6 列 × 150,000 行 ≈ 数 MB。問題なし。

**sire_surf（sire_id × surface_code × race_date）**:
- surface_code は 2 値（芝/ダート）のため sire_daily の約 2 倍
- 推定 **100,000〜300,000 行**
- メモリ: 同上、問題なし。

**sire_dist（sire_id × distance_category × race_date）**:
- distance_category は 4 値（短距離/マイル/中距離/長距離）のため sire_daily の約 4 倍
- ただし距離帯ごとに産駒の出走が集中するため実際の組み合わせは少ない
- 推定 **100,000〜400,000 行**
- メモリ: 同上、問題なし。

**bms_daily（bms_id × race_date）**:
- sire_id より分布が広い（母父は多様）が構造は同じ
- 推定 **80,000〜200,000 行**
- メモリ: 問題なし。

### D-2. 代替案（不採用）

`observed=True` と dtype の適切な指定（category 型）により中間テーブルを削減できるが、
上記の推定サイズが十分小さいため、現時点では複雑な最適化は不要。

---

## E. train_config.json の変更事項

### 変更内容

```json
"data": {
  "features_version": "v4"   // "v3" → "v4" に変更
}
```

### 変更不要な項目

- `model`: LambdaRank パラメータ変更なし（特徴量効果を純粋に測定するため 1 変更ずつ原則を守る）
- `features.categorical`: 新規追加 7 列はすべて数値特徴量（categorical への追加不要）
- `filters`: 変更なし
- `training`: 変更なし

---

## F. NaN 率の見込み

| 列名 | 予想 NaN 率 | 理由・対処 |
|------|-----------|----------|
| `hist_sire_win_rate_ts` | 10〜20% | sire_id の初登場レース（産駒が初出走の父馬）でNaN |
| `hist_sire_surface_win_rate_ts` | 15〜30% | sire × 馬場の組み合わせが初出現でNaN |
| `hist_sire_dist_win_rate_ts` | 20〜35% | sire × 距離帯の組み合わせが初出現でNaN |
| `hist_bms_win_rate_ts` | 15〜25% | bms_id の初登場レースでNaN |
| `hist_best_grade_ever` | 5〜8% | 初出走馬のみNaN（出走歴がある馬は必ず値あり） |
| `hist_grade_diff` | 5〜8% | hist_best_grade_ever と同じ（NaN伝播） |
| `hist_avg_rank_top_grade` | 70〜80% | 条件戦専門馬は重賞出走歴がないため高NaN。期待通り。 |

`hist_avg_rank_top_grade` の高NaN率は設計上の特性であり許容範囲。
LightGBM の欠損値分岐により「重賞未出走」という情報が暗黙的に学習される。

---

## G. 評価基準

### Phase 3 合否判定

| 指標 | Phase 3 合格 | Phase 2b（v3） | 市場ベンチマーク |
|------|------------|--------------|--------------|
| Top-1 的中率 | > 29.0% | 28.9% | 30〜33% |
| Top-3 的中率 | > 53% | — | 60〜65% |
| NDCG@3 | > 0.51 | — | — |
| Spearman相関 | > 0.49 | — | — |

**リーク停止閾値**: Top-1 > 40% または Spearman > 0.6 → 即座に実装停止・evaluator へ報告。

### 期待改善の根拠

- 血統特徴量の時系列正確版: 旧列は全期間統計でバイアスがあったため、精度が下がっていた可能性がある。
  正確版に置き換えることで LambdaRank の勾配計算が改善されることを期待する。
- クラス移動特徴量: `hist_grade_diff` が「降級馬の強さ」を直接表現する。
  現在の `hist_same_grade_win_rate` では今回グレードでの成績しか見られないが、
  `hist_grade_diff` と `hist_avg_rank_top_grade` の組み合わせで上位グレードの実力が評価できる。

---

## H. 全体的な列数変化

| 変化 | 列名 | 数 |
|------|------|---|
| 削除 | `hist_sire_surface_win_rate`, `hist_bms_win_rate` | -2 |
| 追加（血統） | `hist_sire_win_rate_ts`, `hist_sire_surface_win_rate_ts`, `hist_sire_dist_win_rate_ts`, `hist_bms_win_rate_ts` | +4 |
| 追加（クラス） | `hist_best_grade_ever`, `hist_grade_diff`, `hist_avg_rank_top_grade` | +3 |
| **合計** | | **+5** |

`features_v3.parquet`（102列）→ `features_v4.parquet`（約 107列）

---

## I. implementer への引き渡し事項

以下を順番に実施すること。

### 1. train_config.json の変更

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\config\train_config.json`

- `data.features_version`: `"v3"` → `"v4"`

### 2. create_features.py の変更（SECTION 5 の完全置き換え）

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\src\create_features.py`

SECTION 5 の `_build_sire_features()` 関数本体を、仕様書 A-3 に記載した
時系列正確版に**完全置き換え**する（追記ではなく関数の中身を丸ごと入れ替える）。

削除対象:
- 旧 `hist_sire_surface_win_rate` の計算ブロック（全期間版）
- 旧 `hist_bms_win_rate` の計算ブロック（全期間版）

追加対象:
- `hist_sire_win_rate_ts`（sire × race_date 日次集計）
- `hist_sire_surface_win_rate_ts`（sire × surface × race_date 日次集計）
- `hist_sire_dist_win_rate_ts`（sire × distance_category × race_date 日次集計）
- `hist_bms_win_rate_ts`（bms × race_date 日次集計）
- `hist_sire_dist_diff`（既存ロジックを継続使用。関数末尾に保持）

### 3. create_features.py の変更（SECTION 3 末尾への追加）

`_build_hist_features()` 内の `hist_exact_dist_win_rate` の計算ブロックの後に
仕様書 B-5 のクラス移動特徴量 3 列を追加する。

`df = df.drop(columns=["_time_dev", "_is_top_grade", "_dist_bin_100"])` の直前に
クラス移動特徴量の計算ブロックを挿入し、`df = df.drop(columns=[...])` に
`"_rank_top_grade"` を追加すること。

### 4. 特徴量生成の実行

```bash
python pure_rank/src/create_features.py
```

実行後に確認:
- `features_v4.parquet` が生成されていること
- `features_v3.bak.parquet` が自動生成されていること（バックアップ）
- NaN 率レポートで `hist_avg_rank_top_grade` が 70〜80% 前後であることを確認し、
  それ以外の新列が 35% 以下であること

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
- `hist_avg_rank_top_grade` の実際の NaN 率
- 旧列（`hist_sire_surface_win_rate`, `hist_bms_win_rate`）が結果に含まれていないこと

---

## 禁止事項の確認

- [x] 単勝オッズ・人気を特徴量に含めない
- [x] market_log_odds / init_score を使わない
- [x] ROI・回収率で合否を判定しない
- [x] テストデータの結果で特徴量を後付け選択しない
- [x] features_v3.parquet をバックアップなしに上書きしない（v4 として新規生成）
- [x] `hist_avg_rank_top_grade` の高NaN率（70〜80%）を「不合格」と誤判断しない
- [x] Top-1 > 40% または Spearman > 0.6 の場合は即座に実装停止・evaluator へ報告

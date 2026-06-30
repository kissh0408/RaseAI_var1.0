# 実装仕様書: Phase 4 — 騎手・調教師特徴量

**日付**: 2026-06-29
**対象**: RaceAI_var1.0 — 市場情報なし純粋能力LambdaRank
**フェーズ**: Phase 4（騎手・調教師の数値成績特徴量）
**前提**: features_v5a（107列）、Top-1=27.6%（Phase 7基準28.5%に-0.9pp）
**出力**: features_v6.parquet（116列）

---

## 目的

現在 `jockey_code` / `trainer_code` はカテゴリ変数としてすでに学習に使われており、
LightGBM importance（gain）で jockey_code が top9 に入っている（gain=10,230）。
しかし「このコードに対応するエンティティが直近どのくらい勝っているか」という
数値的な能力指標は特徴量に含まれていない。

本 Phase では騎手・調教師の成績を数値特徴量として追加し、
純粋能力評価における人間要素（騎乗技術・調教技術）を定量化する。

**目標**: Top-1 >= 28.5%（Phase 7 基準 28.5% 超え）
**市場ベンチマーク**: 1番人気 Top-1 ≈ 30〜33%

---

## 禁止特徴量の確認

- [x] オッズ系データを一切含まないことを確認した（騎手・調教師成績はオッズと無関係）
- [x] 人気順位を含まないことを確認した
- [x] market_log_odds / init_score を含まないことを確認した
- [x] jockey_code / trainer_code のコード自体は成績集計の「キー」として使うのみ（特徴量としての使用継続は許容）

実装後の確認コマンド（実装後に必ず実行）:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

---

## 設計上の検討事項への回答

### a. 直近N日 vs 累積（shift済み）の選択 → 両方を持つ

| 方式 | 列名 | 特性 | 採用理由 |
|------|------|------|---------|
| 累積（expanding + shift(1)） | `hist_jockey_win_rate_cum` | 長期的な技量・実力の安定指標 | キャリア全体の実力を表現。新人以外はNaN少ない |
| rolling 30D | `hist_jockey_win_rate_30d` | 直近の調子（フォーム）を反映 | ケガ復帰後の不調、夏競馬での得意不得意を捉える |
| rolling 60D | `hist_jockey_win_rate_60d` | rolling 30D の安定版 | 30D はノイズが多い場合に LightGBM が60D を選ぶ |

両方持つ理由: LightGBM が importance を通じてどちらの時間軸が予測に有効かを自ら判断できる。
単一方式に絞ることは情報を事前に捨てる行為であり、Phase 4 の段階では推奨しない。

調教師は活動が安定しているため複勝率の追加は省略し、勝率のみ（cumulative + 30d + 60d）とする。

### b. 騎手×コース適性 → 採用（通算累積のみ）

`hist_jockey_course_win_rate` を採用する。

採用根拠:
- JRA では騎手のコース適性は実際に観測される（東京コースが得意・苦手など）
- `course_code` は 10 種類（10競馬場）と少なく、累積データが蓄積されやすい
- NaN率が高くなる組み合わせ（例: 新人騎手 × 地方場外）は `MIN_JOCKEY_RACES=10` でNaN扱いし LightGBM の欠損値分岐に委ねる

rolling window を使わず通算累積とする理由:
- コース別は1騎手あたり年間数十レース程度で、30日間ではサンプル数が極端に少ない
- 累積の方が分母安定性が高く、コースへの「適性」（安定した得意不得意）の表現に向いている

### c. jockey_code / trainer_code の扱い → categorical として継続使用

数値特徴量を「追加」し、コードのカテゴリ変数を「置き換えない」。

継続使用の理由:
1. `jockey_code` categorical は rolling 勝率では表現できない「特定馬との相性」や「出遅れ癖」などの個体差情報を暗黙的に保持している
2. 新人騎手や稀少騎手は rolling 勝率がNaNになるため、categorical fallback として機能する
3. importance が高い列を除去するとトータル精度が下がるリスクが高い

`train_config.json` の `categorical` リストへの新規追加は不要。
（追加するのは数値特徴量のみ）

---

## 追加する特徴量一覧（9列）

### 騎手特徴量（5列）

| 列名 | 計算グループ | 計算方法 | リーク防止 | 期待NaN率 |
|------|-----------|---------|-----------|---------|
| `hist_jockey_win_rate_cum` | jockey_code × race_date | 日次集計→cumsum→shift(1)→cum_wins_prev/cum_races_prev | 日次shift(1) | 3〜8% |
| `hist_jockey_win_rate_30d` | jockey_code × race_date | 日次集計→rolling(30D, closed='left')→wins/races | closed='left' | 8〜18% |
| `hist_jockey_place_rate_30d` | jockey_code × race_date | 日次集計→rolling(30D, closed='left')→place/races | closed='left' | 8〜18% |
| `hist_jockey_win_rate_60d` | jockey_code × race_date | 日次集計→rolling(60D, closed='left')→wins/races | closed='left' | 5〜12% |
| `hist_jockey_course_win_rate` | jockey_code × course_code × race_date | 日次集計→cumsum→shift(1)→wins_prev/races_prev | 日次shift(1) | 30〜50% |

### 調教師特徴量（4列）

| 列名 | 計算グループ | 計算方法 | リーク防止 | 期待NaN率 |
|------|-----------|---------|-----------|---------|
| `hist_trainer_win_rate_cum` | trainer_code × race_date | 日次集計→cumsum→shift(1)→cum_wins_prev/cum_races_prev | 日次shift(1) | 3〜8% |
| `hist_trainer_win_rate_30d` | trainer_code × race_date | 日次集計→rolling(30D, closed='left')→wins/races | closed='left' | 5〜15% |
| `hist_trainer_win_rate_60d` | trainer_code × race_date | 日次集計→rolling(60D, closed='left')→wins/races | closed='left' | 4〜10% |
| `hist_trainer_surface_win_rate` | trainer_code × surface_code × race_date | 日次集計→cumsum→shift(1)→wins_prev/races_prev | 日次shift(1) | 10〜20% |

---

## 時系列リーク防止の原則

### なぜエントリ単位の shift(1) では不十分か

血統特徴量（Phase 3）と同じ問題が騎手・調教師にも存在する。

**実測データ**:
- 騎手: 76.7% の（jockey_code × race_date）ペアで同日複数レース騎乗
- 調教師: 74.8% の（trainer_code × race_date）ペアで同日複数レース管理

`df.groupby('jockey_code')['is_win'].transform(lambda x: x.shift(1).expanding().mean())`
のように馬単位ではなく騎手単位で shift すると、同日の別レース結果が混入する。

**解決策**: 日次集計（日レベルで先に集約）→ 日次 shift(1) または rolling の closed='left'

### rolling(ND, closed='left') の動作確認

```
jockey_code=5 の日次データ:
  2022-01-01: d_wins=1, d_races=5
  2022-01-15: d_wins=2, d_races=8
  2022-02-10: d_wins=1, d_races=6

rolling('30D', closed='left') at 2022-02-10:
  ウィンドウ = [2022-02-10 - 30D, 2022-02-10) = [2022-01-11, 2022-02-10)
  → 2022-01-15 を含む（✓）、2022-02-10 当日を含まない（✓）
  → roll_wins_30d = 2, roll_races_30d = 8
  → hist_jockey_win_rate_30d = 2/8 = 0.25
```

`closed='left'` により当日の全レース結果が自動的に除外される。

---

## 実装コード: `_build_jockey_trainer_features(df)`

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\src\create_features.py` の
SECTION 5（`_build_sire_features`）と SECTION 6（`_add_training_features`）の間に
SECTION 5.5 として以下の関数を追加する。

```python
# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.5: JOCKEY / TRAINER FEATURES
# 騎手・調教師の成績特徴量（Phase 4: 時系列正確版）。
# 日次集計 → 累積/rolling → shift(1)/closed='left' でリーク防止済み。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の成績特徴量を時系列正確版で計算する。

    アプローチ:
    - 通算勝率: 日次集計 → cumsum → shift(1) でリーク防止
    - 直近N日勝率: 日次集計 → GroupBy.rolling(ND, closed='left') でリーク防止

    騎手/調教師は同日に複数レースに関与しうるため（実測: 騎手76.7%・調教師74.8%）、
    エントリ単位の shift(1) では同日他レースの結果が混入する。
    日次集計後に処理することで当日を完全除外する。
    """
    # 分母が少ない場合は NaN としてノイズを抑制する（LightGBM の欠損値分岐に委ねる）
    MIN_JOCKEY_RACES = 10
    MIN_TRAINER_RACES = 10

    # ══════════════════════════════════════════════════════════════════════
    # 騎手特徴量
    # ══════════════════════════════════════════════════════════════════════

    # ─── Step J-1: 日次集計（jockey × date） ─────────────────────────────────
    jockey_daily = (
        df.groupby(["jockey_code", "race_date"], observed=True)
        .agg(
            d_wins=("is_win", "sum"),
            d_races=("is_win", "count"),
            d_place=("is_place", "sum"),
        )
        .reset_index()
        .sort_values(["jockey_code", "race_date"])
        .reset_index(drop=True)
    )

    # ─── Step J-2: 通算勝率（cumulative + shift(1) で当日を除外） ─────────────
    grp_j = jockey_daily.groupby("jockey_code", observed=True)
    jockey_daily["cum_wins"]       = grp_j["d_wins"].cumsum()
    jockey_daily["cum_races"]      = grp_j["d_races"].cumsum()
    jockey_daily["cum_wins_prev"]  = grp_j["cum_wins"].shift(1)
    jockey_daily["cum_races_prev"] = grp_j["cum_races"].shift(1)
    jockey_daily["hist_jockey_win_rate_cum"] = (
        jockey_daily["cum_wins_prev"] / jockey_daily["cum_races_prev"]
    )
    # 出走数が少ない場合はNaN（デビュー直後のノイズ抑制）
    jockey_daily.loc[
        jockey_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_win_rate_cum",
    ] = np.nan

    df = df.merge(
        jockey_daily[["jockey_code", "race_date", "hist_jockey_win_rate_cum"]],
        on=["jockey_code", "race_date"],
        how="left",
    )

    # ─── Step J-3: rolling 30D・60D 勝率（closed='left' で当日除外） ────────────
    # GroupBy.rolling を使うことで apply より効率的に時系列ウィンドウを計算する。
    # closed='left': ウィンドウ = [race_date - ND, race_date) → 当日を除外する。
    jd_idx = jockey_daily.set_index("race_date")

    for n_days in [30, 60]:
        roll = (
            jd_idx.groupby("jockey_code", observed=True)[["d_wins", "d_races", "d_place"]]
            .rolling(f"{n_days}D", closed="left")
            .sum()
            .reset_index()  # → columns: jockey_code, race_date, d_wins, d_races, d_place
            .rename(columns={
                "d_wins":  f"roll_wins_{n_days}d",
                "d_races": f"roll_races_{n_days}d",
                "d_place": f"roll_place_{n_days}d",
            })
        )

        # 勝率
        roll[f"hist_jockey_win_rate_{n_days}d"] = (
            roll[f"roll_wins_{n_days}d"] / roll[f"roll_races_{n_days}d"]
        )
        roll.loc[
            roll[f"roll_races_{n_days}d"] < MIN_JOCKEY_RACES,
            f"hist_jockey_win_rate_{n_days}d",
        ] = np.nan

        merge_cols = ["jockey_code", "race_date", f"hist_jockey_win_rate_{n_days}d"]

        # 30D のみ複勝率を追加（60D は重複情報となるため省略）
        if n_days == 30:
            roll["hist_jockey_place_rate_30d"] = (
                roll["roll_place_30d"] / roll["roll_races_30d"]
            )
            roll.loc[
                roll["roll_races_30d"] < MIN_JOCKEY_RACES,
                "hist_jockey_place_rate_30d",
            ] = np.nan
            merge_cols.append("hist_jockey_place_rate_30d")

        df = df.merge(roll[merge_cols], on=["jockey_code", "race_date"], how="left")

    # ─── Step J-4: 騎手×競馬場 通算勝率（cumulative + shift(1)） ────────────────
    # rolling ではなく cumulative を採用する理由: コース別は30日間のサンプルが
    # 極端に少なく（数レース程度）、累積の方が安定した適性スコアを提供する。
    jc_daily = (
        df.groupby(["jockey_code", "course_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["jockey_code", "course_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_jc = jc_daily.groupby(["jockey_code", "course_code"], observed=True)
    jc_daily["cum_wins"]       = grp_jc["d_wins"].cumsum()
    jc_daily["cum_races"]      = grp_jc["d_races"].cumsum()
    jc_daily["cum_wins_prev"]  = grp_jc["cum_wins"].shift(1)
    jc_daily["cum_races_prev"] = grp_jc["cum_races"].shift(1)
    jc_daily["hist_jockey_course_win_rate"] = (
        jc_daily["cum_wins_prev"] / jc_daily["cum_races_prev"]
    )
    jc_daily.loc[
        jc_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_course_win_rate",
    ] = np.nan

    df = df.merge(
        jc_daily[["jockey_code", "course_code", "race_date", "hist_jockey_course_win_rate"]],
        on=["jockey_code", "course_code", "race_date"],
        how="left",
    )

    # ══════════════════════════════════════════════════════════════════════
    # 調教師特徴量
    # ══════════════════════════════════════════════════════════════════════

    # ─── Step T-1: 日次集計（trainer × date） ─────────────────────────────────
    trainer_daily = (
        df.groupby(["trainer_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "race_date"])
        .reset_index(drop=True)
    )

    # ─── Step T-2: 通算勝率（cumulative + shift(1)） ─────────────────────────
    grp_t = trainer_daily.groupby("trainer_code", observed=True)
    trainer_daily["cum_wins"]       = grp_t["d_wins"].cumsum()
    trainer_daily["cum_races"]      = grp_t["d_races"].cumsum()
    trainer_daily["cum_wins_prev"]  = grp_t["cum_wins"].shift(1)
    trainer_daily["cum_races_prev"] = grp_t["cum_races"].shift(1)
    trainer_daily["hist_trainer_win_rate_cum"] = (
        trainer_daily["cum_wins_prev"] / trainer_daily["cum_races_prev"]
    )
    trainer_daily.loc[
        trainer_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_win_rate_cum",
    ] = np.nan

    df = df.merge(
        trainer_daily[["trainer_code", "race_date", "hist_trainer_win_rate_cum"]],
        on=["trainer_code", "race_date"],
        how="left",
    )

    # ─── Step T-3: rolling 30D・60D 勝率（closed='left' で当日除外） ────────────
    td_idx = trainer_daily.set_index("race_date")

    for n_days in [30, 60]:
        roll = (
            td_idx.groupby("trainer_code", observed=True)[["d_wins", "d_races"]]
            .rolling(f"{n_days}D", closed="left")
            .sum()
            .reset_index()
            .rename(columns={
                "d_wins":  f"roll_wins_{n_days}d",
                "d_races": f"roll_races_{n_days}d",
            })
        )
        roll[f"hist_trainer_win_rate_{n_days}d"] = (
            roll[f"roll_wins_{n_days}d"] / roll[f"roll_races_{n_days}d"]
        )
        roll.loc[
            roll[f"roll_races_{n_days}d"] < MIN_TRAINER_RACES,
            f"hist_trainer_win_rate_{n_days}d",
        ] = np.nan

        df = df.merge(
            roll[["trainer_code", "race_date", f"hist_trainer_win_rate_{n_days}d"]],
            on=["trainer_code", "race_date"],
            how="left",
        )

    # ─── Step T-4: 調教師×馬場種別 通算勝率（cumulative + shift(1)） ────────────
    # 芝・ダート適性は安定した長期特性のため cumulative を採用する。
    ts_daily = (
        df.groupby(["trainer_code", "surface_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "surface_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_ts = ts_daily.groupby(["trainer_code", "surface_code"], observed=True)
    ts_daily["cum_wins"]       = grp_ts["d_wins"].cumsum()
    ts_daily["cum_races"]      = grp_ts["d_races"].cumsum()
    ts_daily["cum_wins_prev"]  = grp_ts["cum_wins"].shift(1)
    ts_daily["cum_races_prev"] = grp_ts["cum_races"].shift(1)
    ts_daily["hist_trainer_surface_win_rate"] = (
        ts_daily["cum_wins_prev"] / ts_daily["cum_races_prev"]
    )
    ts_daily.loc[
        ts_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_surface_win_rate",
    ] = np.nan

    df = df.merge(
        ts_daily[["trainer_code", "surface_code", "race_date", "hist_trainer_surface_win_rate"]],
        on=["trainer_code", "surface_code", "race_date"],
        how="left",
    )

    return df
```

---

## main() への組み込み

`create_features.py` の `main()` 関数内、
`_build_sire_features(df)` の呼び出しと `_load_hc(cfg)` の呼び出しの間に
以下を挿入する。

```python
    print("\n[5] Building bloodline features...")
    df = _build_sire_features(df)

    # ↓ ここに追加 ↓
    print("\n[5.5] Building jockey/trainer features...")
    df = _build_jockey_trainer_features(df)
    # ↑ ここまで ↑

    print("\n[6] Building training features (HC/WC)...")
    hc = _load_hc(cfg)
```

---

## train_config.json の変更事項

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\config\train_config.json`

```json
"data": {
  "features_version": "v6"   // "v5a" → "v6" に変更
}
```

### 変更不要な項目

- `model`: LambdaRank パラメータ変更なし（特徴量効果を純粋に測定するため1変更ずつ原則を守る）
- `features.categorical`: 追加する9列はすべて数値特徴量（categorical への追加不要）
  - `jockey_code` / `trainer_code` は現状のまま categorical に残す
- `filters`: 変更なし
- `training`: 変更なし

---

## NaN率の見込みと対処方針

| 列名 | 予想NaN率 | 主な原因 | 対処 |
|------|---------|---------|------|
| `hist_jockey_win_rate_cum` | 3〜8% | デビュー直後・MIN_JOCKEY_RACES未満 | LightGBM欠損値分岐 |
| `hist_jockey_win_rate_30d` | 8〜18% | 30日以内の出走数 < MIN_JOCKEY_RACES | LightGBM欠損値分岐 |
| `hist_jockey_place_rate_30d` | 8〜18% | 上に同じ | LightGBM欠損値分岐 |
| `hist_jockey_win_rate_60d` | 5〜12% | 60日以内の出走数 < MIN_JOCKEY_RACES | LightGBM欠損値分岐 |
| `hist_jockey_course_win_rate` | 30〜50% | コース別は分母が積み上がりにくい | 許容（設計上の特性） |
| `hist_trainer_win_rate_cum` | 3〜8% | デビュー直後・MIN_TRAINER_RACES未満 | LightGBM欠損値分岐 |
| `hist_trainer_win_rate_30d` | 5〜15% | 30日以内の出走数 < MIN_TRAINER_RACES | LightGBM欠損値分岐 |
| `hist_trainer_win_rate_60d` | 4〜10% | 60日以内の出走数 < MIN_TRAINER_RACES | LightGBM欠損値分岐 |
| `hist_trainer_surface_win_rate` | 10〜20% | 馬場種別デビュー初期 | LightGBM欠損値分岐 |

`hist_jockey_course_win_rate` の30〜50%NaN率は設計上の特性であり許容範囲。
「この騎手×コースの組み合わせが初出現」というNaN自体が有用な情報として
LightGBM の欠損値分岐（missing direction）に学習される。

---

## 中間テーブルのサイズ概算

メモリ使用量が問題になるかの事前評価。

| 中間テーブル | ユニーク数（推定） | 推定行数 |
|-----------|----------------|--------|
| jockey_daily（jockey × date） | 429騎手 × 3,500開催日 | 50,000〜100,000行 |
| jd_idx rolling計算（30D + 60D） | 上に同じ | 〜100,000行 |
| jc_daily（jockey × course × date） | 429 × 10 × 3,500 | 実稼働: 130,000〜200,000行 |
| trainer_daily（trainer × date） | 469調教師 × 3,500開催日 | 60,000〜120,000行 |
| ts_daily（trainer × surface × date） | 469 × 2 × 3,500 | 80,000〜180,000行 |

全テーブル合計 < 1M行。64bit float 4〜6列でも数十MB以内。問題なし。

---

## 列数の変化

| 変化 | 列名 | 数 |
|------|------|---|
| 追加（騎手） | `hist_jockey_win_rate_cum`, `hist_jockey_win_rate_30d`, `hist_jockey_place_rate_30d`, `hist_jockey_win_rate_60d`, `hist_jockey_course_win_rate` | +5 |
| 追加（調教師） | `hist_trainer_win_rate_cum`, `hist_trainer_win_rate_30d`, `hist_trainer_win_rate_60d`, `hist_trainer_surface_win_rate` | +4 |
| **合計** | | **+9** |

`features_v5a.parquet`（107列）→ `features_v6.parquet`（116列）

---

## 評価基準

### Phase 4 合否判定

| 指標 | Phase 4 合格 | v5a（現在） | 市場ベンチマーク |
|------|------------|-----------|--------------|
| Top-1 的中率 | >= 28.5% | 27.6% | 30〜33% |
| Top-3 的中率 | >= 53% | — | 60〜65% |
| NDCG@3 | >= 0.52 | 0.5144 | — |
| Spearman相関 | >= 0.50 | 0.4983 | — |

**最低条件**: Phase 7 ベースライン（Top-1=28.5%）以上
**リーク停止閾値**: Top-1 > 40% または Spearman > 0.6 → 即座に実装停止・evaluatorへ報告

---

## 禁止事項の確認

- [x] 単勝オッズ・人気を特徴量に含めない
- [x] market_log_odds / init_score を使わない
- [x] ROI・回収率で合否を判定しない
- [x] テストデータの結果で特徴量を後付け選択しない
- [x] features_v5a.parquet をバックアップなしに上書きしない（v6 として新規生成）
- [x] `hist_jockey_course_win_rate` の高NaN率（30〜50%）を「不合格」と誤判断しない
- [x] Top-1 > 40% または Spearman > 0.6 の場合は即座に実装停止・evaluator へ報告

---

## implementer への引き渡し事項

以下を順番に実施すること。

### 1. create_features.py への関数追加

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\src\create_features.py`

SECTION 5（`_build_sire_features`）と SECTION 6（`_add_training_features`）の間に
本仕様書の「実装コード」セクションに記載した `_build_jockey_trainer_features()` 関数を
新規追加する（SECTION 5.5 として）。

既存コードは一切変更しない。追加のみ。

### 2. main() への呼び出し追加

同ファイルの `main()` 関数内で
`df = _build_sire_features(df)` の直後・`hc = _load_hc(cfg)` の直前に
以下を挿入する:

```python
    print("\n[5.5] Building jockey/trainer features...")
    df = _build_jockey_trainer_features(df)
```

### 3. train_config.json の変更

`C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\config\train_config.json`

- `data.features_version`: `"v5a"` → `"v6"`

### 4. 特徴量生成の実行

```bash
python pure_rank/src/create_features.py
```

実行後に確認:
- `features_v6.parquet` が生成されていること
- `features_v5a.bak.parquet` が自動生成されていること（バックアップ）
- 列数が116列前後であること（`manifest.json` で確認）
- NaN率レポートで `hist_jockey_course_win_rate` が 30〜50% 前後であること（設計通り）
- その他の新列が 20% 以下であること

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
- `hist_jockey_course_win_rate` の実際の NaN 率
- 新規追加9列の feature importance（gain）上位を報告
- Top-1 > 40% または Spearman > 0.6 の場合は即座に実装停止してevaluatorへ連絡

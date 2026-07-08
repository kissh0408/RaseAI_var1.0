# 実装仕様書: EV-ROI 相関が成立する条件の診断 — 2026-07-01

## 禁止特徴量の確認

- [x] WideOdds / QuinellaOdds をモデル特徴量に使わないことを確認した（事後評価専用）
- [x] オッズ・人気順位を `features_*.parquet` に混入しないことを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した

---

## 1. 目的と問題設定

### 現状

| 指標 | 値 | 問題点 |
|------|----|----|
| EV>=1.0 wide ROI | 85.7% (n=1478) | EV フィルターがほぼ無効 |
| EV>=1.3 wide ROI | 86.0% (n=698) | 高閾値でも改善しない |
| EV>=0.8 wide ROI | 81.0% (n=2624) | 全体 ROI より低い |
| best_condition (テストデータ上) | weather_code=3, ROI=126.8%, n=60 | n 不足・後出し問題 |

EV スイープの逆転パターン（高閾値ほど hit_rate が下がる）は、EV と ROI の rank-order 相関が成立していないことを意味する。

EV = P_wide × prior_odds が高い馬券ほど的中率が低いという逆転の原因を条件別に特定し、「EV フィルターが有効な条件」を発見することが本仕様の目的である。

### 根本原因の仮説

| 仮説番号 | 内容 |
|---------|------|
| H1 | Harville の確率過大推定が特定条件（頭数・距離・馬場）で極端に悪化する |
| H2 | モデルのランキング精度が低い条件で EV 推定が特に不正確になる |
| H3 | 市場（オッズ）の情報量が低い条件（雨・重馬場・小頭数）でモデルが優位を持つ |
| H4 | 特定競馬場・距離帯でのみ EV と true-hit-rate の相関が成立する |

本診断はこれらの仮説を条件別 EV-ROI リフトで定量的に検証する。

### 後出しじゃんけん防止原則

**条件の探索は 2024 バリデーションデータのみで行う。**
- 発見した条件セットを 2025+ テストデータで独立検証する
- テストデータを見て条件を追加・修正することは禁止
- weather_code=3 は「テストで偶然発見した条件」であるため、2024 バリデーションでの成立確認を先に行う

---

## 2. データ分割

```
TRAIN  : 〜2023-12-31  （特徴量学習用、診断では使用しない）
VALID  : 2024          （条件スクリーニング用）
TEST   : 2025+         （スクリーニング結果の独立検証用）
```

`simulate_ev.py` の現在の実装は TEST データのみを対象とする。
`--diagnose-ev-conditions` フラグ実行時は VALID データの df_bets を別途構築し、条件探索を VALID 上で完結させる。

---

## 3. 診断対象の 8 次元

| 次元 | カラム名 | 値の例 | 備考 |
|------|---------|--------|------|
| 芝/ダート | `surface_code` | 1=芝, 2=ダート | 既存 |
| 距離カテゴリ | `distance_category` | 短距離/マイル/中距離/長距離 | 既存 |
| 天候 | `weather_code` | 1=晴, 2=曇, 3=雨, 4=小雨 | 既存 |
| 競馬場 | `course_code` | 01=札幌〜10=小倉 | **要追加** |
| 馬場状態 | `track_condition_code` | 1=良, 2=稍重, 3=重, 4=不良 | **要追加** |
| フィールドサイズ | `horse_count_band` | "<=10" / "11-14" / "15+" | **要追加（集計列）** |
| モデル信頼度 | `score_diff_band` | "low" / "mid" / "high" | **要追加（計算列）** |
| ワイドオッズ帯 | `odds_band` | "<3" / "3-8" / "8-20" / "20+" | **要追加（計算列）** |

### 集計・計算列の定義

```python
# horse_count_band: レース内の出走頭数
def _horse_count_band(n: int) -> str:
    if n <= 10:
        return "le10"
    elif n <= 14:
        return "11-14"
    else:
        return "15plus"

# score_diff_band: Top-1 と Top-2 のスコア差（モデル確信度）
# scores は pred_score 降順でソート済み
score_diff = scores[0] - scores[1]
def _score_diff_band(d: float, low_q: float, high_q: float) -> str:
    """low_q / high_q は VALID セット全件の 33/67 パーセンタイル"""
    if d < low_q:
        return "low"
    elif d < high_q:
        return "mid"
    else:
        return "high"

# odds_band: ベット対象ペアの事前オッズ
def _odds_band(odds: float | None) -> str:
    if odds is None:
        return "na"
    elif odds < 3.0:
        return "lt3"
    elif odds < 8.0:
        return "3-8"
    elif odds < 20.0:
        return "8-20"
    else:
        return "20plus"
```

---

## 4. `_collect_bets_per_race` への追加カラム

現在の出力 DataFrame に以下を追加する。**EV の計算ロジックは変更しない。**

| 追加カラム | 型 | 取得元 | 備考 |
|-----------|-----|--------|------|
| `course_code` | int | `grp.iloc[0]["course_code"]` | -1 で欠損 |
| `track_condition_code` | int | `grp.iloc[0]["track_condition_code"]` | -1 で欠損 |
| `horse_count` | int | `len(grp)` | レース内の馬数 |
| `horse_count_band` | str | `_horse_count_band(len(grp))` | |
| `score_diff` | float | `scores[0] - scores[1]` | scores は降順ソート済み |
| `prior_odds_wide` | float or NaN | `wide_odds_lookup.get(rid, {}).get(wide_key, NaN)` | ベット選択ペアのオッズ |
| `odds_band` | str | `_odds_band(prior_odds_wide)` | |

`score_diff_band` はグループ全件のパーセンタイルが必要なため、`_collect_bets_per_race` 内ではなく、`analyze_ev_roi_by_condition` の前処理として追加する。

---

## 5. 実装する関数の仕様

### 5.1 `analyze_ev_roi_by_condition`

```python
def analyze_ev_roi_by_condition(
    df_bets: pd.DataFrame,
    condition_col: str,
    ev_threshold: float = 1.0,
    min_bets: int = 30,
) -> pd.DataFrame:
    """条件列ごとに EV lift を計算して返す。

    Parameters
    ----------
    df_bets       : _collect_bets_per_race() の出力 DataFrame
                    必須カラム: ev_wide, hit_wide, payout_wide, [condition_col]
    condition_col : 集計軸となるカラム名（"course_code", "weather_code" 等）
    ev_threshold  : EV フィルター閾値
    min_bets      : この件数未満の条件は "判定保留" とする

    Returns
    -------
    pd.DataFrame: 以下のカラムを持つ DataFrame（ev_lift 降順ソート）
      condition_col, condition_value, n_races_all, n_bets_ev_filtered,
      roi_all, roi_ev_filtered, ev_lift, ev_lift_1_3,
      hit_rate_ev_filtered, mean_ev_filtered, verdict

    verdict の値:
      "有効"   : ev_lift >= 3.0pp かつ n_bets_ev_filtered >= min_bets
      "判定保留": n_bets_ev_filtered < min_bets
      "無効"   : ev_lift < 3.0pp かつ n_bets_ev_filtered >= min_bets
    """
```

**計算式**:

```python
# 全件 ROI（EV フィルターなし、NaN 除外）
df_all = df_bets[df_bets[condition_col] == val]
roi_all = df_all["payout_wide"].sum() / (len(df_all) * STAKE)

# EV フィルター後 ROI
df_ev = df_all[df_all["ev_wide"] >= ev_threshold]
n_bets = len(df_ev)
roi_ev_filtered = df_ev["payout_wide"].sum() / (n_bets * STAKE) if n_bets > 0 else NaN

# EV lift（プラスなら EV フィルターが有効）
ev_lift = roi_ev_filtered - roi_all  # 単位: 倍率差（1.0=100pp）

# EV>=1.3 でのリフト（参考値）
df_ev13 = df_all[df_all["ev_wide"] >= 1.3]
n_13 = len(df_ev13)
roi_ev_13 = df_ev13["payout_wide"].sum() / (n_13 * STAKE) if n_13 > 0 else NaN
ev_lift_1_3 = roi_ev_13 - roi_all if not np.isnan(roi_ev_13) else NaN

# 合否
if n_bets < min_bets:
    verdict = "判定保留"
elif ev_lift >= 0.030:  # 3pp = 0.030 in decimal
    verdict = "有効"
else:
    verdict = "無効"
```

---

### 5.2 `screen_effective_ev_conditions`

```python
def screen_effective_ev_conditions(
    df_bets: pd.DataFrame,
    condition_cols: list[str] | None = None,
    ev_threshold: float = 1.0,
    min_lift: float = 0.030,
    min_bets: int = 30,
) -> dict:
    """全次元をスキャンして有効条件を返す。

    Parameters
    ----------
    df_bets        : _collect_bets_per_race() の出力（追加カラム付き）
    condition_cols : スキャンする次元のリスト
                     None の場合は以下の 8 次元をすべてスキャン:
                     ["surface_code", "distance_category", "weather_code",
                      "course_code", "track_condition_code",
                      "horse_count_band", "score_diff_band", "odds_band"]
    ev_threshold   : EV フィルター閾値（デフォルト 1.0）
    min_lift       : 有効条件の最小 ev_lift（倍率差。デフォルト 0.030 = 3pp）
    min_bets       : 最小ベット件数（デフォルト 30）

    Returns
    -------
    dict: {
        "screened_at": "VALID",          # 実行データセット種別（必ずVALIDで実行）
        "ev_threshold": 1.0,
        "min_lift": 0.030,
        "min_bets": 30,
        "all_results": [                 # 全次元・全条件値の結果（ev_lift 降順）
            {
                "dimension": "weather_code",
                "value": "3",
                "n_races_all": 210,
                "n_bets_ev_filtered": 55,
                "roi_all": 0.890,
                "roi_ev_filtered": 1.268,
                "ev_lift": 0.378,
                "ev_lift_1_3": 0.412,
                "hit_rate_ev_filtered": 0.300,
                "mean_ev_filtered": 1.15,
                "verdict": "有効",
            },
            ...
        ],
        "effective_conditions": [        # verdict=="有効" のみ抽出
            {"dimension": "weather_code", "value": "3", "ev_lift": 0.378},
            ...
        ],
        "summary": {
            "n_dimensions_scanned": 8,
            "n_conditions_total": 42,
            "n_conditions_effective": 3,
            "n_conditions_pending": 12,
            "n_conditions_invalid": 27,
        }
    }
    """
```

**スクリーニング手順**:
1. 全条件を `analyze_ev_roi_by_condition` で計算
2. 全結果を `ev_lift` 降順にソート
3. `verdict == "有効"` の条件を `effective_conditions` に抽出
4. `effective_conditions` が空の場合は `{"effective_conditions": [], "message": "有効条件なし"}` を返す

---

### 5.3 `build_composite_ev_filter`

```python
def build_composite_ev_filter(
    df_bets: pd.DataFrame,
    conditions: list[tuple[str, str]],
    ev_threshold: float = 1.0,
    mode: str = "OR",
) -> pd.DataFrame:
    """複数条件のフィルタを適用してベット結果を返す。

    Parameters
    ----------
    df_bets    : _collect_bets_per_race() の出力
    conditions : [(dimension, value), ...] のリスト
                 例: [("weather_code", "3"), ("track_condition_code", "3")]
    ev_threshold: EV 閾値（各条件に共通適用）
    mode       : "OR"  = いずれか一つの条件を満たすレース
                 "AND" = すべての条件を同時に満たすレース

    Returns
    -------
    pd.DataFrame: フィルタ通過したレースのみの df_bets

    Raises
    ------
    ValueError: conditions が空リストの場合
    ValueError: mode が "OR" でも "AND" でもない場合
    """
```

**計算例（OR モード）**:

```python
mask = pd.Series(False, index=df_bets.index)
for dim, val in conditions:
    mask |= (df_bets[dim].astype(str) == str(val))
df_filtered = df_bets[mask & (df_bets["ev_wide"] >= ev_threshold)]
```

---

## 6. `score_diff_band` の前処理

`score_diff_band` は全件のパーセンタイルに依存するため、VALID セット上で分位点を計算してから TEST に適用する。

```python
def assign_score_diff_band(
    df_bets_valid: pd.DataFrame,
    df_bets_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """VALID で分位点を計算し、VALID と TEST 両方に score_diff_band を付与する。

    VALID の 33/67 パーセンタイルを基準とする。
    テストで分位点を再計算することはデータリークになるため禁止。
    """
    low_q  = df_bets_valid["score_diff"].quantile(0.33)
    high_q = df_bets_valid["score_diff"].quantile(0.67)
    for df in [df_bets_valid, df_bets_test]:
        df["score_diff_band"] = df["score_diff"].apply(
            lambda d: "low" if d < low_q else ("high" if d >= high_q else "mid")
        )
    return df_bets_valid, df_bets_test
```

---

## 7. `main()` への `--diagnose-ev-conditions` フラグ追加

```python
parser.add_argument(
    "--diagnose-ev-conditions",
    action="store_true",
    help=(
        "VALID (2024) で EV-ROI 相関が成立する条件をスクリーニングし、"
        "TEST (2025+) で独立検証する。結果を ev_results.json の "
        "'best_condition_sweep' に保存する。"
    ),
)
```

### `--diagnose-ev-conditions` 実行フロー

```
1. VALID セット (2024) の df_bets を構築
   └─ _collect_bets_per_race(df_valid, ...) で course_code 等も取得

2. assign_score_diff_band() で VALID の分位点を計算

3. screen_effective_ev_conditions(df_bets_valid) で有効条件をスクリーニング
   └─ 結果を "screened_at"="VALID" として記録

4. TEST セット (2025+) の df_bets を構築（既存ロジック）

5. assign_score_diff_band() で VALID 分位点を TEST に適用

6. 有効条件を TEST 上で独立検証
   a. 各有効条件を build_composite_ev_filter() で適用
   b. ROI / hit_rate / n_bets を計算
   c. ev_lift_on_test を記録

7. ev_results.json の "best_condition_sweep" フィールドに保存
```

---

## 8. `ev_results.json` の出力フォーマット拡張

`best_condition_sweep` フィールドを追加する。既存フィールドは変更しない。

```json
{
  "n_races": 4775,
  "overall": { ... },
  "ev_filtered": { ... },
  "ev_sweep_wide": [ ... ],
  "best_condition": { ... },
  "best_condition_sweep": {
    "diagnosis_date": "2026-07-01",
    "valid_n_races": 3500,
    "test_n_races": 4775,
    "ev_threshold": 1.0,
    "min_lift_pp": 3.0,
    "min_bets": 30,
    "valid_screening": {
      "n_dimensions_scanned": 8,
      "n_conditions_total": 42,
      "n_conditions_effective": 0,
      "n_conditions_pending": 12,
      "all_results": [
        {
          "dimension": "weather_code",
          "value": "3",
          "n_races_all": 210,
          "n_bets_ev_filtered": 55,
          "roi_all": 0.890,
          "roi_ev_filtered": 1.268,
          "ev_lift": 0.378,
          "ev_lift_1_3": 0.412,
          "hit_rate_ev_filtered": 0.300,
          "mean_ev_filtered": 1.15,
          "verdict": "有効"
        }
      ],
      "effective_conditions": [
        {"dimension": "weather_code", "value": "3", "ev_lift": 0.378}
      ]
    },
    "test_validation": {
      "individual_conditions": [
        {
          "dimension": "weather_code",
          "value": "3",
          "n_bets_test": 60,
          "roi_test": 1.268,
          "hit_rate_test": 0.300,
          "ev_lift_test": 0.411,
          "verdict": "有効"
        }
      ],
      "composite_or": {
        "conditions_used": [["weather_code", "3"]],
        "n_bets_test": 60,
        "roi_test": 1.268,
        "hit_rate_test": 0.300
      },
      "composite_and": null
    },
    "summary": {
      "n_valid_effective_conditions": 0,
      "n_test_validated_conditions": 0,
      "best_composite_roi_test": null,
      "roi_target_achieved": false,
      "note": "有効条件が 0 件の場合は composite フィルタを構成しない"
    }
  }
}
```

---

## 9. 合否基準

### 各条件の verdict

| 条件 | verdict |
|------|---------|
| ev_lift >= 3.0pp かつ n_bets >= 30 | 有効 |
| n_bets < 30 | 判定保留 |
| ev_lift < 3.0pp かつ n_bets >= 30 | 無効 |

### 診断全体の合否

| 状態 | 次アクション |
|------|------------|
| VALID で有効条件あり、TEST でも ROI >= 1.00 かつ n_bets >= 30 | 条件フィルターを simulate_ev.py の推奨ベット戦略に組み込む（implementer） |
| VALID で有効条件あり、TEST では ROI < 1.00 | 過学習の疑い。条件フィルターを採用しない |
| VALID で有効条件なし | EV-ROI 相関は現モデルでは成立しない。P_wide 推定精度の改善を planner に差し戻す |
| weather_code=3 が VALID でも有効 | n_bets が 2024 で 30 以上かを確認。満たさなければ「判定保留」のままテスト n=60 のデータ蓄積を待つ |

---

## 10. implementer への引き渡し事項

### タスク 1: `_collect_bets_per_race` への追加カラム

対象ファイル: `C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\src\simulate_ev.py`

追加するカラム（`rows.append(...)` の dict に追加）:

```python
"course_code": int(first.get("course_code", -1)) if "course_code" in grp.columns else -1,
"track_condition_code": int(first.get("track_condition_code", -1)) if "track_condition_code" in grp.columns else -1,
"horse_count": len(grp),
"horse_count_band": _horse_count_band(len(grp)),
"score_diff": float(scores[0] - scores[1]) if len(scores) >= 2 else float("nan"),
"prior_odds_wide": float(prior_odds_wide) if prior_odds_wide is not None else float("nan"),
"odds_band": _odds_band(prior_odds_wide),
```

ヘルパー関数 `_horse_count_band` と `_odds_band` はモジュールレベルに定義する。

### タスク 2: 3 関数の実装

- `analyze_ev_roi_by_condition` — 本仕様書セクション 5.1
- `screen_effective_ev_conditions` — セクション 5.2
- `build_composite_ev_filter` — セクション 5.3
- `assign_score_diff_band` — セクション 6

### タスク 3: `main()` への VALID セット構築ロジック追加

現在の `main()` は `df_test = df[df["race_date"] > valid_end_ts]` のみ。
`--diagnose-ev-conditions` 時に以下を追加する:

```python
valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
train_end_ts = pd.Timestamp(cfg["training"]["train_end"])

df_valid = df[
    (df["race_date"] > train_end_ts) &
    (df["race_date"] <= valid_end_ts)
].copy()
valid_years = sorted(df_valid["race_date"].dt.year.unique().tolist())
wide_odds_lookup_valid = _build_odds_lookup(valid_years, odds_dir, "Wide")
quinella_odds_lookup_valid = _build_odds_lookup(valid_years, odds_dir, "Quinella")

df_bets_valid = _collect_bets_per_race(
    df_valid, preds_valid, hr_df, T_opt,
    wide_odds_lookup=wide_odds_lookup_valid,
    quinella_odds_lookup=quinella_odds_lookup_valid,
    bracket_models=bracket_models,
)
```

`preds_valid` はモデルを VALID セットに適用した予測スコア。

### タスク 4: `--diagnose-ev-conditions` フラグの組み込みと `best_condition_sweep` 保存

セクション 7 のフローに従い実装する。`ev_results.json` 保存時に `best_condition_sweep` キーを追加する（既存フィールドは上書きしない）。

### 注意事項

- `roi_by_condition`（既存関数）はそのまま残す。本仕様の関数は新規追加のみ
- `best_condition_sweep` は `--diagnose-ev-conditions` フラグなしの実行では `null` とする
- `score_diff_band` の VALID 分位点は `best_condition_sweep.valid_screening.score_diff_quantiles` に記録する

---

## 11. 期待される診断結果と解釈

### 期待 A: weather_code=3 が VALID で有効

- VALID (2024) の weather_code=3 ベット数が 30 以上かつ ev_lift >= 3pp
- TEST (2025+) で n_bets >= 30 かつ ROI >= 100%
- 解釈: 雨天時にモデルが市場より優れた能力評価ができている証拠
- 次アクション: composite フィルター（weather_code=3 OR track_condition_code=3/4）を組み込む

### 期待 B: VALID で有効条件なし

- weather_code=3 の VALID n_bets < 30（判定保留）
- 他の条件でも ev_lift < 3pp
- 解釈: 現モデルの確率推定精度（Harville + calibration）が EV フィルタを機能させるほど正確でない
- 次アクション: P_wide 推定精度の改善を planner へ差し戻し。具体的には「Harville の代替（Plackett-Luce ペア積分）」または「頭数別 calibration 係数」の仕様策定を依頼する

### 期待 C: VALID で有効だが TEST で無効

- VALID で ev_lift が高い条件が TEST で ev_lift < 0
- 解釈: 条件フィルターが VALID に過適合した（後出しではないが年次変動が大きい）
- 次アクション: 採用しない。データが 2 年以上蓄積されるまで判定保留

---

## 12. 制約事項

1. `features_*.parquet` の書き換えは不要（追加カラムは実行時に計算）
2. `ev_results.json` への保存は `best_condition_sweep` フィールドの追加のみ（既存構造は保持）
3. VALID セットの分位点情報はコード内にハードコードせず、実行時に計算して JSON に記録する
4. 合格条件（ev_lift >= 3pp, n_bets >= 30）の閾値は `train_config.json` に追加しない（診断専用の閾値であり、モデル学習設定ではないため）

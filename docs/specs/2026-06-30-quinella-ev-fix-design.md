# 実装仕様書: Quinella EV 修正 — OR事前オッズ化 — 2026-06-30

## 禁止特徴量の確認

- [x] QuinellaOdds CSV を `features_*.parquet` にマージしないことを確認した
- [x] QuinellaOdds を学習の特徴量として使わないことを確認した
- [x] オッズは EV 計算（事後評価）のみに使用することを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した
- [x] 人気順位を含まないことを確認した

---

## 1. 問題の整理

### 現在の機能不全

`pure_rank/src/simulate_ev.py` の `_collect_bets_per_race()` における馬連 EV 計算:

```python
# 現在の実装（機能不全）
quin_ref_payout = float(hr_df[hr_df["bet_type"] == "quinella"]["payout"].mean())
ref_q = quin_payout if quin_payout > 0 else quin_ref_payout
ev_quin = p_quin * ref_q / STAKE
```

**問題の核心**: `quin_payout` は HR レコードから取得した「実際の払戻金額」であり、レース結果が
判明した後のデータである。

- 的中レース（`quin_payout > 0`）: 実際の払戻額で EV 計算 → 的中した馬連組み合わせの
  EV が実際の払戻で過大評価される
- 外れレース（`quin_payout == 0`）: 全レースの平均払戻 `quin_ref_payout` で代替 → 個別
  レースのオッズ水準を無視した一律値

EV は「ベット前に計算できる期待値」でなければならない。HR払戻（事後データ）を参照した
現在の quinella EV は ROI=45% という機能不全の結果を示している。

### 解決策

`QuinellaOdds_YYYY.csv`（事前オッズ）を使った真の EV 計算:

```
EV_quinella(i, j) = P_quinella(i, j) × quinella_odds(i, j)
```

- `P_quinella(i, j)`: Harville 公式による馬 i が 1 着かつ馬 j が 2 着（またはその逆）の確率
- `quinella_odds(i, j)`: QuinellaOdds CSV から取得した事前オッズ（ベット前に観測可能）

---

## 2. QuinellaOdds CSV の仕様

### ファイル配置

```
C:\Users\syugo\AI\RaceAI_var1.0\common\data\output\odds\
├── QuinellaOdds_2015.csv
├── QuinellaOdds_2016.csv
├── ...
└── QuinellaOdds_2026.csv
```

全年分（2015〜2026）の存在を evaluator が確認済み。

### CSV 列定義

WideOdds CSV と完全に同一のカラム構造である。JV-Data 仕様書（O2 レコード）に対応。

| 列名 | 型 | 内容 |
|------|-----|------|
| `race_id` | int64 (16桁) | `2023010506010101` 形式。WideOdds と同じ形式 |
| `horse_num_1` | int | ペアの馬番 1 |
| `horse_num_2` | int | ペアの馬番 2 |
| `odds_status` | str | `"ok"` = 発売中オッズ。それ以外 = 取消等 |
| `odds` | float | 事前オッズ値 |

**実データ確認（QuinellaOdds_2023.csv 先頭行）:**

```
race_id,horse_num_1,horse_num_2,odds_status,odds
2023010506010101,4,15,ok,777.3
2023010506010101,1,4,ok,7377.6
```

WideOdds との違い: `odds` の水準が異なる（馬連は組み合わせ数が多く高倍率になりやすい）。
カラム名・型・`race_id` の形式は完全に同一。

### race_id の形式

WideOdds と同様に、CSV の `race_id` は **int64 (16桁)** である。
`simulate_ev.py` での変換:

```python
df["race_id_str"] = df["race_id"].apply(lambda x: str(int(x)))
```

`features_*.parquet` の `race_id` も 16桁 str であるため、直接一致する。

### ペア正規化

馬連は組み合わせの順序なし。必ず `_norm_pair()` を使って `(min(h1,h2), max(h1,h2))` に正規化する。

```python
# predict.py からインポート済みの関数を使う
from predict import _norm_pair
key = _norm_pair(int(horse_num_1), int(horse_num_2))  # 常に (小さい馬番, 大きい馬番)
```

---

## 3. 新関数: `_build_quinella_odds_lookup()`

`simulate_ev.py` に以下の関数を追加する。`_build_wide_odds_lookup()` と完全に同一の
実装パターンで、ファイル名のみ異なる。`_build_wide_odds_lookup()` の直後に配置すること。

```python
def _build_quinella_odds_lookup(
    years: list[int],
    odds_dir: Path,
) -> dict[str, dict[PAIR_KEY, float]]:
    """QuinellaOdds_YYYY.csv を複数年読み込み、race_id -> {(h1,h2): odds} の辞書を返す。

    Parameters
    ----------
    years : テストセットの年リスト
    odds_dir : QuinellaOdds CSV が格納されたディレクトリ

    Returns
    -------
    dict[race_id_str, dict[(h1,h2), odds]]
        - race_id_str: str 16桁（int64 を str() 変換したもの。features_*.parquet の race_id と一致）
        - (h1, h2): _norm_pair() で正規化（小さい馬番が先頭）
        - odds: float（事前オッズ）

    除外条件
    --------
    - odds_status != "ok" の行（発売前取消・発売後取消を除外）
    - odds が NaN の行
    - CSV ファイルが存在しない年（警告を出してスキップ）
    """
    lookup: dict[str, dict[PAIR_KEY, float]] = {}
    for year in years:
        path = odds_dir / f"QuinellaOdds_{year}.csv"
        if not path.exists():
            print(f"  [warn] QuinellaOdds_{year}.csv not found, skipping")
            continue
        df = pd.read_csv(path)
        df = df[(df["odds_status"] == "ok") & df["odds"].notna()].copy()
        df["race_id_str"] = df["race_id"].apply(lambda x: str(int(x)))
        df["h_min"] = df[["horse_num_1", "horse_num_2"]].min(axis=1).astype(int)
        df["h_max"] = df[["horse_num_1", "horse_num_2"]].max(axis=1).astype(int)
        df["pair_key"] = list(zip(df["h_min"], df["h_max"]))
        # 同一ペアに複数スナップショットが存在する場合は最後の値を採用
        for rid, grp in df.groupby("race_id_str"):
            lookup[rid] = dict(zip(grp["pair_key"], grp["odds"].astype(float)))
    print(f"  QuinellaOdds loaded: {len(lookup):,} races across {years}")
    return lookup
```

---

## 4. `simulate_ev.py` の修正箇所

### 修正 1: `_collect_bets_per_race()` のシグネチャ変更

```python
def _collect_bets_per_race(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]] | None = None,
    quinella_odds_lookup: dict[str, dict[PAIR_KEY, float]] | None = None,  # 追加
) -> pd.DataFrame:
```

### 修正 2: 馬連 EV 計算の変更

関数本体内の馬連 EV 計算ブロックを以下のように変更する。

変更前（機能不全）:
```python
# 馬連: HR 払戻平均を使った参照 EV（WideOdds に馬連オッズなし）
ref_q = quin_payout if quin_payout > 0 else quin_ref_payout
ev_quin = p_quin * ref_q / STAKE
```

変更後（QuinellaOdds 事前オッズ使用）:
```python
# 馬連: QuinellaOdds 事前オッズによる真の EV
# EV = P_quinella × odds（Wide と同じ計算パターン）
if quinella_odds_lookup is not None:
    prior_odds_quin = quinella_odds_lookup.get(rid, {}).get(quin_key, None)
    ev_quin = (p_quin * prior_odds_quin) if prior_odds_quin is not None else float("nan")
else:
    # フォールバック（quinella_odds_lookup 未提供時の後方互換）
    ref_q = quin_payout if quin_payout > 0 else quin_ref_payout
    ev_quin = p_quin * ref_q / STAKE
```

`quin_ref_payout` の計算行は **削除しない**。フォールバックパスで使うため残す。

### 修正 3: `main()` に QuinellaOdds ローダーを追加

```python
# WideOdds ローダーの呼び出し（既存）の直後に追加する
print(f"\nLoading QuinellaOdds for years: {test_years}")
quinella_odds_lookup = _build_quinella_odds_lookup(test_years, odds_dir)

# _collect_bets_per_race の呼び出しに引数を追加
df_bets = _collect_bets_per_race(
    df_test, preds, hr_df, T_opt,
    wide_odds_lookup=wide_odds_lookup,
    quinella_odds_lookup=quinella_odds_lookup,   # 追加
)
```

### 修正 4: `main()` の EV=NaN 集計を馬連にも追加

wide の NaN 集計の後に quinella の NaN 集計を追加する:

```python
n_quin_ev_na = int(df_bets["ev_quin"].isna().sum())
print(f"  Quinella EV=NaN (no odds): {n_quin_ev_na}/{n_total} ({n_quin_ev_na/n_total*100:.1f}%)")
```

### 変更しないこと

| 禁止事項 | 理由 |
|---------|------|
| `_collect_bets_with_calibration()` の変更 | 馬連 EV は計算していない。対象外 |
| QuinellaOdds を `features_*.parquet` にマージ | 特徴量汚染の防止 |
| Top-1・NDCG@3・Spearman 評価への影響 | EV 計算の変更。モデルは変更しない |
| `quinella_odds_lookup = None` のデフォルト引数を削除 | 後方互換性の維持 |

---

## 5. データフロー

```
[入力]
QuinellaOdds_{year}.csv  ─┐
                           ├→ _build_quinella_odds_lookup()
(テストセットの年を自動検出) ─┘
         ↓
quinella_odds_lookup: dict[str, dict[(h1,h2), float]]
         ↓
_collect_bets_per_race() に wide_odds_lookup と並列に渡す
         ↓
df_bets["ev_quin"] = p_quin × quinella_odds (or NaN if not found)
         ↓
ev_threshold_sweep(df_bets, thresholds, bet_type="quin") で EV フィルタ分析
         ↓
[出力] ev_results.json の "ev_sweep_quinella" フィールド
```

---

## 6. 評価方法

実装後に以下を確認して evaluator に報告する。

| 指標 | 変更前 | 変更後（目標） |
|------|--------|--------|
| 馬連 EV 計算の根拠 | HR 払戻（事後） | QuinellaOdds 事前オッズ |
| EV >= 1.0 馬連ベット数 | 機能不全（ROI=45%） | 計測して報告 |
| EV >= 1.0 馬連 ROI | ~45%（機能不全） | 計測して報告（市場平均 70-75% 付近が期待値） |
| EV = NaN 馬連レース数 | 0（全件に ref_payout 代入） | 計測して報告 |
| Top-1 的中率 | 30.18% | 30.18%（変化しないこと） |
| NDCG@3 | 0.538 | 0.538（変化しないこと） |
| Spearman | 0.506 | 0.506（変化しないこと） |

**Top-1・NDCG@3・Spearman は変化しない**。モデル・特徴量を一切変更しないため。

---

## 7. implementer への引き渡し事項

ブランチ `feature/wide-odds-ev` で以下の順序で実装すること。

1. `simulate_ev.py` に `_build_quinella_odds_lookup()` を追加する（Section 3 の仕様）

2. `_collect_bets_per_race()` に `quinella_odds_lookup` 引数を追加し、馬連 EV 計算を変更する
   （Section 4 修正1・2）

3. `main()` に QuinellaOdds ローダーを追加し、`_collect_bets_per_race()` の呼び出しを更新する
   （Section 4 修正3・4）

4. 実行して結果を報告する:

   ```bash
   cd C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\src
   python simulate_ev.py
   ```

5. 市場情報混入チェックを実行する:

   ```bash
   grep -rn "odds\|popularity\|market_log_odds\|init_score" \
       C:/Users/syugo/AI/RaceAI_var1.0/pure_rank/src/simulate_ev.py
   ```

   QuinellaOdds 参照は EV 評価目的であり特徴量ではない。`quinella_odds_lookup` 変数名が
   grep に引っかかるが、`features_*.parquet` へのマージがなければ問題なし。

---

## 8. リーク停止閾値

この変更は EV 計算の正確性向上のみを目的とする。モデル精度への影響はない。
ただし変更後の実行時に Top-1 が変化していないことを必ず確認すること。

```
Top-1 > 40% または Spearman > 0.6 → 即座に実装停止して evaluator へ報告
```

モデル変更なしでこの閾値を超えることはあり得ない。もし超えた場合は QuinellaOdds が
誤って特徴量に混入しているか、別の実装バグがある。

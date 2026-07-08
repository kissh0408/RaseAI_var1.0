# 実装仕様書: WideOdds 統合 — EV 計算の真オッズ化 — 2026-07-01

## 禁止特徴量の確認

- [x] WideOdds CSV を `features_*.parquet` にマージしないことを確認した
- [x] WideOdds を学習の特徴量として使わないことを確認した
- [x] オッズは EV 計算（事後評価）のみに使用することを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した

---

## 1. 現状分析と変更の目的

### 問題: 現在の EV 計算は事後データを事前オッズの代替に使っている

`pure_rank/src/simulate_ev.py` の `_collect_bets_per_race()` は、EV を次の式で計算している:

```python
ref_w = wide_payout if wide_payout > 0 else wide_ref_payout  # HR 払戻 or 全払戻平均
ev_wide = p_wide * ref_w / STAKE
```

この実装には2つの問題がある。

**問題 1: 的中レースのみ実払戻を使っている**
- 的中したレース: `ev = p_wide * actual_payout / 100`（正しくない。事前に payout は不明）
- 外れたレース: `ev = p_wide * average_payout / 100`（事後平均で代替）

EV は「ベット前に計算できる期待値」でなければならない。実払戻（HR レコード）は結果であり、
ベット時点では知ることができない。現在の実装は EV フィルタの有効性を過大評価している可能性がある。

**問題 2: 全払戻平均は個別レースのオッズの代理指標として不適切**
- 実際のワイドオッズはレースによって 100〜10000 円の範囲で大きくばらつく
- 平均値（例: 1200円程度）では EV > 1.0 の判定が不正確になる

### 解決策: WideOdds CSV（事前オッズ）を使った真の EV 計算

`common/data/output/odds/WideOdds_{year}.csv` には各ペアの事前オッズが含まれる。

```
EV_wide(i, j) = P_wide(i, j) × wide_odds(i, j) / 100
```

- `P_wide(i, j)`: Harville 公式による馬 i・j の共 3 着以内確率（モデル出力）
- `wide_odds(i, j)`: WideOdds CSV の事前オッズ（ベット前に観測可能）
- `/100`: JRA の払戻オッズは「100円あたりの払戻金額」

### 期待効果

- EV > 1.0 のフィルタが「真に期待値プラスのベット」を選別できるようになる
- EV フィルタ後の ROI が現在より正確な値になる
- ROI > 100% 到達の判定が信頼できるものになる

### 変更による Top-1・NDCG@3・Spearman への影響

**変化なし**。モデルの特徴量・学習・スコアリングは一切変更しない。
EV 計算ロジックのみ変更する。

---

## 2. WideOdds CSV の仕様

### ファイル配置

```
C:\Users\syugo\AI\RaceAI_var1.0\common\data\output\odds\
├── WideOdds_2015.csv
├── WideOdds_2016.csv
├── ...
└── WideOdds_2026.csv
```

### CSV 列定義

| 列名 | 型 | 内容 |
|------|-----|------|
| `race_id` | int64 (16桁) | `2023010506010101` 形式。先頭 14 桁がレースID、末尾 2桁はスナップショット番号 |
| `horse_num_1` | int | ペアの馬番 1 |
| `horse_num_2` | int | ペアの馬番 2 |
| `odds_status` | str | `"ok"` = 発売中オッズ。それ以外 = 取消等 |
| `odds` | float | 100円あたりの払戻倍率（例: 172.5 = 172.5円） |

### race_id の形式と既存コードの対応

WideOdds CSV の `race_id` は **int64 (16桁)** である。既存の `simulate_ev.py` が使う
`race_id` は **str (14桁)** である（例: `"20230105060101"`）。

変換ルール:
```python
# WideOdds の race_id (16桁 int) → 既存の race_id (14桁 str)
wide_race_id_str = str(int(race_id))[:14]  # 末尾2桁（スナップショット番号）を除去
```

実例:
```
WideOdds: 2023010506010101 (16桁 int)
→ str変換:  "2023010506010101"
→ 先頭14桁: "20230105060101"  ← features_*.parquet の race_id と一致
```

---

## 3. 新関数の設計: `_build_wide_odds_lookup()`

`simulate_ev.py` に以下の関数を追加する。

```python
def _build_wide_odds_lookup(
    years: list[int],
    odds_dir: Path,
) -> dict[str, dict[tuple[int, int], float]]:
    """WideOdds_YYYY.csv を複数年読み込み、race_id → {(h1,h2): odds} の辞書を返す。

    Parameters
    ----------
    years : 対象年リスト（テストセットの年から自動検出する）
    odds_dir : common/data/output/odds/ ディレクトリ

    Returns
    -------
    {race_id_14: {(h1,h2): odds}} の辞書
        - race_id_14: str 型、14 桁（WideOdds の 16 桁 int の先頭 14 桁）
        - (h1, h2): _norm_pair() で正規化（小さい馬番が先頭）
        - odds: float（100円あたりの払戻倍率）

    除外条件
    --------
    - odds_status != "ok" の行
    - odds が NaN の行
    - CSV ファイルが存在しない年（警告を出してスキップ）
    """
```

### 除外条件の理由

| 除外条件 | 理由 |
|---------|------|
| `odds_status != "ok"` | 発売前・取消後のスナップショットを除く |
| `odds` が NaN | 払戻倍率が未確定のレコードを除く |
| ファイル不存在 | テストセットの年が将来（2026 等）の場合に備える |

### 実装上の注意点

- `race_id` は int64 として読み込まれる。`str(val)[:14]` で 14 桁の文字列に変換する
- 同一 race_id + pair に複数スナップショットが存在する場合は**最後の値**を採用する
  （ファイルは最終スナップショットのみを収録しているため、通常は重複しない）
- ペアの正規化: `_norm_pair(h1, h2)` を使う（predict.py からインポート済み）
- メモリ効率: 1年分の WideOdds_2023.csv は 313,673 行。辞書として保持しても問題ない

---

## 4. EV 計算の変更: `_collect_bets_per_race()` の修正

### 現在の実装（問題あり）

```python
ref_w = wide_payout if wide_payout > 0 else wide_ref_payout
ref_q = quin_payout if quin_payout > 0 else quin_ref_payout
ev_wide = p_wide * ref_w / STAKE
ev_quin = p_quin * ref_q / STAKE
```

### 変更後の実装（真の EV）

```python
# ワイド: WideOdds 事前オッズを使った真の EV
prior_odds_wide = wide_odds_lookup.get(rid, {}).get(wide_key, None)
if prior_odds_wide is not None:
    ev_wide = p_wide * prior_odds_wide / 100.0
else:
    ev_wide = float("nan")  # オッズ未取得のレースは NaN

# 馬連: 現状維持（WideOdds CSV には馬連オッズが含まれない）
# 引き続き HR 払戻平均を参照値として使用する
# TODO: QuinellaOdds CSV が整備された場合は同様の変更を施す
ev_quin = p_quin * quin_ref_payout / STAKE
```

### 関数シグネチャの変更

`_collect_bets_per_race()` に `wide_odds_lookup` 引数を追加する:

```python
def _collect_bets_per_race(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    wide_odds_lookup: dict[str, dict[tuple[int, int], float]],  # 新規追加
) -> pd.DataFrame:
```

### EV が NaN のレースの扱い

- `ev_wide = NaN` のレースは EV フィルタから除外する（`df_bets[ev_col] >= threshold` は NaN を自動除外）
- ROI 計算では NaN を「ベットしなかった」として扱う（分母から除外）
- NaN レース数を `n_ev_na` として集計・報告する

---

## 5. `_collect_bets_with_calibration()` の変更

既存の4手法（baseline / Platt / ROI-T / Isotonic）に対しても同様の変更を施す。

各手法の EV 計算を以下に統一する:

```python
prior_odds_wide = wide_odds_lookup.get(rid, {}).get(key, None)
ev = (p_wide * prior_odds_wide / 100.0) if prior_odds_wide is not None else float("nan")
```

`wide_ref_payout` への依存を全手法から除去する。

---

## 6. `main()` の変更

### WideOdds ローダーの呼び出し

```python
# テストセットの年を自動検出
test_years = sorted(df_test["race_date"].dt.year.unique().tolist())
odds_dir = PROJECT_ROOT / "common" / "data" / "output" / "odds"

# WideOdds の読み込み
print(f"Loading WideOdds for years: {test_years}")
wide_odds_lookup = _build_wide_odds_lookup(test_years, odds_dir)
print(f"  WideOdds races loaded: {len(wide_odds_lookup):,}")
```

### バリデーションセットへの適用

バリデーションセット（`compare_calibration_methods` 経由）でも同様の変更を施す。
バリデーション年（2024年等）の WideOdds を別途読み込む:

```python
valid_years = sorted(df_valid["race_date"].dt.year.unique().tolist())
wide_odds_lookup_valid = _build_wide_odds_lookup(valid_years, odds_dir)
```

### 結果 JSON への追加フィールド

`ev_results.json` に以下を追加する:

```json
{
  "wide_odds_coverage": {
    "n_races_total": 4775,
    "n_races_with_odds": 4750,
    "n_races_ev_na": 25,
    "coverage_rate": 0.9948
  }
}
```

---

## 7. 変更しないこと（禁止事項）

| 禁止事項 | 理由 |
|---------|------|
| WideOdds を `features_*.parquet` にマージしない | 特徴量汚染・データリークの防止 |
| WideOdds を学習の特徴量として使わない | プロジェクト憲法（市場情報排除） |
| EV フィルタで弾かれたレースの払戻をゼロ（損失）としてカウントしない | ベットしないレース = ROI 計算の対象外 |
| `wide_ref_payout`（平均払戻）を ev_wide の代替として残さない | 正確な EV 評価を妨げる |
| EV = NaN のレースを強制的に EV = 0 とみなさない | 未取得オッズを「期待値ゼロ」と解釈するのは誤り |

---

## 8. 評価方法: 変更前後の比較

変更後、以下を比較して報告する:

| 指標 | 変更前（HR払戻代替） | 変更後（WideOdds真のEV） |
|------|---------------------|-------------------------|
| EV 計算の根拠 | HR 払戻結果（事後） | WideOdds 事前オッズ（事前） |
| EV > 1.0 ベット数（ワイド） | ? | ? |
| EV > 1.0 ベット数 / 全レース数 | ? | ? |
| Wide ROI（EV > 1.0 フィルタ後） | 80.99%（全件ベース） | ? |
| Wide ROI（全件、フィルタなし） | 80.99% | 同値（モデル変更なし） |
| Top-1 的中率 | 30.18%（変化なし） | 30.18%（変化なし） |
| NDCG@3 | 0.538（変化なし） | 0.538（変化なし） |
| Spearman | 0.506（変化なし） | 0.506（変化なし） |
| EV = NaN のレース数 | 0（全件に ref_payout 代入） | 計測して報告 |

**Top-1・NDCG@3・Spearman は変化しない**。モデル・特徴量は一切変更しないため。

---

## 9. implementer への引き渡し事項

以下をこの順序で実装すること（ブランチ: `feature/wide-odds-ev`）:

1. `simulate_ev.py` に `_build_wide_odds_lookup()` を追加する（Section 3 の仕様に従う）

2. `_collect_bets_per_race()` のシグネチャに `wide_odds_lookup` を追加し、EV 計算を変更する（Section 4 の仕様に従う）

3. `_collect_bets_with_calibration()` の EV 計算を同様に変更する（Section 5 の仕様に従う）

4. `main()` に WideOdds ローダーの呼び出しを追加し、バリデーションセットにも適用する（Section 6 の仕様に従う）

5. 変更後に `simulate_ev.py` を実行して結果を報告する:
   ```bash
   cd C:\Users\syugo\AI\RaceAI_var1.0
   python pure_rank/src/simulate_ev.py --compare-calibration
   ```

6. 市場情報混入チェックを実行する:
   ```bash
   grep -rn "odds\|popularity\|market_log_odds\|init_score" pure_rank/src/simulate_ev.py --include="*.py"
   ```
   WideOdds 参照は EV 計算目的であり、特徴量ではない。`wide_odds_lookup` の変数名が
   grep に引っかかるが、`features_*.parquet` へのマージがなければ問題なし。

---

## 10. リーク停止閾値（再確認）

この変更は EV 計算の正確性向上のみを目的とする。モデル精度への影響はない。
ただし変更後の evaluate.py 実行時に以下を確認すること:

```
Top-1 > 40% または Spearman > 0.6 → 即座に実装停止して evaluator へ報告
```

EV 計算の変更がモデルのスコアリングに混入していないことを確認するため、
evaluate.py を `--no-ev` オプションなしで実行して Top-1 が 30.18% のままであることを確認する。

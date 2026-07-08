# 実装仕様書: 市場乖離ベット戦略 — 2026-07-01

## 禁止特徴量の確認

- [x] WideOdds を `features_*.parquet` にマージしないことを確認した（事後評価専用）
- [x] implied_probability を学習の特徴量として使わないことを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した
- [x] 人気順位を含まないことを確認した

---

## 1. 目的と背景

### 現状の問題

現在の EV フィルタ戦略（`ev_wide = p_model * odds >= 1.3`）は ROI=86.0% に留まる。
これは「期待値 130%」に対して 44 percentage point のオーバーコンフィデンスを示している。

現行戦略の根本的な限界:

| 問題 | 説明 |
|------|------|
| ペア選択基準 | `argmax(p_model)` でペアを選び EV をチェック |
| 市場との比較なし | p_model が市場の見立てより高いかを直接比較していない |
| ペア選択の機会損失 | 「p_model が高い」と「市場に対して edge がある」は別命題 |

### 本仕様の目的

WideOdds が示す市場 implied_probability との差（乖離スコア）を使い、
`argmax(p_model)` の代わりに `argmax(divergence)` でペアを選ぶ戦略を設計する。

**市場乖離が最大のペアを選ぶ** = 「モデルが市場より最も優位に評価しているペア」を選ぶ。

これは EV フィルタとは独立した発想であり、組み合わせて使うことも可能。

### 前提確認

本施策では WideOdds を「ベット前に観測可能な市場情報」として使用する。
WideOdds は `simulate_ev.py` の評価専用入力であり、`features_*.parquet` には含まない。
この使用方法は既存の EV 計算と同じパターンであり、プロジェクト憲法に違反しない。

---

## 2. implied_probability の計算

### 基本定義

WideOdds の `odds` 列は **decimal multiplier**（倍率）で格納されている。
例: `odds = 3.5` → 100 円ベットで 350 円返却（3.5 倍）。

```python
# 1ペアの raw implied 確率（overround 未補正）
p_implied_raw(i, j) = 1.0 / odds(i, j)
```

### overround の計算

1 レース内の全ペアの raw implied 確率の合計が overround:

```python
overround = sum(1.0 / odds(i, j) for all valid pairs (i, j) in race)
```

18 頭立てレースでは 153 ペア存在し、overround は 1.3〜2.0 程度が典型。

### overround 補正後の implied_probability

```python
p_implied(i, j) = p_implied_raw(i, j) / overround
                = (1.0 / odds(i, j)) / overround
```

補正後は全ペアの sum が 1.0 に正規化される。
補正後の p_implied は補正前より **小さくなる**（overround > 1.0 で割るため）。

### overround 補正の意味

補正前: p_implied_raw = 1/odds （市場の名目的な「公正価格」）
補正後: p_implied_corrected （overround を除去した真の市場確率推定）

Value bet 判定:
- **補正なし**: p_model > 1/odds ↔ EV_raw > 1.0（現行の EV フィルタと同値）
- **補正あり**: p_model > (1/odds) / overround ↔ EV_raw > 1/overround

overround > 1.0 なので 1/overround < 1.0。補正ありの条件は EV_raw 閾値が下がる。
これは「市場が全体として過剰な利益を取っている分、EV=0.8 でも市場より有利」を意味する。

### 代替案との比較

| 手法 | 数式 | 長所 | 短所 |
|------|------|------|------|
| EV_raw（現行） | p_model × odds | シンプル、直感的 | overround 無視 |
| 絶対乖離 | p_model - p_implied | スケール不変でない（大穴に不利） | 小確率ペアで不安定 |
| 対数乖離（推奨） | log(p_model / p_implied) | スケール不変、EV との関係が明確 | log(0) の処理が必要 |
| EV_corrected | p_model × odds × overround | overround 反映 | overround 計算コスト |

**推奨**: 対数乖離 `log_divergence = log(p_model * odds * overround)` を採用。
- EV_raw = 1.0 ↔ log(EV_raw * overround) = log(overround) ≈ 0.26〜0.69（overround=1.3〜2.0）
- 市場より有利 ↔ log_divergence > 0

ただし絶対乖離も出力して比較分析に使う。

---

## 3. 新関数の設計

以下の関数を `simulate_ev.py` に追加する。

### 3.1 `compute_race_overround()`

```python
def compute_race_overround(
    race_id: str,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]],
) -> float:
    """
    1 レースの全ペアから overround を計算する。

    Parameters
    ----------
    race_id         : str 形式の race_id（wide_odds_lookup のキー）
    wide_odds_lookup: _build_odds_lookup() の出力

    Returns
    -------
    float: sum(1/odds) for all valid pairs in race
           レース未発見の場合は 1.0 を返す（overround なし扱い）
    """
    pairs = wide_odds_lookup.get(race_id, {})
    if not pairs:
        return 1.0
    total = sum(1.0 / v for v in pairs.values() if v > 0)
    return max(total, 1.0)  # overround は常に >= 1.0
```

### 3.2 `collect_divergence_bets_per_race()`

現行の `_collect_bets_per_race()` が `argmax(p_model)` でペアを選ぶのに対し、
本関数は `argmax(divergence)` でペアを選ぶ。

```python
def collect_divergence_bets_per_race(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    hr_df: pd.DataFrame,
    T_opt: float,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]],
) -> pd.DataFrame:
    """
    各レースで log_divergence が最大のペアを選ぶ戦略のベット DataFrame を返す。

    アルゴリズム:
    1. 全ペアの p_model（Harville wide_matrix）を計算
    2. 各ペアの WideOdds から overround を計算
    3. log_divergence = log(p_model_ij * odds_ij * overround) を計算
    4. WideOdds が取得できないペアは log_divergence = NaN（スキップ）
    5. argmax(log_divergence) のペアを選択

    Returns
    -------
    pd.DataFrame with columns:
        race_id      : str
        horse_num_1  : int（小さい馬番）
        horse_num_2  : int（大きい馬番）
        p_model      : float（Harville p_wide）
        p_implied_raw: float（1/odds、overround 補正なし）
        p_implied    : float（overround 補正後）
        overround    : float
        ev_raw       : float（p_model * odds）
        log_divergence: float（log(p_model * odds * overround)、log(EV_raw * overround)）
        abs_divergence: float（p_model - p_implied）
        odds_wide    : float（WideOdds の prior odds）
        payout_wide  : int（HR 実払戻、0 = 外れ）
        hit_wide     : int（0/1）
        race_date    : Timestamp（time-sequential ordering 用）
        surface_code : int
        distance_category: object
        weather_code : int
    """
```

### 3.3 `sweep_divergence_threshold()`

```python
def sweep_divergence_threshold(
    df_div_bets: pd.DataFrame,
    thresholds: list[float],
    div_col: str = "log_divergence",
) -> pd.DataFrame:
    """
    乖離スコア閾値をスイープして ROI・的中率・ベット数を返す。

    Parameters
    ----------
    df_div_bets : collect_divergence_bets_per_race() の出力
    thresholds  : 閾値リスト（log_divergence の場合: [-0.5, 0, 0.1, 0.2, 0.3, 0.5]）
    div_col     : "log_divergence" または "abs_divergence"

    Returns
    -------
    pd.DataFrame: threshold / n_bets / hit_rate / return_rate / total_profit
    """
```

### 3.4 `compare_ev_vs_divergence()`

```python
def compare_ev_vs_divergence(
    df_bets_ev: pd.DataFrame,       # _collect_bets_per_race() の出力（argmax p_model）
    df_bets_div: pd.DataFrame,      # collect_divergence_bets_per_race() の出力（argmax divergence）
    ev_threshold: float = 1.3,
    div_threshold: float = 0.0,     # log_divergence > 0 = p_model > p_implied
) -> dict:
    """
    2戦略の ROI・的中率・ベット数を比較して返す。

    Returns
    -------
    dict:
        strategy_ev     : {"n_bets": int, "hit_rate": float, "roi": float}
        strategy_div    : {"n_bets": int, "hit_rate": float, "roi": float}
        strategy_combined: EV >= threshold AND log_divergence > threshold の積集合
    """
```

---

## 4. ベット選択戦略の詳細設計

### 戦略 A: EV フィルタのみ（現行）

```
ペア選択: argmax(p_model) by Harville
ベット条件: ev_wide >= threshold
```

### 戦略 B: divergence 選択 + EV フィルタ

```
ペア選択: argmax(log_divergence)
ベット条件: ev_wide >= threshold（従来と同じ閾値で比較）
```

### 戦略 C: divergence 選択 + divergence フィルタ

```
ペア選択: argmax(log_divergence)
ベット条件: log_divergence > threshold
```

### 戦略 D: 複合条件（最厳格）

```
ペア選択: argmax(log_divergence)
ベット条件: ev_wide >= 1.0 AND log_divergence > 0
```

戦略 D は「EV プラス かつ 市場に対して edge あり」の両条件を満たすベットのみを選ぶ。

---

## 5. 閾値探索の設計（後出しじゃんけん防止）

### 探索データ: valid_year = 2024

```python
# バリデーション用 WideOdds ローダー
valid_years = [2024]
wide_odds_lookup_valid = _build_odds_lookup(valid_years, odds_dir, "Wide")
```

2024 年のバリデーションセットで閾値を決定する。

探索する閾値:
```python
LOG_DIV_THRESHOLDS = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
EV_THRESHOLDS      = [0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
```

選択基準（バリデーションセット）: `n_bets >= 30` を満たす中で `return_rate` 最大の閾値を採用。

### 評価データ: test_year = 2025+

確定した閾値で `ev_results_divergence.json` に結果を保存する。

---

## 6. `main()` への追加

```python
# --divergence フラグが渡された場合
if args.divergence:
    print("\n=== 市場乖離ベット戦略 ===")
    df_div_bets = collect_divergence_bets_per_race(
        df_test, preds, hr_df, T_opt, wide_odds_lookup
    )
    comparison = compare_ev_vs_divergence(
        df_bets, df_div_bets,
        ev_threshold=1.3,
        div_threshold=0.0,
    )
    sweep_div = sweep_divergence_threshold(
        df_div_bets, LOG_DIV_THRESHOLDS, div_col="log_divergence"
    )
    print(sweep_div.to_string(index=False))
```

コマンドライン引数: `python simulate_ev.py --divergence`

---

## 7. 出力フォーマット（`ev_results_divergence.json`）

```json
{
  "divergence_strategy": {
    "strategy_a_ev13": {
      "ペア選択": "argmax_p_model",
      "n_bets": 698,
      "hit_rate": 0.155,
      "roi": 0.860
    },
    "strategy_b_div_ev13": {
      "ペア選択": "argmax_log_divergence",
      "ev_threshold": 1.3,
      "n_bets": null,
      "hit_rate": null,
      "roi": null
    },
    "strategy_c_div0": {
      "ペア選択": "argmax_log_divergence",
      "log_divergence_threshold": 0.0,
      "n_bets": null,
      "hit_rate": null,
      "roi": null
    },
    "log_divergence_sweep": []
  }
}
```

`null` は implementer が実行後に埋める。

---

## 8. 変更しないこと

| 禁止事項 | 理由 |
|---------|------|
| WideOdds を `features_*.parquet` にマージしない | 特徴量汚染防止 |
| implied_probability を特徴量として学習データに含めない | プロジェクト憲法 |
| テストデータで閾値を決定しない | 後出しじゃんけん禁止 |
| 既存の `_collect_bets_per_race()` を削除・変更しない | 後方互換 |

---

## 9. 評価基準

### 最低条件

- `n_bets >= 100` のベット数が確保できること（統計的有意性）
- ROI >= 80%（Phase 7 ベースラインの overall ROI を下回らない）

### 採用条件（テストセットで判定）

- 戦略 B〜D のいずれかが戦略 A（ROI=86.0%, n=698）を上回ること
- または同等 ROI でベット数が大きく増加すること（収益額の増加）

### リーク停止閾値（再確認）

本施策は EV 計算ロジックのみの変更であり、モデルスコアには影響しない。
Top-1・NDCG@3・Spearman は変化しないため、リーク停止閾値のチェックは
`simulate_ev.py` 実行前に `evaluate.py` で確認済みであることが前提。

---

## 10. implementer への引き渡し事項

1. `compute_race_overround()` を `simulate_ev.py` に追加（Section 3.1）
2. `collect_divergence_bets_per_race()` を `simulate_ev.py` に追加（Section 3.2）
   - race_date カラムの追加が必要（df_test から join）
3. `sweep_divergence_threshold()` を追加（Section 3.3）
4. `compare_ev_vs_divergence()` を追加（Section 3.4）
5. `main()` に `--divergence` フラグを追加（Section 6）
6. バリデーション年 2024 で閾値探索を実行
7. テストセット 2025+ で全戦略を評価し `ev_results_divergence.json` に保存

実行コマンド:
```bash
cd C:\Users\syugo\AI\RaceAI_var1.0
python pure_rank/src/simulate_ev.py --divergence --output pure_rank/data/02_features/ev_results_divergence.json
```

# 実装仕様書: リスク調整評価指標の追加 — 2026-07-01

## 禁止特徴量の確認

- [x] リスク指標の計算に市場情報を特徴量として使わないことを確認した
- [x] WideOdds はベット額計算の入力のみ（事後評価専用）
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した
- [x] Kelly 基準のパラメータをモデルの特徴量に含めないことを確認した

---

## 1. 目的と背景

### 現状の問題

現在の評価指標は **平均 ROI のみ**:

```
EV >= 1.3 → ROI = 86.0%（698 ベット / 4,775 レース）
EV >= 1.0 → ROI = 85.7%（1478 ベット）
```

この数値から「期待値プラスかどうか」はわかるが、以下が不明:

| 問題 | 影響 |
|------|------|
| 最大ドローダウン（MDD）が不明 | 100,000 円元本がいつ何円まで減るかわからない |
| シャープレシオが不明 | ROI の「安定性」が評価できない |
| Kelly 基準による推奨ベット額が不明 | 「いくら賭けるか」の根拠がない |
| 固定ベット以外の戦略比較がない | 資金管理戦略の比較ができない |

ROI = 86% は損失（ROI < 100%）であるにもかかわらず、ある戦略が「他の戦略より優れている」を
言うには単一の ROI だけでは不十分。リスク調整後の指標が必要。

### 実装対象

以下の 4 指標を追加する:

| 指標 | 計算対象 | 意味 |
|------|---------|------|
| 最大ドローダウン（MDD） | 固定ベット累積 P&L | ピークから谷への最大落ち幅（円・率） |
| シャープレシオ | ベット単位の収益率 | 平均収益 / 標準偏差（安定性） |
| Kelly ベット分率 | 推薦ベット額の分布 | 「適切なベット額」の根拠 |
| Kelly シミュレーション | 1/4 Kelly 資金推移 | 時系列的な資金増減の可視化 |

---

## 2. 実装先の判断: `simulate_ev.py`

### `evaluate.py` に追加しない理由

`evaluate.py` は **着順予測精度**（Top-1、NDCG@3、Spearman）の評価専用スクリプト。
ベット P&L の計算ロジックを混在させると責務が分散し、将来の保守コストが増える。

### `simulate_ev.py` に追加する理由

- EV 計算・ROI 計算は既に `simulate_ev.py` に実装済み
- ベット DataFrame (`df_bets`) が既に生成されている
- WideOdds lookup も既にロード済み
- リスク指標は EV フィルタ後の df_bets に対して計算する（EV との結合自然）

---

## 3. 追加する評価指標の定義

### 3.1 最大ドローダウン（MDD）

**定義**: 累積 P&L 時系列のピークから谷への最大落ち幅。

```
固定ベット（100 円/レース）の累積 P&L を race_date 昇順で計算する。
  MDD_yen = max over all t: (peak_cumulative_pnl_up_to_t - cumulative_pnl_at_t)
  MDD_pct = MDD_yen / initial_capital
```

**解釈**:
- MDD_yen が大きい → 途中で大きく資金が減る → 精神的・資金的なリスクが高い
- ROI が同じでも MDD が小さい方が優れた戦略

**計算対象**: EV フィルタ後のベットのみ（EV < threshold のレースはベットしない）。

### 3.2 シャープレシオ（ベット単位）

**定義**: ベット当たりの収益率の平均 / 標準偏差。

```
固定ベット（100 円）の場合:
  r_i = (payout_wide_i - 100) / 100  if hit
  r_i = -1.0                          if miss

sharpe = mean(r_i) / std(r_i)
```

**注意**: 年率換算は行わない（レース間の時間が不規則なため）。
「ベット当たりシャープレシオ」として報告する。

**解釈**:
- sharpe > 0: 平均収益率 > 0（ROI > 100%）
- sharpe < 0: 平均損失（ROI < 100%、現状）
- sharpe が高いほど ROI が安定している（高 ROI + 高シャープが理想）

### 3.3 Kelly 基準の適用

**定式化（Fractional Kelly, 1/4 Kelly）**:

```python
# b = 純利益オッズ（1 円賭けて勝った時の純利益）
# wide bets では: b = prior_odds_wide - 1（decimal odds - 1）
# p = モデル推定確率（p_wide from Harville）
# q = 1 - p

b = prior_odds_wide - 1  # net odds
f_kelly = (b * p - q) / b
        = p - (1 - p) / b
        = p - (1 - p) / (prior_odds_wide - 1)

# フラクショナル（1/4 Kelly）
f_quarter = max(f_kelly / 4.0, 0.0)  # 負は 0 にクリップ
```

**重要な等価性**:
```
f_kelly > 0  ↔  (b * p - q) / b > 0  ↔  b * p > q  ↔  p * odds > 1  ↔  EV > 1.0
```

つまり **EV > 1.0 かつ Kelly > 0 は同値**。EV フィルタと Kelly フィルタは一致する。
ただし EV は「どのレースにベットするか」を決め、Kelly は「いくら賭けるか」を決める。

**prior_odds の復元**:
`df_bets` には `ev_wide` と `p_wide` が格納されている。
`prior_odds_wide = ev_wide / p_wide`（どちらも NaN でない場合）

ただし `p_wide = 0` の場合は算出不能 → Kelly 対象外（`f_quarter = 0`）。

### 3.4 Kelly シミュレーション

**設定**:

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `initial_capital` | 100,000 円 | 開始時の資金 |
| `kelly_fraction` | 0.25 | 1/4 Kelly |
| `ev_threshold` | 1.0 | ベット条件（EV >= 1.0 のみ） |
| 順序 | `race_date` 昇順 | 時系列順にシミュレート |

**各ベットのルール**:
```python
bet_size_i = f_quarter_i * current_balance  # 可変ベット額
if hit:
    profit_i = (prior_odds_wide_i - 1) * bet_size_i
else:
    profit_i = -bet_size_i

new_balance = current_balance + profit_i
```

**最小ベット額**: `bet_size >= 10 円`（端数切り捨て）。
残高が 1,000 円未満になった場合はシミュレーションを終了し `ruined=True` を記録。

---

## 4. 関数設計

### 4.1 `compute_max_drawdown()`

```python
def compute_max_drawdown(
    pnl_series: np.ndarray,
) -> tuple[float, float]:
    """
    累積 P&L 時系列から最大ドローダウンを計算する。

    Parameters
    ----------
    pnl_series : 各ベットの P&L（+100, -100 など）の配列（時系列順）

    Returns
    -------
    tuple[float, float]:
        mdd_yen : 最大ドローダウン（円）、常に >= 0
        mdd_pct : mdd_yen / cumulative_max（最大資産比の最大 drawdown）

    アルゴリズム:
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative  # 各時点の drawdown
        mdd_yen = drawdown.max()
        mdd_pct = mdd_yen / max(running_max.max(), 1.0)
    """
```

### 4.2 `compute_sharpe_ratio()`

```python
def compute_sharpe_ratio(
    returns: np.ndarray,
) -> float:
    """
    ベット当たりのシャープレシオを計算する。

    Parameters
    ----------
    returns : 各ベットの収益率配列（hits: (payout-100)/100, miss: -1.0）

    Returns
    -------
    float: mean(returns) / std(returns)
           std が 0 の場合（全ベット同一結果）は 0.0 を返す
    """
```

### 4.3 `compute_kelly_fractions()`

```python
def compute_kelly_fractions(
    df_bets: pd.DataFrame,
    kelly_fraction: float = 0.25,
    ev_col: str = "ev_wide",
    p_col: str = "p_wide",
) -> pd.Series:
    """
    df_bets の各行に Kelly ベット分率を計算して返す。

    Parameters
    ----------
    df_bets       : _collect_bets_per_race() の出力
    kelly_fraction: フラクション（0.25 = 1/4 Kelly）
    ev_col        : EV 列名
    p_col         : p_model 列名

    Returns
    -------
    pd.Series: 各行の f_quarter（EV NaN または p=0 の行は 0.0）

    計算:
        prior_odds = df_bets[ev_col] / df_bets[p_col]
        b = prior_odds - 1
        f_full = (df_bets[p_col] - (1 - df_bets[p_col]) / b).clip(lower=0)
        f_quarter = f_full * kelly_fraction
    """
```

### 4.4 `simulate_kelly_quarter()`

```python
def simulate_kelly_quarter(
    df_bets: pd.DataFrame,
    initial_capital: float = 100_000.0,
    kelly_fraction: float = 0.25,
    ev_threshold: float = 1.0,
    ev_col: str = "ev_wide",
    pay_col: str = "payout_wide",
    hit_col: str = "hit_wide",
    p_col: str = "p_wide",
    min_bet: float = 10.0,
    ruin_threshold: float = 1_000.0,
) -> dict:
    """
    1/4 Kelly でのシミュレーション結果を返す。

    Parameters
    ----------
    df_bets           : race_date でソート済みの _collect_bets_per_race() 出力
                        （race_date カラムが必要）
    initial_capital   : 初期資金（円）
    kelly_fraction    : Kelly 分率（デフォルト 0.25）
    ev_threshold      : ベット条件（EV >= この値のみ）
    min_bet           : 最小ベット額（円）
    ruin_threshold    : 残高がこれを下回ったらシミュレーション終了（破産）

    Returns
    -------
    dict:
        initial_capital   : float（入力値そのまま）
        final_balance     : float（シミュレーション終了時の残高）
        total_profit_yen  : float（final_balance - initial_capital）
        final_return_pct  : float（total_profit / initial_capital）
        n_bets            : int（EV フィルタ通過後のベット数）
        hit_rate          : float
        mdd_yen           : float（最大ドローダウン、円）
        mdd_pct           : float（最大ドローダウン、初期資金比）
        sharpe_per_bet    : float（Kelly ベット収益率のシャープレシオ）
        ruined            : bool（破産フラグ）
        balance_series    : list[float]（残高時系列、分析用）

    実装メモ:
        1. df_bets を ev_col >= ev_threshold でフィルタ
        2. race_date 昇順でソート
        3. 各行で f_quarter を計算、bet_size = max(f_quarter * balance, min_bet)
        4. hit なら balance += (payout_wide / 100 - 1) * bet_size
           miss なら balance -= bet_size
        5. balance < ruin_threshold になったら break
        6. MDD は balance_series から compute_max_drawdown() を呼ぶ
    """
```

### 4.5 `compute_risk_metrics()`

```python
def compute_risk_metrics(
    df_bets: pd.DataFrame,
    ev_thresholds: list[float] = [1.0, 1.3],
    initial_capital: float = 100_000.0,
    kelly_fraction: float = 0.25,
    bet_type: str = "wide",
) -> dict:
    """
    複数 EV 閾値でリスク調整評価指標をまとめて計算する。

    Parameters
    ----------
    df_bets       : _collect_bets_per_race() の出力（race_date カラム必須）
    ev_thresholds : 評価する EV 閾値リスト
    bet_type      : "wide" または "quin"

    Returns
    -------
    dict: {
        "ev_{threshold}": {
            "fixed_stake": {
                "n_bets"      : int,
                "hit_rate"    : float,
                "roi"         : float,
                "mdd_yen"     : float,
                "mdd_pct"     : float,
                "sharpe_per_bet": float,
                "total_profit_yen": float,
            },
            "kelly_quarter": {
                "initial_capital": float,
                "final_balance"  : float,
                "n_bets"         : int,
                "hit_rate"       : float,
                "mdd_yen"        : float,
                "mdd_pct"        : float,
                "sharpe_per_bet" : float,
                "total_profit_yen": float,
                "final_return_pct": float,
                "ruined"         : bool,
            }
        }
    }
    """
```

---

## 5. race_date カラムの追加

現在の `_collect_bets_per_race()` は `race_date` を df_bets に含めていない。
時系列順の MDD 計算のために追加が必要。

### 修正: `_collect_bets_per_race()` への `race_date` 追加

```python
# rows.append({...}) の中に追加:
"race_date": first["race_date"] if "race_date" in grp.columns else pd.NaT,
```

`df_test` は `race_date` カラムを持っているため、`grp.iloc[0]["race_date"]` で取得できる。

---

## 6. `ev_results.json` への追加フィールド

```json
{
  "risk_metrics": {
    "wide": {
      "ev_1.0": {
        "fixed_stake": {
          "n_bets": 1478,
          "hit_rate": 0.205,
          "roi": 0.857,
          "mdd_yen": null,
          "mdd_pct": null,
          "sharpe_per_bet": null,
          "total_profit_yen": -21110
        },
        "kelly_quarter": {
          "initial_capital": 100000,
          "final_balance": null,
          "n_bets": 1478,
          "hit_rate": 0.205,
          "mdd_yen": null,
          "mdd_pct": null,
          "sharpe_per_bet": null,
          "total_profit_yen": null,
          "final_return_pct": null,
          "ruined": null
        }
      },
      "ev_1.3": {
        "fixed_stake": { "...": "..." },
        "kelly_quarter": { "...": "..." }
      }
    }
  }
}
```

`null` は implementer が実行後に数値で埋める。

---

## 7. `main()` への追加

```python
# 既存の EV スイープ後、risk_metrics 計算を追加:
print(f"\n--- Risk-Adjusted Metrics ---")
risk_metrics = compute_risk_metrics(
    df_bets,
    ev_thresholds=[1.0, 1.3],
    initial_capital=100_000.0,
    kelly_fraction=0.25,
    bet_type="wide",
)

for ev_thr, metrics in risk_metrics.items():
    print(f"\n[{ev_thr}]")
    fs = metrics["fixed_stake"]
    kq = metrics["kelly_quarter"]
    print(f"  Fixed stake  : n={fs['n_bets']:,}, ROI={fs['roi']*100:.2f}%, "
          f"MDD={fs['mdd_yen']:.0f}円 ({fs['mdd_pct']*100:.1f}%), "
          f"Sharpe={fs['sharpe_per_bet']:.3f}")
    print(f"  Kelly (1/4)  : initial=¥{kq['initial_capital']:,.0f}, "
          f"final=¥{kq['final_balance']:,.0f}, "
          f"MDD=¥{kq['mdd_yen']:,.0f} ({kq['mdd_pct']*100:.1f}%), "
          f"ruined={kq['ruined']}")
```

コマンドライン引数は追加不要（デフォルトで常に計算）。

---

## 8. 読み方ガイド（评価基準）

### MDD（最大ドローダウン）

| MDD_pct | 解釈 | アクション |
|---------|------|-----------|
| < 20% | 良好 | そのまま継続 |
| 20〜40% | 注意 | Kelly 分率を下げる（1/8 Kelly 等） |
| > 40% | 危険 | 戦略の根本的見直しが必要 |

### シャープレシオ（ベット当たり）

| Sharpe | 解釈 |
|--------|------|
| > 0.1 | 良好（安定したプラス収益） |
| -0.1〜0.1 | ニュートラル（ROI でのみ判断） |
| < -0.1 | 損失が安定している（ROI 改善を優先） |

### Kelly シミュレーション

- `ruined = True` → 1/4 Kelly でも破産するリスクあり → Kelly 分率をさらに下げる
- `final_return_pct > 0` かつ `ruined = False` → 複利効果でプラスになった
- `final_return_pct < 0` かつ `mdd_pct < 30%` → 損失は固定ベットより少ない可能性

---

## 9. 変更しないこと

| 禁止事項 | 理由 |
|---------|------|
| `evaluate.py` にベット指標を追加しない | 責務分離（着順精度 vs ベット収益は別） |
| Kelly ベット額を過去データで最適化しない | 後出しじゃんけん禁止 |
| MDD 最小化を目的関数に加えない | モデルの学習目標は着順予測のみ |
| リーク停止閾値を変更しない | Top-1>40% の場合は依然として実装停止 |

---

## 10. implementer への引き渡し事項

1. `simulate_ev.py` に `compute_max_drawdown()` を追加（Section 4.1）
2. `simulate_ev.py` に `compute_sharpe_ratio()` を追加（Section 4.2）
3. `simulate_ev.py` に `compute_kelly_fractions()` を追加（Section 4.3）
4. `simulate_ev.py` に `simulate_kelly_quarter()` を追加（Section 4.4）
5. `simulate_ev.py` に `compute_risk_metrics()` を追加（Section 4.5）
6. `_collect_bets_per_race()` に `race_date` カラムを追加（Section 5）
7. `main()` に risk_metrics 計算・表示を追加（Section 7）
8. `ev_results.json` の保存ロジックに `risk_metrics` セクションを追加（Section 6）

実行コマンド:
```bash
cd C:\Users\syugo\AI\RaceAI_var1.0
python pure_rank/src/simulate_ev.py --output pure_rank/data/02_features/ev_results.json
```

出力確認:
```
--- Risk-Adjusted Metrics ---
[ev_1.0]
  Fixed stake  : n=1,478, ROI=85.72%, MDD=XXXX円 (XX.X%), Sharpe=-X.XXX
  Kelly (1/4)  : initial=¥100,000, final=¥XXXXX, MDD=¥XXXXX (XX.X%), ruined=False/True
[ev_1.3]
  Fixed stake  : n=698, ROI=86.00%, MDD=XXXX円 (XX.X%), Sharpe=-X.XXX
  Kelly (1/4)  : initial=¥100,000, final=¥XXXXX, MDD=¥XXXXX (XX.X%), ruined=False/True
```

# RaceAI_var1.0 現状システム指南書（2026-07-09時点）

本ドキュメントは、現行の学習方法・予測方法・特徴量・評価結果を一元的にまとめた参照資料である。新しい改善案を検討する前に必ず読むこと。数字は全て実測値（誇張なし）。

---

## 1. アーキテクチャ全体像

4層構成（2026-07-08 Benter型再構築）。

```
L0: common/data/      JV-Link データ取得・前処理
L1: pure_rank/        LambdaRank 着順予測（市場情報を完全排除）
L2: prob_fusion/      条件付きロジットによる確率統合（Benter型）
L3: betting/          EV/Kelly 馬券推奨
評価: evaluation/     分割定義・市場ベースライン・OOS測定基盤
```

**目標**: 市場（1番人気）を超える的中率・回収率を、市場情報を使わない純粋能力モデルで達成すること。

---

## 2. L1: pure_rank（LambdaRank 着順予測）

### 2.1 モデル

- LightGBM `objective=lambdarank`, `metric=ndcg`
- `label_gain=[0,1,3,7,15,31,100]`（1着の重みを強く設定）
- `num_leaves=63, min_child_samples=50, reg_alpha=1.0, reg_lambda=2.0, learning_rate=0.05, n_estimators=800`
- 5シードアンサンブル（seed 42〜46）× 3fold = 15モデル
- `init_score` は使わない（var2.0.0との根本的な違い。市場残差学習をしない）

設定は `pure_rank/config/train_config.json` が単一の真実。

### 2.2 時系列分割

```
train_end  = 2023-12-31
valid_end  = 2024-12-31
TEST       = 2025-01-01 以降（4,775レース）
fold_valid_years = [2022, 2023, 2024]  # fold1/2/3 の early-stopping年
```

**fold2モデル**（train<2023、2023はearly stopping専用、2024/2025+は完全未見）が、唯一「完全にOOSな」測定に使えるモデル。**本番アンサンブル（15モデル）の全期間平均スコアは、ある期間に対して別foldが学習済みという意味で「in-sample汚染」がある**ため、精度の正式判定には使わない（`evaluation/reports/gate_summary.json`の`contaminated_reference_runs`に格下げ保存）。

### 2.3 特徴量（`features_v39_course_slim.parquet`, 132列）

| プレフィックス | 列数 | 内容 |
|---|---|---|
| `hist_` | 51 | 過去走成績の集計（勝率・連対率・上がり3F・タイム偏差・賞金等）。全てshift(1)でリーク防止 |
| `trn_` | 22 | 調教データ（坂路HC・ウッドWC）。加速度・時計差・頻度等 |
| `field_` | 9 | レース内相対Zスコア（既に安全な列を集計） |
| `horse_`, `corner_`, `race_`, `wakuban_` 等 | 各数列 | 属性・枠番・コーナー通過順位等 |

**市場情報（オッズ・人気）は特徴量に一切含まれない**。`pure_rank/src/common.py::FORBIDDEN_MARKET_COLS`（完全一致ブロックリスト）+ `SUSPICIOUS_MARKET_NAME_PATTERN`（正規表現の第二防波堤、2026-07-09追加）で二重に防御している。

### 2.4 現在の性能（fold2 OOS、TEST 2025+、4,775レース）

| 指標 | 値 |
|---|---|
| Top-1的中率 | 約29-31%（fold2単体）／本番15モデルアンサンブルで30.24% |
| Spearman相関 | 約0.50 |
| 1番人気Top-1（市場ベンチマーク） | **32.90%** |

→ **市場に対して-2.66pp未達**。これがプロジェクトの中心的な未解決問題。

---

## 3. L2: prob_fusion（確率統合）

### 3.1 Benter型条件付きロジット

```
p_i ∝ exp(α・z_i + β・ln(q_i) + γ・x_i)
```

- `z_i`: L1のレース内zスコア（`pure_score_z`）
- `q_i`: 市場確率（単勝オッズから比例法 or べき乗法で算出、`prob_fusion/src/market_prob.py`）
- `x_i`: 追加候補特徴量（αゲートで使用、通常はγ=0固定）
- α, β は最尤推定（`prob_fusion/src/fit_fusion.py::fit_fusion_mle`）

### 3.2 確定した中心的事実: α = 0

fold2 OOS正式測定（fit=2023-2024, TEST=2025+）:

```json
{
  "alpha": 0.0,
  "beta": 1.0343,
  "lrt_p_value": 0.99999,
  "test_logloss_fusion": 1.93124,
  "test_logloss_market": 1.93124,  // fusionと同一
  "test_top1": 0.3290              // 市場と同一値
}
```

**αが下限0に張り付く = L1のスコアは市場オッズが持つ情報を一切追加しない。** 統合確率は実質「市場確率そのもの」に収束する。これは以下の複数の独立した検証で繰り返し確認されている（詳細は§6参照）:

- 通常のL1→L2統合（上記）
- 市場情報（確定オッズ）を直接L1特徴量に混入させた実験
- var2.0.0のinit_scoreトリック（市場相関の強いJRA公式予想を残差学習のベースに使う手法）の再現実験
- JRA公式マイニング指数をαゲート候補として検定（γ=0）

### 3.3 複勝・ワイド・馬連確率の算出方法

**単勝確率p_winのみ直接推定し、2着・3着・複勝・ペア確率は全て数式で逆算している**（直接予測ではない）。

| 券種 | 算出式 | 実装 |
|---|---|---|
| 複勝（top3） | Stern型逐次確率（λ2, λ3パラメータ） | `prob_fusion/src/place_prob.py` |
| ワイド | Stern式（`pair_probs.py`）**採用**、Harville式（`ev_filters.py`）で比較 | `betting/src/pair_probs.py`, `betting/src/ev_filters.py` |
| 馬連 | Harville式**採用**（Sternより較正誤差が良い） | 同上 |

**モデル選定根拠**（`betting/analysis/pair_probability_model_comparison.json`, 2026-07-09実測）:

| | ワイド logloss | ワイド較正誤差 | 馬連 logloss | 馬連較正誤差 |
|---|---|---|---|---|
| Stern式 | 0.1096 | 8.35pp | 0.0460 | 28.65pp |
| Harville式 | 0.1103 | 13.55pp | 0.0460（僅差） | 16.01pp |

→ ワイドはStern、馬連はHarville、と**券種ごとに使い分け**ている。

λ2, λ3は `prob_fusion/src/place_prob.py::fit_stern_lambda()` でfit期間（2023-2024）の複勝的中データに対する尤度最大化により推定（現行値: λ2=0.6018, λ3=0.6381）。

---

## 4. L3: betting（EV/Kelly推奨）

### 4.1 単勝

`betting/src/backtest.py::run_backtest_oos.py` — VALID(2024)でEV閾値を選定、TEST(2025+)で1回だけ判定（Rule 3遵守）。

**結果: 正のEVベットが存在しない**（α=0のため統合確率≒市場確率、市場に対する優位性ゼロ）。

### 4.2 複勝

`evaluation/place_baseline.py` — HR確定払戻データを使った決済ベースの診断（事前オッズではない、上限診断としてのみ有効）。

```
モデルTop-1複勝ROI: 84.63%
1番人気複勝ROI:     85.15%
差:                 -0.52pp
```

→ 市場にわずかに劣る。

### 4.3 ワイド・馬連

`betting/src/run_backtest_oos_pairs.py` — argmax EV選択（レース内の全ペアから最高EVの1つを選ぶ）で構築。

**重大な方法論的問題を発見・対処済み**: レース内で100超の候補ペアからargmax選択すると、統計的に「稀な大穴の的中」に依存した見かけのROIが生まれる（winner's curse）。実測: ワイドROI 303%のうち**98.65%がたった1件の的中（14,257倍）に依存**、それを除くと実質ROIは4.09%まで崩壊。この問題を検出するため`payout_not_concentrated_top1_lte_30pct`, `n_hits_gte_10`ゲートを追加。

**結果**:
- ワイド: `phase3_pass = false`（払戻集中度ゲートで不合格）
- 馬連: VALID期間で正のEV閾値が見つからず`skipped`

---

## 5. 評価プロトコル（OOS規律）

### 5.1 fold2限定OOS測定

`evaluation/reports/fusion_oos_fold2.json` が正式判定の一次資料。

```
fit期間:  2023-01-01 〜 2024-12-31（2023はfold2のearly-stopping年、弱い接触注記あり）
TEST期間: 2025-01-01 〜（4,775レース、完全未見）
```

### 5.2 Rule 3（後出しじゃんけん禁止）

パラメータ・閾値の選定はVALID期間のみで行い、TESTは1回だけ最終判定に使う。`betting/src/ev_engine.py::select_ev_threshold_on_valid`, `pure_rank/experiments/place_calibration/`等、全ての新規実験がこの規律を踏襲している。

### 5.3 汚染測定の隔離

15モデル全fold平均スコアによる旧測定値は `evaluation/reports/gate_summary.json` の `contaminated_reference_runs` に格下げ保存され、合否判定には使われない。

---

## 6. これまでに試して「効果なし」と判明した手法一覧（重複実験防止用）

| # | 手法 | 結果 | 参照 |
|---|---|---|---|
| 1 | L1特徴量拡張（v40〜v50、条件別特徴量等） | 全て不合格・Phase 7基準以下 | CLAUDE.md Phase進捗 |
| 2 | 確定単勝オッズを直接L1特徴量に投入 | α=0.021（有意差なし）、Top-1は市場と同値に収束するのみ | `pure_rank/experiments/market_leak_diagnostic/` |
| 3 | var2.0.0式init_scoreトリック（JRA公式予想を残差学習ベースに） | VALID期間では改善して見えるが、真のOOSでTop-1が27.64%まで悪化（本番30.24%・市場32.90%を下回る） | `pure_rank/experiments/jra_mining_diagnostic/` |
| 4 | JRA公式マイニング指数をαゲート候補に | γ=0、LRT p=1.0（市場条件付きで無情報）。加えて候補とpure_score_zの相関0.73で「情報過多」警告 | 同上 |
| 5 | 複勝(top3)確率の直接binary予測（市場情報なしのL1特徴量のみ） | Stern逆算式にlogloss有意に劣後（ブートストラップCI）。較正誤差のみわずかに改善 | `pure_rank/experiments/place_direct/` |
| 6 | Stern式λ2/λ3の再フィット改善 | 点推定はわずかに改善するが実質誤差の範囲内 | `pure_rank/experiments/place_calibration/` |
| 7 | Stern式へのisotonic事後較正 | logloss・較正誤差とも点推定は改善するが、ブートストラップ95%CIがゼロを含み統計的有意性なし | 同上 |
| 8 | ワイド/馬連のargmax EV選択によるバックテスト | 数字上のROIは巨大だが払戻集中度（winner's curse）が原因と判明、集中度ゲート追加で不合格化 | `betting/src/run_backtest_oos_pairs.py` |

**共通する結論**: L1（市場情報なし）の情報は、市場オッズに完全に織り込まれている。市場情報を直接混ぜても、真のOOSでは改善しないか悪化する。これまでのアプローチ（特徴量追加・モデル形式変更・確率算出方式変更）は全て同じ壁（α=0）に突き当たっている。

---

## 7. 未解決・未着手の領域

- **調教データの深掘り**（HC/WC時系列、坂路加速度の変化・頻度軸）— v50失敗の教訓を踏まえ「水準」ではなく「変化・頻度・形状」軸に限定した5候補が`docs/specs/2026-07-08-alpha-gate-data-expansion-plan.md`のTrack Bとして事前登録済みだが未着手
- **1着・2着・3着それぞれの個別直接予測**（複勝のみ検証済み、単勝・2着・3着は未検証。ただし複勝の結果からは大きな改善は期待しにくい）
- **L4当日パイプライン**の完全復旧（`main/main.py`等、一部import修復済みだが本体は未完）
- **真のレース前オッズデータの欠如**（現在の`WinOdds_*.csv`等は全て確定後オッズ。市場効率性の検証は「最も有利なケース」での検証にとどまる）

---

## 8. ディレクトリ早見表

```
pure_rank/config/train_config.json   # L1学習設定（単一の真実）
pure_rank/src/create_features.py     # 特徴量生成
pure_rank/src/train.py               # LambdaRank学習
prob_fusion/src/fit_fusion.py        # Benter統合MLE
prob_fusion/src/place_prob.py        # Stern型複勝確率
prob_fusion/src/oos_protocol.py      # fold2 OOSプロトコル定義
betting/src/pair_bets.py             # ワイド/馬連EV選定・決済
betting/src/run_backtest_oos_pairs.py # ワイド/馬連OOSバックテスト
evaluation/reports/gate_summary.json # 正式合否判定の集約
evaluation/alpha_gate.py             # 新規候補特徴量のαゲート審査
pure_rank/experiments/               # 隔離実験（本番非接触）
```

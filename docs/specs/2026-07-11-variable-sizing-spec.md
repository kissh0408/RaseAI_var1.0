# 実装仕様書: margin対応可変サイジング（variable sizing） — 2026-07-11

**作成者**: planner
**承認**: ユーザーの明示的判断による実装指示（性能改善の根拠に基づくものではない。§0参照）
**実装先**: `betting/experiments/variable_sizing/`（隔離実験・本番非接触）
**実装担当**: implementer（本仕様書はコードを含まない）
**先例書式**: `docs/specs/2026-07-11-confidence-tiers-spec.md`（margin定義・凍結境界・隔離パターン）、
`docs/specs/2026-07-10-loss-minimization-implementation-spec.md`（flat top-1・決定規則v2・DISCLAIMER）

---

## 0. 決定的な前提（本Phaseの位置づけ。仕様書・README・結果JSONに必ず反映）

### 0.1 否定的結果の確定事実（先行Phase confidence tiers）

`docs/specs/2026-07-11-confidence-tiers-spec.md` §15 において、自信度（margin =
`pure_score_z` の1位−2位差）と対1番人気ROI優位の関連を検証した結果、
**全仮説不通過**（H1〜H4・順序仮説すべて p ≥ 0.24 ≫ Bonferroni閾値0.002、
verdict = `confidence_does_not_predict_market_edge`）が確定している。

さらに記述的診断で以下が確認済み:

> 自信度が高い階層ほどモデル予測1位が1番人気と一致する割合が明確に上昇する
> （T1: 37.4% → T4: 74.2%。平均オッズも4.64倍→3.26倍と単調低下）。
> **自信度で重み付けすると、モデルの数少ない優位性の源泉（過剰人気馬回避）を
> 薄めるリスクがある。**

### 0.2 本Phaseの目的（性能改善ではない）

**本Phaseは性能改善（ROI向上）を目的としない・主張しない・示唆しない。**

位置づけは「モデル自身の自信度に応じてリスク配分を変える」という**ユーザーの設計選好の実装**
である。前提は「ROIは変わらない（または悪化しうる）」であり、期待できるのは
リスクプロファイルの形状変更（自信度の低いベットへのエクスポージャ縮小）のみである。
§0.1の記述的診断より、本設計はむしろ優位性の源泉を薄める方向に作用しうる。
この事実を隠さず、仕様書（本節）・実験README・結果JSONの `caveats` 配列・DISCLAIMER の
**4箇所すべて**に埋め込むこと（§4.4・§10）。

### 0.3 黒字化表現の全面禁止

「黒字化」を主張・示唆する表現をコード・ログ・JSON・レポート・コミットメッセージに
一切書かない。`betting/src/flat_top1.py::DISCLAIMER` を踏襲し、本Phase専用に拡張した
定型文（§4.4）を全出力に埋め込む。

### 0.4 期待値は負・ケリー不採用

- 期待値は**負**（fold2 OOS実測: ROI 81.89%、元本の約18%の期待損失）。
- ケリー基準は**不採用**（confidence-tiers仕様 §7.4: 負のEVではケリーの数学的前提が
  成立せず、最適解は「賭けない」。本Phaseで実装するのは有界な階層別配分であり、
  比例ケリーとは無関係である）。

### 0.5 倍率のROI最適化は無意味かつ禁止

§0.1より margin は対市場優位を予測しないことが確定しているため、倍率をROI（または Δ）で
最適化して選ぶことは統計的に無意味であり、後出しじゃんけん（Rule 3違反）にもなる。
**倍率は「リスク予算保存＋単調性＋有界性」という設計制約のみから決める**（§3）。
VALID(2024)の着順・払戻データを使ってよいのは**リスク検証（月次MDD・エクスポージャ測定、
§5）のみ**であり、倍率・階層境界・関数形の選択には一切使わない（outcome-blind設計）。

---

## 1. 市場情報境界・禁止事項の確認

- [x] **L1（`pure_rank/src/`）に一切変更を加えない**。既存 fold2 OOS スコア
  （`pure_score_z`）とオッズ（L3許可範囲: 除外条件・決済のみ）を読むだけである。
- [x] **サイジング関数はmarginのみの関数**とする。stake の決定にオッズ・人気・
  `market_log_odds`・`init_score`・融合確率 `win_prob_est` を一切使わない
  （§9のテストで機械的に担保。オッズの使用は `select_top1_bets` の min/max 除外と
  `settle_win_bets` の決済の2箇所のみ = 既存 flat 運用と同一）。
- [x] z の二重使用なし（L2統合は本Phaseに登場しない）。
- [x] 本番凍結値 `betting/config/betting_config.json`（`stake_fraction=0.001`,
  `stake_fraction_frozen=true`）を**変更しない**（§8隔離宣言）。
- [x] ROI は記述的報告のみ（§6.2）。いかなる系列のROIが100%を超えても「黒字」と
  記述せず、§7の危険信号として扱う。

---

## 2. 入力・再利用資産（コピー禁止・import再利用）

| 資産 | 用途 |
|---|---|
| `betting/src/flat_top1.py::select_top1_bets` | ベット選定（min_odds=2.0 / max_odds=50.0、本番と同値）。**import再利用** |
| `betting/src/flat_top1.py::settle_win_bets` | 決済（payout = stake × odds）。**import再利用** |
| `betting/src/flat_top1.py::DISCLAIMER` | 基底定型文（§4.4で拡張） |
| `betting/experiments/confidence_tiers/tiers_lib.py::compute_race_margin` | margin計算（1位・2位の決定的順序含む）。**import再利用** |
| `betting/experiments/confidence_tiers/tiers_lib.py::assign_tier` / `assign_tier_batch` | 階層割当。境界値ちょうどの扱いは confidence-tiers §13 の確定記法（`np.searchsorted(boundaries, margin, side="left")` = 境界値ちょうどは下位階層）と**同一** |
| `betting/src/backtest.py::load_scored_odds_frame` | scores + features + オッズ結合（既存除外フィルタ込み） |
| `betting/src/derive_flat_fraction.py::_monthly_max_drawdown` / `_busiest_day_exposure` | 月次MDD・最繁忙日エクスポージャ。**同一セマンティクス**（暦月ごとの独立P&L・対初期bankroll比・monthly_mdd_limit=0.15）を可変系列に適用する。import可能ならimport、モジュール構造上不可なら同型実装＋出典コメント |

### 階層境界（凍結値の再利用。新たな境界導出は行わない）

confidence-tiers §13 の凍結境界をそのまま使う（2023年margin分布のみから導出済みの
outcome-blind 値。出典: `betting/experiments/confidence_tiers/results/stage1_boundaries.json`）:

```
b1 = 0.11895233392715454
b2 = 0.28122442960739136
b3 = 0.5161097198724747
```

割当: `tier = 1 + np.searchsorted([b1, b2, b3], margin, side="left")`
（T1 = margin最小 = 低自信 〜 T4 = margin最大 = 高自信。margin=b1ちょうど→T1）。

---

## 3. 可変サイジング関数の設計（事前登録。後出し変更禁止）

### 3.1 関数形: 階層別固定倍率（有界・単調・階段状）

```
stake(t) = base_stake × m_t
base_stake = 初期bankroll × f_var        （f_varは§5で機械的に導出・凍結）
m = (m1, m2, m3, m4) = (0.5, 0.75, 1.25, 1.5)   ← 事前登録・固定
```

- **有界性**: min倍率 = 0.5、max倍率 = 1.5（max/min比 = 3.0）。これを上限・下限として
  凍結する。単純比例（stake ∝ margin）は非有界であり不採用。
- **単調性**: m1 < m2 < m3 < m4（自信度が高いほどstakeが大きい = ユーザーの設計選好）。
- **倍率値の導出根拠（outcome-blindの構造的担保）**: 1を中心に対称
  `m_t = 1 + d×c_t`（d = 0.25、c = (−2, −1, +1, +2)）という**設計制約のみ**から定めた。
  （記法訂正 2026-07-11: 初版の式 `m_t = 1 + d×(2t−5)/2` は登録倍率
  (0.5, 0.75, 1.25, 1.5) を再現しないためevaluator指摘により式表記のみ修正。
  登録倍率の数値自体は初版から不変であり、実装・判定への影響なし。）着順・払戻・ROI・
  Δのいかなる実データも参照していない（参照してはならない。§0.5）。
  d = 0.25 の選定理由も設計制約のみ: (i) 1/4刻みは§3.3の100円丸めを厳密に成立させる
  最粗の粒度であり、(ii) max/min比3.0は「配分を変えたことが観測可能な最小限の振れ幅」
  として事前に固定する。d をこれ以外の値に変更する場合は本仕様書の改訂（planner差し戻し）
  を要し、VALID/TESTの結果を見た後の変更は禁止。

### 3.2 リスク予算保存則（事前登録）

平均stakeが現行 flat（f=0.001相当）を大きく超えないことを以下で保証する:

1. **占有率加重平均倍率 ≈ 1.0**: VALID(2024)ベット集合の階層占有率
   `w_t = n_t / Σn_t`（margin分布のみから計算する outcome-blind 統計量。
   着順・払戻列を読まない関数シグネチャとする — §9-3）に対し

   ```
   M̄ = Σ_t w_t × m_t   が   0.95 ≤ M̄ ≤ 1.05   を満たすこと
   ```

   参考: confidence-tiers §15 の実測占有（666/677/641/624）で計算すると
   M̄ ≈ 0.996 であり成立見込み。**成立しない場合は倍率を調整せず planner へ差し戻す**
   （倍率をいじって通すことは事前登録違反）。
2. **最大倍率上限**: m4 = 1.5 を超える倍率は導入しない（§3.1で凍結済み）。
3. **総リスク上限は§5の機械的導出が最終防衛線**: 上記1・2は設計時保証であり、
   実測の月次MDD・エクスポージャは§5で別途検証する（保存則が成立しても
   MDD検証は省略しない）。

### 3.3 100円丸め問題への対処（事前登録）

JRA最低購入単位は100円。倍率 {0.5, 0.75, 1.25, 1.5} が**丸め誤差ゼロ**で100円単位に
乗るためには、`base_stake` が **400円の倍数**でなければならない
（0.75 × 400 = 300円。0.75 × 100 = 75円は購入不能）。

- **最低運用bankroll（可変サイジング運用時）**:
  `min_bankroll_variable = 4 × stake_rounding_yen / f_var`。
  f_var = 0.001 なら **400,000円**（flat運用の100,000円の4倍）。
- `base_stake` が400円の倍数でない bankroll は `apply_flat_sizing` と同様に
  **ValueError で拒否**する（暗黙の切り捨てで実効倍率が設計値からずれることを禁止する。
  例: bankroll=100,000 では base=100円、m2=0.75 → 75円 → 0円に退化 or 100円に切り上げ、
  いずれも実効倍率 {1,1,?,?} となり設計 {0.5,0.75,1.25,1.5} と乖離する）。
- これにより **stake は常に厳密に base_stake × m_t**（丸め演算は防御的検証のみで
  実値を変えない）となり、「丸めで実効倍率がずれる」問題は構造的に発生しない。
- バックテスト（§5・§6）は bankroll = 400,000円・f_var（§5導出値）で実施する。
  比較対象の flat 系列は**同一bankroll**で base_stake = 400円の flat とし、
  総ステーク規模を揃えて比較する（stake規模が違うと DD 比較が無意味になるため）。

### 3.4 サイジング関数のシグネチャ制約（市場情報遮断の構造的担保)

サイジング関数（倍率決定・stake計算）は
`(margin または tier, base_stake, multipliers)` のみを入力とし、
オッズ・払戻・着順・人気・確率のいかなる列も**引数に取らない**シグネチャとする
（§9-6の静的検査対象）。

---

## 4. 手順（Rule 3 遵守: VALID凍結 → TEST 1回）

### Stage V0: データセット構築（outcome-blind部分）

1. `build_dataset.py`（confidence-tiers の同名スクリプトのパターン踏襲）:
   `load_scored_odds_frame` → `select_top1_bets` → `compute_race_margin` →
   凍結境界で `assign_tier_batch`。全期間を含んでよいが、以後のスクリプトは
   io直後に期間フィルタを適用し、Stage V1/V2 は 2024年のみを読む（TEST行に触れない）。
2. `run_v0_occupancy.py`: VALID(2024)の階層占有率 w_t を算出し、リスク予算保存則
   （§3.2-1: 0.95 ≤ M̄ ≤ 1.05）を検証 → `results/occupancy_valid.json`。
   **このスクリプトは着順・払戻列を読まない**（outcome-blind。§9-3で担保）。
   不成立なら以後に進まず planner へ差し戻し。

### Stage V1: リスク再検証と f_var の機械的導出（VALID 2024のみ）

`run_v1_risk_valid.py`: 可変stake系列（§3の倍率適用済み）に対し
`derive_flat_fraction.py` と**同一セマンティクス**の分析を実行する:

1. VALID(2024)ベット集合に可変サイジングを適用（f0 = 0.001, bankroll = 400,000）し
   決済（`settle_win_bets`）。
2. **月次MDD**（暦月ごとの独立P&L ÷ 初期bankroll、`_monthly_max_drawdown` と同一定義）
   の worst 月 `worst_month_dd_var@f0` を実測。
3. **決定規則（v2の可変版。線形性を利用した機械的導出）**:

   ```
   f_scale  = monthly_mdd_limit / (worst_month_dd_var@f0 ÷ f0)     （monthly_mdd_limit = 0.15、不変）
   f_capped = 0.5 × f_scale                                         （安全係数 k = 0.5、不変・緩和禁止）
   f_var    = グリッド {0.001, 0.0005, 0.00025} のうち f_capped 以下の最大値
   ```

   - グリッドは flat の凍結値 0.001 を上限とし**下方にのみ**拡張する（上方拡張禁止 =
     可変化を口実に総リスクを増やさない）。
   - 参考見積り（設計時点。判定には使わない）: flat の worst 月は f=0.001 で 6.4%
     （`flat_fraction_valid_2024.json` の線形スケーリング）。可変系列は倍率上限1.5より
     worst 月 ≤ 9.6% が上界。f_capped = 0.5 × 0.15/(0.096/0.001) ≈ 0.00078 となった場合
     f_var = 0.0005 に落ちる。**f_var = 0.001 で通ることを前提にしない**こと。
   - f_var < 0.001 が採用された場合、§3.3の最低bankrollは `4×100/f_var` に上がる
     （f_var=0.0005 なら 800,000円）。**係数・グリッド・閾値の緩和は禁止**
     （調整は下方のみ）。
4. **最繁忙日エクスポージャ**: `busiest_day_exposure@f_var ≤ 0.5 × max_daily_exposure
   (=0.125)` を確認（`_busiest_day_exposure` の可変stake版: Σstake(その日)/bankroll）。
   不成立なら f_var をグリッド内で1段下げて再確認（下方調整のみ）。
5. 出力 `results/risk_valid.json`: f0系列・f_var系列の月次DD一覧、f_scale・f_capped・
   採用f_var、flat（同一bankroll・base 400円）との月次DD比較、caveats、disclaimer。
6. **f_var を実験config（`config.json`）に凍結追記**してから Stage V2 へ。
   この時点まで TEST(2025+) を一切読まない。

### Stage V2: VALID記述的報告（判定に使わない参考測定）

`run_v2_valid_report.py`: VALID(2024)で可変系列 vs flat系列（同一bankroll）の
ROI・的中率・階層別stake内訳を**記述的に**算出 → `results/valid_report.json`。
ROI差は報告するが、**倍率・f_varの再調整には一切使わない**（凍結済み。使えば§0.5違反）。
結果JSONの `caveats` に §0.1・§0.2 の文言を必ず含める。

### Stage V3: TEST(2025+)確認 — 1回のみ・リスク指標が確認対象

**実行条件**: Stage V0〜V2 完了・§9テスト全パス・f_var凍結コミット・evaluator の
Stage V1/V2 確認済み。揃ってから**1回だけ**実行（バグ修正による再実行はレポートに
理由明記の場合のみ可。結果を見てのパラメータ変更・再実行は禁止）。

`run_v3_test.py` が1回の実行で以下をすべて算出する（事前登録済み出力）:

1. **リスク指標（合否の確認対象はこれのみ）**:
   - worst月次DD（可変系列）≤ monthly_mdd_limit = 0.15
   - 最繁忙日エクスポージャ ≤ max_daily_exposure = 0.25
   - flat系列（同一bankroll）とのDD特性比較（worst月・月次DD分布・最大連敗時損失）
2. **再現性アンカー**: 100円均等flat換算の合算 n_bets・ROI が
   `betting_backtest_oos_flat.json`（n=3,758、ROI_model 83.35%）と一致（±0.1pp / nは完全一致）。
   不一致は選定・結合バグとして修正対象。
3. **記述的報告（合否判定・優位性主張に使わない。事前登録）**: 可変系列ROI・flat系列ROI・
   その差、階層別内訳。**ROI差がいかなる符号・大きさでも「改善」「優位」とは記述しない**。

出力 `results/test_risk.json`。判定は evaluator が行う。

---

### 4.4 DISCLAIMER（本Phase専用の拡張定型文。定数として定義し全出力に埋め込む）

```
本可変サイジングはユーザーの設計選好（自信度に応じたリスク配分）の実装であり、
ROI改善を目的・根拠としない。先行検証（docs/specs/2026-07-11-confidence-tiers-spec.md §15、
verdict=confidence_does_not_predict_market_edge）で自信度は対1番人気ROI優位を予測しないことが
確定しており、自信度加重はモデルの優位性の源泉（過剰人気馬回避）を薄めるリスクが記述的に
示唆されている。本推奨は市場に対する相対的な損失最小化の枠内の配分変更であり、黒字化を
保証するものではない（fold2 OOS実測: ROI 81.89%、元本の約18%の期待損失）
```

埋め込み先: (a) 実験README、(b) 全結果JSONの `disclaimer` キー、(c) 実行ログ（print）、
(d) 結果JSONの `caveats` 配列には加えて §0.1 の要点
（`confidence_does_not_predict_market_edge` への参照、T1:37.4%→T4:74.2% の希釈リスク）を
個別項目として記載する。

---

## 5. リスク上限（既存値。変更・緩和禁止）

| 項目 | 値 | 出典 |
|---|---|---|
| `monthly_mdd_limit` | 0.15 | `betting_config.json`（不変） |
| 安全係数 k | 0.5 | 決定規則v2（`derive_flat_fraction.py`。fを小さくする方向にのみ働く） |
| `max_daily_exposure` | 0.25（headroom検査は0.5倍の0.125） | 同上 |
| f_var グリッド | {0.001, 0.0005, 0.00025}（下方のみ） | 本仕様§4 Stage V1 |
| 月次MDD定義 | 暦月独立P&L ÷ 初期bankroll | `_monthly_max_drawdown`（本番 `risk_limits.py` セマンティクスと同一） |

守れない場合の対処は**係数の下方調整のみ**。閾値・安全係数・グリッド上限の緩和、
月次ウィンドウ定義の変更（暦月→ローリング等）は禁止
（loss-minimization仕様 §1.2 R2 の「採らなかった選択肢」の記録と同趣旨）。

---

## 6. 評価・解釈の事前固定

### 6.1 合否（evaluatorが判定。対象はリスクと規律のみ）

| ゲート | 基準 |
|---|---|
| `budget_preserved` | M̄ ∈ [0.95, 1.05]（Stage V0） |
| `f_var_derived_mechanically` | Stage V1 の決定規則を機械適用した値と config 凍結値が一致 |
| `valid_mdd_ok` | VALID worst月次DD@f_var ≤ 0.5 × 0.15（= f_capped 定義と等価） |
| `valid_exposure_ok` | VALID 最繁忙日エクスポージャ ≤ 0.125 |
| `test_mdd_ok` | TEST worst月次DD ≤ 0.15 |
| `test_exposure_ok` | TEST 最繁忙日エクスポージャ ≤ 0.25 |
| `reproduction_ok` | TEST 合算（100円flat換算）= 83.35% ±0.1pp / n=3,758 |
| `no_performance_claim` | 全出力に「改善・優位・黒字」の主張が存在しない（§9-9） |

### 6.2 ROIの扱い（事前登録）

- VALID・TEST いずれの ROI 差（可変 − flat）も**記述的報告のみ**。正でも負でも
  合否判定・優位性主張・倍率再調整に使わない。
- 可変系列の ROI が flat を下回った場合も本Phaseの失敗ではない（§0.2の前提通り）。
  その事実を結果JSONに `roi_note` として中立に記録する。

### 6.3 本番採用の扱い

全ゲート合格でも**本番 `betting_config.json` への反映は行わない**。可変サイジング設定は
実験ディレクトリ内 `config.json` に並存させ、本番採用の是非は**ユーザー判断に委ねる**
（README・結果JSONに明記）。採用判断の材料は「リスク上限内で運用できるか」のみであり、
「採用するとROIが良くなるか」ではない。

---

## 7. 危険信号（事前登録。confidence-tiers §9 踏襲）

1. **ROI > 100% の系列・階層**: 「黒字」と記述せず、データ結合バグ（race_id正規化・
   払戻重複計上）を疑い即停止 → evaluator へ検証依頼。payout集中度ゲート
   （`top1_payout_share ≤ 0.30` かつ `n_hits ≥ 10`、
   `divergence_lib.py::payout_concentration_gate` パターン）を診断として適用する。
2. **階層別Top-1的中率 > 40%**: 即停止 → evaluator 報告。同一階層内の1番人気的中率と
   比較し、モデル側のみ突出していればリーク疑いとして implementer へ差し戻し
   （confidence-tiers §9-1 と同一手順。手順自体を事前登録とし結果後の基準変更禁止）。
3. **可変系列と flat 系列の n_bets 不一致**: サイジングはstakeのみを変えるため
   ベット集合は完全同一のはず。不一致は選定ロジックの二重実装バグ。
4. **実効倍率の設計乖離**: 実測 stake(t)/base_stake が {0.5, 0.75, 1.25, 1.5} と
   1件でも一致しない → 丸め退化バグ（§3.3違反）として即停止。
5. いずれの報告にも確定/前日水準オッズの限界と §4.4 DISCLAIMER を併記。

---

## 8. 実装構成（隔離実験）

```
betting/experiments/variable_sizing/
├── README.md              # 目的（§0の全文脈）・隔離宣言・否定的結果への参照・実行手順
├── config.json            # 倍率(0.5,0.75,1.25,1.5)・凍結境界(b1,b2,b3)・f_varグリッド・
│                          #   bankroll=400000・保存則tolerance・期間定数・アンカー値。
│                          #   f_var は Stage V1 後に凍結追記。ハードコード禁止の受け皿
├── sizing_lib.py          # 純関数のみ: 倍率適用・保存則検証・可変stake系列構築・
│                          #   月次MDD/エクスポージャ（derive_flat_fraction と同一セマンティクス）。
│                          #   サイジング関数は margin/tier 以外の市場・結果列を引数に取らない
├── build_dataset.py       # scores+features+オッズ → margin・tier付きベット候補parquet（data/）
├── run_v0_occupancy.py    # VALID占有率・保存則検証（outcome-blind）→ results/occupancy_valid.json
├── run_v1_risk_valid.py   # VALID月次MDD・f_var機械導出 → results/risk_valid.json
├── run_v2_valid_report.py # VALID記述的報告 → results/valid_report.json
├── run_v3_test.py         # TEST 1回のみ（リスク指標確認）→ results/test_risk.json
├── data/
├── results/
└── tests/                 # §9 の TDD テスト
```

### 隔離宣言（READMEに明記すること）

本実験は上記ディレクトリに完結する。以下には**書き込まない**:
`betting/config/betting_config.json`（凍結 `stake_fraction=0.001` /
`stake_fraction_frozen=true` を変更しない）、`betting/src/`（`flat_top1.py` /
`derive_flat_fraction.py` は import・参照のみ）、`pure_rank/`（読み取りのみ）、
`prob_fusion/`、`evaluation/reports/gate_summary.json`、
`betting/experiments/confidence_tiers/`（`tiers_lib.py` は import のみ、
results/config への書き込み禁止）。本実験のスクリプトは自ディレクトリの
`data/` と `results/` のみに出力する。本番採用はユーザー判断（§6.3）。

---

## 9. TDDテスト項目（テストファースト。合成データのみで走ること）

`betting/experiments/variable_sizing/tests/` に実装:

1. **有界性**: 任意のmargin入力（0、負値ガード、極大値含む）で
   0.5 ≤ 実効倍率 ≤ 1.5。倍率配列が config の {0.5, 0.75, 1.25, 1.5} と一致し、
   狭義単調増加であること。
2. **階層割当（凍結境界・境界値ちょうど）**: 凍結境界 [b1, b2, b3] に対し
   margin = b1 → T1（`side="left"`: 境界値ちょうどは下位階層。confidence-tiers §13 と
   同一）、b1+ε → T2、b3 → T3、b3+ε → T4。`tiers_lib.assign_tier` の import 再利用で
   あること（同等ロジックの再実装 `def assign_tier` が実験ディレクトリに存在しない
   ことの静的検査）。
3. **リスク予算保存則**: 合成占有率 w = (0.25,0.25,0.25,0.25) で M̄ = 1.0 ちょうど。
   偏った占有率で M̄ が正しく計算され、[0.95, 1.05] 外で `budget_preserved=false`。
   占有率計算関数が着順・払戻列を受け取らないシグネチャであること（outcome-blindの
   構造的担保）。
4. **100円丸め**: bankroll=400,000・f=0.001 で stake = {200, 300, 500, 600} 円ちょうど
   （全て100円の倍数、丸め誤差ゼロ）。base_stake が400円の倍数にならない bankroll
   （例: 100,000 や 350,000）で ValueError。f_var=0.0005 なら最低 bankroll 800,000 で
   stake = {200, 300, 500, 600}。
5. **実効倍率の一致**: 各階層の stake/base_stake が設計倍率と厳密一致（§7-4の検出器）。
6. **市場情報混入の静的検査**: `sizing_lib.py` のサイジング関数群（倍率決定・stake計算・
   占有率・保存則）のソースに `odds` / `popularity` / `ninki` / `market_log_odds` /
   `init_score` / `win_prob` / `payout` / `finish_rank` が現れないこと
   （決済・MDD計算関数は payout/odds を扱うため検査対象から除外し、除外関数名を
   テスト内に明示列挙。禁止トークンは文字列結合でエンコードし自己ヒット回避 —
   confidence-tiers §11-10 パターン）。
7. **flat比較の決済一致性**: 同一合成ベット集合で可変系列とflat系列の n_bets・
   的中フラグ・レース集合が完全一致し、差は stake 列のみであること。
8. **月次MDDセマンティクス**: 合成P&L系列で暦月独立・対初期bankroll比の worst 月が
   手計算と一致（`derive_flat_fraction._monthly_max_drawdown` と同値であることの
   照合テスト）。f に対する線形性（f を2倍にすると月次DDが厳密に2倍）。
9. **f_var機械導出**: 合成 worst_month_dd から f_scale・f_capped・グリッド採用値が
   決定規則通りに出ること。f_capped 未満の候補が無い場合に None（＝planner差し戻し）
   となり、勝手にグリッドを上方拡張しないこと。
10. **性能主張の不在**: 全結果JSON生成関数の出力に `disclaimer` キー（§4.4定型文）と
    `caveats`（`confidence_does_not_predict_market_edge` への参照を含む）が存在すること。
    「黒字」「改善」「優位」等の禁止語がJSON生成コードの文字列リテラルに現れないこと
    （記述的 `roi_note` は中立文言テンプレートで固定）。
11. **危険信号フラグ**: ROI=1.01 注入 → `danger_roi_gt_100=true`、階層Top-1=0.41 注入 →
    `leak_review_required=true`（`tiers_lib` の既存フラグ関数を import 再利用）。
12. **再現性**: 全テストが seed 固定（42）で決定的に通ること。ブートストラップ等の
    乱数処理は本Phaseに無いが、pandas ソートは `kind="mergesort"`（安定）で決定的で
    あること。

---

## 10. implementer への引き渡し事項（順序付きタスクリスト）

1. `betting/experiments/variable_sizing/` を作成し、README（§0の前提全文・隔離宣言・
   §4.4 DISCLAIMER・実行手順）と `config.json`（§3・§5の全定数集約。ハードコード禁止）
   を書く。
2. **テストを先に書く**（§9の1〜12。
   `python -m pytest betting/experiments/variable_sizing/tests/ -v`）。
3. `sizing_lib.py`（純関数）を実装しテストを通す。`flat_top1` / `tiers_lib` は
   import 再利用（コピー禁止）。
4. `build_dataset.py` → `run_v0_occupancy.py` 実行。保存則不成立なら停止・planner差し戻し。
5. `run_v1_risk_valid.py` 実行 → f_var を config に凍結追記 → orchestrator/planner に報告
   （本仕様書 §12 への追記を待つ。**TEST には触れない**）。
6. `run_v2_valid_report.py` 実行（記述的報告のみ。倍率・f_var は変更しない）。
7. 市場情報混入チェックと隔離確認をコマンドラインでも実行しログを残す:
   ```bash
   grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
     pure_rank/src/ --include="*.py"   # → 増分ゼロであること
   git status --short                   # → 変更が betting/experiments/variable_sizing/ と
                                        #    docs/specs/ に限られること
   git diff betting/config/betting_config.json  # → 差分ゼロであること
   ```
8. evaluator へ引き渡し（Stage V0〜V2 の独立検証）。
9. evaluator 承認後のみ `run_v3_test.py` を実装・実行（**TEST 1回のみ。再実行禁止**。
   バグ修正による再実行はレポートに理由明記の場合のみ可）。
10. 全結果を evaluator の最終判定（§11）に回す。`gate_summary.json` への反映は
    判定後の別タスク（本実験からは書き込まない）。

---

## 11. 評価基準（evaluator 向けサマリ）

以下を独立に検証する:

1. **性能改善を主張していないか**: 全出力（README・JSON・ログ・コミットメッセージ）に
   「黒字」「改善」「優位」の主張が無いこと。§4.4 DISCLAIMER が全出力に存在すること。
2. **否定的結果を正しく参照しているか**: `caveats` に
   `confidence_does_not_predict_market_edge`（confidence-tiers §15）と希釈リスク
   （T1:37.4%→T4:74.2%）への参照が含まれること。
3. **リスク上限遵守**: §6.1 の全ゲート再計算。f_var が決定規則の機械適用値と一致し、
   閾値・安全係数・グリッドの緩和が無いこと。
4. **Rule 3 遵守**: 倍率・境界・f_var がすべて VALID 凍結後に TEST を1回だけ読んだこと
   （ファイル更新時刻・コミット履歴で確認。loss-min仕様 §7.5 の検査手法を踏襲）。
   倍率・f_var の選択に VALID/TEST の着順・ROI が使われていないこと
   （outcome-blind 関数シグネチャ＋コードレビュー）。
5. **本番資産非接触**: `betting_config.json`・`betting/src/`・`pure_rank/`・
   `prob_fusion/`・`gate_summary.json`・`confidence_tiers/results/` に変更差分ゼロ。
6. **再現性アンカー**: TEST 合算（100円flat換算）= n 3,758 / ROI 83.35% ±0.1pp。
7. 危険信号（§7）該当時は合格ではなく停止・検証。

---

## 12. Stage V0/V1 確定結果（2026-07-11 追記。追記後は変更禁止）

出典: `betting/experiments/variable_sizing/results/occupancy_valid.json` / `results/risk_valid.json`

- **Stage V0（占有率・保存則、outcome-blind）**: VALID(2024) n_bets = 2,608
  （再現性アンカーと一致）、階層占有 {T1: 666, T2: 677, T3: 641, T4: 624}、
  **M̄ = 0.9885** ∈ [0.95, 1.05] → `budget_preserved = true`。
- **Stage V1（リスク導出、決定規則v2可変版の機械適用）**:
  - worst_month_dd_var@f0(0.001) = **0.08005**（2024-10。同一bankroll flat は 0.0798/2024-02）
  - f_scale = 0.15 / (0.08005/0.001) = **0.0018738**、f_capped = 0.5 × f_scale = **0.00093691**
  - **採用 f_var = 0.0005**（グリッド {0.001, 0.0005, 0.00025} 中 f_capped 以下の最大値。
    下方調整のみ・緩和なし）
  - 最低運用bankroll = 4 × 100 / 0.0005 = **800,000円**、base_stake = 400円、
    stake = {200, 300, 500, 600}円
  - busiest-day exposure @f_var = **0.01638** ≤ 0.125（1段目候補で成立、step-down不要）
  - f_var を `betting/experiments/variable_sizing/config.json` に凍結追記済み。
    この時点まで TEST(2025+) 非接触。

## 13. Stage V2 確定結果（2026-07-11 追記。追記後は変更禁止）

出典: `betting/experiments/variable_sizing/results/valid_report.json`

- VALID(2024) 記述的報告（判定には使わない）: 可変系列 n_bets=2,608, ROI=79.158%,
  的中率=24.233%。flat比較系列（同一bankroll・base_stake=400円固定）n_bets=2,608,
  ROI=79.099%。ROI差 = **+0.059pp**（符号・大きさとも判定に使用しない）。
- 階層別内訳: T1 n=666 ROI=82.85%、T2 n=677 ROI=74.05%、T3 n=641 ROI=78.55%、
  T4 n=624 ROI=81.14%。
- `n_bets_match=true`, `race_set_match=true`, `win_flags_match=true`,
  `effective_multiplier_matches_design=true`, `danger_signals.any_danger=false`。

## 14. Stage V3 確定結果（2026-07-11 追記。1回のみ実行・追記後は変更禁止）

出典: `betting/experiments/variable_sizing/results/test_risk.json`
（`run_v3_test.py` を1回実行。単一実行ガードにより再実行不可）

- **リスク指標（唯一の合否対象）**:
  - worst月次DD（可変系列, TEST）= **0.04235** ≤ monthly_mdd_limit(0.15) → **PASS**
  - 最繁忙日エクスポージャ（可変系列, TEST）= **0.0175** ≤ max_daily_exposure(0.25) → **PASS**
  - flat比較系列（同一bankroll）とのDD特性: 同水準（可変系列がわずかに高いが両者とも
    上限を大きく下回る）
  - `risk_gates_pass = true`
- **再現性アンカー**: 100円均等flat換算 n_bets=3,758・ROI=83.34752527940394% が
  `evaluation/reports/betting_backtest_oos_flat.json`（n=3,758、ROI=83.34752527940394%）
  と完全一致（n完全一致・ROI差0.0pp ≤ 許容0.1pp）→ `reproduction_ok = true`
- **整合性チェック**: `n_bets_match=true`, `race_set_match=true`, `win_flags_match=true`,
  `effective_multiplier_matches_design=true`, `danger_signals.any_danger=false`
- **記述的報告（判定・優位性主張に使わない。事前登録）**: 可変系列 n_bets=3,758,
  ROI=81.205%, 的中率=24.987%。flat比較系列（同一bankroll）ROI=83.348%。
  ROI差 = **-2.142pp**（可変系列がflat比較系列を下回るが、この符号・大きさは
  性能の優劣を意味しない。§6.2の通り事前登録された参考測定に過ぎない）。
  階層別内訳: T1 n=869 ROI=91.48%、T2 n=938 ROI=86.29%、T3 n=1010 ROI=79.49%、
  T4 n=941 ROI=77.05%。
- **結論**: リスク上限は全て遵守され、再現性アンカーも完全一致した。
  confidence-tiers §15（`confidence_does_not_predict_market_edge`）の否定的結果と
  整合的に、可変サイジングはTESTでもROI優位を示さない（記述的にはむしろ
  flat比較系列をわずかに下回る）。本Stageはユーザーの設計選好としてのリスク配分
  変更が既存のリスク上限内に収まることを確認するものであり、性能改善を主張しない。

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-07-11 | 初版。否定的結果（confidence-tiers §15）確定後のユーザー設計選好としての可変サイジングを事前登録: 階層別固定倍率 (0.5, 0.75, 1.25, 1.5)・凍結境界再利用・リスク予算保存則（占有率加重平均 ∈ [0.95, 1.05]）・400円倍数base stakeによる丸め厳密化・f_var の決定規則v2型機械導出（下方のみ）・TESTはリスク指標のみ1回確認・本番採用はユーザー判断 |
| 2026-07-11 | Stage V0〜V3 実行完了。V1: f_var=0.0005（bankroll 800,000円）を機械導出・凍結。V2: VALID記述的ROI差+0.059pp（判定に不使用）。V3: リスク上限を全て遵守（worst月次DD=0.04235≤0.15、最繁忙日エクスポージャ=0.0175≤0.25）、再現性アンカー完全一致、記述的ROI差-2.142pp（可変系列がflat比較系列をわずかに下回る。判定に不使用）。§12〜§14に確定結果を追記。 |

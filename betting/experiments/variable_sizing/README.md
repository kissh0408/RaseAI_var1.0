# variable_sizing — margin対応可変サイジング（P6）

**仕様書**: `docs/specs/2026-07-11-variable-sizing-spec.md`
**先例**: `docs/specs/2026-07-11-confidence-tiers-spec.md`（margin定義・凍結境界・隔離パターン）、
`docs/specs/2026-07-10-loss-minimization-implementation-spec.md`（flat top-1・決定規則v2・DISCLAIMER）

---

## 0. 決定的な前提（本Phaseの位置づけ。必読）

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
この事実を隠さず、仕様書・本README・結果JSONの `caveats` 配列・DISCLAIMER の
**4箇所すべて**に埋め込む（§4.4）。

### 0.3 黒字化表現の全面禁止

「黒字化」を主張・示唆する表現をコード・ログ・JSON・レポート・コミットメッセージに
一切書かない。`betting/src/flat_top1.py::DISCLAIMER` を踏襲し、本Phase専用に拡張した
定型文（§4.4、`sizing_lib.py::DISCLAIMER`）を全出力に埋め込む。

### 0.4 期待値は負・ケリー不採用

- 期待値は**負**（fold2 OOS実測: ROI 81.89%、元本の約18%の期待損失）。
- ケリー基準は**不採用**（負のEVではケリーの数学的前提が成立せず、最適解は
  「賭けない」）。本Phaseで実装するのは有界な階層別配分であり、比例ケリーとは無関係。

### 0.5 倍率のROI最適化は無意味かつ禁止

§0.1より margin は対市場優位を予測しないことが確定しているため、倍率をROI（または Δ）で
最適化して選ぶことは統計的に無意味であり、後出しじゃんけん（Rule 3違反）にもなる。
**倍率は「リスク予算保存＋単調性＋有界性」という設計制約のみから決める**。
VALID(2024)の着順・払戻データを使ってよいのは**リスク検証（月次MDD・エクスポージャ測定）
のみ**であり、倍率・階層境界・関数形の選択には一切使わない（outcome-blind設計）。

---

## 4.4 DISCLAIMER（本Phase専用の拡張定型文。全出力に埋め込む）

```
本可変サイジングはユーザーの設計選好（自信度に応じたリスク配分）の実装であり、
ROI改善を目的・根拠としない。先行検証（docs/specs/2026-07-11-confidence-tiers-spec.md §15、
verdict=confidence_does_not_predict_market_edge）で自信度は対1番人気ROI優位を予測しないことが
確定しており、自信度加重はモデルの優位性の源泉（過剰人気馬回避）を薄めるリスクが記述的に
示唆されている。本推奨は市場に対する相対的な損失最小化の枠内の配分変更であり、黒字化を
保証するものではない（fold2 OOS実測: ROI 81.89%、元本の約18%の期待損失）
```

定数: `sizing_lib.py::DISCLAIMER`。埋め込み先: (a) 本README、(b) 全結果JSONの
`disclaimer` キー、(c) 実行ログ（print）、(d) 結果JSONの `caveats` 配列
（`confidence_does_not_predict_market_edge` への参照 + 希釈リスク T1:37.4%→T4:74.2%）。

---

## 隔離宣言（本番非接触）

本実験は `betting/experiments/variable_sizing/` 配下に完結する。以下には**書き込まない**:

- `betting/config/betting_config.json`（凍結 `stake_fraction=0.001` /
  `stake_fraction_frozen=true` を変更しない）
- `betting/src/`（`flat_top1.py` / `derive_flat_fraction.py` は import・参照のみ）
- `pure_rank/`（読み取りのみ）
- `prob_fusion/`
- `evaluation/reports/gate_summary.json`
- `betting/experiments/confidence_tiers/`（`tiers_lib.py` は import のみ、
  results/config への書き込み禁止）

本実験のスクリプトは自ディレクトリの `data/` と `results/` のみに出力する。
本番採用はユーザー判断（仕様書§6.3）。

## 市場情報境界

- **L1（`pure_rank/src/`）に一切変更を加えない**。既存 fold2 OOS スコア
  （`pure_score_z`）とオッズ（L3許可範囲: 除外条件・決済のみ）を読むだけである。
- **サイジング関数はmarginのみの関数**とする。stake の決定にオッズ・人気・
  `market_log_odds`・`init_score`・融合確率 `win_prob_est` を一切使わない
  （`tests/test_static_guards.py` で機械的に担保。オッズの使用は `select_top1_bets`
  の min/max 除外と `settle_win_bets` の決済の2箇所のみ = 既存 flat 運用と同一）。
- z の二重使用なし（L2統合は本Phaseに登場しない）。
- 本番凍結値 `betting/config/betting_config.json`（`stake_fraction=0.001`,
  `stake_fraction_frozen=true`）を**変更しない**。
- ROI は記述的報告のみ。いかなる系列のROIが100%を超えても「黒字」と記述せず、
  危険信号として扱う。

## 可変サイジング関数の設計（事前登録。後出し変更禁止）

```
stake(t) = base_stake × m_t
base_stake = 初期bankroll × f_var        （f_varはStage V1で機械的に導出・凍結）
m = (m1, m2, m3, m4) = (0.5, 0.75, 1.25, 1.5)   ← 事前登録・固定
```

- 有界性: min倍率=0.5、max倍率=1.5（max/min比=3.0）。
- 単調性: m1<m2<m3<m4。
- 倍率値は 1 を中心に等間隔対称 `m_t = 1 + d×(2t−5)/2`（d=0.25）という**設計制約のみ**
  から定めた。着順・払戻・ROI・Δのいかなる実データも参照していない。
- 階層境界（凍結値の再利用。新たな境界導出は行わない）: confidence-tiers §13 の
  凍結境界をそのまま使う: `b1=0.11895233392715454`, `b2=0.28122442960739136`,
  `b3=0.5161097198724747`。割当: `tier = 1 + searchsorted([b1,b2,b3], margin, side="left")`
  （境界値ちょうどは下位階層）。

### リスク予算保存則

占有率加重平均倍率 `M̄ = Σ w_t × m_t` が `0.95 ≤ M̄ ≤ 1.05` を満たすこと（Stage V0で検証。
不成立なら倍率を調整せず planner へ差し戻す）。

### 100円丸め問題への対処

`base_stake` は必ず400円（stake_rounding_yenの4倍）の倍数でなければならない
（0.75×400=300円のように丸め誤差ゼロで100円単位に乗るため）。base_stakeが400円の
倍数にならない bankroll は ValueError で拒否する（`sizing_lib.compute_base_stake`）。

## Rule 3（期間規律）

| 段階 | 期間 | 用途 |
|---|---|---|
| Stage V0 | 全期間（outcome-blind） | 占有率・保存則検証 |
| Stage V1 | VALID=2024年のみ | 月次MDD実測・f_var機械導出・config凍結追記 |
| Stage V2 | VALID=2024年のみ | 記述的報告（判定に使わない） |
| Stage V3（本実装では未実施） | TEST=2025年以降 | evaluator承認後に1回のみ |

`build_dataset.py` は全期間を対象にビルドしてよい（着順・payout列を含む）が、
Stage V0/V1/V2 の各スクリプトは **io直後に期間フィルタ**を適用し、TEST(2025+)の行には
一切触れない。**Stage V3（TEST）は本タスクでは実行しない**（evaluator承認後の別タスク）。

## 実行手順

```bash
# 0. テスト（TDD。合成データのみ、実データ不要）
python -m pytest betting/experiments/variable_sizing/tests/ -v

# 1. データセット構築（全期間。scores+features+odds+margin+tier付与）
python betting/experiments/variable_sizing/build_dataset.py

# 2. Stage V0: 占有率・保存則検証（outcome-blind）
python betting/experiments/variable_sizing/run_v0_occupancy.py

# --- 保存則不成立ならここで停止し planner へ差し戻し ---

# 3. Stage V1: VALID月次MDD実測・f_var機械導出 → config.json に凍結追記
python betting/experiments/variable_sizing/run_v1_risk_valid.py

# 4. Stage V2: VALID記述的報告（倍率・f_varは変更しない）
python betting/experiments/variable_sizing/run_v2_valid_report.py

# --- ここまでで evaluator への引き渡し。Stage V3 は evaluator 承認後の別タスク ---
```

## ディレクトリ構成

```
betting/experiments/variable_sizing/
├── README.md
├── config.json            # 倍率・凍結境界・f_varグリッド・bankroll・保存則tolerance等
├── sizing_lib.py           # 純関数のみ: 倍率適用・保存則検証・base_stake検証・
│                          #   月次MDD/エクスポージャ（derive_flat_fraction を import）
├── build_dataset.py        # scores+features+オッズ → margin・tier付きベット候補parquet
├── run_v0_occupancy.py     # VALID占有率・保存則検証（outcome-blind）
├── run_v1_risk_valid.py    # VALID月次MDD・f_var機械導出 → config凍結追記
├── run_v2_valid_report.py  # VALID記述的報告
├── data/
├── results/
└── tests/                  # TDDテスト（合成データのみ）
```

## 既存モジュールの import 再利用（コピー禁止）

- ベット選定・決済: `betting/src/flat_top1.py::select_top1_bets` / `settle_win_bets` / `DISCLAIMER`
- スコア+特徴量+オッズ結合: `betting/src/backtest.py::load_scored_odds_frame`
- margin計算・階層割当: `betting/experiments/confidence_tiers/tiers_lib.py::compute_race_margin` /
  `assign_tier` / `assign_tier_batch`
- 危険信号フラグ: `betting/experiments/confidence_tiers/tiers_lib.py::leak_review_flag` /
  `danger_roi_gt_100`
- 月次MDD・最繁忙日エクスポージャ: `betting/src/derive_flat_fraction.py::_monthly_max_drawdown` /
  `_busiest_day_exposure`

## 判定基準（要約。詳細は仕様書§6, §11）

判定者は evaluator。本実験スクリプトは `results/` に測定値を出力するのみ。危険信号
（ROI>100%系列、階層別Top-1的中率>40%、可変系列とflat系列のn_bets不一致、実効倍率の
設計乖離）該当時は合格ではなく即時停止・検証。「黒字化」表現は禁止。

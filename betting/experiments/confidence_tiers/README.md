# confidence_tiers — 自信度階層別の対1番人気ROI優位検定（P5）

**目的**: flat top-1 戦略（モデル予測1位への定額単勝ベット）が示した対1番人気ROI優位
（TEST(2025+) で +4.99pp、`evaluation/reports/betting_backtest_oos_flat.json`）が、
モデルの自信度（margin = レース内1位-2位スコア差）が高い階層ほど統計的に大きいかを
事前登録・OOS規律で検証する。ユーザーの可変サイジング提案の前提仮説検証であり、
**可変サイジングの実装は行わない**。

**仕様書**: `docs/specs/2026-07-11-confidence-tiers-spec.md`

> **本 Phase の限界（必読）**: 全体の期待値は負（fold2 OOS実測で元本の約18%の期待損失）
> であり、測っているのは「自信度と損失の相対差」の関連である。いかなる結果も
> 「黒字化」を意味しない（`betting/src/flat_top1.py::DISCLAIMER` を全結果JSONの
> `disclaimer` キーに含める）。

## 隔離宣言（本番非接触）

本実験は `betting/experiments/confidence_tiers/` 配下に完結する。以下には**書き込まない**:
`betting/config/betting_config.json`（凍結 `stake_fraction=0.001` を変更しない）、
`pure_rank/models/`、`pure_rank/data/`（読み取りのみ）、`prob_fusion/`、
`betting/src/`（`flat_top1.py`・`backtest.py` は import のみ）、
`evaluation/reports/gate_summary.json`、`evaluation/reports/betting_backtest_oos_flat.json`。
`gate_summary.json` への結果反映は **evaluator の判定後に別タスクとして**
`evaluation/update_gate_summary.py` 経由で行う。本実験のスクリプトは
`data/` と `results/` のみに出力する。

## 市場情報境界

- **L1（`pure_rank/src/`）に一切変更を加えない**。既存 flat top-1 戦略の出力
  （fold2 OOS スコア由来の `pure_score_z`）とオッズを階層化するだけである。
- **自信度指標（margin）はオッズ・人気・市場由来の列を一切使わない**
  （`pure_score_z` のみ。`tests/test_static_guards.py` で機械的に担保）。
- オッズの使用箇所は (i) `select_top1_bets` の min/max 除外、(ii) 決済
  （payout = stake×odds）、(iii) 1番人気ベースラインの特定、の3箇所のみ
  （すべて L3 許可範囲）。
- z の二重使用なし（L2 統合は本 Phase に登場しない）。
- 検定対象は margin の単一指標のみ（win_prob_est は市場確率の単調変換に一致するため
  不採用。仕様書§3.2）。

## 自信度指標・階層設計（事前登録。後出し変更禁止）

- margin(r) = pure_score_z(レースr内1位馬) − pure_score_z(レースr内2位馬)。
  1位・2位の特定は `select_top1_bets` と同一の決定的タイブレーク
  （スコア降順→同値は馬番昇順）。
- K=4（四分位）。境界 [b1,b2,b3] は **2023年の「実際にベット対象となったレース」の
  margin 分布のみ**（outcome-blind）で決定し、Stage 1 実行後は変更禁止。
- 割当規則: `tier = 1 + searchsorted([b1,b2,b3], margin)`（境界値ちょうどは下位階層）。

## Rule 3（期間規律）

| 段階 | 期間 | 用途 |
|---|---|---|
| Stage 1（境界決定） | 2023年 | margin四分位の算出のみ（着順・ROIは見ない） |
| Stage 2（一次判定） | VALID=2024年 | 階層別Δの測定・検定・一次判定 |
| Stage 3（二次判定） | TEST=2025年以降 | 事前登録条件成立時のみ1回 |

- `build_dataset.py` は全期間を対象にビルドする（scores/features/oddsの結合・
  margin付与・select_top1_bets適用まで）。
- Stage 1/2/3 の各スクリプトは **io直後に自身の対象期間でフィルタ**し、他期間の行
  には一切触れない（多重ガード）。
- Stage 1 は margin の四分位のみを算出し、**着順・払戻・ROI・的中率は一切計算・
  出力しない**（`run_stage1_boundaries.py` が読み込む列は race_id/race_date/margin
  のみ）。

## 実行手順

```bash
# 0. テスト（TDD。合成データのみ、実データ不要）
python -m pytest betting/experiments/confidence_tiers/tests/ -v

# 1. データセット構築（全期間。scores+features+odds+margin+select_top1_bets適用）
python betting/experiments/confidence_tiers/build_dataset.py

# 2. Stage 1: 2023年margin四分位境界の決定（outcome-blind）
python betting/experiments/confidence_tiers/run_stage1_boundaries.py

# --- ここで境界を docs/specs/2026-07-11-confidence-tiers-spec.md §13 に追記し、
#     orchestrator/planner の確認を得てから Stage 2 に進む ---

# 3.（Stage 2 一次判定。境界確定後に別途実装・実行）
# python betting/experiments/confidence_tiers/run_stage2_valid.py

# 4.（Stage 3 二次判定。一次通過+evaluator承認後、1回のみ）
# python betting/experiments/confidence_tiers/run_stage3_test.py
```

## ディレクトリ構成

```
betting/experiments/confidence_tiers/
├── README.md
├── config.json               # K=4・n_min=200・B=10000・seed=42・Bonferroni 0.01/5、
│                             #   min_odds/max_odds、期間定数、境界値（Stage1後に凍結追記）
├── tiers_lib.py               # 純関数のみ: margin計算・階層割当・四分位境界・Δ/ペアド
│                             #   ブートストラップ・順序コントラスト・最小サンプル判定・
│                             #   Bonferroni・危険信号フラグ
├── build_dataset.py           # scores + features + オッズ → ベット候補parquet（margin付き）
├── run_stage1_boundaries.py   # 2023年margin四分位 → results/stage1_boundaries.json
├── run_stage2_valid.py        # （未実装）VALID 2024 測定・検定・一次判定材料
├── run_stage3_test.py         # （未実装）一次通過+承認時のみ TEST 1回
├── data/                      # bets_dataset.parquet, build_log.json
├── results/                   # stage1_boundaries.json 等
└── tests/                     # TDDテスト（合成データのみ）
```

## 既存モジュールの import 再利用（コピー禁止）

- スコア+特徴量+オッズ結合: `betting/src/backtest.py::load_scored_odds_frame`
- モデル1位馬選定・オッズ除外: `betting/src/flat_top1.py::select_top1_bets`
  （`tests/test_reuse_guard.py` で再実装が無いことを機械的に担保）
- payout集中度ゲート: `betting/experiments/cross_pool_divergence/divergence_lib.py::payout_concentration_gate`
- disclaimer: `betting/src/flat_top1.py::DISCLAIMER`

## 判定基準（要約。詳細は仕様書§6, §7, §14）

- 一次（VALID 2024のみ）: H4（Δ(T4)>0）または H_ord（C=Δ(T4)-Δ(T1)>0）の少なくとも
  一方が片側ブートストラップ p<0.002（0.01/5）。各階層n≥200。再現性アンカー成立。
- 二次（TEST、1回のみ・一次通過時のみ）: Δ(T4)>0 かつ C>0（点推定）、再現性アンカー成立。
  TESTでは有意性を要求しない。
- 判定者は evaluator。本実験スクリプトは `results/` に測定値を出力するのみ。
- 危険信号（仕様書§9）該当時は合格ではなく即時停止・検証。「黒字化」表現は禁止。

# cross_pool_divergence — 券種間整合性の系統的乖離測定（P4）

**目的**: 単勝オッズから Stern（複勝・ワイド）/ Harville（馬連）で逆算した理論確率と、
複勝・ワイド・馬連の確定払戻が示す実効確率との**系統的乖離**（特定の人気帯 × 頭数帯で
常に過小/過大評価される構造）を fit 期間（2023-01-01〜2024-12-31）で測定する。

**本 Phase は L1 を一切使わない「市場 vs 市場」の検証である。** 理論確率の入力は
市場確率 q（単勝オッズ由来）のみ。`pure_score_z`・L1 特徴量・L1 スコアは不使用。

**仕様書**: `docs/specs/2026-07-10-p4-cross-pool-divergence-spec.md`

> **成功確率に関する注記**: P4 は事前評価で成功確率が最も低いとされている（「控除率の壁が
> 厚い」）。本 Phase の価値は「乖離が存在しない/控除率で説明され尽くす」ことを定量的に
> 確定させて撤退判断の材料にすることを含む。数値目標は設定しない。

## 隔離宣言（本番非接触）

本実験は `betting/experiments/cross_pool_divergence/` 配下に完結する。以下には**書き込まない**:
`evaluation/reports/gate_summary.json`、`betting/config/`、`prob_fusion/data/`、
`pure_rank/models/`、`pure_rank/data/`（読み取りは `01_preprocessed` のみ）、
`prob_fusion/src/`、`betting/src/`。
`gate_summary.json` への結果反映は **evaluator の判定後に別タスクとして**
`evaluation/update_gate_summary.py` 経由で行う。本実験のスクリプトは
`data/` と `results/` のみに出力する。

## 市場情報境界・L1 不使用宣言

- 本実験は L1 の産物（`pure_score_z`、`scores_*.parquet`、`features_*.parquet`）を
  **一切読み込まない**。ベースデータは `pure_rank/data/01_preprocessed/`（L0 前処理層）の
  SE / RA parquet と、`common/data/output/odds/`・`common/data/output/race_hr/` の
  確定オッズ・確定払戻のみ。
- オッズ・払戻の使用は本 Phase の目的そのもの（市場 vs 市場）であり、ベッティング
  レイヤー（L3 相当）の隔離実験として正当。**L1（`pure_rank/src/`）へは何も還流しない**。
- z の使用ゼロ（α·z すら登場しない）。q は理論確率算出の入力としてのみ使用。
- L1 不使用は `tests/test_static_guards.py` の静的検査（禁止トークン grep・import 検査）
  で機械的に担保する。

検証コマンド:

```bash
grep -rn "pure_score\|scores_v39\|03_scores\|features_v39\|02_features" \
  betting/experiments/cross_pool_divergence/ --include="*.py"
# → テスト内のエンコード済み定数（文字列結合）以外は 0 件であること
```

## Rule 3（期間規律）

- `build_dataset.py` は `config.json` の `protocol.fit_start`〜`protocol.fit_end`
  （2023-01-01〜2024-12-31）のみをビルド対象とする（TEST(2025+) 行は生成しない）。
- Stage 1 / Stage 2 のスクリプトは io 直後に `race_date <= 2024-12-31` フィルタを
  さらに適用し、TEST 行を一切読まない（build_dataset 側の制限と多重にガード）。
- Stage 3（TEST 各 1 回）は一次通過セグメントが存在し、evaluator が承認した場合のみ
  実装・実行する。本実験は現時点では Stage 3 を実装していない
  （build_dataset の対象期間拡張が前提）。

## セグメント定義（K_max=30、確定後変更禁止）

- 複勝: 人気帯 POP1〜POP4（4）× 頭数帯 FS_S/FS_M/FS_L（3）= 12
- ワイド: ペア人気帯 PAIR_TOP/MIX/LONG（3）× 頭数帯（3）= 9
- 馬連: 同上 = 9
- 合計 K_max = 30。Stage 1 の最小サンプル基準（n_units≥300 かつ Σp_theo≥30）を
  満たしたセグメントのみ確定 K として Stage 2 で検定する。

## 実行手順

```bash
# 0. テスト（TDD。合成データのみ、実データ不要）
python -m pytest betting/experiments/cross_pool_divergence/tests/ -v

# 1. データセット構築（fit期間のみ。SE+RA+オッズ+払戻 → 券種別ユニットparquet）
python betting/experiments/cross_pool_divergence/build_dataset.py

# 2. Stage 1: セグメント別 n_units・Σp_theo 集計、K 確定（TEST 非接触）
python betting/experiments/cross_pool_divergence/run_stage1_counts.py

# 3. Stage 2: 確定セグメント別 D_cal/D_adj/ROI_flat/クラスタブートストラップp値
python betting/experiments/cross_pool_divergence/run_stage2_divergence.py

# 4.（一次通過セグメントが存在し evaluator が承認した場合のみ、TEST 各1回・未実装）
# python betting/experiments/cross_pool_divergence/run_stage3_test.py
```

## ディレクトリ構成

```
betting/experiments/cross_pool_divergence/
├── README.md
├── config.json               # セグメント定義・期間・λ参照元・閾値（ハードコード禁止の受け皿）
├── divergence_lib.py         # 純関数のみ: 人気順位付与・帯割当・p_pool/OR_r・D_cal/D_adj・
│                             #   クラスタブートストラップ・除外判定・Bonferroni閾値
├── build_dataset.py          # SE+RA+オッズ+払戻 → 券種別ユニットデータセット parquet
├── run_stage1_counts.py      # fit期間セグメント別 n・Σp_theo 集計 → K 確定
├── run_stage2_divergence.py  # fit期間 D_cal/D_adj/ROI_flat/ブートストラップp → 一次判定材料
├── data/                     # units_place.parquet / units_wide.parquet / units_quinella.parquet
├── results/                  # stage1_counts.json / divergence_fit.json
└── tests/                    # TDD テスト（合成データのみ）
```

## 既存モジュールの import 再利用（コピー禁止）

- 複勝確率: `prob_fusion/src/place_prob.py::stern_place_probs`
- ワイド確率: `betting/src/pair_probs.py::stern_wide_pair_prob`
- 馬連確率: `betting/src/ev_filters.py::harville_quinella_pair_prob`
- ペアオッズ・オーバーラウンド: `betting/src/wide_ev_core.py::load_wide_odds_lookup`,
  `get_pair_odds`, `compute_race_overround`, `norm_pair`
- オッズ付与: `evaluation/odds_loader.py::attach_odds_from_se_parquet`
- 市場確率 q: `prob_fusion/src/market_prob.py::attach_market_q`（method="proportional"）
- 複勝確定払戻: `evaluation/place_payout_loader.py::build_place_payout_lookup`,
  `attach_place_payout`
- fusion formal パラメータ（λ2, λ3）: `evaluation/reports/fusion_oos_fold2.json`

## 判定基準（要約。詳細は仕様書 §3, §5, §13）

- 一次判定: D_adj(s) ≥ +0.03（+3pp）**かつ** レース単位クラスタ・ブートストラップ
  両側 p < 0.01/K（Bonferroni）**かつ** 2023 年・2024 年の各年で D_adj > 0（符号一貫）。
- 打ち切り: 全確定セグメントで D_adj < +0.03 →
  `verdict = "cross_pool_divergence_within_takeout_wall"`、P4 終了（延長禁止）。
- 判定者は evaluator。本実験スクリプトは `results/` に測定値を出力するのみ。
- **限界**: 確定払戻ベースである（購入時点のオッズではない）。乖離が見えても購入時点で
  消えている可能性は残る。本測定は上限診断。

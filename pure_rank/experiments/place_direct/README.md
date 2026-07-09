# place_direct — 複勝(top3)確率 直接予測実験

**目的**: 複勝（3着以内）確率を LightGBM **binary 分類で直接予測**し、既存の
「単勝確率 p_win → Harville / Stern 式で逆算した複勝確率」と
logloss・較正誤差（最大乖離pp）・Brier score で比較する。

**答える問い**: 「複勝的中を最もよく予測するのは、直接 binary 予測か、
Stern/Harville 逆算か」。市場超えの判定ではない。

**仕様書**: `docs/specs/2026-07-09-place-direct-prediction-spec.md`

**位置づけ**: 隔離実験。本番 `pure_rank/models/`, `pure_rank/config/train_config.json`,
`evaluation/reports/`, `prob_fusion/data/`, `pure_rank/data/02_features/*.parquet`
には一切触れない。`evaluation/reports/gate_summary.json` の合否判定にはこの実験結果を
反映しない。特徴量は本番 v39_course_slim をそのまま使い、追加・削除しない。

## 特徴量に市場情報を使わない（プロジェクト憲法遵守）

- odds / popularity / ninki / market_log_odds / init_score は特徴量に一切使わない。
- 比較対象 (a) Stern 逆算・(b) Harville 逆算のみ、既存承認済みコードパス
  （`prob_fusion/src/*`, `evaluation/odds_loader.py`）経由でオッズを使う
  （L2/betting 層で許容されている用途。L1 特徴量には入れない）。

検証コマンド:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/place_direct/ --include="*.py"
```

## 実行手順

```bash
# 0. テスト（先行して書いた §8 のテスト。実装前に RED を確認済み）
python -m pytest pure_rank/experiments/place_direct/tests/ -v

# 1. データセット構築（features_v39_course_slim.parquet 読込 + target付与 + fold2分割）
python pure_rank/experiments/place_direct/build_dataset.py

# 2. 5シード binary 学習（train<2023, early stopping=2023, 2024/2025は完全未見）
python pure_rank/experiments/place_direct/train_fold2.py

# 3. TEST(2025+) で4系列の確率を算出・マージ
python pure_rank/experiments/place_direct/export_probs.py

# 4. 評価レポート生成（logloss / Brier / 較正誤差 / ブートストラップCI / 判定）
python pure_rank/experiments/place_direct/evaluate_place.py
```

## ディレクトリ構成

```
pure_rank/experiments/place_direct/
├── README.md
├── config.json              # 実験用ハイパーパラメータ（仕様書§3.2固定値。探索なし）
├── place_lib.py             # 純関数（target / filters / normalize / logloss）
├── build_dataset.py
├── train_fold2.py
├── export_probs.py
├── evaluate_place.py
├── data/                    # train/es/test parquet（実験専用）
├── models/                  # place_direct_seed{42..46}.txt
├── scores/                  # probs_place_direct_fold2_oos.parquet
├── reports/                 # place_direct_comparison.json, train_log.txt
└── tests/                   # §8 のテスト（TDD）
```

## 既知の制約

- **7頭以下レースの複勝ルール**: JRA複勝は5〜7頭レースで2着まで払戻だが、
  本実験は target を top3 に固定して統一する（比較対象 Stern/Harville も
  top3 を返すため公平な比較になる）。5〜7頭レースの確率をそのまま
  複勝 EV 計算に使ってはならない（betting 層への接続は本実験のスコープ外）。
- (a)(b) の p_win は `evaluation/reports/fusion_oos_fold2.json` の formal 値
  （alpha=0.0, beta=1.0343...）を使う。alpha=0 のため実質的に市場確率ベースの
  p_win であり、L2 層の既存承認済み用途（q は統合変数のみ）の範囲内で再利用している。
- ハイパーパラメータ探索は行っていない（本番 LambdaRank と同一複雑度制約で比較するため）。
